"""Synthetic integration test; does not touch real caches or experiment outputs."""
from pathlib import Path
import json
import tempfile

import numpy as np
import pandas as pd
import torch

import gated_attention_experiment as ga


def main():
    torch.manual_seed(42)
    model = ga.GatedAttentionHead([5,7,9,11],hidden=16,dropout=0,project_dim=16,gate_hidden=8)
    x = torch.randn(8,32)
    logits, aux = model(x)
    assert logits.shape == (8,4) and aux is None
    gates = model.gate_weights(x)
    assert gates.shape == (8,4) and torch.allclose(gates.sum(1),torch.ones(8))
    assert torch.all(gates >= 0)
    torch.nn.functional.cross_entropy(logits,torch.arange(8)%4).backward()
    for module in [model.attention,model.gate_v,model.gate_u,model.gate_w,*model.projections]:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    with tempfile.TemporaryDirectory(prefix='gated_attention_fixture_', dir=str(Path(__file__).parent)) as temp:
        root=Path(temp);data=root/'data';cache=root/'cache';data.mkdir()
        rng=np.random.RandomState(42)
        for split,n in [('train',40),('val',12),('test',8)]:
            y=np.arange(n)%4;names=np.array([f'{split}_{i}.jpg' for i in range(n)])
            template=pd.DataFrame(np.eye(4,dtype=int)[y] if split!='test' else np.zeros((n,4),dtype=int),columns=ga.fb.CLASSES)
            template.insert(0,'new_filename',names)
            template.to_csv(data/({'train':'train_data.csv','val':'val_data.csv','test':'test_data_sample.csv'}[split]),index=False)
            for branch,d in zip(ga.BRANCHES,[5,7,9,11]):
                folder=cache/branch;folder.mkdir(exist_ok=True,parents=True)
                features=rng.normal(size=(n,d)).astype('float32');features[:,:4]+=np.eye(4)[y]*2
                np.savez(folder/(split+'.npz'),x=features,names=names,y=y if split!='test' else np.array([],dtype=int))
        ga.main(['--cache',str(cache),'--data',str(data),'--out',str(root/'runs'),
                 '--fusion','all','--epochs','2','--patience','2','--batch-size','16',
                 '--hidden','16','--project-dim','16','--gate-hidden','8','--predict-test','--allow-cpu'])
        run=next((root/'runs').iterdir());summary=pd.read_csv(run/'summary.csv')
        assert set(summary.fusion)==set(ga.MODES) and len(summary)==6
        for mode in ga.MODES:
            dest=run/mode
            sub=pd.read_csv(dest/'submission.csv');prob=pd.read_csv(dest/'submission_probabilities.csv')
            assert sub.new_filename.tolist()==[f'test_{i}.jpg' for i in range(8)]
            assert np.allclose(prob[['p_'+c for c in ga.fb.CLASSES]].sum(axis=1),1)
            assert np.array_equal(sub[ga.fb.CLASSES].to_numpy().argmax(1),prob[['p_'+c for c in ga.fb.CLASSES]].to_numpy().argmax(1))
            cp=torch.load(dest/'best.pt',map_location='cpu',weights_only=True)
            net=ga.make_head(mode,cp['feature_dims'],cp['hidden'],cp['dropout'],0,cp['project_dim'],cp['gate_hidden'])
            net.load_state_dict(cp['state_dict'],strict=True);net.eval()
            X,y,names=ga.fb.load_features(cache,ga.BRANCHES,'val')
            p,pred,score=ga.fb.score(net,X,y,torch.device('cpu'))
            saved=pd.read_csv(dest/'val_probabilities.csv')
            assert np.allclose(p,saved[['p_'+c for c in ga.fb.CLASSES]],atol=1e-6)
            assert abs(score-cp['best_macro_f1'])<1e-8
        # Exact order checks reject a corrupt cache before training.
        bad=cache/'densenet'/'val.npz'
        with np.load(bad) as d: payload={k:d[k] for k in d.files}
        payload['names']=payload['names'][::-1]
        np.savez(bad,**payload)
        try: ga.check_split(data,cache,ga.BRANCHES,'val')
        except ValueError: pass
        else: raise AssertionError('Accepted misaligned cache')
    print('PASS: gradients, normalized gates, six training modes, checkpoint reload, validation reproduction, test alignment, corrupt-cache rejection.')


if __name__=='__main__':main()
