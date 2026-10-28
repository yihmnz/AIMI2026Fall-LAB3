"""Synthetic integration checks for ensemble_search_lab3.py (numpy/pandas required)."""
import importlib.util
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name('ensemble_search_lab3.py')
spec = importlib.util.spec_from_file_location('lab3', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def main():
    with tempfile.TemporaryDirectory(prefix='lab3_fixture_', dir=str(Path(__file__).parent)) as tmp:
        root = Path(tmp)
        data, outputs = root/'data', root/'outputs'
        data.mkdir(); outputs.mkdir()
        rng = np.random.RandomState(42)
        n, nt = 80, 12
        y = np.arange(n) % 4
        names = ['v%03d.jpeg' % i for i in range(n)]
        tn = ['t%03d.jpeg' % i for i in range(nt)]
        template = pd.DataFrame(np.eye(4, dtype=int)[y], columns=m.CLASSES)
        template.insert(0, 'new_filename', names)
        template.to_csv(data/'val_data.csv', index=False)
        test = pd.DataFrame(np.zeros((nt,4),dtype=int),columns=m.CLASSES)
        test.insert(0,'new_filename',tn)
        test.to_csv(data/'test_data_sample.csv',index=False)
        combos = [''.join(c) for k in range(1,5) for c in itertools.combinations('1234',k)]
        models = ['f'+c+suffix for suffix in ['', '_lora_s42'] for c in combos]
        for i, model in enumerate(models):
            folder = outputs/model; folder.mkdir()
            p = rng.dirichlet(np.ones(4),n)
            p = .6*p + .4*np.eye(4)[y] if model not in m.DEFAULT_BASE else p
            frame = pd.DataFrame(p, columns=m.PCOLS)
            frame.insert(0,'true', [m.CLASSES[j] for j in y] if i%2 else y)
            frame.insert(0,'new_filename',names)
            frame.iloc[rng.permutation(n)].to_csv(folder/'val_probabilities.csv',index=False)
            tp = rng.dirichlet(np.ones(4),nt)
            frame = pd.DataFrame(tp,columns=m.PCOLS); frame.insert(0,'new_filename',tn)
            frame.iloc[::-1].to_csv(outputs/('submission_'+model+'_probabilities.csv'),index=False)
        # A legacy fusion folder must not enter the pool.
        legacy = outputs/'f24_aux25'; legacy.mkdir()
        template.to_csv(legacy/'val_probabilities.csv',index=False)
        def run(*extra, success=True):
            result = subprocess.run([sys.executable,str(SCRIPT),'--root',str(outputs),'--data',str(data),'--out',str(root/'runs'),*extra],capture_output=True,text=True)
            assert (result.returncode == 0) == success, result.stdout+result.stderr
            runs = sorted((root/'runs').iterdir(),key=lambda p:p.stat().st_mtime_ns)
            return runs[-1]
        out = run('--max-trials','200','--max-rounds','3','--submit-top','2')
        manifest = json.loads((out/'manifest.json').read_text())
        assert manifest['baseline'] == dict.fromkeys(m.DEFAULT_BASE,.2)
        assert len(pd.read_csv(out/'models.csv')) == 30
        assert len(pd.read_csv(out/'pairwise.csv')) == 435
        assert 1 < manifest['evaluated_trials'] <= 200
        assert max(r['round'] for r in manifest['ranking']) >= 2
        assert all(len(r['weights']) <= 8 and np.isclose(sum(r['weights'].values()),1) for r in manifest['ranking'])
        assert all(r['status']=='created' for r in manifest['submissions'])
        rank = manifest['ranking'][0]
        total = sum(pd.read_csv(outputs/('submission_'+model+'_probabilities.csv')).set_index('new_filename').loc[tn,m.PCOLS].to_numpy()*w for model,w in rank['weights'].items())
        actual = pd.read_csv(out/'submission_rank01_probabilities.csv')
        assert actual.new_filename.tolist()==tn
        assert np.allclose(actual[m.PCOLS],total)
        labels = pd.read_csv(out/'submission_rank01.csv')
        assert np.array_equal(labels[m.CLASSES].to_numpy().argmax(1),total.argmax(1))
        for a,b in [('f1','f2'),('f134','f3')]:
            pa,_r,_l=m.probabilities(outputs/a/'val_probabilities.csv', names,y)
            pb,_r,_l=m.probabilities(outputs/b/'val_probabilities.csv', names,y)
            pair=pd.read_csv(out/'pairwise.csv').query('model_a == @a and model_b == @b').iloc[0]
            ea,eb=pa.argmax(1)!=y,pb.argmax(1)!=y
            assert np.isclose(pair.double_fault,(ea & eb).mean())
            assert np.isclose(pair.a_errors_rescued_by_b,(ea & ~eb).sum()/ea.sum())
        # Strict rejection: mismatched IDs, labels, duplicates, NaN, invalid sums.
        source=pd.read_csv(outputs/'f3'/'val_probabilities.csv')
        bad=root/'bad.csv'
        for kind in ['id','label','duplicate','nan','sum']:
            frame=source.copy()
            if kind=='id': frame.loc[0,'new_filename']='unknown'
            if kind=='label':
                frame['true']=frame['true'].astype(object)
                frame.loc[0,'true']='invalid'
            if kind=='duplicate': frame.loc[0,'new_filename']=frame.loc[1,'new_filename']
            if kind=='nan': frame.loc[0,m.PCOLS[0]]=np.nan
            if kind=='sum': frame.loc[0,m.PCOLS]=0
            frame.to_csv(bad,index=False)
            try: m.probabilities(bad,names,y)
            except ValueError: pass
            else: raise AssertionError('accepted '+kind)
        out=run('--max-trials','1')
        assert len(pd.read_csv(out/'ranking.csv'))==1
        # Missing test pairs skip with an explicit report; no fabricated submission.
        (outputs/'submission_f3_probabilities.csv').unlink()
        out=run('--max-trials','1','--submit-top','1')
        assert pd.read_csv(out/'submissions.csv').iloc[0]['status']=='skipped'
        assert not (out/'submission_rank01.csv').exists()
        # Missing baseline fails, and records why in discovery.csv.
        source.iloc[:-1].to_csv(outputs/'f3'/'val_probabilities.csv',index=False)
        run('--max-trials','1',success=False)
    print('PASS: 30 models, 435 pairs, multi-round search, bounds, aligned submissions, invalid inputs, missing pairs/baseline.')

if __name__=='__main__': main()

