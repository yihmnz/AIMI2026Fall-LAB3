"""Separate LoRA fine-tuning, audited feature regeneration, and unchanged concat fusion."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import hashlib
import json
import random
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from sklearn.metrics import f1_score, classification_report

from lora_models import Encoder, file_hash
from lora_layers import inject, adapter_state, restore_adapter, checkpoint_blocks
from lora_validation import CLASSES, BRANCHES, audit, curate, digest, read_frame, checked_cache, save_cache


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def audit_args(args):
    return audit(args.data,args.patient_map,getattr(args,'strict_split_checks',False))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_for(args):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for model training/extraction; audit is CPU-only')
    device = torch.device('cuda')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This runner requires BF16 support (RTX 5070 Ti)')
    # Disable the native MHA inference shortcut for consistent parametrized paths.
    torch.backends.mha.set_fastpath_enabled(False)
    return device


def autocast(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()


class Rows(Dataset):
    def __init__(self,data,split,encoder,augment=False):
        self.data=Path(data); self.split=split
        self.frame=read_frame(data,split)
        self.transform=encoder.transform
        self.rgb=encoder.branch=='chexfound'
        self.augment=T.Compose([T.RandomRotation(5,fill=0),
                                T.RandomApply([T.ColorJitter(brightness=.06,contrast=.06)],p=.5)]) if augment else None
        self.targets=self.frame[CLASSES].to_numpy().argmax(1) if split!='test' else np.full(len(self.frame),-1)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self,i):
        with Image.open(self.data/f'{self.split}_images'/str(self.frame.iloc[i].new_filename)) as image:
            image=image.convert('RGB' if self.rgb else 'L')
            if self.augment is not None:
                image=self.augment(image)
            x=self.transform(image)
        if not torch.isfinite(x).all():
            raise ValueError(f'Nonfinite transformed image at {self.split}/{i}; check constant-valued images')
        return x,int(self.targets[i])


def model_options(args):
    return {k:str(getattr(args,k)) if isinstance(getattr(args,k),Path) else getattr(args,k)
            for k in ('size','rad_model','rad_revision','chexfound_repo','chexfound_config',
                      'chexfound_weights','chexzero_repo','chexzero_weights') if getattr(args,k,None) is not None}


def build(args, options=None, lora=None):
    encoder=Encoder(args.branch,options or model_options(args))
    settings=lora or {'rank':args.rank,'alpha':args.alpha,'last_blocks':args.last_blocks}
    targets=inject(encoder.net,args.branch,**settings)
    if not getattr(args,'no_checkpointing',False):
        checkpoint_blocks(encoder.net,args.branch)
    return encoder,settings,targets


def trainable(encoder,head):
    return [p for p in encoder.parameters() if p.requires_grad]+list(head.parameters())


def evaluate(encoder,head,loader,device):
    encoder.eval(); head.eval()
    probabilities=[]; labels=[]; loss_sum=0.
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device)
            with autocast(device):
                logits=head(encoder(x))
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite validation logits')
            loss_sum+=F.cross_entropy(logits.float(),y,reduction='sum').item()
            probabilities.append(logits.float().softmax(-1).cpu().numpy()); labels.extend(y.cpu().tolist())
    p=np.concatenate(probabilities); y=np.asarray(labels)
    return float(f1_score(y,p.argmax(1),labels=[0,1,2,3],average='macro',zero_division=0)),loss_sum/len(y),p,y


def save_checkpoint(path,payload):
    temporary=path.with_suffix('.tmp.pt')
    torch.save(payload,temporary)
    temporary.replace(path)


def run_train(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory; existing experiments are never overwritten')
    report=audit_args(args)
    print(report['patient_status'],flush=True)
    device=device_for(args); seed_all(args.seed)
    encoder,settings,targets=build(args)
    encoder.to(device)
    head=nn.Sequential(nn.LayerNorm(encoder.dim),nn.Dropout(.25),nn.Linear(encoder.dim,4)).to(device)
    training=Rows(args.data,'train',encoder,augment=not args.no_augment)
    validation=Rows(args.data,'val',encoder)
    counts=np.bincount(training.targets,minlength=4).astype(float)
    sampler=None
    if args.sampler=='balanced':
        sampler=WeightedRandomSampler(torch.tensor(1/np.sqrt(counts[training.targets]),dtype=torch.double),len(training),replacement=True)
    loader=DataLoader(training,batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler,
                      num_workers=args.workers,pin_memory=True)
    val_loader=DataLoader(validation,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    weights=None
    if args.loss=='weighted_ce':
        w=np.sqrt(counts.sum()/counts); w=w/w.mean()
        weights=torch.tensor(w,dtype=torch.float32,device=device)
    opt=torch.optim.AdamW([{'params':[p for p in encoder.parameters() if p.requires_grad],'lr':args.lr},
                            {'params':head.parameters(),'lr':args.head_lr}],weight_decay=args.weight_decay)
    args.output.mkdir(parents=True,exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    write_json(args.output/'audit.json',report)
    write_json(args.output/'run_config.json',config)
    metadata={'schema':1,'kind':'cxr-lora','branch':args.branch,'classes':CLASSES,'lora':settings,'targets':targets,
              'base_identity':encoder.identity,'model_options':encoder.options,'audit':report,'seed':args.seed,
              'head_dim':encoder.dim,'config':config}
    best=-1.; stale=0; history=[]
    print(f'{args.branch}: {encoder.size}px, {encoder.dim} dims, {sum(p.numel() for p in trainable(encoder,head)):,} trainable',flush=True)
    for epoch in range(1,args.epochs+1):
        # Frozen base stays in eval mode; gradients still flow through its operations.
        encoder.eval(); head.train(); opt.zero_grad(set_to_none=True)
        total=0.; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for step,(x,y) in enumerate(loader):
            group_start=(step//args.accum_steps)*args.accum_steps*args.batch_size
            group_count=min(args.accum_steps*args.batch_size,len(training)-group_start)
            x=x.to(device); y=y.to(device)
            with autocast(device):
                logits=head(encoder(x))
            loss=F.cross_entropy(logits.float(),y,weight=weights,reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; no checkpoint saved for this epoch')
            (loss/group_count).backward(); total+=loss.item()
            if (step+1)%args.accum_steps==0 or step+1==len(loader):
                nn.utils.clip_grad_norm_(trainable(encoder,head),1.,error_if_nonfinite=True)
                opt.step(); opt.zero_grad(set_to_none=True)
            if (step+1)%100==0:
                print(f'epoch {epoch}: {step+1}/{len(loader)} batches',flush=True)
        score,vl,probs,y=evaluate(encoder,head,val_loader,device)
        row={'epoch':epoch,'train_loss':total/len(training),'val_loss':vl,'val_macro_f1':score,
             'seconds':time.perf_counter()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        history.append(row); pd.DataFrame(history).to_csv(args.output/'history.csv',index=False)
        cp=dict(metadata,adapter=adapter_state(encoder.net),head={k:v.detach().cpu() for k,v in head.state_dict().items()},
                epoch=epoch,val_macro_f1=score,history=history)
        if score>best+1e-8:
            best=score; stale=0
            save_checkpoint(args.output/'best.pt',dict(cp,selection='full-validation-macro-f1'))
            write_json(args.output/'best_val_report.json',classification_report(y,probs.argmax(1),labels=[0,1,2,3],
                        target_names=CLASSES,output_dict=True,zero_division=0))
            pd.DataFrame(probs,columns=[f'p_{c}' for c in CLASSES]).assign(
                new_filename=validation.frame.new_filename.astype(str).tolist(),true=y).to_csv(args.output/'val_probabilities.csv',index=False)
        else:
            stale+=1
        save_checkpoint(args.output/'last.pt',dict(cp,optimizer=opt.state_dict(),selection='last-epoch'))
        print(json.dumps(row),flush=True)
        if stale>=args.patience:
            break
    print(f'Best validation macro-F1: {best:.6f}. Adapter: {args.output / "best.pt"}',flush=True)


def load_for_extract(args,report):
    cp=torch.load(args.adapter,map_location='cpu',weights_only=True)
    if cp.get('kind')!='cxr-lora' or cp.get('classes')!=CLASSES or cp.get('branch')!=args.branch:
        raise ValueError('Wrong adapter kind/classes/branch')
    if cp.get('selection')!='full-validation-macro-f1':
        raise ValueError('Extract from a best.pt selected on full validation')
    if cp['audit']['id']!=report['id']:
        raise ValueError('Dataset or patient mapping changed since LoRA training')
    if args.size is not None and args.size!=cp['base_identity']['size']:
        raise ValueError('Extraction size must match the trained adapter; omit --size to use checkpoint settings')
    encoder,_,targets=build(args,cp['model_options'],cp['lora'])
    if digest(encoder.identity)!=digest(cp['base_identity']) or targets!=cp['targets']:
        raise ValueError('Base weights, source, preprocessing, or attention targets changed')
    restore_adapter(encoder.net,cp['adapter'])
    return encoder,cp


def run_extract(args):
    report=audit_args(args)
    device=device_for(args); seed_all(args.seed)
    if args.frozen:
        encoder=Encoder(args.branch,model_options(args))
        adapter_hash=None; selection='frozen-official-base'; settings=None
    else:
        encoder,cp=load_for_extract(args,report)
        adapter_hash=file_hash(args.adapter); selection=cp['selection']; settings=cp['lora']
    encoder.to(device).eval()
    identity=digest({'adapter_sha256':adapter_hash,'base':encoder.identity,'lora':settings})
    for split in args.splits:
        dest=args.cache/args.branch/f'{split}.npz'
        if dest.exists():
            raise FileExistsError(f'Use a fresh cache directory: {dest}')
        ds=Rows(args.data,split,encoder)
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
        parts=[]
        with torch.no_grad():
            for step,(x,_) in enumerate(loader):
                with autocast(device):
                    z=encoder(x.to(device))
                if z.ndim!=2 or z.shape[1]!=encoder.dim:
                    raise ValueError('Unexpected feature dimension')
                parts.append(z.float().cpu().numpy())
                if (step+1)%100==0:
                    print(f'{args.branch} {split}: {step+1}/{len(loader)} batches',flush=True)
        y=ds.targets.astype(np.int64) if split!='test' else np.empty(0,dtype=np.int64)
        save_cache(dest,np.concatenate(parts),y,ds.frame.new_filename.astype(str).to_numpy(),args.branch,encoder.size,
                   {'encoder_id':identity,'audit_id':report['id'],'split_fingerprint':report['splits'][split],'adapter_sha256':adapter_hash,
                    'base_identity':encoder.identity,'selection':selection})
        print(f'Saved {dest}',flush=True)


def run_dense(args):
    """Rebuild the supervised DenseNet on the audited split, then extract its features."""
    report=audit_args(args)
    if args.checkpoint is None:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError('Use a fresh DenseNet output directory')
        subprocess.run([sys.executable,'train_lab3.py','--data',str(args.data),'--model','xrv_densenet',
                        '--output',str(args.output),'--epochs',str(args.epochs),'--patience',str(args.patience),
                        '--batch-size',str(args.batch_size),'--finetune-encoder','--loss','ce','--sampler','balanced',
                        '--seed',str(args.seed),'--workers','0'],check=True)
        args.checkpoint=args.output/'best.pt'
        cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        cp['lora_audit_id']=report['id']
        save_checkpoint(args.checkpoint,cp)
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if cp.get('lora_audit_id')!=report['id']:
        raise ValueError('DenseNet checkpoint not trained through this runner on the same audited split; old supervised weights may contain leakage')
    import fourbranch as original
    device=device_for(args)
    opts=argparse.Namespace(xrv_checkpoint=args.checkpoint)
    encoder=original.FeatureEncoder('densenet',opts).to(device).eval()
    identity=digest({'checkpoint':file_hash(args.checkpoint),'preprocessing':'fourbranch-densenet-224-v1'})
    for split in ('train','val','test'):
        df=read_frame(args.data,split)
        ds=original.ImageRows(args.data/f'{split}_images',df,'densenet',224)
        loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
        parts=[]
        with torch.no_grad():
            for step,x in enumerate(loader):
                with autocast(device):
                    z=encoder(x.to(device))
                parts.append(z.float().cpu().numpy())
                if (step+1)%100==0:
                    print(f'densenet {split}: {step+1}/{len(loader)} batches',flush=True)
        y=df[CLASSES].to_numpy().argmax(1).astype(np.int64) if split!='test' else np.empty(0,dtype=np.int64)
        save_cache(args.cache/'densenet'/f'{split}.npz',np.concatenate(parts),y,df.new_filename.astype(str).to_numpy(),'densenet',224,
                   {'encoder_id':identity,'audit_id':report['id'],'checkpoint_sha256':file_hash(args.checkpoint),'split_fingerprint':report['splits'][split]})
    print('DenseNet cache rebuilt from a checkpoint trained on the audited split.')


def cache_identities(args,report,splits=('train','val','test')):
    identities={}; dims={}
    for split in splits:
        for branch in BRANCHES:
            x,_,_,m=checked_cache(args.cache,branch,split,args.data,True,report)
            if branch in identities and (identities[branch]!=m['encoder_id'] or dims[branch]!=x.shape[1]):
                raise ValueError(f'Mixed encoder versions/dimensions across splits: {branch}')
            identities[branch]=m['encoder_id']; dims[branch]=x.shape[1]
    return identities,dims


def reuse_dense(args):
    """Reuse original DenseNet embeddings, preserving every value and CSV row."""
    report=audit_args(args)
    sources={s:file_hash(args.source_cache/'densenet'/f'{s}.npz') for s in ('train','val','test')}
    identity=digest({'kind':'existing-densenet-cache','sources':sources})
    checked=[]
    for split in ('train','val','test'):
        x,y,names,_=checked_cache(args.source_cache,'densenet',split,args.data)
        with np.load(args.source_cache/'densenet'/f'{split}.npz',allow_pickle=False) as d:
            size=int(d['size'])
        if x.shape[1]!=1024:
            raise ValueError('Expected existing DenseNet embeddings with 1024 dimensions')
        destination=args.cache/'densenet'/f'{split}.npz'
        if destination.exists():
            raise FileExistsError(destination)
        checked.append((split,x,y,names,size,destination))
    for split,x,y,names,size,destination in checked:
        save_cache(destination,x,y,names,'densenet',size,
                   {'encoder_id':identity,'audit_id':report['id'],'split_fingerprint':report['splits'][split],
                    'existing_cache_sha256':sources,'selection':'existing-project-densenet'})
        print(f'Reused existing DenseNet features: {destination}',flush=True)


def fuse(args):
    report=audit_args(args)
    identities,dims=cache_identities(args,report)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new fusion output directory')
    import fourbranch as original
    args.branches=BRANCHES; args.fusion='concat'; args.aux_weight=0.; args.project_dim=128; args.gate_hidden=64
    original.train(args)
    cp=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
    cp.update(lora_cache_identities=identities,lora_audit_id=report['id'],classes=CLASSES)
    save_checkpoint(args.output/'best.pt',cp)
    write_json(args.output/'provenance.json',{'audit':report,'cache_identities':identities,'dims':dims,
                                            'fusion':'unchanged fusion_heads.ConcatHead','aux_weight':0.})


def predict(args):
    report=audit_args(args)
    identities,_=cache_identities(args,report,splits=(args.split,))
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if cp.get('lora_cache_identities')!=identities or cp.get('lora_audit_id')!=report['id']:
        raise ValueError('Fusion checkpoint does not belong to these features/dataset')
    if args.output.exists() or args.output.with_name(args.output.stem+'_probabilities.csv').exists():
        raise FileExistsError('Prediction output already exists')
    import fourbranch as original
    original.predict(args)


def smoke(args):
    """Real image/GPU backward and save-reload probe, never produces a training adapter."""
    device=device_for(args); seed_all(args.seed)
    encoder=Encoder(args.branch,model_options(args)).to(device)
    ds=Rows(args.data,'train',encoder)
    x=torch.stack([ds[i][0] for i in range(args.batch_size)]).to(device)
    y=torch.tensor([ds[i][1] for i in range(args.batch_size)],device=device)
    encoder.eval(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),autocast(device):
        baseline=encoder(x)
    settings={'rank':args.rank,'alpha':args.alpha,'last_blocks':args.last_blocks}
    targets=inject(encoder.net,args.branch,**settings)
    if not args.no_checkpointing:
        checkpoint_blocks(encoder.net,args.branch)
    with torch.no_grad(),autocast(device):
        zero=encoder(x)
    torch.testing.assert_close(baseline,zero,rtol=1e-3,atol=1e-3)
    initial=adapter_state(encoder.net)
    frozen={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
            for n,p in encoder.net.named_parameters() if not p.requires_grad}
    head=nn.Linear(encoder.dim,4).to(device)
    opt=torch.optim.AdamW(trainable(encoder,head),lr=1e-3)
    losses=[]
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        with autocast(device):
            loss=F.cross_entropy(head(encoder(x)).float(),y)
        loss.backward()
        adapter_grads={n:p.grad for n,p in encoder.net.named_parameters() if p.requires_grad}
        if any(g is None or not torch.isfinite(g).all() for g in adapter_grads.values()):
            raise RuntimeError('Missing or nonfinite LoRA gradient')
        if not any(g.abs().sum()>0 for g in adapter_grads.values()):
            raise RuntimeError('All adapter gradients are zero')
        opt.step(); losses.append(loss.item())
    trained=adapter_state(encoder.net)
    if not any(not torch.equal(initial[n],trained[n]) for n in initial):
        raise RuntimeError('Adapters did not update')
    if any(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()!=frozen[n]
           for n,p in encoder.net.named_parameters() if not p.requires_grad):
        raise RuntimeError('Frozen base changed')
    with torch.no_grad(),autocast(device):
        adapted=encoder(x)
    if torch.equal(adapted,zero):
        raise RuntimeError('Updated adapters do not affect embeddings')
    restore_adapter(encoder.net,initial)
    with torch.no_grad(),autocast(device):
        reset=encoder(x)
    torch.testing.assert_close(zero,reset,rtol=1e-3,atol=1e-3)
    import io
    buffer=io.BytesIO(); torch.save(trained,buffer); buffer.seek(0)
    restore_adapter(encoder.net,torch.load(buffer,weights_only=True))
    with torch.no_grad(),autocast(device):
        restored=encoder(x)
    torch.testing.assert_close(adapted,restored,rtol=1e-3,atol=1e-3)
    result={'branch':args.branch,'size':encoder.size,'dim':encoder.dim,'targets':targets,'losses':losses,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'torch':str(torch.__version__),
            'gpu':torch.cuda.get_device_name(),'zero_adapter_equivalence':True,'base_unchanged':True,
            'adapter_update_affects_embeddings':True,'adapter_roundtrip':True,'base_identity':encoder.identity}
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/f'smoke_{args.branch}.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for command in ('audit','curate','train','extract','dense','reuse-dense','fuse','predict','smoke'):
        q=sub.add_parser(command)
        q.add_argument('--data',type=Path,default=Path('data'))
        q.add_argument('--patient-map',type=Path)
        q.add_argument('--strict-split-checks',action='store_true',help='Opt in to blocking cross-split duplicate images; default preserves original splits')
        q.add_argument('--cache',type=Path,default=Path('cache_embeddings_lora'))
        q.add_argument('--seed',type=int,default=42)
        if command in ('train','extract','smoke'):
            q.add_argument('--branch',choices=BRANCHES[1:],required=True)
            q.add_argument('--batch-size',type=int,default=1)
            q.add_argument('--workers',type=int,default=0)
            q.add_argument('--no-checkpointing',action='store_true')
            q.add_argument('--rank',type=int,default=8); q.add_argument('--alpha',type=float,default=16.)
            q.add_argument('--last-blocks',type=int,default=4,help='0 adapts all blocks')
            q.add_argument('--size',type=int,help='Default RAD=224, CheXFound=512; CheXzero inferred from weights')
            q.add_argument('--rad-model',default='microsoft/rad-dino')
            q.add_argument('--rad-revision',default='110cbc18d5133582e320b43d53bf5c44e410c936')
            q.add_argument('--chexfound-repo',type=Path,default=Path('external/CheXFound'))
            q.add_argument('--chexfound-config',type=Path,default=Path('pretrain_weight/chexfound/config.yaml'))
            q.add_argument('--chexfound-weights',type=Path,default=Path('pretrain_weight/chexfound/teacher_checkpoint.pth'))
            q.add_argument('--chexzero-repo',type=Path,default=Path('external/CheXzero'))
            q.add_argument('--chexzero-weights',type=Path,default=Path('pretrain_weight/chexzero/best_64_5e-05_original_22000_0.864.pt'))
        if command in ('train','fuse','dense'):
            q.add_argument('--output',type=Path,required=True)
            q.add_argument('--epochs',type=int,default=20 if command=='train' else 60 if command=='dense' else 80)
            q.add_argument('--patience',type=int,default=5 if command=='train' else 15)
            q.add_argument('--lr',type=float,default=1e-4 if command=='train' else 1e-3)
            q.add_argument('--weight-decay',type=float,default=1e-4)
            q.add_argument('--sampler',choices=['natural','balanced'],default='balanced')
            q.add_argument('--loss',choices=['ce','weighted_ce'],default='ce')
        if command=='train':
            q.add_argument('--head-lr',type=float,default=1e-3)
            q.add_argument('--accum-steps',type=int,default=16)
            q.add_argument('--no-augment',action='store_true')
        elif command=='extract':
            mode=q.add_mutually_exclusive_group(required=True)
            mode.add_argument('--adapter',type=Path)
            mode.add_argument('--frozen',action='store_true',help='Official frozen baseline for a fair comparison')
            q.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test'])
        elif command=='dense':
            q.add_argument('--checkpoint',type=Path,help='Only checkpoints previously trained and bound by this runner')
            q.add_argument('--batch-size',type=int,default=8)
        elif command=='reuse-dense':
            q.add_argument('--source-cache',type=Path,default=Path('cache_embeddings'))
        elif command=='fuse':
            q.add_argument('--batch-size',type=int,default=64)
            q.add_argument('--hidden',type=int,default=256); q.add_argument('--dropout',type=float,default=.35)
        elif command=='predict':
            q.add_argument('--checkpoint',type=Path,required=True)
            q.add_argument('--split',choices=['val','test'],default='test')
            q.add_argument('--output',type=Path,required=True)
        elif command in ('audit','smoke'):
            q.add_argument('--output',type=Path,default=Path('outputs/lora_checks'))
            if command=='audit':
                q.add_argument('--check-cache',action='store_true',help='Also require all existing branch caches to match CSV')
        elif command=='curate':
            q.add_argument('--output',type=Path,default=Path('data_lora'))
    args=p.parse_args()
    for key in ('epochs','patience','batch_size','accum_steps','rank'):
        if hasattr(args,key) and getattr(args,key)<1:
            p.error(f'{key} must be positive')
    if hasattr(args,'last_blocks') and args.last_blocks<0:
        p.error('last-blocks must be >= 0')
    if hasattr(args,'size') and args.size is not None and (args.size<32 or args.size% (16 if args.branch=='chexfound' else 14 if args.branch=='raddino' else 1)):
        p.error('size must be >=32 and divisible by the model patch size')
    if hasattr(args,'workers') and args.workers!=0:
        p.error('Use workers=0 on Windows for compatibility with upstream transforms')
    for key in ('lr','head_lr','alpha'):
        if hasattr(args,key) and (not np.isfinite(getattr(args,key)) or getattr(args,key)<=0):
            p.error(f'{key} must be finite and positive')
    if hasattr(args,'weight_decay') and (not np.isfinite(args.weight_decay) or args.weight_decay<0):
        p.error('weight-decay must be finite and nonnegative')
    if hasattr(args,'dropout') and not 0<=args.dropout<1:
        p.error('dropout must be in [0,1)')
    if args.command=='curate':
        print(json.dumps(curate(args.data,args.output),indent=2))
    elif args.command=='audit':
        result=audit_args(args)
        if args.check_cache:
            for split in ('train','val','test'):
                for branch in BRANCHES:
                    x,_,_,_=checked_cache(args.cache,branch,split,args.data)
                    print(f'{branch}/{split}: {x.shape}')
        args.output.mkdir(parents=True,exist_ok=True); write_json(args.output/'audit.json',result)
        print(json.dumps(result,indent=2))
    else:
        {'train':run_train,'extract':run_extract,'dense':run_dense,'reuse-dense':reuse_dense,'fuse':fuse,'predict':predict,'smoke':smoke}[args.command](args)


if __name__=='__main__':
    main()
