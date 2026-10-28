"""LAB3: cached features for four original CXR branches.
Requires separately cloned official CheXFound and CheXzero repositories.
"""
from __future__ import annotations
import argparse, csv, json, random, sys, time
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from fusion_heads import make_head, MODES
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES=['normal','bacteria','virus','COVID-19']
BRANCHES=['densenet','raddino','chexfound','chexzero']

def load_csv(root, split):
    name={'train':'train_data.csv','val':'val_data.csv','test':'test_data_sample.csv'}[split]
    p=next((p for p in [root/name,root.parent/name] if p.exists()),None)
    if p is None: raise FileNotFoundError(name)
    df=pd.read_csv(p,dtype={'new_filename':'string'})
    if df.columns.tolist()!=['new_filename']+CLASSES: raise ValueError(f'Wrong columns: {df.columns.tolist()}')
    if df.new_filename.isna().any() or df.new_filename.duplicated().any(): raise ValueError('Missing/duplicate filenames')
    if split!='test':
        arr=df[CLASSES].to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all() or not (np.isin(arr,[0,1]).all() and np.all(arr.sum(1)==1)): raise ValueError('Invalid one-hot labels')
    return df

def image_transform(branch, size, augment=False):
    # Keep branch-specific preprocessing; no data augmentation for val/test.
    pipeline=[]
    if augment:
        pipeline += [T.RandomRotation(5,fill=0),T.RandomAffine(0,translate=(0.025,0.025)),
                     T.RandomApply([T.ColorJitter(brightness=.06,contrast=.06)],p=.5)]
    pipeline += [T.Resize((size,size),interpolation=InterpolationMode.BICUBIC),T.ToTensor()]
    if branch=='densenet':
        pipeline += [T.Lambda(lambda x:(x*2-1)*1024)]
    elif branch=='raddino':
        from transformers import AutoImageProcessor
        proc=AutoImageProcessor.from_pretrained('microsoft/rad-dino')
        pipeline += [T.Lambda(lambda x:x.repeat(3,1,1)),T.Normalize(proc.image_mean,proc.image_std)]
    elif branch=='chexfound':
        # Use exact official classification eval transform from CheXFound repo instead (see extractor).
        pass
    elif branch=='chexzero':
        pipeline += [T.Lambda(lambda x:x.repeat(3,1,1)),
                     T.Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))]
    return T.Compose(pipeline)

def load_chexfound_windows(config_path, checkpoint_path):
    import torch
    from omegaconf import OmegaConf
    from chexfound.models import build_model_from_cfg
    from chexfound.utils.utils import load_pretrained_weights

    # 不呼叫 setup()，避免初始化 NCCL distributed
    cfg = OmegaConf.load(str(config_path))

    # 官方建立 teacher backbone 的方式
    model, _ = build_model_from_cfg(
        cfg,
        only_teacher=True
    )

    # 使用官方權重轉換與載入函式
    load_pretrained_weights(
        model,
        str(checkpoint_path),
        "teacher"
    )

    model.eval()
    model.requires_grad_(False)

    print("CheXFound pretrained encoder loaded.")
    return model

class ImageRows(torch.utils.data.Dataset):
    def __init__(self,root,df,branch,size,augment=False,chexfound_transform=None):
        self.root=root;self.df=df;self.branch=branch
        self.transform=chexfound_transform or image_transform(branch,size,augment)
    def __len__(self):return len(self.df)
    def __getitem__(self,i):
        name=str(self.df.iloc[i].new_filename)
        if not (self.root/name).is_file():raise FileNotFoundError(self.root/name)
        with Image.open(self.root/name) as img: img=img.convert('RGB' if self.branch=='chexfound' else 'L');x=self.transform(img)
        return x

class FeatureEncoder(nn.Module):
    def __init__(self,branch,args):
        super().__init__();self.branch=branch
        if branch=='densenet':
            from lab3_common import create_model
            # Reuse exact original architecture and checkpoint. Verify compatibility strictly.
            cp=torch.load(args.xrv_checkpoint,map_location='cpu',weights_only=True)
            if cp['config']['model']!='xrv_densenet':raise ValueError('DenseNet checkpoint is not xrv_densenet')
            net=create_model(cp['config'],pretrained=False)
            net.load_state_dict(cp['state_dict'],strict=True)
            self.net=net.encoder
        elif branch=='raddino':
            from lab3_common import create_model
            cp=torch.load(args.rad_checkpoint,map_location='cpu',weights_only=True)
            if cp['config']['model']!='rad_dino':raise ValueError('RAD checkpoint is not rad_dino')
            net=create_model(cp['config'],pretrained=False)
            net.load_state_dict(cp['state_dict'],strict=True)
            self.net=net.encoder
        elif branch=='chexfound':
            repo=Path(args.chexfound_repo).resolve()
            if not (repo/'chexfound'/'eval'/'setup.py').exists():raise FileNotFoundError(f'Clone RPIDIAL/CheXFound at {repo}')
            sys.path.insert(0,str(repo))
            from argparse import Namespace
            opts=Namespace(config_file=str(args.chexfound_config),pretrained_weights=None,
                           output_dir=str(Path(args.cache).resolve()),opts=[],image_size=args.chexfound_size,
                           patch_size=16,n_register_tokens=4,n_last_blocks=4,return_class_token=True,
                           num_classes=40,num_heads=8)
            net = load_chexfound_windows(
                config_path=args.chexfound_config,
                checkpoint_path=args.chexfound_weights
            )
            raw=torch.load(args.chexfound_weights,map_location='cpu',weights_only=False)
            if 'teacher' not in raw:raise RuntimeError('Expected CheXFound checkpoint["teacher"]')
            mapped={}
            for key,value in raw['teacher'].items():
                if key.startswith('backbone'):
                    parts=key.split('.')
                    name='.'.join([parts[1],*parts[3:]]) if 'blocks' in key else '.'.join(parts[1:])
                    mapped[name]=value
            if not mapped:raise RuntimeError('No CheXFound backbone tensors found')
            result=net.load_state_dict(mapped,strict=False)
            matched=len(mapped)-len(result.unexpected_keys)
            print('CheXFound tensors matched:',matched,'unexpected:',len(result.unexpected_keys),'missing:',len(result.missing_keys),flush=True)
            if matched < 250 or result.unexpected_keys:
                raise RuntimeError('CheXFound loading incompatible; refusing to extract random/partial features')
            self.net=net
        elif branch=='chexzero':
            repo=Path(args.chexzero_repo).resolve()
            if not (repo/'model.py').exists(): raise FileNotFoundError(f'Clone rajpurkarlab/CheXzero at {repo}')
            sys.path.insert(0,str(repo))
            from model import CLIP
            raw=torch.load(args.chexzero_weights,map_location='cpu',weights_only=False)
            state=raw.get('state_dict',raw) if isinstance(raw,dict) else raw
            if not isinstance(state,dict):
                raise TypeError('CheXzero checkpoint must contain a state_dict')
            if any(k.startswith('module.') for k in state):
                state={k.removeprefix('module.'):v for k,v in state.items()}
            # Infer CLIP architecture from its checkpoint instead of assuming ViT-B/16.
            # The supplied CheXzero weights use ViT-B/32 (224 px, 512-d output).
            required=['visual.conv1.weight','visual.positional_embedding',
                      'text_projection','positional_embedding','token_embedding.weight',
                      'ln_final.weight']
            missing=[k for k in required if k not in state]
            if missing: raise KeyError(f'Unsupported CheXzero checkpoint; missing: {missing}')
            width,channels,patch_h,patch_w=state['visual.conv1.weight'].shape
            if channels!=3 or patch_h!=patch_w: raise ValueError('Invalid visual patch convolution')
            n_pos=int(state['visual.positional_embedding'].shape[0]);grid=round((n_pos-1)**0.5)
            if grid*grid+1!=n_pos: raise ValueError('Visual positional embeddings are not a square grid')
            resolution=grid*patch_h
            embed_dim=int(state['text_projection'].shape[1])
            vision_layers=len([k for k in state if k.startswith('visual.transformer.resblocks.') and k.endswith('.attn.in_proj_weight')])
            text_layers=len({k.split('.')[2] for k in state if k.startswith('transformer.resblocks.')})
            text_width=int(state['ln_final.weight'].shape[0])
            net=CLIP(embed_dim=embed_dim,image_resolution=resolution,
                     vision_layers=vision_layers,vision_width=width,vision_patch_size=patch_h,
                     context_length=int(state['positional_embedding'].shape[0]),
                     vocab_size=int(state['token_embedding.weight'].shape[0]),
                     transformer_width=text_width,transformer_heads=text_width//64,
                     transformer_layers=text_layers)
            # Metadata values are not trainable parameters and may occur in CLIP exports.
            for key in ['input_resolution','context_length','vocab_size']:
                state.pop(key,None)
            net.load_state_dict(state,strict=True)
            self.input_size=resolution
            print(f'CheXzero loaded strictly: ViT-{vision_layers} layers, '
                  f'patch {patch_h}, input {resolution}px, embedding {embed_dim}',flush=True)
            self.net=net
        else:raise ValueError(branch)
        self.net.eval()
        for p in self.net.parameters():p.requires_grad_(False)
    def forward(self,x):
        if self.branch=='chexfound':
            z=self.net.get_intermediate_layers(x,n=4,return_class_token=True)
            # CLS token from final block, consistent shape [B, 1024].
            return z[-1][1]
        if self.branch=='chexzero':return self.net.encode_image(x).float()
        return self.net(x)

def extract(args):
    branch=args.branch
    if branch not in BRANCHES:raise ValueError(branch)
    size={'densenet':224,'raddino':224,'chexfound':args.chexfound_size,'chexzero':224}[branch]
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {branch} on {device}...',flush=True)
    model=FeatureEncoder(branch,args).to(device).eval()
    if branch=='chexzero': size=model.input_size
    tr=None
    if branch=='chexfound':
        from chexfound.data.transforms import make_classification_eval_transform
        tr=make_classification_eval_transform(resize_size=size,crop_size=size)
    output=Path(args.cache)/branch;output.mkdir(parents=True,exist_ok=True)
    for split in args.splits:
        df=load_csv(args.data,split)
        views=args.train_views if split=='train' else 1
        feats=[];names=[];labels=[]
        for view in range(views):
            ds=ImageRows(args.data/f'{split}_images',df,branch,size,
                         augment=(split=='train' and view>0 and branch!='chexfound'),chexfound_transform=tr)
            dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
            chunks=[]
            for batch in tqdm(dl,desc=f'{branch} {split} view {view+1}/{views}'):
                batch=batch.to(device)
                with torch.inference_mode():
                    with (torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' and torch.cuda.is_bf16_supported() and branch!='chexzero' else nullcontext()):
                        f=model(batch)
                if f.ndim!=2:raise RuntimeError(f'Invalid embedding dimension {f.shape}')
                chunks.append(f.float().cpu().numpy())
            feats.append(np.concatenate(chunks));names.extend(df.new_filename.astype(str).tolist())
            if split!='test':labels.extend(df[CLASSES].to_numpy().argmax(1).tolist())
        X=np.concatenate(feats).astype(np.float32)
        if not np.isfinite(X).all():raise ValueError('Nonfinite embeddings')
        np.savez_compressed(output/f'{split}.npz',x=X,names=np.array(names),
                            y=np.array(labels,dtype=np.int64),branch=branch,size=size)
        print(f'Saved {output}/{split}.npz : {X.shape}',flush=True)
    print('Important: these are frozen embeddings; view=0 uses unaugmented inputs.')

def feature_dims(cache,branches,split="train"):
    dims=[]
    for b in branches:
        with np.load(cache/b/f"{split}.npz",allow_pickle=False) as d:
            dims.append(int(d["x"].shape[1]))
    return dims

def load_features(cache,branches,split):
    xs=[];reference=None;targets=None
    for b in branches:
        path=cache/b/f'{split}.npz'
        if not path.exists():raise FileNotFoundError(f'{path}; run extract --branch {b} --splits {split}')
        with np.load(path,allow_pickle=False) as d:
            names=d['names'].astype(str);x=d['x'].astype(np.float32);y=d['y'].astype(np.int64)
        if x.ndim!=2 or len(x)!=len(names) or not np.isfinite(x).all():
            raise ValueError(f'Invalid/nonfinite features: {path}')
        if split=='test' and len(y):raise ValueError(f'Test cache must have no labels: {path}')
        if split!='test' and (len(y)!=len(x) or not np.isin(y,[0,1,2,3]).all()):
            raise ValueError(f'Invalid labels: {path}')
        if reference is None:reference=names;targets=y
        elif not np.array_equal(names,reference):raise ValueError(f'Inconsistent image order / views for {b} {split}')
        elif not np.array_equal(y,targets):raise ValueError(f'Inconsistent labels for {b} {split}')
        xs.append(x)
    return np.concatenate(xs,axis=1),targets,reference

class Head(nn.Module):
    def __init__(self,dim,dropout,hidden,aux):
        super().__init__();self.body=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(dropout),nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.classifier=nn.Linear(hidden,4);self.aux=nn.Linear(hidden,1) if aux>0 else None
    def forward(self,x):
        h=self.body(x);return self.classifier(h),self.aux(h).flatten() if self.aux is not None else None

def score(model,X,y,device,batch=512):
    model.eval();parts=[]
    with torch.inference_mode():
        for i in range(0,len(X),batch):
            p,_=model(torch.from_numpy(X[i:i+batch]).to(device))
            parts.append(p.softmax(-1).cpu().numpy())
    probs=np.concatenate(parts);pred=probs.argmax(1)
    return probs,pred,float(f1_score(y,pred,average='macro',labels=[0,1,2,3],zero_division=0)) if len(y) else float('nan')

def write_eval(out,names,y,probs,pred):
    report=classification_report(y,pred,labels=[0,1,2,3],target_names=CLASSES,output_dict=True,zero_division=0)
    (out/'best_val_report.json').write_text(json.dumps(report,indent=2))
    cm=confusion_matrix(y,pred,labels=[0,1,2,3]);np.savetxt(out/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
    fig,ax=plt.subplots(figsize=(6,5));ax.imshow(cm,cmap='Blues')
    for (i,j),val in np.ndenumerate(cm):ax.text(j,i,str(val),ha='center',va='center')
    ax.set_xticks(range(4),CLASSES,rotation=25);ax.set_yticks(range(4),CLASSES);ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
    fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    d=pd.DataFrame(probs,columns=[f'p_{k}' for k in CLASSES]);d.insert(0,'new_filename',names);d.insert(1,'true',np.array(CLASSES)[y]);d.insert(2,'pred',np.array(CLASSES)[pred]);d.to_csv(out/'val_probabilities.csv',index=False)

def write_gate_analysis(model,X,y,names,branches,device,out):
    if not hasattr(model,'gate_weights'):
        return
    model.eval();pieces=[]
    with torch.inference_mode():
        for i in range(0,len(X),512):
            values=model.gate_weights(torch.from_numpy(X[i:i+512]).to(device))
            if values is None:return
            pieces.append(values.cpu().numpy())
    weights=np.concatenate(pieces)
    per_image=pd.DataFrame(weights,columns=[f'gate_{b}' for b in branches])
    per_image.insert(0,'new_filename',names)
    per_image.insert(1,'true',np.array(CLASSES)[y])
    per_image.to_csv(out/'val_gate_weights.csv',index=False)
    means=per_image.groupby('true',sort=False)[[f'gate_{b}' for b in branches]].mean()
    means.loc['all']=weights.mean(axis=0)
    means.to_csv(out/'val_gate_summary.csv')

def train(args):
    branches=args.branches
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X,y,names=load_features(args.cache,branches,'train');V,vy,vnames=load_features(args.cache,branches,'val')
    if X.shape[1]!=V.shape[1]:raise ValueError('Feature dims mismatch')
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    dims=feature_dims(args.cache,branches)
    if dims!=feature_dims(args.cache,branches,'val'):raise ValueError('Per-branch train/val feature dims mismatch')
    model=make_head(args.fusion,dims,args.hidden,args.dropout,args.aux_weight,args.project_dim,args.gate_hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    counts=np.bincount(y,minlength=4).astype(float)
    if np.any(counts==0):raise ValueError('Training split must contain all four classes')
    weights=np.sqrt(counts.sum()/counts);weights=weights/weights.mean()
    lossfn=nn.CrossEntropyLoss(weight=torch.from_numpy(weights).float().to(device) if args.loss=='weighted_ce' else None)
    if args.sampler=='balanced':
        w=1/np.sqrt(counts[y]);sampler=WeightedRandomSampler(torch.from_numpy(w),num_samples=len(y),replacement=True)
    else:sampler=None
    dl=DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y)),batch_size=args.batch_size,shuffle=sampler is None,sampler=sampler)
    best=-1;stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();loss_sum=0;tr_pred=[];tr_true=[]
        for xb,yb in tqdm(dl,desc=f'Epoch {epoch}/{args.epochs}',leave=False):
            xb=xb.to(device);yb=yb.to(device);opt.zero_grad();logits,aux=model(xb)
            loss=lossfn(logits,yb)
            if aux is not None:
                positive=min(12.,np.sqrt((counts.sum()-counts[3])/counts[3]))
                loss=loss+args.aux_weight*F.binary_cross_entropy_with_logits(aux,(yb==3).float(),pos_weight=torch.tensor(positive,device=device))
            loss.backward();opt.step();loss_sum+=loss.item()*len(yb);tr_pred.extend(logits.argmax(1).detach().cpu().tolist());tr_true.extend(yb.cpu().tolist())
        train_f1=f1_score(tr_true,tr_pred,average='macro',labels=[0,1,2,3],zero_division=0)
        _,vp,vf=score(model,V,vy,device)
        with torch.inference_mode():
            val_logits,_=model(torch.from_numpy(V).to(device));val_loss=lossfn(val_logits,torch.from_numpy(vy).to(device)).item()
        trcov=f1_score(np.array(tr_true)==3,np.array(tr_pred)==3,zero_division=0)
        vcov=f1_score(vy==3,vp==3,zero_division=0)
        record=dict(epoch=epoch,train_loss=loss_sum/len(tr_true),val_loss=val_loss,train_macro_f1=train_f1,val_macro_f1=vf,train_covid_f1=trcov,val_covid_f1=vcov)
        history.append(record);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        fig,axs=plt.subplots(1,3,figsize=(15,4))
        for ax,k in zip(axs,['loss','macro_f1','covid_f1']):
            for phase in ['train','val']:ax.plot([r['epoch'] for r in history],[r[phase+'_'+k] for r in history],label=phase)
            ax.set_title(k);ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(out/'curves.png',dpi=150);plt.close(fig)
        print(f'Epoch {epoch:03d} train_loss={record["train_loss"]:.4f} val_loss={val_loss:.4f} train_F1={train_f1:.4f} val_F1={vf:.4f} COVID_F1={vcov:.4f}',flush=True)
        if vf>best+1e-8:
            best=vf;stale=0
            torch.save({'state_dict':model.state_dict(),'branches':branches,'dim':X.shape[1],'hidden':args.hidden,'dropout':args.dropout,'aux_weight':args.aux_weight,'best_epoch':epoch,'best_macro_f1':vf,'fusion':args.fusion,'feature_dims':dims,'project_dim':args.project_dim,'gate_hidden':args.gate_hidden,'seed':args.seed},out/'best.pt')
            probs,pred,_=score(model,V,vy,device);write_eval(out,vnames,vy,probs,pred)
            write_gate_analysis(model,V,vy,vnames,branches,device,out)
        else:stale+=1
        if stale>=args.patience:print('Early stopping');break
    print(f'Best val macro-F1={best:.5f}; checkpoint={out/"best.pt"}')

def predict(args):
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    branches=cp['branches'];X,y,names=load_features(args.cache,branches,args.split)
    dims=cp.get('feature_dims') or feature_dims(args.cache,branches,args.split)
    model=make_head(cp.get('fusion','concat'),dims,cp['hidden'],cp['dropout'],cp['aux_weight'],cp.get('project_dim',128),cp.get('gate_hidden',64))
    model.load_state_dict(cp['state_dict'],strict=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    probabilities,pred,_=score(model,X,y,device)
    template=load_csv(args.data,args.split)
    if not np.array_equal(names,template.new_filename.astype(str).to_numpy()):raise ValueError('Names/order do not match official CSV template (or train views >1)')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    if args.split=='test':
        onehot=np.eye(4,dtype=int)[pred];df=pd.DataFrame(onehot,columns=CLASSES);df.insert(0,'new_filename',names);df.to_csv(path,index=False)
        print('Predicted counts:',dict(zip(CLASSES,np.bincount(pred,minlength=4).tolist())))
    else:
        print('Val macro-F1:',f1_score(y,pred,average='macro',labels=[0,1,2,3]))
    prob=pd.DataFrame(probabilities,columns=[f'p_{c}' for c in CLASSES]);prob.insert(0,'new_filename',names);prob.to_csv(path.with_name(path.stem+'_probabilities.csv'),index=False)
    print('Saved',path)

def ensemble(args):
    # Join by filename rather than relying on matching row positions.
    template=load_csv(args.data,args.split);names=template.new_filename.astype(str).tolist()
    total=None
    for file,weight in zip(args.prob_files,args.weights):
        df=pd.read_csv(file,dtype={'new_filename':'string'})
        if df.new_filename.duplicated().any():raise ValueError(f'Duplicated names: {file}')
        aligned=df.set_index('new_filename').reindex(names)
        values=aligned[[f'p_{c}' for c in CLASSES]].to_numpy(dtype=float)
        if not np.isfinite(values).all():raise ValueError(f'Bad probabilities or filename mismatch: {file}')
        total=weight*values if total is None else total+weight*values
    total=total/sum(args.weights);pred=total.argmax(1)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if args.split!='test':
        y=template[CLASSES].to_numpy().argmax(1)
        print('Ensemble validation Macro-F1:',f1_score(y,pred,average='macro',labels=[0,1,2,3]))
        print(classification_report(y,pred,labels=[0,1,2,3],target_names=CLASSES,zero_division=0))
    else:
        df=pd.DataFrame(np.eye(4,dtype=int)[pred],columns=CLASSES);df.insert(0,'new_filename',names);df.to_csv(out,index=False)
        print('Saved Kaggle submission:',out)
    p=pd.DataFrame(total,columns=[f'p_{c}' for c in CLASSES]);p.insert(0,'new_filename',names);p.to_csv(out.with_name(out.stem+'_probabilities.csv'),index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('extract');e.add_argument('--branch',required=True,choices=BRANCHES)
    e.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val','test'])
    e.add_argument('--train-views',type=int,default=1)
    e.add_argument('--batch-size',type=int,default=8);e.add_argument('--workers',type=int,default=0)
    e.add_argument('--seed',type=int,default=42);e.add_argument('--chexfound-size',type=int,default=512)
    e.add_argument('--xrv-checkpoint',type=Path,default=Path('outputs/xrv_oversampling/best.pt'))
    e.add_argument('--rad-checkpoint',type=Path,default=Path('outputs/raddino_oversampling/best.pt'))
    e.add_argument('--chexfound-repo',type=Path,default=Path('external/CheXFound'))
    e.add_argument('--chexfound-config',type=Path,default=Path('pretrain_weight/chexfound/config.yaml'))
    e.add_argument('--chexfound-weights',type=Path,default=Path('pretrain_weight/chexfound/teacher_checkpoint.pth'))
    e.add_argument('--chexzero-repo',type=Path,default=Path('external/CheXzero'))
    e.add_argument('--chexzero-weights',type=Path,default=Path('pretrain_weight/chexzero/best_64_5e-05_original_22000_0.864.pt'))
    t=sub.add_parser('train');t.add_argument('--branches',nargs='+',choices=BRANCHES,required=True)
    t.add_argument('--output',type=Path,required=True);t.add_argument('--epochs',type=int,default=80)
    t.add_argument('--patience',type=int,default=15);t.add_argument('--batch-size',type=int,default=64)
    t.add_argument('--hidden',type=int,default=256);t.add_argument('--dropout',type=float,default=.35)
    t.add_argument('--lr',type=float,default=.001);t.add_argument('--weight-decay',type=float,default=.0001)
    t.add_argument('--sampler',choices=['natural','balanced'],default='balanced')
    t.add_argument('--loss',choices=['ce','weighted_ce'],default='ce')
    t.add_argument('--aux-weight',type=float,default=0.0);t.add_argument('--seed',type=int,default=42)
    t.add_argument('--fusion',choices=MODES,default='concat');t.add_argument('--project-dim',type=int,default=128)
    t.add_argument('--gate-hidden',type=int,default=64)
    q=sub.add_parser('predict');q.add_argument('--checkpoint',type=Path,required=True)
    q.add_argument('--split',choices=['train','val','test'],default='test');q.add_argument('--output',type=Path,required=True)
    z=sub.add_parser('ensemble');z.add_argument('--prob-files',nargs='+',type=Path,required=True)
    z.add_argument('--weights',nargs='+',type=float,required=True)
    z.add_argument('--split',choices=['val','test'],default='val');z.add_argument('--output',type=Path,required=True)
    for cmd in (e,t,q,z):cmd.add_argument('--data',type=Path,default=Path('data'));cmd.add_argument('--cache',type=Path,default=Path('cache_embeddings'))
    args=p.parse_args()
    if args.command=='extract':
        if args.train_views<1: p.error('train-views >= 1')
        extract(args)
    if args.command=='train':
        if len(args.branches)!=len(set(args.branches)):p.error('Duplicate branches')
        train(args)
    if args.command=='predict':predict(args)
    if args.command=='ensemble':
        if len(args.prob_files)!=len(args.weights) or any(w<0 for w in args.weights) or sum(args.weights)<=0:p.error('Provide one nonnegative weight per file, sum > 0')
        ensemble(args)
if __name__=='__main__':main()
