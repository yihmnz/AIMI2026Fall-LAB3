"""Train/predict all 15 original concat configurations from a prepared cache."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--setting',choices=['frozen','lora'],required=True)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,default=Path('outputs'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--plan',action='store_true')
    a=p.parse_args();project=Path(__file__).resolve().parent
    jobs=[]
    for size in range(1,5):
        for group in itertools.combinations(range(4),size):
            combo=''.join(str(i+1) for i in group)
            name='f'+combo+('_lora_s'+str(a.seed) if a.setting=='lora' else '')
            dest=a.out/name
            branches=[['densenet','raddino','chexfound','chexzero'][i] for i in group]
            prediction=a.out/('submission_'+name+'.csv')
            if dest.exists() or prediction.exists() or prediction.with_name(prediction.stem+'_probabilities.csv').exists():
                raise FileExistsError('Existing output preserved; choose a new --out: '+str(dest))
            common=['--data',str(a.data),'--cache',str(a.cache)]
            train=[sys.executable,str(project/'fourbranch.py'),'train',*common,'--branches',*branches,
                   '--output',str(dest),'--epochs','80','--patience','15','--batch-size','64',
                   '--sampler','balanced','--loss','ce','--seed',str(a.seed)]
            predict=[sys.executable,str(project/'fourbranch.py'),'predict',*common,
                     '--checkpoint',str(dest/'best.pt'),'--split','test','--output',str(prediction)]
            jobs.append((name,combo,dest,train,predict))
    rows=[]
    for name,combo,dest,train,predict in jobs:
        for cmd in [train,predict]:
            print(subprocess.list2cmdline(cmd),flush=True)
            if not a.plan:subprocess.run(cmd,check=True)
        if not a.plan:
            r=json.loads((dest/'best_val_report.json').read_text(encoding='utf-8'))
            rows.append(dict(experiment=name,branches=combo,validation_macro_f1=r['macro avg']['f1-score'],
                             **{c+'_f1':r[c]['f1-score'] for c in ['normal','bacteria','virus','COVID-19']}))
    if not a.plan:
        path=a.out/('fusion_summary_'+a.setting+'.csv')
        with path.open('x',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('Saved:',path)


if __name__=='__main__':main()
