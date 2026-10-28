"""Run paired loss/sampling comparisons for all 15 cached branch subsets."""
import argparse
import itertools
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BRANCHES = ['densenet', 'raddino', 'chexfound', 'chexzero']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=Path('cache_embeddings'))
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--out', type=Path, default=Path('outputs/imbalance_ablation_s42'))
    p.add_argument('--strategies', nargs='+', choices=['oversampling','weighted_ce'],
                   default=['oversampling','weighted_ce'])
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--predict-test', action='store_true')
    p.add_argument('--plan', action='store_true', help='print all commands without running or writing')
    args = p.parse_args()
    if min(args.epochs,args.patience) < 1: p.error('epochs and patience must be positive')
    args.cache,args.data=args.cache.resolve(),args.data.resolve()
    runner=Path(__file__).with_name('gated_attention_experiment.py')
    if not runner.exists(): raise FileNotFoundError(runner)
    if args.plan:
        run=args.out.resolve()/'NEW_UNIQUE_RUN'
    else:
        args.out.mkdir(parents=True,exist_ok=True)
        run=Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'),dir=str(args.out))).resolve()
    rows=[]
    for size in range(1,5):
        for subset in itertools.combinations(range(4),size):
            name='B'+''.join(str(i+1) for i in subset)
            branches=[BRANCHES[i] for i in subset]
            for strategy in dict.fromkeys(args.strategies):
                dest=run/name/strategy
                sampler,loss=('balanced','ce') if strategy=='oversampling' else ('natural','weighted_ce')
                cmd=[sys.executable,str(runner),'--cache',str(args.cache),'--data',str(args.data),
                     '--branches',*branches,'--fusion','concat','--sampler',sampler,'--loss',loss,
                     '--epochs',str(args.epochs),'--patience',str(args.patience),'--seed',str(args.seed),
                     '--out',str(dest)]
                if args.predict_test:cmd.append('--predict-test')
                print(subprocess.list2cmdline(cmd),flush=True)
                if args.plan:continue
                dest.mkdir(parents=True,exist_ok=True)
                with (dest/'training.log').open('x',encoding='utf-8') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                reports=list(dest.glob('*/concat/best_val_report.json'))
                if len(reports)!=1:raise RuntimeError('Expected exactly one result: '+str(dest))
                report=json.loads(reports[0].read_text(encoding='utf-8'))
                row={'configuration':name,'branches':'+'.join(branches),'strategy':strategy,
                     'validation_macro_f1':report['macro avg']['f1-score'],
                     **{c+'_f1':report[c]['f1-score'] for c in ['normal','bacteria','virus','COVID-19']},
                     'result_directory':str(reports[0].parent)}
                rows.append(row)
                import csv
                with (run/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(rows)
                print(name,strategy,'Validation Macro-F1=%.3f%%'%(row['validation_macro_f1']*100),flush=True)
    if not args.plan:print('Summary:',run/'summary.csv')


if __name__=='__main__':main()
