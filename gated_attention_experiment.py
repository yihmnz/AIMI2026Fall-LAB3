"""Cached-feature fusion experiment. Requires fourbranch.py and fusion_heads.py.

No encoder training. Existing outputs are preserved in unique run directories.
Gated attention = branch-token self-attention followed by gated attention pooling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import fourbranch as fb
from fusion_heads import make_head as original_make_head

BRANCHES = ['densenet', 'raddino', 'chexfound', 'chexzero']
MODES = ['concat', 'projected_concat', 'static_weighted', 'dynamic_gated',
         'gated_concat', 'gated_attention']


class GatedAttentionHead(nn.Module):
    """Input [batch,sum(dims)] -> branch tokens -> contextual gated pooling.

    Attention has four heads and one pre-norm residual block. Pooling scores
    are w^T(tanh(Vh) * sigmoid(Uh)); softmax is across branches. Gates are
    diagnostic pooling weights, not causal feature-importance estimates.
    """
    def __init__(self, dims, hidden=256, dropout=.35, project_dim=128, gate_hidden=64):
        super().__init__()
        if project_dim % 4:
            raise ValueError('project_dim must be divisible by four attention heads')
        self.dims = list(dims)
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, project_dim), nn.GELU())
            for d in dims])
        # Identity embedding refers to branch order saved in the checkpoint.
        self.branch_embedding = nn.Parameter(torch.zeros(1, len(dims), project_dim))
        nn.init.normal_(self.branch_embedding, std=.02)
        self.norm1 = nn.LayerNorm(project_dim)
        self.attention = nn.MultiheadAttention(project_dim, 4, dropout=dropout, batch_first=True)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(project_dim)
        self.ffn = nn.Sequential(nn.Linear(project_dim, 2*project_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2*project_dim, project_dim))
        self.gate_v = nn.Linear(project_dim, gate_hidden)
        self.gate_u = nn.Linear(project_dim, gate_hidden)
        self.gate_w = nn.Linear(gate_hidden, 1, bias=False)
        self.body = nn.Sequential(nn.LayerNorm(project_dim), nn.Dropout(dropout),
                                  nn.Linear(project_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden, 4)

    def contextual_tokens(self, x):
        z = torch.stack([p(part) for p, part in zip(
            self.projections, torch.split(x, self.dims, dim=1))], dim=1)
        z = z + self.branch_embedding
        q = self.norm1(z)
        attended, _ = self.attention(q, q, q, need_weights=False)
        z = z + self.residual_dropout(attended)
        return z + self.residual_dropout(self.ffn(self.norm2(z)))

    def pool_weights(self, z):
        scores = self.gate_w(torch.tanh(self.gate_v(z)) * torch.sigmoid(self.gate_u(z)))
        return scores.squeeze(-1).softmax(dim=1)

    def gate_weights(self, x):
        return self.pool_weights(self.contextual_tokens(x))

    def forward(self, x):
        z = self.contextual_tokens(x)
        a = self.pool_weights(z)
        fused = (z * a.unsqueeze(-1)).sum(dim=1)
        return self.classifier(self.body(fused)), None


def make_head(mode, dims, hidden, dropout, aux, project_dim=128, gate_hidden=64):
    if mode == 'gated_attention':
        if aux != 0:
            raise ValueError('This experiment uses no auxiliary loss')
        return GatedAttentionHead(dims, hidden, dropout, project_dim, gate_hidden)
    return original_make_head(mode, dims, hidden, dropout, aux, project_dim, gate_hidden)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def check_split(data, cache, branches, split):
    """Require exact sample/label matches to the official split, no multi-view cache."""
    frame = fb.load_csv(data, split)
    names = frame.new_filename.astype(str).to_numpy()
    expected_y = frame[fb.CLASSES].to_numpy().argmax(1) if split != 'test' else np.array([], dtype=int)
    dimensions, identities = [], {}
    if len(names) == 0 or pd.Series(names).str.strip().eq('').any():
        raise ValueError('Empty split or blank sample ID: ' + split)
    for branch in branches:
        path = cache / branch / (split + '.npz')
        with np.load(path, allow_pickle=False) as d:
            x, y, n = d['x'], d['y'], d['names'].astype(str)
            if x.ndim != 2 or x.shape[0] != len(names) or x.shape[1] < 1 or not np.isfinite(x).all():
                raise ValueError('Invalid features: ' + str(path))
            if not np.array_equal(n, names):
                raise ValueError('Cache sample order/set differs from official CSV: ' + str(path))
            if not np.array_equal(y, expected_y):
                raise ValueError('Cache labels differ from official CSV: ' + str(path))
            dimensions.append(int(x.shape[1]))
        identities[str(path.resolve())] = sha256(path)
    return dimensions, identities, set(names)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--branches', nargs='+', choices=BRANCHES, default=BRANCHES)
    p.add_argument('--fusion', choices=MODES+['all'], default='gated_attention')
    p.add_argument('--out', type=Path, default=Path('outputs/gated_attention_runs'))
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--project-dim', type=int, default=128)
    p.add_argument('--gate-hidden', type=int, default=64)
    p.add_argument('--dropout', type=float, default=.35)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--sampler', choices=['natural', 'balanced'], default='balanced')
    p.add_argument('--loss', choices=['ce', 'weighted_ce'], default='ce')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--predict-test', action='store_true', help='also predict test with each best checkpoint')
    p.add_argument('--allow-cpu', action='store_true', help='allow CPU (intended for synthetic smoke tests)')
    args = p.parse_args(argv)
    if len(set(args.branches)) != len(args.branches):
        p.error('Specify distinct branches')
    if len(args.branches) < 2 and args.fusion != 'concat':
        p.error('Attention/fusion comparisons require at least two branches; single-branch concat is supported')
    if any(v < 1 for v in [args.epochs,args.patience,args.batch_size,args.hidden,args.project_dim,args.gate_hidden]):
        p.error('Integer training settings must be positive')
    if args.project_dim % 4 or not 0 <= args.dropout < 1:
        p.error('project-dim must be divisible by 4; dropout must be in [0,1)')
    if not all(np.isfinite(v) for v in [args.lr,args.weight_decay,args.dropout]) or args.lr <= 0 or args.weight_decay < 0:
        p.error('Invalid learning rate, weight decay or dropout')
    if not torch.cuda.is_available() and not args.allow_cpu:
        p.error('CUDA unavailable; activate your PyTorch CUDA environment (or --allow-cpu for tests)')
    args.cache, args.data = args.cache.resolve(), args.data.resolve()
    splits = ['train','val'] + (['test'] if args.predict_test else [])
    identities, sets, reference_dims = {}, {}, None
    for split in splits:
        dims, hashes, names = check_split(args.data,args.cache,args.branches,split)
        if reference_dims is not None and dims != reference_dims:
            raise ValueError('Feature dimensions differ across splits')
        reference_dims = dims
        identities.update(hashes)
        sets[split] = names
    for i,a in enumerate(splits):
        for b in splits[i+1:]:
            if sets[a] & sets[b]:
                raise ValueError('Sample IDs overlap between splits: '+a+' / '+b)
    args.out.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'),dir=str(args.out)))
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(classes=fb.CLASSES, feature_dims=reference_dims,
                  cache_sha256=identities, attention_heads=4, attention_blocks=1,
                  input='cached features; encoders are not trained',
                  code_sha256={f:sha256(Path(__file__).parent/f) for f in
                               [Path(__file__).name,'fourbranch.py','fusion_heads.py']})
    (run/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Run:',run.resolve(),flush=True)
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',flush=True)
    # Adapt factory in memory only: original project files are not modified.
    fb.make_head = make_head
    results = []
    modes = MODES if args.fusion == 'all' else [args.fusion]
    for mode in modes:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed); np.random.seed(args.seed)
        dest = run/mode
        opts = argparse.Namespace(**vars(args))
        opts.output, opts.fusion, opts.aux_weight = dest, mode, 0.
        fb.train(opts)
        cp_path = dest/'best.pt'
        cp = torch.load(cp_path,map_location='cpu',weights_only=True)
        cp.update(experiment_kind='branch-gated-attention-v1', run_config=config)
        torch.save(cp,cp_path)
        report = json.loads((dest/'best_val_report.json').read_text(encoding='utf-8'))
        row = dict(fusion=mode,best_epoch=cp['best_epoch'],validation_macro_f1=report['macro avg']['f1-score'],
                   **{c+'_f1':report[c]['f1-score'] for c in fb.CLASSES})
        results.append(row)
        if args.predict_test:
            # Check all bytes again before using cached inputs for prediction.
            for path,digest in identities.items():
                if sha256(path) != digest: raise ValueError('Cache changed during training: '+path)
            fb.predict(argparse.Namespace(checkpoint=cp_path,cache=args.cache,data=args.data,
                                          split='test',output=dest/'submission.csv'))
        pd.DataFrame(results).sort_values('validation_macro_f1',ascending=False).to_csv(
            run/'summary.csv',index=False,encoding='utf-8-sig')
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(pd.DataFrame(results).sort_values('validation_macro_f1',ascending=False).to_string(index=False))
    print('Results:',run.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
