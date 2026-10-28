"""Evaluate the 11 reported ensembles using validation probabilities only."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']
PCOLS = ['p_' + c for c in CLASSES]
CORE = ['f3', 'f134', 'f3_lora_s42', 'f134_lora_s42']
SPECS = [
    (1, CORE + ['f23'], [1]*5, 90.819),
    (2, CORE + ['f23', 'f23_lora_s42'], [1]*6, 90.813),
    (3, CORE + ['f234_lora_s42'], [1]*5, 90.726),
    (4, ['f12_lora_s42', 'f134_lora_s42', 'f23', 'f234_lora_s42', 'f3'], [1]*5, 90.667),
    (5, CORE, [1]*4, 90.662),
    (6, ['f12_lora_s42', 'f134_lora_s42', 'f23', 'f234', 'f3', 'f3_lora_s42'], [.16,.16,.16,.20,.16,.16], 90.098),
    (7, ['f3', 'f134'], [1,1], 89.941),
    (8, ['f3', 'f134', 'f123'], [.2,.6,.2], 89.937),
    (9, ['f3', 'f134', 'f1234_lora_s42'], [1]*3, 89.801),
    (10, ['f12_lora_s42', 'f134_lora_s42', 'f23', 'f3', 'f34_lora_s42'], [1]*5, 89.729),
    (11, ['f3_lora_s42', 'f134_lora_s42'], [1]*2, 89.126),
]


def read(path):
    df = pd.read_csv(path, dtype={'new_filename': 'string'})
    if 'new_filename' not in df:
        raise ValueError(f'{path}: missing new_filename')
    ids = df['new_filename']
    if df.empty or ids.isna().any() or ids.str.strip().eq('').any() or ids.duplicated().any():
        raise ValueError(f'{path}: empty, missing, blank or duplicate sample IDs')
    return df.set_index('new_filename')


def decode(series):
    names = series.map({c: i for i, c in enumerate(CLASSES)})
    result = names.where(names.notna(), pd.to_numeric(series, errors='coerce')).to_numpy(dtype=float)
    if not np.isfinite(result).all() or not np.isin(result, range(4)).all():
        raise ValueError('Invalid true labels')
    return result.astype(int)


def load(path, ids, y):
    df = read(path)
    if set(df.index) != set(ids):
        raise ValueError(f'{path}: validation sample set mismatch')
    df = df.loc[ids]
    p = df[PCOLS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-4, rtol=0):
        raise ValueError(f'{path}: invalid probabilities')
    for key in ['true', 'true_label']:
        if key in df and not np.array_equal(decode(df[key]), y):
            raise ValueError(f'{path}: labels disagree with validation template')
    if set(CLASSES).issubset(df.columns):
        labels = df[CLASSES].to_numpy(dtype=float)
        if not np.isin(labels, [0,1]).all() or not (labels.sum(1) == 1).all() or not np.array_equal(labels.argmax(1), y):
            raise ValueError(f'{path}: one-hot labels disagree with validation template')
    return p


def scores(y, pred):
    values = []
    for c in range(4):
        denominator = int((y == c).sum() + (pred == c).sum())
        values.append(2 * int(((y == c) & (pred == c)).sum()) / denominator if denominator else 0.)
    return [float(np.mean(values))] + values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('outputs'))
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('ensemble_validation_runs'))
    args = parser.parse_args()
    template = read(args.data / 'val_data.csv')
    labels = template[CLASSES].to_numpy(dtype=float)
    if not np.isin(labels, [0,1]).all() or not (labels.sum(1) == 1).all():
        raise ValueError('Validation template requires one-hot labels')
    ids, y = template.index.tolist(), labels.argmax(1)
    cache, rows = {}, []
    for rank, members, weights, public in SPECS:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        row = dict(public_rank=rank, members='+'.join(members),
                   weights=json.dumps(dict(zip(members, weights.tolist()))),
                   public_macro_f1_pct=public)
        try:
            for m in members:
                if m not in cache:
                    cache[m] = load(args.root / m / 'val_probabilities.csv', ids, y)
            total = sum(cache[m]*w for m,w in zip(members, weights))
            values = scores(y, total.argmax(1))
            row.update(dict(zip(['validation_macro_f1_pct'] + [c+'_f1_pct' for c in CLASSES], [100*v for v in values])))
            row.update(status='ok', error='')
            print(f'Rank {rank:02d}: Val Macro-F1={100*values[0]:.3f}%  Public={public:.3f}%')
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            row.update(status='error', error=str(exc))
            print(f'Rank {rank:02d}: ERROR: {exc}')
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='validation_', dir=str(args.out)))
    frame = pd.DataFrame(rows)
    frame.to_csv(run/'ensemble_validation_summary.csv', index=False, encoding='utf-8-sig')
    print('Saved:', (run/'ensemble_validation_summary.csv').resolve())
    return 0 if all(r['status'] == 'ok' for r in rows) else 2


if __name__ == '__main__':
    raise SystemExit(main())
