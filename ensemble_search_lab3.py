"""Bounded LAB3 probability ensemble search; no training, no existing-file writes.

Requires numpy and pandas. Run python ensemble_search.py --help.
Model IDs are validation parent paths relative to --root (forward slashes).
Test correspondence is explicit or a documented filename alias, never inferred
from prediction similarity. Verify that each mapped pair used the same checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import re
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['normal', 'bacteria', 'virus', 'COVID-19']
PCOLS = ['p_' + c for c in CLASSES]
DEFAULT_BASE = ['f3', 'f134', 'f3_lora_s42', 'f134_lora_s42', 'f23']
# Only the 15 nonempty ordered branch subsets, frozen or LoRA.
MODEL_RE = re.compile(r"f(?:1|2|3|4|12|13|14|23|24|34|123|124|134|234|1234)(?:_lora(?:_s[0-9]+)?)?$")


def read_csv(path):
    df = pd.read_csv(path, dtype={'new_filename': 'string'})
    if 'new_filename' not in df:
        raise ValueError('missing new_filename')
    ids = df.new_filename
    if len(df) == 0 or ids.isna().any() or ids.str.strip().eq('').any():
        raise ValueError('empty CSV or missing/blank filename')
    if ids.duplicated().any():
        raise ValueError('duplicate filenames')
    return df


def targets(df):
    arr = df[CLASSES].to_numpy(dtype=float)
    if not np.isfinite(arr).all() or not np.isin(arr, [0, 1]).all() or not (arr.sum(1) == 1).all():
        raise ValueError('validation labels must be finite one-hot values')
    return arr.argmax(1)


def decode_labels(series):
    names = series.astype(str).map({c: i for i, c in enumerate(CLASSES)})
    numeric = pd.to_numeric(series, errors='coerce')
    result = names.where(names.notna(), numeric).to_numpy(dtype=float)
    if not np.isfinite(result).all() or not np.isin(result, range(4)).all():
        raise ValueError('invalid true labels; expected class names or indices 0..3')
    return result.astype(int)


def probabilities(path, names, y=None):
    df = read_csv(path)
    if set(df.new_filename) != set(names):
        missing = len(set(names) - set(df.new_filename))
        extra = len(set(df.new_filename) - set(names))
        raise ValueError('sample set mismatch: missing=%d extra=%d' % (missing, extra))
    reordered = df.new_filename.tolist() != list(names)
    df = df.set_index('new_filename').loc[list(names)]
    actual = {c for c in df.columns if c.startswith('p_')}
    if actual != set(PCOLS):
        raise ValueError('probability class columns must be exactly ' + ','.join(PCOLS))
    p = df[PCOLS].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('probabilities must be finite and in [0,1]')
    if not np.allclose(p.sum(1), 1, rtol=0, atol=1e-4):
        raise ValueError('probability rows do not sum to 1 (tolerance 1e-4)')
    # Keep the original probabilities, matching fourbranch.py arithmetic.
    label_columns = [c for c in ['true', 'true_label'] if c in df]
    if y is not None:
        for col in label_columns:
            if not np.array_equal(decode_labels(df[col]), y):
                raise ValueError('labels disagree with validation template: ' + col)
        if set(CLASSES).issubset(df.columns) and not np.array_equal(targets(df), y):
            raise ValueError('one-hot labels disagree with validation template')
    elif any(df[c].notna().any() for c in label_columns):
        raise ValueError('test probability CSV unexpectedly contains true labels')
    return p, reordered, bool(label_columns or set(CLASSES).issubset(df.columns))


def metrics(y, pred):
    f = []
    for c in range(4):
        tp = int(((y == c) & (pred == c)).sum())
        den = int((y == c).sum() + (pred == c).sum())
        f.append(2 * tp / den if den else 0.)
    return {'macro_f1': float(np.mean(f)), 'accuracy': float((y == pred).mean()),
            **{'f1_' + c: f[i] for i, c in enumerate(CLASSES)}}


def save_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, mode='x', encoding='utf-8-sig')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('outputs'), help='LAB3 outputs directory; discover immediate f*/val_probabilities.csv')
    p.add_argument('--data', type=Path, default=Path('data'), help='official val_data.csv / test_data_sample.csv directory')
    p.add_argument('--baseline', nargs='+', default=DEFAULT_BASE, help='exact model IDs; default is the five-model LAB3 baseline, equal weights')
    p.add_argument('--weights', nargs='+', type=float, help='baseline weights; default equal')
    p.add_argument('--max-trials', type=int, default=200, help='maximum ensemble evaluations INCLUDING baseline; default 200')
    p.add_argument('--max-rounds', type=int, default=3, help='maximum greedy search rounds')
    p.add_argument('--max-models', type=int, default=8, help='maximum members in any ensemble')
    p.add_argument('--add-weight', type=float, default=.2, help='new model fraction for additions; default .2')
    p.add_argument('--exclude', nargs='*', default=[], help='model IDs to exclude from candidate pool')
    p.add_argument('--test-map', type=Path, help='JSON object mapping model ID to test probability path (relative to JSON file)')
    p.add_argument('--submit-top', type=int, default=0, help='try top N validation ranks; default 0, no submissions; missing pairs are skipped')
    p.add_argument('--out', type=Path, default=Path('ensemble_search_runs'), help='parent for a new unique run directory')
    args = p.parse_args()
    if args.max_rounds < 1 or args.max_models < len(args.baseline) or args.max_trials < 1 or args.submit_top < 0 or not 0 < args.add_weight < 1:
        p.error('max-rounds >= 1; max-models >= baseline size; max-trials >= 1; submit-top >= 0; 0 < add-weight < 1 required')
    if len(args.baseline) != len(set(args.baseline)):
        p.error('baseline IDs must be unique')
    w = np.array(args.weights if args.weights is not None else [1.] * len(args.baseline))
    if len(w) != len(args.baseline) or not np.isfinite(w).all() or (w <= 0).any() or not np.isfinite(w.sum()):
        p.error('provide one finite, strictly positive weight per baseline model')
    args.weights = w / w.sum()
    return args


def main():
    args = parse_args()
    template = read_csv(args.data / 'val_data.csv')
    names, y = template.new_filename.tolist(), targets(template)
    # Never permit split overlap even when submissions are not requested.
    test_template = read_csv(args.data / 'test_data_sample.csv') if (args.data / 'test_data_sample.csv').exists() else None
    if test_template is not None and set(names) & set(test_template.new_filename):
        raise ValueError('validation/test templates overlap')
    args.out.mkdir(parents=True, exist_ok=True)
    import tempfile
    out = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=str(args.out)))
    models, sources, hashes, audit = {}, {}, {}, []
    for path in sorted(args.root.glob('f*/val_probabilities.csv')):
        model = path.parent.relative_to(args.root).as_posix()
        row = {'model': model, 'path': str(path.resolve())}
        if any(part.lower() in {'lora_checks', 'integration_run', 'integration_data', '__pycache__'} for part in path.parts):
            audit.append(dict(row, status='excluded', reason='integration/test fixture directory'))
            continue
        if not MODEL_RE.fullmatch(model):
            audit.append(dict(row, status='excluded', reason='not a LAB3 fusion model ID'))
            continue
        if model in args.exclude:
            audit.append(dict(row, status='excluded', reason='--exclude'))
            continue
        try:
            if path.name != 'val_probabilities.csv':
                raise ValueError('noncanonical copy; only val_probabilities.csv is eligible')
            p, reordered, has_labels = probabilities(path, names, y)
            digest = hashlib.sha256(p.tobytes()).hexdigest()
            models[model], sources[model], hashes[model] = p, path.resolve(), digest
            audit.append(dict(row, status='accepted', reason='', reordered=reordered,
                              labels_checked=has_labels, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
            audit.append(dict(row, status='rejected', reason=str(e)))
    save_csv(audit, out / 'discovery.csv')
    missing = set(args.baseline) - set(models)
    if missing:
        raise ValueError('baseline missing/rejected: %s; see %s; available IDs: %s' %
                         (sorted(missing), out / 'discovery.csv', sorted(models)))
    individual = {m: metrics(y, p.argmax(1)) for m, p in models.items()}
    save_csv([dict(model=m, **v) for m, v in sorted(individual.items(), key=lambda x: (-x[1]['macro_f1'], x[0]))], out / 'models.csv')
    pairs = []
    for a, b in itertools.combinations(sorted(models), 2):
        pa, pb = models[a].argmax(1), models[b].argmax(1)
        ea, eb = pa != y, pb != y
        pairs.append({'model_a': a, 'model_b': b, 'disagreement': float((pa != pb).mean()),
                      'double_fault': float((ea & eb).mean()),
                      'a_errors_rescued_by_b': float((ea & ~eb).sum() / ea.sum()) if ea.any() else None,
                      'b_errors_rescued_by_a': float((eb & ~ea).sum() / eb.sum()) if eb.any() else None,
                      'oracle_accuracy': float((~(ea & eb)).mean())})
    save_csv(pairs, out / 'pairwise.csv')
    base = dict(zip(args.baseline, args.weights.tolist()))
    base_p = sum(models[m] * w for m, w in base.items())
    base_errors = base_p.argmax(1) != y
    # Candidate screening uses individual F1 and rescue of baseline errors only.
    # These are diagnostics, not additional ensemble evaluations.
    seen = {hashes[m] for m in base}
    pool = []
    for m in sorted(models, key=lambda m: (-individual[m]['macro_f1'], m)):
        if m in base or hashes[m] in seen:
            continue
        seen.add(hashes[m])
        correct = models[m].argmax(1) == y
        rescue = float((base_errors & correct).sum() / base_errors.sum()) if base_errors.any() else 0.
        pool.append({'model': m, 'macro_f1': individual[m]['macro_f1'], 'baseline_error_rescue': rescue,
                     'screen_score': individual[m]['macro_f1'] + .1 * rescue})
    pool.sort(key=lambda r: (-r['screen_score'], r['model']))
    save_csv(pool, out / 'candidate_screen.csv')
    # Greedy best-improvement search; each round can add or replace one member.
    # All scored candidates are retained, including those that do not improve.
    ranked = []
    visited = set()

    def evaluate(desc, weights, parent, round_number):
        total_weight = sum(weights.values())
        weights = {m: float(w / total_weight) for m, w in sorted(weights.items())}
        key = tuple((m, round(w, 12)) for m, w in weights.items())
        if key in visited or len(ranked) >= args.max_trials:
            return None
        visited.add(key)
        prob = sum(models[m] * w for m, w in weights.items())
        row = dict(trial=len(ranked), parent_trial=parent, round=round_number,
                   operation=desc, weights=weights, **metrics(y, prob.argmax(1)))
        ranked.append(row)
        return row

    current = evaluate('baseline', base, None, 0)
    # Screening order only prioritizes trials when the evaluation budget is tight.
    # Include original members so later rounds may reintroduce a replaced model.
    order = [r['model'] for r in pool] + list(base)
    for round_number in range(1, args.max_rounds + 1):
        start = len(ranked)
        weights = current['weights']
        candidates = [m for m in order if m not in weights]
        slots = sorted(weights, key=lambda m: (individual[m]['macro_f1'], m))
        # Round-robin across candidates for each edit type.
        for slot in [None] + slots:
            for m in candidates:
                if len(ranked) >= args.max_trials:
                    break
                if slot is None:
                    if len(weights) >= args.max_models:
                        continue
                    new = {k: v * (1 - args.add_weight) for k, v in weights.items()}
                    new[m] = args.add_weight
                    desc = 'add:' + m
                else:
                    new = {k: v for k, v in weights.items() if k != slot}
                    new[m] = weights[slot]
                    desc = 'replace:%s->%s' % (slot, m)
                evaluate(desc, new, current['trial'], round_number)
        if len(ranked) == start:
            break
        best = max(ranked[start:], key=lambda r: (r['macro_f1'], -r['trial']))
        if best['macro_f1'] <= current['macro_f1'] + 1e-12:
            break
        current = best
    ranked.sort(key=lambda r: (-r['macro_f1'], r['trial']))
    baseline_f1 = metrics(y, base_p.argmax(1))['macro_f1']
    for rank, row in enumerate(ranked, 1):
        row['rank'] = rank
        row['delta_vs_baseline'] = row['macro_f1'] - baseline_f1
    save_csv([dict(r, weights=json.dumps(r['weights'], sort_keys=True)) for r in ranked], out / 'ranking.csv')
    mapping = {}
    if args.test_map:
        raw = json.loads(args.test_map.read_text(encoding='utf-8-sig'))
        if not isinstance(raw, dict) or not all(isinstance(v, str) for v in raw.values()):
            raise ValueError('test-map must be a JSON object of model IDs to path strings')
        unknown = set(raw) - set(models)
        if unknown:
            raise ValueError('unknown test-map IDs: ' + str(sorted(unknown)))
        mapping = {k: (args.test_map.parent / v).resolve() for k, v in raw.items()}
    for m, source in sources.items():
        if m in mapping:
            continue
        # Canonical LAB3 prediction output; explicit --test-map overrides this.
        path = args.root / ('submission_' + m + '_probabilities.csv')
        if path.is_file():
            mapping[m] = path.resolve()
    submissions = []
    for row in ranked[:args.submit_top]:
        status = {'rank': row['rank'], 'status': 'skipped'}
        try:
            if test_template is None:
                raise ValueError('test_data_sample.csv is required')
            test_names = test_template.new_filename.tolist()
            missing = set(row['weights']) - set(mapping)
            if missing:
                raise ValueError('missing/ambiguous test mapping: ' + ', '.join(sorted(missing)))
            paths = [mapping[m] for m in row['weights']]
            if len(paths) != len(set(paths)):
                raise ValueError('different models map to the same test file')
            total = np.zeros((len(test_names), 4))
            for m, w in row['weights'].items():
                p, _, _ = probabilities(mapping[m], test_names)
                total += w * p
            total /= sum(row['weights'].values())
            frame = pd.DataFrame(np.eye(4, dtype=int)[total.argmax(1)], columns=CLASSES)
            frame.insert(0, 'new_filename', test_names)
            dest = out / ('submission_rank%02d.csv' % row['rank'])
            frame.to_csv(dest, index=False, mode='x')
            prob = pd.DataFrame(total, columns=PCOLS)
            prob.insert(0, 'new_filename', test_names)
            prob.to_csv(dest.with_name(dest.stem + '_probabilities.csv'), index=False, mode='x')
            status.update(status='created', path=str(dest.resolve()), reason='')
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as e:
            status['reason'] = str(e)
        submissions.append(status)
    save_csv(submissions or [{'status': 'disabled', 'reason': '--submit-top 0'}], out / 'submissions.csv')
    manifest = {'classes': CLASSES, 'baseline': base, 'max_trials': args.max_trials,
                'max_rounds': args.max_rounds, 'max_models': args.max_models,
                'evaluated_trials': len(ranked), 'validation_samples': len(names),
                'validation_template': str((args.data / 'val_data.csv').resolve()),
                'validation_sha256': hashlib.sha256((args.data / 'val_data.csv').read_bytes()).hexdigest(),
                'test_mapping': {m: str(p) for m, p in mapping.items()},
                'ranking': ranked, 'submissions': submissions}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    note = '''LAB3 baseline: f3, f134, f3_lora_s42, f134_lora_s42, f23; equal weights by default.
All F1 values use the four fixed classes, zero_division=0. Ranking is validation only.
Selection on this validation set can overfit; gains are not an unbiased generalization estimate.
Pairwise rescue is conditional on the named model being wrong; blank means no errors.
oracle_accuracy means either model correct, not achievable ensemble accuracy.
Missing true columns are checked against the official template by sample ID; discovery.csv records labels_checked=False.
Exact probability duplicates are excluded from the new-candidate pool. Reordered rows are aligned by exact filename.
Root test path: outputs/submission_<model>_probabilities.csv; override with --test-map.
Test mapping must refer to the same checkpoint, preprocessing and class order as its validation probabilities.
No labels/predictions are invented for missing test models; top ranks without complete test pairs are skipped.
Search is bounded by max-trials, max-rounds and max-models. Each round expands the best improving ensemble. Additions use add-weight; replacements inherit the removed weight. No exhaustive weight tuning.
'''
    (out / 'README.txt').write_text(note, encoding='utf-8')
    print('Baseline:', base)
    print('Accepted %d models; evaluated %d combinations.' % (len(models), len(ranked)))
    for row in ranked:
        print('%2d F1=%.6f delta=%+.6f %s' % (row['rank'], row['macro_f1'], row['delta_vs_baseline'], row['operation']))
    print('Reports:', out.resolve())
    for status in submissions:
        print('Submission:', status)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)
