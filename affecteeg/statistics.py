"""Paper aggregation and paired inference over independently selected fits."""
from __future__ import annotations
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from .common import write_json

TONE_SEEDS = (42, 123, 456)
GENERATION_SEEDS = (42, 123, 456, 789, 1024)
GENERATION_MODELS = ('neurotalk', 'convconcatnet', 'fesde', 'ssm2mel', 'dmf2mel', 'neurosonic', 'cat_wavenet')


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Expected finite p-values in [0, 1]')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def centered_bootstrap_p(observed, boot):
    observed, boot = np.asarray(observed), np.asarray(boot)
    scale = np.maximum(np.abs(observed), np.max(np.abs(boot), axis=0))
    tolerance = 64 * np.finfo(float).eps * scale
    return (1 + (np.abs(boot-observed) >= np.abs(observed)-tolerance).sum(axis=0)) / (len(boot)+1)


def stratified_bootstrap(values, emotions, draws=10_000, seed=20261010):
    """Model x key input, after averaging items and seeds; shared key resamples."""
    values, emotions = np.asarray(values, dtype=np.float64), np.asarray(emotions)
    if values.ndim != 2 or values.shape[1] != len(emotions) or not np.isfinite(values).all():
        raise ValueError('Expected finite model x key values and one emotion per key')
    classes = sorted(set(emotions.tolist()))
    rng = np.random.default_rng(seed)
    result = np.zeros((draws, len(values)))
    for emotion in classes:
        indices = np.flatnonzero(emotions == emotion)
        for start in range(0, draws, 1000):
            sample = rng.integers(len(indices), size=(min(1000, draws-start), len(indices)))
            result[start:start+len(sample)] += values[:, indices[sample]].mean(2).T / len(classes)
    return result


def summarize_tone(rows, output_root):
    """Only complete three-seed/ten-participant pairs receive paper intervals."""
    keyed = {}
    for row in rows:
        key = (row['backbone'], int(row['seed']), int(row['fold']), row['variant'])
        if key in keyed:
            raise ValueError(f'Duplicate evaluation coordinate: {key}')
        keyed[key] = row
    models = sorted({row['backbone'] for row in rows})
    draws = np.random.default_rng(17003).integers(10, size=(50_000, 10))
    summaries = []
    for model in models:
        expected = {(model, s, f, v) for s in TONE_SEEDS for f in range(1, 11) for v in ('base', 'affecteeg')}
        if not expected.issubset(keyed):
            summaries.append({'backbone': model, 'status': 'partial', 'missing_fits': len(expected-set(keyed))})
            continue
        means = {v: {metric: np.array([
            np.mean([keyed[(model, s, f, v)][metric] for s in TONE_SEEDS])
            for f in range(1, 11)]) for metric in ('balanced_accuracy', 'macro_f1')}
            for v in ('base', 'affecteeg')}
        difference = means['affecteeg']['balanced_accuracy'] - means['base']['balanced_accuracy']
        interval = np.quantile(difference[draws].mean(1), [.025, .975])
        summaries.append({'backbone': model, 'status': 'complete',
            'base_ba': float(means['base']['balanced_accuracy'].mean()),
            'affecteeg_ba': float(means['affecteeg']['balanced_accuracy'].mean()),
            'delta_ba': float(difference.mean()), 'delta_ba_ci95': interval.tolist(),
            'delta_macro_f1': float((means['affecteeg']['macro_f1']-means['base']['macro_f1']).mean())})
    complete = [r for r in summaries if r['status'] == 'complete']
    mean_rows = []
    for label, subset, required in [('eeg_only_mean', [r for r in complete if r['backbone'] != 'cat_net'], 9),
                                    ('all_backbone_mean', complete, 10)]:
        if len(subset) == required:
            mean_rows.append({'backbone': label, 'status': 'descriptive', **{
                metric: float(np.mean([r[metric] for r in subset]))
                for metric in ('base_ba', 'affecteeg_ba', 'delta_ba', 'delta_macro_f1')}})
    result = {'status': 'complete' if len(complete) == 10 else 'partial',
              'units': 'fractions (multiply BA and F1 by 100 for percentages/percentage points)',
              'aggregation': 'average seeds within participant, then equal participants',
              'bootstrap': {'draws': 50_000, 'seed': 17003, 'unit': 'participant', 'multiplicity': 'none'},
              'models': summaries+mean_rows}
    write_json(Path(output_root)/'tone'/'paper_summary.json', result)
    return result


def _read_key_metrics(path):
    grouped, identities = defaultdict(list), {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['condition_key']
            identity = (key, int(row['content']), int(row['emotion_source']))
            pair = int(row['pair_index'])
            if pair in identities:
                raise ValueError(f'Duplicate test item {pair} in {path}')
            identities[pair] = identity
            grouped[key].append([float(row[m]) for m in ('pcc', 'l1', 'rfe')])
    if any(len(values) != 100 for values in grouped.values()):
        raise ValueError(f'Test keys must each contain 100 items: {path}')
    return {key: np.mean(values, axis=0) for key, values in grouped.items()}, identities


def summarize_reconstruction(output_root):
    """Infer the 16-comparison family only after all 140 evaluation fits exist."""
    root = Path(output_root)/'reconstruction'
    values, support, missing = {}, {}, []
    for track in ('seediv', 'seedvii'):
        for model in GENERATION_MODELS:
            for variant in ('base', 'affecteeg'):
                for seed in GENERATION_SEEDS:
                    coordinate = (track, model, variant, seed)
                    path = root/track/model/variant/f'seed_{seed}'/'test_metrics.csv'
                    if not path.is_file():
                        missing.append('/'.join(map(str, coordinate)))
                        continue
                    keys, identity = _read_key_metrics(path)
                    if len(keys) != (37 if track == 'seediv' else 71):
                        raise ValueError(f'Unexpected test-key count: {path}')
                    if track in support and identity != support[track]:
                        raise ValueError(f'Test item identities differ across paired runs: {path}')
                    support[track] = identity
                    values[coordinate] = keys
    result = {'status': 'partial' if missing else 'complete', 'completed_fits': len(values),
              'expected_fits': 140, 'missing': missing, 'models': [], 'paired_pcc': []}
    if missing:
        write_json(root/'paper_summary.json', result)
        return result
    inference = []
    for ti, track in enumerate(('seediv', 'seedvii')):
        keys = sorted(values[(track, GENERATION_MODELS[0], 'base', GENERATION_SEEDS[0])])
        emotions = np.array([int(key.rsplit('|', 1)[-1]) for key in keys])
        classes = sorted(set(emotions))
        tensor = np.array([[[np.mean([values[(track, m, v, s)][key] for s in GENERATION_SEEDS], axis=0)
                             for key in keys] for m in GENERATION_MODELS] for v in ('base', 'affecteeg')])
        point = np.mean([tensor[:, :, emotions == e, :].mean(2) for e in classes], axis=0)
        point = np.concatenate([point, point.mean(1, keepdims=True)], axis=1)
        for vi, variant in enumerate(('base', 'affecteeg')):
            for mi, model in enumerate((*GENERATION_MODELS, 'backbone_mean')):
                result['models'].append({'track': track, 'backbone': model, 'variant': variant,
                    **dict(zip(('pcc', 'l1', 'rfe'), point[vi, mi].tolist()))})
        differences = tensor[1, :, :, 0]-tensor[0, :, :, 0]
        differences = np.concatenate([differences, differences.mean(0, keepdims=True)], axis=0)
        boot = stratified_bootstrap(differences, emotions, seed=20261010+ti)
        observed = point[1, :, 0]-point[0, :, 0]
        intervals = np.quantile(boot, [.025, .975], axis=0)
        p_values = centered_bootstrap_p(observed, boot)
        for mi, model in enumerate((*GENERATION_MODELS, 'backbone_mean')):
            inference.append({'track': track, 'backbone': model, 'delta_pcc': float(observed[mi]),
                              'ci95': intervals[:, mi].tolist(), 'p_raw': float(p_values[mi])})
    for row, adjusted in zip(inference, holm_adjust([r['p_raw'] for r in inference])):
        row.update(p_holm=float(adjusted), significant_positive=bool(adjusted < .05 and row['ci95'][0] > 0))
    result.update(paired_pcc=inference, aggregation='equal items per key, five seeds, equal keys within emotion, equal emotions; equal models for mean',
                  bootstrap={'draws': 10_000, 'unit': 'condition key', 'strata': 'emotion',
                             'seeds': {'seediv': 20261010, 'seedvii': 20261011}, 'holm_family_size': 16})
    write_json(root/'paper_summary.json', result)
    return result
