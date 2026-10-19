"""Main-paper tone protocol and supplementary candidate-search settings."""
from pathlib import Path
import copy
import json
import numpy as np
from ..common import TONE_SEEDS

BACKBONES = ('cat_net', 'eegnet', 'atcnet', 'eeg_conformer', 'rgnn',
             'cbramod', 'st_eegformer', 'eeg_deformer', 'eegmamba', 'reve')
SEEDS = TONE_SEEDS
OFFSETS = {42: 1, 123: -1, 456: 2}
SUBJECTS = tuple(range(1, 11))
CAT_CHANNELS = ('T7', 'T8', 'Fp1', 'Fpz', 'Fp2')
EEG20 = ('AF7', 'PO3', 'PO6', 'F7', 'FC6', 'Fp1', 'CP6', 'FC4', 'F1', 'PO4',
         'O1', 'O2', 'FT7', 'CP4', 'FT8', 'CP3', 'PO7', 'C3', 'PO5', 'F6')
FEATURE_DIMS = dict(zip(BACKBONES, (256, 240, 32, 32, 64, 200, 512, 1216, 200, 512)))
BASE_RECIPES = {
    'cat_net': dict(batch_size=64, lr=1e-3, head_lr=1e-3, amp=False),
    'eegnet': dict(batch_size=128, lr=1e-3, head_lr=1e-3, amp=True),
    **{name: dict(batch_size=256, lr=1e-3, head_lr=1e-3, amp=True)
       for name in ('atcnet', 'eeg_conformer', 'rgnn', 'eeg_deformer')},
    'cbramod': dict(batch_size=16, lr=1e-4, head_lr=1e-3, amp=True),
    'st_eegformer': dict(batch_size=8, lr=1e-4, head_lr=1e-3, amp=True),
    'eegmamba': dict(batch_size=16, lr=1e-4, head_lr=1e-3, amp=True),
    'reve': dict(batch_size=8, lr=2e-5, head_lr=1e-3, amp=True),
}
STRUCTURAL_MODES = dict(cat_net='cat_dual_pool', eeg_conformer='conformer256',
                        reve='reve_after_norm', eeg_deformer='deformer_multiscale')


def split(subjects, seed, fold):
    seed, fold = int(seed), int(fold)
    if seed not in SEEDS or fold not in SUBJECTS:
        raise ValueError('Tone seeds are 42/123/456 and folds are test-subject IDs 1..10')
    subjects = np.asarray(subjects)
    if set(np.unique(subjects)) != set(SUBJECTS):
        raise ValueError('Tone data must contain exactly participants 1..10')
    validation = (fold - 1 + OFFSETS[seed]) % 10 + 1
    return dict(train=np.flatnonzero((subjects != fold) & (subjects != validation)),
                validation=np.flatnonzero(subjects == validation),
                test=np.flatnonzero(subjects == fold))


def templates(backbone):
    rows = json.loads(Path(__file__).with_name('candidate_templates.json').read_text())
    dimension = FEATURE_DIMS[backbone]
    for row in rows:
        if row['rank'] == 'conservative_rank':
            row['rank'] = 4 if dimension <= 64 else 8
        elif row['rank'] == 'expanded_rank':
            row['rank'] = 16 if dimension <= 64 else 64 if backbone == 'eeg_deformer' else 32
        row['feature_mode'] = 'native'
    # Refit the six fixed-grid anchors, rather than relying on historical results.
    for lr in (1e-4, 1e-3):
        for weight in (1., 10., 100.):
            rows.append(dict(id=f'grid_lr{lr:g}_cov{weight:g}', adapter_lr=lr,
                head_lr=lr, rank=16, rho=.1, kd_weight=.5, covariance_weight=weight,
                teacher_view='original', covariance='global_centered',
                covariance_batch_size=128, feature_mode='native'))
    return rows


def structural_templates(backbone):
    mode = STRUCTURAL_MODES[backbone]
    return [dict(id=f's{i}_{mode}', feature_mode=mode, adapter_lr=3e-4, head_lr=3e-5,
                 rank=rank, rho=rho, kd_weight=.1, covariance_weight='calibrated',
                 teacher_view=f'train_only_pca{pca}_no_whitening', covariance=cov,
                 covariance_batch_size=size)
            for i, (rank, rho, pca, cov, size) in enumerate([
                (16, .1, 32, 'global_centered', 128),
                (32, .2, 64, 'within_tone_centered_equal_tone_weight', 512)])]


def select(rows):
    if not rows:
        raise ValueError('No validation candidates')
    return min(rows, key=lambda r: (-round(r['validation_metrics']['balanced_accuracy'], 12),
        -round(r['validation_metrics']['macro_f1'], 12),
        0 if r['selected_epoch'] == 0 else 1, r['candidate_id'], r['selected_epoch']))
