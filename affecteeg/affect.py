"""Train the frozen reconstruction affect encoder and subject-balanced prototypes."""
from __future__ import annotations
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from vendor.maet.architecture import MAET, default_maet_kwargs
from .common import RECONSTRUCTION_SEEDS

SEEDS = RECONSTRUCTION_SEEDS
TRACKS = ('seediv', 'seedvii')
BANDS = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 49))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def coordinates(args):
    tracks = TRACKS if args.track == 'all' else (args.track,)
    seeds = SEEDS if str(args.seed) == 'all' else (int(args.seed),)
    if any(t not in TRACKS for t in tracks) or any(s not in SEEDS for s in seeds):
        raise ValueError('use tracks seediv/seedvii and training seeds 42/123/456/789/1024')
    return tracks, seeds


def output_directory(output_root, track, seed, smoke=False):
    return Path(output_root)/'affect'/track/f'seed_{seed}'


def extract_de(windows):
    """Compute MAET five-band PSD/DE directly from the input windows."""
    raw = np.asarray(windows, dtype=np.float64)
    single = raw.ndim == 2
    if single:
        raw = raw[None]
    if raw.shape[1:] != (800, 62):
        raise ValueError(f'MAET-DE expects [N,800,62], got {raw.shape}')
    energy = np.abs(np.fft.fft(raw.transpose(0, 2, 1)*np.hanning(800), n=1024, axis=-1)[..., :512])
    values = []
    for low, high in BANDS:
        start = max(int(np.floor(low/200*1024)), 1)
        end = max(int(np.floor(high/200*1024)), start+1)
        power = (energy[..., start-1:end]**2).mean(-1)
        values.append(np.log2(100*np.maximum(power, 1e-12)))
    features = np.stack(values, axis=1).reshape(len(raw), 310).astype(np.float32)
    return features[0] if single else features


def inner_split(records, seed):
    """Global row IDs; every original trial stays wholly in one inner split."""
    groups = {}
    for i, row in enumerate(records):
        if row['split'] == 'train':
            group = (int(row['subject']), int(row.get('session', 0)), int(row['trial']))
            groups.setdefault(group, []).append(i)
    ordered = sorted(groups)
    if len(ordered) < 2:
        raise ValueError('MAET needs at least two outer-training original trials')
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    n_val = min(len(ordered)-1, max(1, int(round(.2*len(ordered)))))
    validation_groups = {ordered[i] for i in permutation[:n_val]}
    train, val = [], []
    for group, rows in groups.items():
        (val if group in validation_groups else train).extend(rows)
    return np.asarray(sorted(train)), np.asarray(sorted(val))


def fit_de_stats(features, train_rows):
    value = np.asarray(features[train_rows], dtype=np.float64)
    return {'mean': value.mean(0).astype(np.float32), 'std': value.std(0).clip(1e-6).astype(np.float32)}


class MAETProjector(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = MAET(**default_maet_kwargs(num_classes=7, domain_generalization=False))
        self.project = nn.Linear(32, 128)
        self.classifier = nn.Linear(128, num_classes)

    def encode(self, value):
        return self.project(self.encoder.encode_eeg(value))

    def forward(self, value):
        z = self.encode(value)
        return z, self.classifier(z)


def classification_metrics(labels, predictions, n_classes):
    matrix = np.bincount(np.asarray(labels)*n_classes + np.asarray(predictions), minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    tp = matrix.diagonal().astype(float)
    recall = tp/np.maximum(matrix.sum(1), 1)
    f1 = 2*tp/np.maximum(matrix.sum(1)+matrix.sum(0), 1)
    return {'balanced_accuracy': float(recall.mean()), 'macro_f1': float(f1.mean())}


@torch.no_grad()
def subject_balanced_prototypes(model, features, labels, subjects, train_rows, device):
    model.eval()
    by_subject = {}
    for start in range(0, len(train_rows), 256):
        rows = train_rows[start:start+256]
        values = model.encode(torch.as_tensor(features[rows], device=device)).cpu()
        for z, index in zip(values, rows):
            by_subject.setdefault((int(subjects[index]), int(labels[index])), []).append(z)
    prototypes = []
    for label in range(model.classifier.out_features):
        subject_means = [torch.stack(v).mean(0) for (_, y), v in by_subject.items() if y == label]
        if not subject_means:
            raise ValueError(f'inner training has no prototype support for class {label}')
        prototypes.append(torch.stack(subject_means).mean(0))
    return F.layer_norm(torch.stack(prototypes), (128,))


def prepare_features(data, folder):
    """Features stay outside the distributable package in the user's outputs."""
    folder = Path(folder)
    path = folder/'de_features.npy'
    metadata_path = folder/'de_features.json'
    records = data['manifest']['emotion']
    source = getattr(data['emotion_eeg'], 'filename', None)
    version = None
    if source is not None:
        stat = Path(source).stat()
        version = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    identity = {'format': 1, 'source_ids': [r['source_id'] for r in records],
                'input_version': version, 'definition': 'raw_800_200Hz_hann_fft1024_maet310'}
    if path.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == identity:
        existing = np.load(path, mmap_mode='r')
        if existing.shape != (len(records), 310):
            raise ValueError(f'stale DE features at {path}; remove and rerun')
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npy')
    values = np.lib.format.open_memmap(temporary, mode='w+', dtype='float32', shape=(len(records), 310))
    for start in range(0, len(values), 128):
        values[start:start+128] = extract_de(data['emotion_eeg'][start:start+128])
    values.flush()
    del values
    temporary.replace(path)
    metadata_path.write_text(json.dumps(identity, indent=2)+'\n')
    return np.load(path, mmap_mode='r')


def train_one(data, output_root, track, seed, device, smoke=False):
    folder = output_directory(output_root, track, seed, smoke)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'selected.pt').exists():
        raise FileExistsError(f'{folder}/selected.pt already exists; choose a fresh output root')
    feature_folder = Path(output_root)/'affect'/track
    values = prepare_features(data, feature_folder)
    records = data['manifest']['emotion']
    train_rows, val_rows = inner_split(records, seed)
    active = data['manifest']['active_emotions']
    mapping = {int(label): i for i, label in enumerate(active)}
    labels = np.array([mapping[int(r['emotion'])] for r in records])
    subjects = np.array([int(r['subject']) for r in records])
    counts = np.bincount(labels[train_rows], minlength=len(active))
    if (counts == 0).any():
        raise ValueError('an affect class has no inner-training support')
    stats = fit_de_stats(values, train_rows)
    standardized = (np.asarray(values)-stats['mean'])/stats['std']
    seed_all(seed)
    model = MAETProjector(len(active)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    sampler = WeightedRandomSampler(torch.as_tensor(1/counts[labels[train_rows]], dtype=torch.double), len(train_rows), replacement=True, generator=torch.Generator().manual_seed(seed))
    loader = DataLoader(TensorDataset(torch.from_numpy(standardized[train_rows]), torch.from_numpy(labels[train_rows])), batch_size=256, sampler=sampler)
    best, best_state, history = None, None, []
    for epoch in range(1, (1 if smoke else 30)+1):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x.to(device))[1], y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if smoke:
                break
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(val_rows), 256):
                rows = val_rows[start:start+256]
                predictions.extend(model(torch.from_numpy(standardized[rows]).to(device))[1].argmax(1).cpu().tolist())
        metrics = classification_metrics(labels[val_rows], predictions, len(active))
        candidate = (metrics['macro_f1'], metrics['balanced_accuracy'], -epoch)
        selected = best is None or candidate > best
        if selected:
            best = candidate
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({'epoch': epoch, 'train_ce': float(np.mean(losses)), **metrics, 'selected': selected})
        print(f'affect {track} seed={seed} epoch={epoch}: {metrics}', flush=True)
    model.load_state_dict(best_state)
    prototypes = subject_balanced_prototypes(model, standardized, labels, subjects, train_rows, device)
    state = {'state_dict': best_state, 'prototypes': prototypes.cpu(), 'active_emotions': active, 'de_stats': stats,
             'inner_train_rows': train_rows, 'inner_val_rows': val_rows, 'seed': seed, 'track': track,
             'selected_epoch': -best[2], 'smoke': smoke}
    torch.save(state, folder/'selected.pt')
    (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
    return folder/'selected.pt'


class FrozenAffectResolver:
    def __init__(self, output_root, track, seed, data, device, smoke=False):
        folder = output_directory(output_root, track, seed, smoke)
        path = folder/'selected.pt'
        if not path.exists():
            raise FileNotFoundError(f'{path}: run train-affect first for this track and seed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state['track'] != track or state['seed'] != seed or state['smoke'] != smoke:
            raise ValueError('affect checkpoint coordinate differs')
        self.device = device
        self.model = MAETProjector(len(state['active_emotions'])).to(device)
        self.model.load_state_dict(state['state_dict'])
        self.model.requires_grad_(False).eval()
        self.prototypes = state['prototypes'].to(device)
        self.stats = state['de_stats']
        self.features = prepare_features(data, folder.parent)

    @torch.no_grad()
    def __call__(self, rows):
        raw = np.asarray(self.features[np.asarray(rows)], dtype=np.float32)
        values = torch.from_numpy((raw-self.stats['mean'])/self.stats['std']).to(self.device)
        logits = self.model(values)[1]
        return F.layer_norm(torch.softmax(logits, -1)@self.prototypes, (128,))


def run(args):
    from .data import load_cevb
    from .teacher import extract_content_features
    tracks, seeds = coordinates(args)
    device = torch.device(args.device)
    for track in tracks:
        data = load_cevb(args.processed_root, track)
        for seed in seeds:
            train_one(data, args.output_root, track, seed, device, args.smoke)
        # One frozen MD-JPT content feature set is shared across seeds.
        extract_content_features(args.processed_root, track, args.output_root, device=str(device), batch_size=128)
