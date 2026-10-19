"""Independent 50-epoch Base/AffectEEG fits; test access is evaluation-only."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ..affect import FrozenAffectResolver, coordinates, seed_all
from ..data import load_cevb, make_pairs
from .models import ReconstructionModel, BACKBONES
from .arra import raw_pool, fit_targets, AuxiliarySampler, covariance_correction, POOL_SCHEMA
from .losses import MelStandardizer, reconstruction_loss, item_metrics, equal_metrics, Selector


def normalize_content(raw):
    value = np.asarray(raw, dtype=np.float32)
    return (value-value.mean(axis=-2, keepdims=True))/(value.std(axis=-2, keepdims=True)+1e-6)


class Pairs(Dataset):
    def __init__(self, data, pairs):
        self.data, self.pairs = data, np.asarray(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        ci, ei, ki = map(int, self.pairs[index])
        return {'eeg': normalize_content(self.data['content_eeg'][ci]), 'mel': np.asarray(self.data['targets'][ki]).copy(),
                'content': ci, 'emotion': ei, 'key': ki, 'pair_index': index}


def batches(data, pairs, batch_size=32):
    return DataLoader(Pairs(data, pairs), batch_size=batch_size, shuffle=False, num_workers=0)


def content_tensor(data, rows, device):
    return torch.from_numpy(normalize_content(data['content_eeg'][np.asarray(rows)])).to(device)


@torch.no_grad()
def fit_initial_statistics(model, data, device):
    """One vote per unique training source; fixed position-wise and pooled stats."""
    rows = np.array([i for i, r in enumerate(data['manifest']['content']) if r['split'] == 'train'])
    if not len(rows):
        raise ValueError('empty reconstruction training sources')
    total = square = None
    pooled = []
    for start in range(0, len(rows), 32):
        memory = model.encode_auxiliary(content_tensor(data, rows[start:start+32], device))
        if memory.shape[1:] != (800, 256) or not torch.isfinite(memory).all():
            raise ValueError('invalid content memory')
        values = memory.double()
        sums, squares = values.sum(0).cpu().numpy(), values.square().sum(0).cpu().numpy()
        total = sums if total is None else total+sums
        square = squares if square is None else square+squares
        pooled.append(raw_pool(values).cpu().numpy())
    mean = total/len(rows)
    sigma = np.maximum(square/len(rows)-mean**2, 0)**.5
    features = {'mean': mean.astype(np.float32), 'sigma': sigma.astype(np.float32), 'training_rows': rows}
    pool = np.concatenate(pooled)
    sigma = pool.std(0).astype(np.float32)
    pools = {'schema': POOL_SCHEMA, 'mean': pool.mean(0).astype(np.float32), 'sigma': sigma,
             'active': sigma > 1e-6, 'statistics_split': 'train', 'training_rows': rows}
    return features, pools


def teacher_targets(data, embeddings):
    records = data['manifest']['content']
    groups = np.asarray([f"{r['vocal']}|{r['tone']}" for r in records])
    splits = np.asarray([r['split'] for r in records])
    targets, stats = fit_targets(embeddings, groups, splits, dimensions=32)
    return {'targets': targets, 'groups': groups, 'splits': splits, 'statistics': stats}


def auxiliary_sampler(teacher, seed, calibration=False):
    aux = AuxiliarySampler(teacher, seed)
    aux.generator.manual_seed(seed+70003+(11 if calibration else 0))
    return aux


def make_optimizer(model, track):
    rate = {'seediv': 1e-3, 'seedvii': 3e-4}[track]
    if model.variant == 'base':
        groups = model.parameters()
    else:
        groups = [{'params': model.content_encoder.parameters(), 'lr': rate},
                  {'params': list(model.decoder.parameters())+list(model.fusion.parameters()), 'lr': rate},
                  {'params': model.adapter.parameters(), 'lr': .3*rate}]
    optimizer = torch.optim.Adam(groups, lr=rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=5)
    return optimizer, scheduler


def calibration_weight(model, data, pairs, resolver, scaler, teacher, seed, device):
    aux = auxiliary_sampler(teacher, seed, calibration=True)
    modes = [(m, m.training) for m in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    ratios = []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            shared = tuple(model.adapter.parameters())
            for batch in batches(data, pairs[:256]):
                raw_target = batch['mel'].to(device)
                memory = model.encode_auxiliary(batch['eeg'].to(device))
                rec = reconstruction_loss(model.forward_memory(memory, resolver(batch['emotion'])).mel, raw_target, scaler)
                task = torch.autograd.grad(rec, shared)
                rows, groups = aux.sample()
                cov = model.auxiliary_covariance(content_tensor(data, rows, device), torch.as_tensor(teacher['targets'][rows], device=device), groups)
                covariance = torch.autograd.grad(cov, shared)
                _, geometry = covariance_correction(task, covariance)
                norm_task, norm_cov = geometry['task_squared_norm']**.5, geometry['covariance_squared_norm']**.5
                if not np.isfinite([norm_task, norm_cov]).all() or min(norm_task, norm_cov) <= 0:
                    raise ValueError('zero/nonfinite calibration gradients')
                ratios.append(norm_task/norm_cov)
    finally:
        for module, mode in modes:
            module.training = mode
    if len(ratios) != 8:
        raise ValueError('exactly eight 32-item calibration batches required')
    raw = .03*float(np.median(ratios))
    return float(np.clip(raw, 1e-4, 300)), {'gamma': .03, 'raw_lambda': raw, 'pairs': 8}


def update(model, optimizer, data, batch, resolver, scaler, teacher, aux, weight, epoch, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    token = resolver(batch['emotion']) if resolver else None
    result = model(batch['eeg'].to(device), token)
    rec = reconstruction_loss(result.mel, batch['mel'].to(device), scaler)
    if model.variant == 'base':
        objective = rec
    else:
        rows, groups = aux.sample()
        cov = model.auxiliary_covariance(content_tensor(data, rows, device), torch.as_tensor(teacher['targets'][rows], device=device), groups)
        effective = weight*min(1., epoch/5.)
        objective = rec+effective*cov+.01*result.diagnostics['preservation']
        shared = tuple(model.adapter.parameters())
        task = torch.autograd.grad(rec, shared, retain_graph=True)
        covariance = torch.autograd.grad(cov, shared, retain_graph=True)
        correction, _ = covariance_correction(task, covariance)
    if not torch.isfinite(objective):
        raise ValueError('nonfinite reconstruction objective')
    objective.backward()
    if model.variant == 'affecteeg' and correction is not None:
        for parameter, delta in zip(shared, correction):
            parameter.grad.add_(delta, alpha=effective)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(rec.detach()), float(objective.detach())


@torch.no_grad()
def evaluate_pairs(model, data, pairs, resolver, scaler, device):
    model.eval()
    rows = []
    for batch in batches(data, pairs):
        token = resolver(batch['emotion']) if resolver else None
        # Use the same inference encoder path as statistics/auxiliary encoding.
        memory = model.encode_auxiliary(batch['eeg'].to(device))
        prediction = scaler.inverse(model.forward_memory(memory, token).mel)
        metrics = item_metrics(prediction, batch['mel'].to(device))
        for values, key, ci, ei, index in zip(metrics, batch['key'], batch['content'], batch['emotion'], batch['pair_index']):
            rows.append({'key': data['manifest']['keys'][int(key)], 'condition_key': data['manifest']['keys'][int(key)], 'key_index': int(key), 'content': int(ci), 'emotion_source': int(ei),
                         'pair_index': int(index), **dict(zip(('pcc', 'l1', 'rfe'), map(float, values)))})
    return equal_metrics(rows), rows


def run_directory(output_root, track, backbone, variant, seed, smoke=False):
    return Path(output_root)/'reconstruction'/track/backbone/variant/f'seed_{seed}'


def save_checkpoint(path, model, scaler, feature_stats, pool_stats, identity, selector, epoch, optimizer, scheduler, weight, history):
    state = {'identity': identity, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
             'standardizer': scaler.state_dict(), 'feature_statistics': feature_stats, 'pool_statistics': pool_stats,
             'selected_epoch': selector.epoch, 'selected_scores': selector.scores, 'epoch': epoch,
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'lambda_cov': weight, 'history': history}
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(path)


def train_one(args, data, track, backbone, variant, seed, teacher=None):
    device = torch.device(args.device)
    folder = run_directory(args.output_root, track, backbone, variant, seed, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'last.pt').exists() or (folder/'selected.pt').exists():
        raise FileExistsError(f'existing fit at {folder}; use a fresh output root')
    seed_all(seed)
    model = ReconstructionModel(backbone, variant).to(device)
    train_keys = [i for i, split in enumerate(data['manifest']['key_splits']) if split == 'train']
    scaler = MelStandardizer.fit(data['targets'], train_keys)
    feature_stats = pool_stats = resolver = aux = None
    weight = 0.
    training_pairs = make_pairs(data['manifest'], 'train', seed=seed, epoch=0)
    if variant == 'affecteeg':
        feature_stats, pool_stats = fit_initial_statistics(model, data, device)
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed+77003)
            model.initialize_adapter(feature_stats, pool_stats)
            resolver = FrozenAffectResolver(args.output_root, track, seed, data, device, args.smoke)
        aux = auxiliary_sampler(teacher, seed)
        weight, calibration = calibration_weight(model, data, training_pairs, resolver, scaler, teacher, seed, device)
        (folder/'calibration.json').write_text(json.dumps(calibration | {'lambda_cov': weight}, indent=2)+'\n')
    optimizer, scheduler = make_optimizer(model, track)
    selector, history = Selector(), []
    identity = {'track': track, 'backbone': backbone, 'variant': variant, 'seed': seed, 'smoke': args.smoke, 'epochs': 1 if args.smoke else 50}
    val_pairs = data['val_pairs']
    if args.smoke:
        # Keep one example for every key so validation aggregation remains valid.
        _, first = np.unique(val_pairs[:, 2], return_index=True)
        val_pairs = val_pairs[np.sort(first)]
    scores, _ = evaluate_pairs(model, data, val_pairs, resolver, scaler, device)
    selector.consider(scores, 0)
    history.append({'epoch': 0, 'validation': scores, 'selected': True})
    save_checkpoint(folder/'selected.pt', model, scaler, feature_stats, pool_stats, identity, selector, 0, optimizer, scheduler, weight, history)
    for epoch in range(1, identity['epochs']+1):
        pairs = make_pairs(data['manifest'], 'train', seed=seed, epoch=epoch)
        losses = []
        for batch in batches(data, pairs):
            losses.append(update(model, optimizer, data, batch, resolver, scaler, teacher, aux, weight, epoch, device))
            if args.smoke:
                break
        means = np.mean(losses, axis=0)
        scheduler.step(float(means[0]))
        scores, _ = evaluate_pairs(model, data, val_pairs, resolver, scaler, device)
        selected = selector.consider(scores, epoch)
        history.append({'epoch': epoch, 'reconstruction': float(means[0]), 'objective': float(means[1]), 'validation': scores, 'selected': selected})
        if selected:
            save_checkpoint(folder/'selected.pt', model, scaler, feature_stats, pool_stats, identity, selector, epoch, optimizer, scheduler, weight, history)
        save_checkpoint(folder/'last.pt', model, scaler, feature_stats, pool_stats, identity, selector, epoch, optimizer, scheduler, weight, history)
        (folder/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(f'reconstruction {track}/{backbone}/{variant}/seed={seed} epoch={epoch}: {scores}', flush=True)
    return folder/'selected.pt'


def selections(args):
    tracks, seeds = coordinates(args)
    backbones = BACKBONES if args.backbone == 'all' else (args.backbone,)
    variants = ('base', 'affecteeg') if args.variant in ('both', 'all') else (args.variant,)
    if any(b not in BACKBONES for b in backbones) or any(v not in ('base', 'affecteeg') for v in variants):
        raise ValueError('unknown reconstruction backbone or variant')
    return tracks, seeds, backbones, variants


def run(args):
    from ..teacher import extract_content_features
    tracks, seeds, backbones, variants = selections(args)
    for track in tracks:
        data = load_cevb(args.processed_root, track)
        teacher = None
        if 'affecteeg' in variants:
            path = extract_content_features(args.processed_root, track, args.output_root, device=args.device, batch_size=128)
            teacher = teacher_targets(data, np.load(path, mmap_mode='r'))
        for backbone in backbones:
            for seed in seeds:
                for variant in variants:
                    train_one(args, data, track, backbone, variant, seed, teacher)


def evaluate(args):
    tracks, seeds, backbones, variants = selections(args)
    device = torch.device(args.device)
    results = []
    for track in tracks:
        data = load_cevb(args.processed_root, track)
        for backbone in backbones:
            for variant in variants:
                for seed in seeds:
                    folder = run_directory(args.output_root, track, backbone, variant, seed, args.smoke)
                    state = torch.load(folder/'selected.pt', map_location='cpu', weights_only=False)
                    identity = {'track': track, 'backbone': backbone, 'variant': variant, 'seed': seed, 'smoke': args.smoke, 'epochs': 1 if args.smoke else 50}
                    if state['identity'] != identity:
                        raise ValueError('selected reconstruction checkpoint identity differs')
                    completed = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
                    if completed['epoch'] != identity['epochs']:
                        raise ValueError('evaluate requires a completed training budget')
                    model = ReconstructionModel(backbone, variant, state['feature_statistics'], state['pool_statistics']).to(device)
                    model.load_state_dict(state['state_dict'])
                    scaler = MelStandardizer(**state['standardizer'])
                    resolver = FrozenAffectResolver(args.output_root, track, seed, data, device, args.smoke) if variant == 'affecteeg' else None
                    scores, rows = evaluate_pairs(model, data, data['test_pairs'], resolver, scaler, device)
                    for row in rows:
                        row.update(track=track, backbone=backbone, variant=variant, seed=seed)
                    result = identity | {'selected_epoch': state['selected_epoch'], **scores}
                    results.append(result)
                    (folder/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
                    with (folder/'test_metrics.csv').open('w', newline='') as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
    grouped = defaultdict(list)
    for result in results:
        grouped[(result['track'], result['backbone'], result['variant'])].append(result)
    summary = []
    for (track, backbone, variant), group in grouped.items():
        row = {'track': track, 'backbone': backbone, 'variant': variant, 'seeds': [g['seed'] for g in group]}
        for metric in ('pcc', 'l1', 'rfe'):
            values = [g[metric] for g in group]
            row[metric+'_mean'] = float(np.mean(values))
            row[metric+'_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    destination = Path(args.output_root)/'reconstruction'/'summary.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'complete_paper_matrix': len(results) == 140 and not args.smoke, 'runs': results, 'summary': summary}, indent=2)+'\n')
    return destination
