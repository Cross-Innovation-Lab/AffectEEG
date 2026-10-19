"""Scratch backbone fitting and train/validation-only AffectEEG selection."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from affecteeg.common import seed_all, write_json, read_json
from . import protocol as p, adaptation as a, adaptation_math as am
from .models import build_model, head_modules
from .interfaces import FeatureExtractor, contract_from_model
from .structures import StructuralModel, cached_transform
from .io import load_checkpoint, save_checkpoint


class ToneDataset(Dataset):
    def __init__(self, data, rows, backbone):
        self.data, self.rows, self.backbone = data, np.asarray(rows), backbone
        names = [str(c).upper() for c in data['metadata']['channels']]
        wanted = p.CAT_CHANNELS if backbone == 'cat_net' else p.EEG20
        missing = [c for c in wanted if c.upper() not in names]
        if missing:
            raise ValueError(f'Mandarin data missing required EEG electrodes: {missing}')
        self.indices = [names.index(c.upper()) for c in wanted]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        eeg = np.asarray(self.data['eeg'][row], dtype=np.float32)[:, self.indices]
        if self.backbone == 'cat_net':
            emg = np.asarray(self.data['emg'][row], dtype=np.float32)
            eeg = np.concatenate((eeg[1:], np.diff(eeg, axis=0)), -1)
            emg = np.concatenate((emg[1:], np.diff(emg, axis=0)), -1)
        else:
            emg = np.empty((500, 0), dtype=np.float32)
        return (torch.from_numpy(eeg.copy()), torch.from_numpy(emg.copy()),
                int(self.data['labels'][row]), int(self.data['subjects'][row]) - 1)


def loader(data, rows, backbone, batch_size, shuffle=False, seed=42):
    return DataLoader(ToneDataset(data, rows, backbone), batch_size=batch_size,
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0)


@torch.no_grad()
def native_logits(model, data, rows, backbone, device, batch_size=128):
    model.eval()
    return np.concatenate([model(e.to(device), g.to(device))[0].float().cpu().numpy()
        for e, g, _, _ in loader(data, rows, backbone, batch_size)])


@torch.no_grad()
def features(model, data, rows, backbone, device, mode='native'):
    model.eval().requires_grad_(False)
    extractor = FeatureExtractor(backbone, model)
    output = {}
    # Feature extraction is FP32 for exact parity with the frozen downstream tail.
    for eeg, emg, _, _ in loader(data, rows, backbone, min(64, p.BASE_RECIPES[backbone]['batch_size'])):
        eeg, emg = eeg.to(device), emg.to(device)
        if mode == 'conformer256':
            tokens = model.backbone.encoder(model.backbone.embedding(eeg.permute(0, 2, 1).unsqueeze(1)))
            state = {'target': model.backbone.classifier[:3](tokens)}
        else:
            state = extractor.extract(eeg, emg)
        for key, value in state.items():
            output.setdefault(key, []).append(value.float().cpu().numpy())
    return {key: np.concatenate(values) for key, values in output.items()}


def _fit_base(backbone, data, part, seed, device, recipe, epochs=50, pilot=False):
    seed_all(seed)
    model = build_model(backbone).to(device)
    head_ids = {id(v) for module in head_modules(backbone, model) for v in module.parameters()}
    groups = []
    for is_head, lr in ((False, recipe['lr']), (True, recipe['head_lr'])):
        params = [v for v in model.parameters() if v.requires_grad and (id(v) in head_ids) == is_head]
        if params:
            groups.append(dict(params=params, lr=lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    amp = recipe['amp'] and torch.device(device).type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
        factor=.5, patience=5, min_lr=1e-5) if backbone == 'cat_net' else None)
    train_loader = loader(data, part['train'], backbone, recipe['batch_size'], True, seed)
    best, best_key, stale, history = None, (-float('inf'),), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for eeg, emg, labels, domains in train_loader:
            eeg, emg, labels, domains = (x.to(device) for x in (eeg, emg, labels, domains))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(torch.device(device).type, enabled=amp):
                logits, domain = model(eeg, emg)
                loss = F.cross_entropy(logits, labels)
                if domain.shape[-1] > 1:
                    loss = loss + .05 * F.cross_entropy(domain, domains)
                if hasattr(model, 'regularization_loss'):
                    loss = loss + model.regularization_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{backbone}: non-finite base training loss')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if hasattr(model, 'apply_constraints'):
                model.apply_constraints()
        pred = native_logits(model, data, part['validation'], backbone, device, recipe['batch_size'])
        metrics = am._metrics(data['labels'][part['validation']], pred)
        key = ((metrics['accuracy'],) if backbone == 'cat_net' else am._metric_key(metrics))
        history.append(dict(epoch=epoch, validation_metrics=metrics))
        if key > best_key:
            best_key, stale = key, 0
            best = dict(model_state=am._cpu_state(model), selected_epoch=epoch,
                        validation_metrics=metrics)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(metrics['accuracy'])
        if not pilot and stale >= 10:
            break
    model.load_state_dict(best['model_state'])
    return model, dict(**best, history=history, recipe=recipe, initialization='scratch')


def coordinate_partitions(data, seed, fold, smoke=False):
    part = p.split(data['subjects'], seed, fold)
    if not smoke and tuple(len(part[k]) for k in ('train','validation','test')) != (3840,480,480):
        raise ValueError('Main tone protocol requires 3840/480/480 train/validation/test trials')
    if smoke:
        # Preserve all tones for balanced covariance batches in diagnostic fits.
        for key in ('train','validation'):
            part[key] = np.concatenate([part[key][data['labels'][part[key]] == tone][:32 if key=='train' else 4] for tone in range(4)])
    return part


def validate_coordinate(saved, backbone, seed, fold, smoke, partitions):
    expected = dict(backbone=backbone, seed=int(seed), fold=int(fold), smoke=bool(smoke))
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f'Tone checkpoint coordinate mismatch for {key}: {saved.get(key)!r} != {value!r}')
    observed = saved.get('partitions', {})
    if set(observed) != set(partitions) or any(not np.array_equal(observed[key], partitions[key]) for key in partitions):
        raise ValueError('Tone checkpoint coordinate partitions differ')


def fit_base(backbone, data, part, seed, device, folder, smoke=False):
    folder = Path(folder)
    path = folder / 'base.pt'
    test_subjects = np.unique(data['subjects'][part['test']])
    if len(test_subjects) != 1:
        raise ValueError('A tone fold must hold out exactly one participant')
    fold = int(test_subjects[0])
    if path.exists():
        saved = load_checkpoint(path)
        validate_coordinate(saved, backbone, seed, fold, smoke, part)
        model = build_model(backbone).to(device)
        model.load_state_dict(saved['model_state'])
        return model, saved
    recipe = copy.deepcopy(p.BASE_RECIPES[backbone])
    pilot_scores = []
    if backbone == 'cat_net':
        for i, lr in enumerate((1e-4, 3e-4, 1e-3)):
            pilot_recipe = dict(recipe, lr=lr, head_lr=lr)
            pilot_model, result = _fit_base(backbone, data, part, seed + 1009*(i+1),
                device, pilot_recipe, 1 if smoke else 15, pilot=True)
            pilot_scores.append(dict(lr=lr, validation_accuracy=result['validation_metrics']['accuracy']))
            del pilot_model
        recipe['lr'] = max(pilot_scores, key=lambda r: r['validation_accuracy'])['lr']
        recipe['head_lr'] = recipe['lr']
    model, result = _fit_base(backbone, data, part, seed, device, recipe, 1 if smoke else 50)
    result.update(backbone=backbone, seed=seed, fold=fold, smoke=smoke, partitions=part,
                  lr_pilot=pilot_scores, protocol='main-paper-tone-v1')
    save_checkpoint(path, result)
    return model, result


def _reference(contract, train, val, y, vy, statistics, seed, device, epochs, config=None,
               factory=a.FitModel):
    config = dict(id='reference', feature_mode='native', head_lr=1e-3,
        adapter_lr=0., rank=16, rho=.1, kd_weight=0., covariance_weight=0.,
        teacher_view='original', covariance='global_centered', covariance_batch_size=128) if config is None else config
    original = dict(head_state=contract['head']['state_dict'])
    model = a.make_model(contract, statistics, config, original, 'identity', seed, device, factory)
    original.update(train_logits=a.logits(model, a.tensors(train, device)),
                    validation_logits=a.logits(model, a.tensors(val, device)))
    return a.fit(contract=contract, train=train, validation=val, train_labels=y,
        validation_labels=vy, nuisance=np.zeros((len(y), 1), dtype=np.float32), statistics=statistics, config=config,
        reference=original, arm='identity', calibration={'lambda': 0.}, seed=seed,
        device=device, epochs=epochs, model_factory=factory)


def fit_affect(backbone, model, data, part, teacher, seed, fold, device, folder, smoke=False):
    folder = Path(folder)
    selected_path = folder / 'affecteeg.pt'
    if selected_path.exists():
        saved = load_checkpoint(selected_path)
        validate_coordinate(saved, backbone, seed, fold, smoke, part)
        return saved
    model.eval().requires_grad_(False)
    contract = contract_from_model(backbone, model)
    train = features(model, data, part['train'], backbone, device)
    val = features(model, data, part['validation'], backbone, device)
    statistics = am.feature_statistics(train['target'])
    y, vy = data['labels'][part['train']], data['labels'][part['validation']]
    teacher = np.asarray(teacher[part['train']], dtype=np.float32)
    candidate_seed = seed + 1009*fold + 53
    epochs = 1 if smoke else 30
    reference = _reference(contract, train, val, y, vy, statistics, candidate_seed,
                           device, epochs)
    save_checkpoint(folder / 'reference.pt', reference)
    rows, artifacts, identities = [], {}, {}

    def fit_configs(configs, current_train, current_val, current_contract, stats, ref, factory):
        for config in configs:
            nuisance, transform = a.nuisance_view(teacher, y, config['teacher_view'])
            calibration = a.calibrate(current_contract, current_train, y, nuisance, stats,
                config, ref, candidate_seed, device, factory)
            # Head-only and no-covariance are internal selection controls only.
            for arm in ('arra', 'identity', 'no_cov'):
                cid = config['id'] + '__' + arm
                identity_key = (config['feature_mode'], config['head_lr'], config['kd_weight'])
                if arm == 'identity' and identity_key in identities:
                    continue
                path = folder / 'candidates' / f'{cid}.pt'
                if path.exists():
                    fitted = load_checkpoint(path)
                else:
                    fitted = a.fit(contract=current_contract, train=current_train,
                        validation=current_val, train_labels=y, validation_labels=vy,
                        nuisance=nuisance, statistics=stats, config=config, reference=ref,
                        arm=arm, calibration=calibration, seed=candidate_seed, device=device,
                        epochs=epochs, model_factory=factory)
                    fitted.update(contract=current_contract, nuisance_transform=transform)
                    save_checkpoint(path, fitted)
                row = dict(candidate_id=cid, arm=arm,
                    selected_epoch=fitted['selected_epoch'], validation_metrics=fitted['validation_metrics'])
                rows.append(row)
                artifacts[cid] = path
                if arm == 'identity':
                    identities[identity_key] = row
            print(f'{backbone} seed={seed} fold={fold}: {config["id"]} selected on validation', flush=True)

    configs = p.templates(backbone)
    fit_configs(configs[:1] if smoke else configs, train, val, contract, statistics, reference, a.FitModel)
    chosen = {arm: p.select([r for r in rows if r['arm'] == arm]) for arm in ('arra','identity','no_cov')}
    metrics = {arm: row['validation_metrics']['balanced_accuracy'] for arm,row in chosen.items()}
    enabled = (backbone in p.STRUCTURAL_MODES and
        (metrics['arra'] - metrics['identity'] < .005 - 1e-12 or metrics['arra'] <= metrics['no_cov'] + 1e-12))
    if enabled:
        mode = p.STRUCTURAL_MODES[backbone]
        alternate_contract = copy.deepcopy(contract)
        if mode == 'conformer256':
            alternate_train = features(model, data, part['train'], backbone, device, mode)
            alternate_val = features(model, data, part['validation'], backbone, device, mode)
            alternate_contract['structural_tail_state'] = am._cpu_state(model.backbone.classifier[3:7])
        else:
            alternate_train = cached_transform(train, contract, mode)
            alternate_val = cached_transform(val, contract, mode)
        alternate_stats = am.feature_statistics(alternate_train['target'])
        config = p.structural_templates(backbone)[0]
        # Structural views preserve the selected reference head. Recompute its
        # logits through the exact alternate tail before zero-residual parity.
        alternate_reference = copy.deepcopy(reference)
        ref_model = a.make_model(alternate_contract, alternate_stats, config,
            reference, 'identity', candidate_seed, device, StructuralModel)
        alternate_reference['train_logits'] = a.logits(ref_model, a.tensors(alternate_train, device))
        alternate_reference['validation_logits'] = a.logits(ref_model, a.tensors(alternate_val, device))
        alternate_configs = p.structural_templates(backbone)
        fit_configs(alternate_configs[:1] if smoke else alternate_configs,
            alternate_train, alternate_val, alternate_contract, alternate_stats,
            alternate_reference, StructuralModel)
    best = p.select([r for r in rows if r['arm'] == 'arra'])
    selected = load_checkpoint(artifacts[best['candidate_id']])
    selected.update(backbone=backbone, seed=seed, fold=fold, smoke=smoke, partitions=part,
        candidate_id=best['candidate_id'], structural_enabled=enabled)
    save_checkpoint(selected_path, selected)
    write_json(folder / 'selection.json', dict(backbone=backbone, seed=seed, fold=fold,
        smoke=smoke, structural_enabled=enabled, candidates=rows, selected=best,
        test_accessed=False))
    return selected


def _coordinates(args):
    names = p.BACKBONES if args.backbone == 'all' else (args.backbone,)
    seeds = p.SEEDS if str(args.seed) == 'all' else (int(args.seed),)
    folds = p.SUBJECTS if str(args.fold) == 'all' else (int(args.fold),)
    if any(n not in p.BACKBONES for n in names) or any(s not in p.SEEDS for s in seeds) or any(f not in p.SUBJECTS for f in folds):
        raise ValueError('Invalid tone backbone, seed, or fold')
    return [(n,s,f) for n in names for s in seeds for f in folds]


def output_folder(args):
    return Path(args.output_root) / 'tone'


def run(args):
    from affecteeg.data import load_tone
    from affecteeg.teacher import extract_tone_features
    data = load_tone(args.processed_root)
    teacher = None
    root = output_folder(args)
    for backbone, seed, fold in _coordinates(args):
        part = coordinate_partitions(data, seed, fold, args.smoke)
        folder = root / backbone / f'seed_{seed}' / f'fold_{fold:02d}'
        model, base = fit_base(backbone, data, part, seed, args.device, folder, args.smoke)
        if args.variant in ('both','affecteeg'):
            if teacher is None:
                teacher_path = extract_tone_features(args.processed_root, args.output_root,
                    device=args.device, batch_size=128)
                teacher = np.load(teacher_path, mmap_mode='r') if isinstance(teacher_path, (str, Path)) else teacher_path
            fit_affect(backbone, model, data, part, teacher, seed, fold, args.device, folder, args.smoke)
        del model
    write_json(root / 'training_complete.json', dict(coordinates=_coordinates(args),
               variant=args.variant, smoke=args.smoke, test_accessed=False))
    return dict(output=str(root), status='training_complete', test_accessed=False)
