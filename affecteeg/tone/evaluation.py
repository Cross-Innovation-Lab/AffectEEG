"""Evaluate frozen selections only, then export paired participant-level records."""
from pathlib import Path
import numpy as np
from affecteeg.common import write_json
from . import adaptation as a, adaptation_math as am, protocol as p
from .models import build_model
from .structures import StructuralModel, cached_transform
from .training import (_coordinates, output_folder, native_logits, features,
                       coordinate_partitions, validate_coordinate)
from .io import load_checkpoint


def evaluate(args):
    from affecteeg.data import load_tone
    data = load_tone(args.processed_root)
    root = output_folder(args)
    variants = ('base', 'affecteeg') if args.variant == 'both' else (args.variant,)
    coordinates = _coordinates(args)
    # Validate all requested selections before reading any test EEG.
    for backbone, seed, fold in coordinates:
        folder = root / backbone / f'seed_{seed}' / f'fold_{fold:02d}'
        for name in ('base', *(["affecteeg"] if 'affecteeg' in variants else [])):
            if not (folder / f'{name}.pt').exists():
                raise FileNotFoundError(f'Finish train-tone before evaluation: {folder / (name + ".pt")}')
    records = []
    for backbone, seed, fold in coordinates:
        folder = root / backbone / f'seed_{seed}' / f'fold_{fold:02d}'
        base = load_checkpoint(folder / 'base.pt')
        part = coordinate_partitions(data, seed, fold, args.smoke)
        validate_coordinate(base, backbone, seed, fold, args.smoke, part)
        rows = part['test']
        model = build_model(backbone).to(args.device)
        model.load_state_dict(base['model_state'])
        model.eval().requires_grad_(False)
        for variant in variants:
            if variant == 'base':
                pred = native_logits(model, data, rows, backbone, args.device,
                                     p.BASE_RECIPES[backbone]['batch_size'])
            else:
                saved = load_checkpoint(folder / 'affecteeg.pt')
                validate_coordinate(saved, backbone, seed, fold, args.smoke, part)
                mode = saved['config']['feature_mode']
                state = features(model, data, rows, backbone, args.device,
                                 'conformer256' if mode == 'conformer256' else 'native')
                if mode not in ('native', 'conformer256'):
                    state = cached_transform(state, saved['contract'], mode)
                factory = a.FitModel if mode == 'native' else StructuralModel
                pred = a.predict(saved, saved['contract'], state, args.device, factory)
            labels = np.asarray(data['labels'][rows])
            record = dict(backbone=backbone, seed=seed, fold=fold, variant=variant,
                          n_trials=len(rows), **am._metrics(labels, pred))
            records.append(record)
            np.savez_compressed(folder / f'predictions_{variant}.npz',
                rows=rows, labels=labels, logits=pred, subjects=np.asarray(data['subjects'][rows]))
        del model
    result = dict(rows=records, output=str(root), smoke=args.smoke)
    write_json(root / 'evaluation.json', result)
    return result
