"""Run with python -m affecteeg from the standalone package directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import PACKAGE_ROOT, RECONSTRUCTION_SEEDS, TONE_SEEDS


def parser():
    root = argparse.ArgumentParser(description="AffectEEG main experiments: prepare, train and evaluate.")
    sub = root.add_subparsers(dest="command", required=True)

    def paths(p, runtime=True, seeds=RECONSTRUCTION_SEEDS):
        p.add_argument("--processed-root", type=Path, default=PACKAGE_ROOT / "data" / "processed")
        p.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "outputs")
        if runtime:
            p.add_argument("--device", default="auto", help="cpu, cuda, cuda:N, or auto")
            p.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
            p.add_argument("--seed", choices=("all", *(str(seed) for seed in seeds)),
                           default="all", help="One manuscript training seed or all seeds for the selected task")

    prepare = sub.add_parser("prepare", help="Build prepared data from downloaded source files")
    paths(prepare, runtime=False)
    prepare.add_argument("--raw-root", type=Path, default=PACKAGE_ROOT / "data" / "raw")
    prepare.add_argument("--task", choices=("all", "tone", "cevb"), default="all")
    prepare.add_argument("--track", choices=("all", "seediv", "seedvii"), default="all")

    affect = sub.add_parser("train-affect", help="Extract MD-JPT features and train the MAET-DE affect models")
    paths(affect)
    affect.add_argument("--task", choices=("all", "tone", "reconstruction"), default="all")
    affect.add_argument("--track", choices=("all", "seediv", "seedvii"), default="all")
    affect.add_argument("--smoke", action="store_true", help="Short diagnostic fit saved under the smoke output directory")

    for name, description in (("train-tone", "Train native and adapted tone classifiers"),
                               ("train-reconstruction", "Train independent Base and full AffectEEG reconstructors")):
        p = sub.add_parser(name, help=description)
        paths(p, seeds=TONE_SEEDS if name == "train-tone" else RECONSTRUCTION_SEEDS)
        p.add_argument("--backbone", default="all")
        p.add_argument("--variant", choices=("both", "base", "affecteeg"), default="both")
        p.add_argument("--smoke", action="store_true", help="Short diagnostic fit saved under the smoke output directory")
        if name == "train-tone":
            p.add_argument("--fold", default="all", help="Held-out participant 1..10 or all")
        else:
            p.add_argument("--track", choices=("all", "seediv", "seedvii"), default="all")

    evaluate = sub.add_parser("evaluate", help="Evaluate selected checkpoints and summarize paper metrics")
    paths(evaluate)
    evaluate.add_argument("--task", choices=("tone", "reconstruction"), required=True)
    evaluate.add_argument("--track", choices=("all", "seediv", "seedvii"), default="all")
    evaluate.add_argument("--backbone", default="all")
    evaluate.add_argument("--variant", choices=("both", "base", "affecteeg"), default="both")
    evaluate.add_argument("--fold", default="all")
    evaluate.add_argument("--smoke", action="store_true", help="Evaluate diagnostic outputs")
    return root


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "prepare":
        from .preparation import prepare
        return prepare(args.raw_root, args.processed_root, task=args.task, track=args.track)
    tone_task = args.command == "train-tone" or getattr(args, "task", None) == "tone"
    seeds = TONE_SEEDS if tone_task else RECONSTRUCTION_SEEDS
    if args.seed != "all" and int(args.seed) not in seeds:
        cli.error(f"--seed for the selected task must be one of {seeds}, or all")
    import torch
    from .common import resolve_device
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    args.device = resolve_device(args.device)
    if args.smoke:
        # Store diagnostic checkpoints in a dedicated output directory.
        args.output_root = args.output_root / "smoke"
    if args.command == "train-affect":
        from .teacher import extract_tone_features, extract_content_features
        if args.task in ("all", "tone"):
            extract_tone_features(args.processed_root, args.output_root, device=args.device)
        if args.task in ("all", "reconstruction"):
            tracks = ("seediv", "seedvii") if args.track == "all" else (args.track,)
            for track in tracks:
                extract_content_features(args.processed_root, track, args.output_root, device=args.device)
            from .affect import run
            return run(args)
        return None
    if args.command == "train-tone":
        from .tone import run
        return run(args)
    if args.command == "train-reconstruction":
        from .reconstruction import run
        return run(args)
    if args.task == "tone":
        from .tone import evaluate
        result = evaluate(args)
        if not args.smoke:
            from .statistics import summarize_tone
            summarize_tone(result["rows"], args.output_root)
    else:
        from .reconstruction import evaluate
        result = evaluate(args)
        if not args.smoke:
            from .statistics import summarize_reconstruction
            summarize_reconstruction(args.output_root)
    return result


if __name__ == "__main__":
    main()
