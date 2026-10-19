"""Create the source release, excluding all downloaded/prepared data and outputs."""
from __future__ import annotations
import argparse
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"affecteeg", "vendor", "assets", "configs", "scripts", "tests"}
FILES = {"README.md", "THIRD_PARTY.md", "VALIDATION.md", "requirements.txt",
         "requirements-mamba.txt", "requirements-test.txt", "pyproject.toml", ".gitignore"}
EXCLUDED = {"__pycache__", ".pytest_cache", ".git", "outputs", "results", "logs", "checkpoints"}


def release_files(root=ROOT):
    root = Path(root)
    candidates = [root / name for name in FILES if (root / name).is_file()]
    for name in sorted(DIRECTORIES):
        directory = root / name
        if directory.is_symlink():
            raise ValueError(f"Release source directory is a symlink: {name}")
        if directory.is_dir():
            candidates.extend(directory.rglob("*"))
    for name in ("data/raw/mandarin", "data/raw/seed_iv", "data/raw/seed_vii", "data/raw/mcae", "data/processed"):
        placeholder = root / name / ".gitkeep"
        if placeholder.is_file() and not placeholder.is_symlink():
            candidates.append(placeholder)
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Release must be self-contained; symlink found: {relative}")
        if not path.is_file() or path.suffix in (".pyc", ".pyo"):
            continue
        if relative.parts[0] == "data":
            if path.name == ".gitkeep":
                yield path
            continue
        if relative.parts[0] not in DIRECTORIES and relative.as_posix() not in FILES:
            continue
        if path.suffix in (".pt", ".pth", ".ckpt", ".safetensors"):
            if relative.as_posix() != "assets/mdjpt/epoch=19.ckpt":
                raise ValueError(f"Unexpected model weight in release: {relative}")
        yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.with_suffix(".tar.gz"))
    args = parser.parse_args()
    files = list(release_files())
    with tarfile.open(args.output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=Path(ROOT.name) / path.relative_to(ROOT), recursive=False)
    print(f"Created {args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
