"""Portable array readers and the fixed CEVB pairing protocol."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "assets" / "protocol"
SPLITS = ("train", "val", "test")


def load_protocol(track: str) -> dict:
    if track not in ("seediv", "seedvii"):
        raise ValueError("track must be seediv or seedvii")
    with gzip.open(PROTOCOL_DIR / f"{track}.json.gz", "rt", encoding="utf8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict) -> None:
    keys, splits = manifest["keys"], manifest["key_splits"]
    if len(keys) != len(splits) or len(set(keys)) != len(keys):
        raise ValueError("condition keys must be unique and aligned with key_splits")
    if set(splits) != set(SPLITS):
        raise ValueError("condition partitions must include train, val and test")
    for table in ("content", "emotion"):
        seen_ids, seen_instances = set(), set()
        for row in manifest[table]:
            identity = (row["relative_path"], row["trial"])
            if table == "emotion":
                identity += (row["window_index"],)
            if row["source_id"] in seen_ids or identity in seen_instances:
                raise ValueError(f"duplicate or cross-partition source instance in {table}: {identity}")
            if row["split"] not in SPLITS:
                raise ValueError(f"unknown source partition: {row['split']}")
            seen_ids.add(row["source_id"])
            seen_instances.add(identity)
    for split, pairs in manifest.get("evaluation_pairs", {}).items():
        validate_pairs(manifest, np.asarray(pairs, dtype=np.int64), split)


def validate_pairs(manifest: dict, pairs: np.ndarray, split: str) -> None:
    if pairs.ndim != 2 or pairs.shape[1] != 3:
        raise ValueError("pairs must have shape (N, 3)")
    counts = defaultdict(int)
    for ci, ei, ki in pairs:
        if ci < 0 or ei < 0 or ki < 0:
            raise ValueError("pair indices must be nonnegative")
        c, e = manifest["content"][ci], manifest["emotion"][ei]
        key = manifest["keys"][ki]
        vocal, tone, emotion = key.rsplit("|", 2)
        if (c["split"] != split or e["split"] != split or manifest["key_splits"][ki] != split):
            raise ValueError("pair crosses a source or condition partition")
        if (c["vocal"], c["tone"], e["emotion"]) != (vocal, int(tone), int(emotion)):
            raise ValueError("pair source labels do not match its target key")
        counts[int(ki)] += 1
    expected = {i for i, value in enumerate(manifest["key_splits"]) if value == split}
    if set(counts) != expected or any(value != 100 for value in counts.values()):
        raise ValueError("each partition key must have exactly 100 pairs")


def _protocol_seed(value: dict) -> str:
    # Fixed numerical random seeds for the pairing sampler.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_pairs(manifest: dict, split: str, seed: int = 20260729, epoch: int = 0) -> np.ndarray:
    """100 pairs per key; fixed evaluation pairs and source-stable epoch sampling.

    Training exhausts each key's emotion donors before repeating one. Content
    choices use fixed random seeds with round-robin key interleaving.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split}")
    if split != "train":
        pairs = np.asarray(manifest["evaluation_pairs"][split], dtype=np.int64)
        validate_pairs(manifest, pairs, split)
        return pairs
    contents, emotions = defaultdict(list), defaultdict(list)
    for i, row in enumerate(manifest["content"]):
        if row["split"] == split:
            contents[row["vocal"], row["tone"]].append((row["source_id"], i))
    for i, row in enumerate(manifest["emotion"]):
        if row["split"] == split:
            emotions[row["emotion"]].append((row["source_id"], i))
    selected = {}
    dataset = manifest["dataset"]
    for ki, (key, assignment) in enumerate(zip(manifest["keys"], manifest["key_splits"])):
        if assignment != split:
            continue
        vocal, tone, emotion = key.rsplit("|", 2)
        content = sorted(contents[vocal, int(tone)])
        donors = sorted(emotions[int(emotion)])
        if not content or not donors:
            raise ValueError(f"empty training source pool for {key}")
        key_id = manifest["key_ids"][ki]
        base = _protocol_seed(dict(training_seed=seed, epoch=epoch, dataset=dataset, target_key_id=key_id))
        rows, occurrences, cycle = [], defaultdict(int), 0
        while len(rows) < 100:
            cycle_seed = _protocol_seed(dict(base_seed_sha256=base, donor_cycle=cycle))
            order = np.arange(len(donors))
            np.random.default_rng(int(cycle_seed[:16], 16)).shuffle(order)
            for index in order[:100 - len(rows)]:
                donor, ei = donors[index]
                choice = _protocol_seed(dict(namespace="class_style_train_vocal_choice_v1",
                    training_seed=seed, epoch=epoch, dataset=dataset, target_key_id=key_id,
                    emotion_source_id=donor, donor_occurrence=occurrences[donor]))
                occurrences[donor] += 1
                ci = content[int(choice, 16) % len(content)][1]
                rows.append((ci, ei, ki))
            cycle += 1
        selected[key] = rows
    result = []
    for draw in range(100):
        order = np.asarray(sorted(selected), dtype=object)
        order_seed = _protocol_seed(dict(namespace="class_style_train_key_interleave_v1",
            training_seed=seed, epoch=epoch, dataset=dataset, draw_index=draw))
        np.random.default_rng(int(order_seed[:16], 16)).shuffle(order)
        result.extend(selected[str(key)][draw] for key in order)
    return np.asarray(result, dtype=np.int64)


def _array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing processed array: {path}; run prepare first")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_tone(root: str | Path) -> dict:
    directory = Path(root) / "tone"
    with (directory / "metadata.json").open(encoding="utf8") as handle:
        metadata = json.load(handle)
    data = {name: _array(directory / f"{name}.npy") for name in ("eeg", "emg", "labels", "subjects")}
    n = len(data["labels"])
    if (data["eeg"].shape != (n, 500, 59) or data["emg"].shape != (n, 500, 5)
            or data["subjects"].shape != (n,) or len(metadata["trials"]) != n):
        raise ValueError("tone arrays do not follow the 500-sample 59-EEG/5-EMG schema")
    data["metadata"] = metadata
    return data


def load_cevb(root: str | Path, track: str) -> dict:
    if track not in ("seediv", "seedvii"):
        raise ValueError("track must be seediv or seedvii")
    directory = Path(root) / "cevb" / track
    with (directory / "manifest.json").open(encoding="utf8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    data = {name: _array(directory / f"{name}.npy") for name in
            ("content_eeg", "emotion_eeg", "targets", "val_pairs", "test_pairs")}
    for name, shape in (("content_eeg", (len(manifest["content"]), 800, 59)),
                        ("emotion_eeg", (len(manifest["emotion"]), 800, 62)),
                        ("targets", (len(manifest["keys"]), 100, 187))):
        if data[name].shape != shape:
            raise ValueError(f"{name} shape {data[name].shape} does not match {shape}")
    for split in ("val", "test"):
        if not np.array_equal(data[f"{split}_pairs"], make_pairs(manifest, split)):
            raise ValueError(f"{split} pair array differs from the fixed protocol")
    data["manifest"] = manifest
    return data
