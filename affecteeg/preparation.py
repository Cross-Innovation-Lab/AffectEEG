"""Build tone inputs and CEVB directly from the separately acquired raw corpora.

No network access or historical feature caches are used. SEED-IV deliberately
retains the paper experiments' historical 4,000 -> 800 sample conversion:
4,000 samples at the actual 200 Hz source rate span TWENTY seconds.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.signal import resample

from .data import PROTOCOL_DIR, load_protocol, make_pairs

CONTENT_CHANNELS = (
    "Fpz Fp1 Fp2 AF3 AF4 AF7 AF8 Fz F1 F2 F3 F4 F5 F6 F7 F8 FCz FC1 FC2 FC3 FC4 FC5 FC6 FT7 FT8 "
    "Cz C1 C2 C3 C4 C5 C6 T7 T8 CP1 CP2 CP3 CP4 CP5 CP6 TP7 TP8 Pz P3 P4 P5 P6 P7 P8 "
    "POz PO3 PO4 PO5 PO6 PO7 PO8 Oz O1 O2"
).split()
EMOTION_CHANNELS = (
    "Fp1 Fpz Fp2 AF3 AF4 F7 F5 F3 F1 Fz F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 "
    "T7 C5 C3 C1 Cz C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 P7 P5 P3 P1 Pz P2 P4 P6 P8 "
    "PO7 PO5 PO3 POz PO4 PO6 PO8 CB1 O1 Oz O2 CB2"
).split()
EMOTION_NAMES = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")
EMOTION_ALIASES = {name: i for i, name in enumerate(EMOTION_NAMES)} | {"happiness": 1, "sadness": 2, "anger": 3}
SEED_IV_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
SEED_IV_TO_GLOBAL = {0: 0, 1: 2, 2: 4, 3: 1}


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required source file is missing: {path}")
    return path


def _write_completion(path: Path, value: dict) -> None:
    """Expose the completed dataset only after all arrays have been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf8")
    temporary.replace(path)


def vocabulary() -> dict[tuple[int, int], str]:
    with (PROTOCOL_DIR / "vocabulary.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {(int(r["id"]), int(r["tone"]) - 1): r["vocal"].strip() for r in rows}
    if len(rows) != 120 or set(mapping) != {(g, t) for g in range(1, 31) for t in range(4)}:
        raise ValueError("vocabulary.csv must contain all 120 unique (group, tone) rows")
    return mapping


def event_groups(directory: Path) -> dict[int, list[int]]:
    files = [p for p in directory.glob("*evtvar.txt") if not p.name.startswith("._")]
    if len(files) != 1:
        raise FileNotFoundError(f"expected one *evtvar.txt event file in {directory}, found {len(files)}")
    groups = {t: [] for t in range(4)}
    for line in files[0].read_text(encoding="utf8").splitlines():
        if "Time-group-tone-times" not in line:
            continue
        fields = line.split("\t")
        try:
            group, tone = int(fields[5]), int(fields[6]) - 1
        except (ValueError, IndexError) as exc:
            raise ValueError(f"invalid stimulus event in {files[0]}: {line}") from exc
        if tone not in groups or not 1 <= group <= 30:
            raise ValueError(f"invalid group/tone in {files[0]}: {group}/{tone + 1}")
        groups[tone].append(group)
    if any(len(value) != 120 for value in groups.values()):
        raise ValueError(f"each tone needs 120 events in {files[0]}")
    return groups


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def tone_preprocess(eeg: np.ndarray, emg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Raw (trial, time, channel) -> standardized 500-sample inputs, no differences.

    The 50-sample EMG envelope and [-1.5, 0] EEG baseline operate on 2,000
    samples, reproducing the original 500 Hz preprocessing intermediate.
    """
    eeg = resample(eeg, 2000, axis=1).astype(np.float32)
    emg = np.abs(resample(emg, 2000, axis=1).astype(np.float32))
    times = np.linspace(-2, 2, 2000)
    baseline = eeg[:, (times >= -1.5) & (times <= 0), :]
    eeg = (eeg - baseline.mean(axis=1, keepdims=True)) / (baseline.std(axis=1, keepdims=True) + 1e-6)
    eeg = _zscore(eeg)
    kernel = np.ones(50) / 50
    for trial in range(len(emg)):
        for channel in range(emg.shape[2]):
            emg[trial, :, channel] = np.convolve(emg[trial, :, channel], kernel, mode="same")
    emg = _zscore(emg)
    return resample(eeg, 500, axis=1).astype(np.float32), resample(emg, 500, axis=1).astype(np.float32)


def read_mandarin_pair(eeg_path: Path, emg_path: Path | None = None):
    import mne
    eeg = mne.read_epochs_eeglab(str(_require(eeg_path)), verbose=False)
    if len(eeg) != 120 or set(eeg.ch_names) != set(CONTENT_CHANNELS) or len(eeg.ch_names) != 59:
        raise ValueError(f"expected 120 trials and 59 EEG channels: {eeg_path}")
    eeg.reorder_channels(CONTENT_CHANNELS)
    eeg_array = eeg.get_data(copy=True).transpose(0, 2, 1)
    if emg_path is None:
        return eeg_array, None, None
    emg = mne.read_epochs_eeglab(str(_require(emg_path)), verbose=False)
    if len(emg) != 120 or len(emg.ch_names) != 5:
        raise ValueError(f"expected 120 trials and 5 EMG channels: {emg_path}")
    for field in ("events", "selection"):
        if not np.array_equal(getattr(eeg, field), getattr(emg, field)):
            raise ValueError(f"EEG/EMG {field} alignment mismatch: {eeg_path}")
    return eeg_array, emg.get_data(copy=True).transpose(0, 2, 1), list(emg.ch_names)


def _xlsx_rows(path: Path):
    import openpyxl
    workbook = openpyxl.load_workbook(_require(path), read_only=True, data_only=True)
    try:
        yield from workbook.active.iter_rows(values_only=True)
    finally:
        workbook.close()


def build_audio_index(root: Path) -> list[dict]:
    """Match the official MCAE spreadsheet to the 120-row Mandarin vocabulary."""
    iterator = iter(_xlsx_rows(root / "corpus information.xlsx"))
    header = [str(value).strip() for value in next(iterator)]
    required = {"filename", "pinyin", "tone", "emotioncode"}
    if not required <= set(header):
        raise ValueError(f"MCAE spreadsheet is missing columns: {required - set(header)}")
    pairs = {(vocal, tone): group for (group, tone), vocal in vocabulary().items()}
    index = []
    for values in iterator:
        row = dict(zip(header, values))
        if row["filename"] is None:
            continue
        pair = (str(row["pinyin"]).strip(), int(row["tone"]) - 1)
        if pair not in pairs:
            continue
        filename = str(row["filename"]).strip().removesuffix(".0").removesuffix(".wav")
        name = str(row["emotioncode"]).strip().lower()
        if name not in EMOTION_ALIASES or len(filename) != 6 or filename[0] not in "123456":
            raise ValueError(f"invalid MCAE matched row: {row}")
        relative_path = f"Audio files/{name}/{filename}.wav"
        _require(root / relative_path)
        index.append(dict(filename=filename, relative_path=relative_path, actor=int(filename[0]),
                          vocal=pair[0], tone=pair[1], group=pairs[pair], emotion=EMOTION_ALIASES[name]))
    if len(index) != 1307 or len({r["filename"] for r in index}) != 1307:
        raise ValueError(f"expected 1,307 distinct matched MCAE clips; found {len(index)}")
    return index


def canonical_log_mel(path: Path, mel_filter=None) -> np.ndarray:
    """Two-second 24 kHz BigVGAN/Slaney log-mel with one right padding frame."""
    import librosa
    import torch
    from torch.nn import functional as F
    sample_rate, audio = wavfile.read(_require(path))
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(np.iinfo(audio.dtype).max, 1)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 24000:
        old = np.linspace(0, 1, len(audio), endpoint=False)
        new = np.linspace(0, 1, round(len(audio) * 24000 / sample_rate), endpoint=False)
        audio = np.interp(new, old, audio).astype(np.float32)
    audio = np.pad(audio[:48000], (0, max(0, 48000 - len(audio))))
    wave = F.pad(torch.from_numpy(audio)[None, None], (384, 384), mode="reflect").squeeze(1)
    spec = torch.stft(wave, 1024, hop_length=256, win_length=1024,
                     window=torch.hann_window(1024), center=False, normalized=False,
                     onesided=True, return_complex=True)
    magnitude = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    if mel_filter is None:
        mel_filter = torch.from_numpy(librosa.filters.mel(sr=24000, n_fft=1024,
            n_mels=100, fmin=0, fmax=None, htk=False, norm="slaney")).float()
    mel = torch.log((mel_filter @ magnitude).clamp_min(1e-5))
    if mel.shape[-1] != 187:
        raise ValueError(f"expected 187 valid mel frames, got {mel.shape}")
    return F.pad(mel, (0, 1), value=float(np.log(1e-5)))[0].numpy()


def actor_trimmed_mean(mels: list[np.ndarray], actors: list[int]) -> np.ndarray:
    if not mels or len(mels) != len(actors):
        raise ValueError("canonical targets require matched nonempty mel/actor inputs")
    grouped = defaultdict(list)
    for mel, actor in zip(mels, actors):
        grouped[actor].append(mel)
    actor_means = np.stack([np.asarray(grouped[a], dtype=np.float64).mean(axis=0) for a in sorted(grouped)])
    if len(actor_means) >= 3:
        actor_means = np.sort(actor_means, axis=0)[1:-1]
    return actor_means.mean(axis=0).astype(np.float32)


def emotion_window(trial: np.ndarray, index: int, track: str) -> np.ndarray:
    if track not in ("seediv", "seedvii"):
        raise ValueError("track must be seediv or seedvii")
    trial = np.asarray(trial)
    if trial.ndim != 2:
        raise ValueError("emotion trial must be a channel/time matrix")
    if trial.shape[0] != 62 and trial.shape[1] == 62:
        trial = trial.T
    if trial.shape[0] != 62:
        raise ValueError(f"expected 62 emotion channels, got {trial.shape}")
    width = 4000 if track == "seediv" else 800
    chunk = trial[:, index * width:(index + 1) * width]
    if index < 0 or chunk.shape[1] != width:
        raise ValueError(f"incomplete emotion window {index}: {chunk.shape}")
    # Legacy SEED-VII reader cast before slicing, whereas SEED-IV used float64.
    chunk = chunk.astype(np.float64 if track == "seediv" else np.float32)
    if width != 800:
        chunk = resample(chunk, 800, axis=1)
    return chunk.T.astype(np.float32)


def _seed_vii_labels(root: Path) -> dict[int, int]:
    labels, pending = {}, []
    for row in _xlsx_rows(root / "emotion_label_and_stimuli_order.xlsx"):
        first = str(row[0]).strip().lower() if row else ""
        if first == "video index":
            pending = [int(value) for value in row[1:] if isinstance(value, (int, float))]
        elif first.startswith("session") and pending:
            for trial, name in zip(pending, row[1:]):
                labels[trial] = EMOTION_ALIASES[str(name).strip().lower()]
            pending = []
    if set(labels) != set(range(1, 81)):
        raise ValueError("SEED-VII spreadsheet must label all 80 trials")
    return labels


def _validate_emotion_files(root: Path, manifest: dict) -> None:
    for relative in sorted({row["relative_path"] for row in manifest["emotion"]}):
        _require(root / relative)
    names = [str(row[0]).strip().lower() for row in _xlsx_rows(root / "Channel Order.xlsx") if row and row[0] is not None]
    if names != [name.lower() for name in EMOTION_CHANNELS]:
        raise ValueError(f"emotion channel order differs from the official 62-channel order: {root}")
    if manifest["track"] == "seedvii":
        _seed_vii_labels(root)


def _write_emotion(root: Path, destination: Path, manifest: dict) -> None:
    rows, track = manifest["emotion"], manifest["track"]
    output = np.lib.format.open_memmap(destination / "emotion_eeg.npy", mode="w+", dtype=np.float32,
                                       shape=(len(rows), 800, 62))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["relative_path"]].append((i, row))
    labels = _seed_vii_labels(root) if track == "seedvii" else None
    width = 4000 if track == "seediv" else 800
    for relative, records in groups.items():
        mat = loadmat(root / relative)
        trials = defaultdict(list)
        for i, row in records:
            trials[row["trial"]].append((i, row))
        for trial, windows in trials.items():
            if track == "seediv":
                matches = [key for key in mat if key.endswith(f"_eeg{trial + 1}")]
                if len(matches) != 1:
                    raise ValueError(f"missing or ambiguous trial {trial + 1} in {root / relative}")
                array = mat[matches[0]]
                expected_label = SEED_IV_TO_GLOBAL[SEED_IV_LABELS[windows[0][1]["session"]][trial]]
            else:
                if str(trial) not in mat:
                    raise ValueError(f"missing trial {trial} in {root / relative}")
                array, expected_label = mat[str(trial)], labels[trial]
            if array.ndim != 2 or 62 not in array.shape:
                raise ValueError(f"invalid shape for {relative}, trial {trial}: {array.shape}")
            length = array.shape[1] if array.shape[0] == 62 else array.shape[0]
            if len(windows) != length // width:
                raise ValueError(f"window count mismatch for {relative}, trial {trial}: {length // width} != {len(windows)}")
            for i, row in windows:
                if row["emotion"] != expected_label:
                    raise ValueError(f"emotion label differs from protocol: {relative}, trial {trial}")
                output[i] = emotion_window(array, row["window_index"], track)
        print(f"prepared {track}: {relative}", flush=True)
    output.flush()


def prepare(data_root: str | Path, processed_root: str | Path, task: str = "all", track: str = "all") -> dict:
    """data_root contains mandarin/, seed_iv/, seed_vii/, mcae/ (normally data/raw)."""
    if task not in ("all", "tone", "cevb") or track not in ("all", "seediv", "seedvii"):
        raise ValueError("task must be all/tone/cevb; track must be all/seediv/seedvii")
    raw, processed = Path(data_root), Path(processed_root)
    tone = task in ("all", "tone")
    tracks = (["seediv", "seedvii"] if track == "all" else [track]) if task != "tone" else []
    manifests = {name: load_protocol(name) for name in tracks}
    reference = next(iter(manifests.values()), None) or load_protocol("seediv")
    vocab, events = vocabulary(), {}
    # Fail missing inputs before allocating any dataset-sized output arrays.
    for subject in range(1, 11):
        folder = raw / "mandarin" / f"S{subject}" / "T2"
        events[subject] = event_groups(folder)
        for tone_id in range(1, 5):
            _require(folder / "All" / f"EEGnosoundstim{tone_id}.set")
            if tone:
                _require(folder / "All" / f"EMGnosoundstim{tone_id}.set")
    for row in reference["content"]:
        group = events[row["subject"]][row["tone"]][row["trial"]]
        if (group, vocab[group, row["tone"]]) != (row["group"], row["vocal"]):
            raise ValueError(f"raw Mandarin labels differ from fixed T2 protocol: {row['relative_path']} trial {row['trial']}")
    audio = build_audio_index(raw / "mcae") if tracks else []
    for name, manifest in manifests.items():
        _validate_emotion_files(raw / ("seed_iv" if name == "seediv" else "seed_vii"), manifest)
    destinations = ([processed / "tone"] if tone else []) + [processed / "cevb" / name for name in tracks]
    for directory in destinations:
        if any((directory / filename).exists() for filename in ("metadata.json", "manifest.json")):
            raise FileExistsError(f"completed prepared data already exists: {directory}; select a new output directory to rebuild")
        directory.mkdir(parents=True, exist_ok=True)
    outputs = {name: np.lib.format.open_memmap(processed / "cevb" / name / "content_eeg.npy",
        mode="w+", dtype=np.float32, shape=(4800, 800, 59)) for name in tracks}
    if tone:
        tone_eeg = np.lib.format.open_memmap(processed / "tone/eeg.npy", mode="w+", dtype=np.float32, shape=(4800, 500, 59))
        tone_emg = np.lib.format.open_memmap(processed / "tone/emg.npy", mode="w+", dtype=np.float32, shape=(4800, 500, 5))
    offset, emg_names = 0, None
    for subject in range(1, 11):
        folder = raw / "mandarin" / f"S{subject}" / "T2" / "All"
        for tone_id in range(1, 5):
            eeg, emg, names = read_mandarin_pair(folder / f"EEGnosoundstim{tone_id}.set",
                folder / f"EMGnosoundstim{tone_id}.set" if tone else None)
            if emg_names is not None and emg_names != names:
                raise ValueError(f"EMG channel order differs at {folder}")
            emg_names = names
            if tone:
                tone_eeg[offset:offset + 120], tone_emg[offset:offset + 120] = tone_preprocess(eeg, emg)
            if outputs:
                content = resample(eeg.transpose(0, 2, 1), 800, axis=2).transpose(0, 2, 1).astype(np.float32)
                for output in outputs.values():
                    output[offset:offset + 120] = content
            offset += 120
        print(f"prepared Mandarin S{subject}/T2", flush=True)
    for output in outputs.values():
        output.flush()
    if tone:
        tone_eeg.flush()
        tone_emg.flush()
        for field, filename in (("tone", "labels"), ("subject", "subjects")):
            np.save(processed / "tone" / f"{filename}.npy", np.asarray([r[field] for r in reference["content"]], dtype=np.int64))
        metadata = dict(channels=CONTENT_CHANNELS, emg_channels=emg_names,
            trials=reference["content"], condition="T2", sampling_rate=125, duration_seconds=4,
            preprocessing="500 Hz intermediate; EEG baseline [-1.5,0] and trial z-score; EMG abs/50-point moving mean and trial z-score; resample to 500; no differences stored")
        _write_completion(processed / "tone/metadata.json", metadata)
    if tracks:
        mels = defaultdict(list)
        actors = defaultdict(list)
        for row in audio:
            key = f"{row['vocal']}|{row['tone']}|{row['emotion']}"
            mels[key].append(canonical_log_mel(raw / "mcae" / row["relative_path"]))
            actors[key].append(row["actor"])
        canonical = {key: actor_trimmed_mean(value, actors[key]) for key, value in mels.items()}
        for name, manifest in manifests.items():
            available = {key for key in canonical if int(key.rsplit("|", 1)[1]) in manifest["active_emotions"]}
            if available != set(manifest["keys"]):
                raise ValueError(f"MCAE condition key inventory differs for {name}")
            directory = processed / "cevb" / name
            _write_emotion(raw / ("seed_iv" if name == "seediv" else "seed_vii"), directory, manifest)
            np.save(directory / "targets.npy", np.stack([canonical[key] for key in manifest["keys"]]))
            for split in ("val", "test"):
                np.save(directory / f"{split}_pairs.npy", make_pairs(manifest, split))
            manifest.update(content_channels=CONTENT_CHANNELS, emotion_channels=EMOTION_CHANNELS,
                emotion_names=EMOTION_NAMES, source_sample_rate_hz=200,
                emotion_source_window_samples=4000 if name == "seediv" else 800,
                emotion_source_window_seconds=20 if name == "seediv" else 4,
                emotion_output_samples=800, mel_valid_frames=187, audio_index=audio)
            _write_completion(directory / "manifest.json", manifest)
    return dict(tone_trials=4800 if tone else 0, tracks=tracks, processed_root=str(processed))
