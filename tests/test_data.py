import copy
import json

import numpy as np
import pytest
from scipy.io import wavfile

from affecteeg.data import load_cevb, load_protocol, load_tone, make_pairs, validate_manifest, validate_pairs
from affecteeg.preparation import actor_trimmed_mean, canonical_log_mel, emotion_window, event_groups, prepare, vocabulary


@pytest.mark.parametrize("track,windows,counts", [
    ("seediv", 37575, (12700, 3800, 3700)),
    ("seedvii", 69742, (21500, 7100, 7100)),
])
def test_full_protocol_counts_labels_and_source_partitions(track, windows, counts):
    manifest = load_protocol(track)
    assert len(manifest["content"]) == 4800
    assert len(manifest["emotion"]) == windows
    vocab = vocabulary()
    for row in manifest["content"]:
        assert "/T2/" in row["relative_path"]
        assert vocab[row["group"], row["tone"]] == row["vocal"]
    for split, count in zip(("train", "val", "test"), counts):
        pairs = make_pairs(manifest, split, seed=42, epoch=0)
        assert pairs.shape == (count, 3)
        validate_pairs(manifest, pairs, split)
        if split != "train":
            np.testing.assert_array_equal(pairs, make_pairs(manifest, split, seed=123, epoch=9))


def test_training_sampler_changes_epoch_and_exhausts_donors():
    manifest = load_protocol("seediv")
    first = make_pairs(manifest, "train", 42, 0)
    np.testing.assert_array_equal(first, make_pairs(manifest, "train", 42, 0))
    assert not np.array_equal(first, make_pairs(manifest, "train", 42, 1))
    for ki in np.unique(first[:, 2]):
        selected = first[first[:, 2] == ki, 1]
        emotion = int(manifest["keys"][ki].rsplit("|", 1)[1])
        donor_count = sum(r["split"] == "train" and r["emotion"] == emotion for r in manifest["emotion"])
        assert len(np.unique(selected[:min(donor_count, 100)])) == min(donor_count, 100)


def test_duplicate_source_and_cross_partition_pair_rejected():
    manifest = load_protocol("seediv")
    broken = copy.deepcopy(manifest)
    broken["content"][1] = dict(broken["content"][0], source_id="other", split="test")
    with pytest.raises(ValueError, match="duplicate or cross-partition"):
        validate_manifest(broken)
    broken = copy.deepcopy(manifest)
    ci = broken["evaluation_pairs"]["val"][0][0]
    broken["content"][ci]["split"] = "train"
    with pytest.raises(ValueError, match="crosses"):
        validate_manifest(broken)


def test_seediv_uses_actual_four_second_window():
    raw = np.random.default_rng(0).normal(size=(62, 1900))
    actual = emotion_window(raw, 1, "seediv")
    expected = raw[:, 800:1600].T.astype(np.float32)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="incomplete"):
        emotion_window(raw, 2, "seediv")


def test_seedvii_uses_actual_four_second_window():
    raw = np.random.default_rng(1).normal(size=(1600, 62))
    np.testing.assert_array_equal(emotion_window(raw, 1, "seedvii"), raw[800:].astype(np.float32))


def test_canonical_actor_mean_before_per_bin_trim():
    mels = [np.array([[0., 10.]]), np.array([[2., 12.]]), np.array([[5., 3.]]), np.array([[9., 7.]])]
    # Actor 1 mean=[1,11]; per-bin median of [1,5,9]/[11,3,7] is [5,7].
    np.testing.assert_array_equal(actor_trimmed_mean(mels, [1, 1, 2, 3]), [[5., 7.]])
    np.testing.assert_array_equal(actor_trimmed_mean(mels[:3], [1, 1, 2]), [[3., 7.]])


def test_canonical_mel_matches_two_second_stft(tmp_path):
    import librosa
    path = tmp_path / "two_seconds.wav"
    audio = (np.sin(2 * np.pi * 440 * np.arange(48000) / 24000) * 16000
             + np.random.default_rng(2).normal(0, 500, 48000)).astype(np.int16)
    wavfile.write(path, 24000, audio)
    mel = canonical_log_mel(path)
    assert mel.shape == (100, 187)
    assert np.isfinite(mel).all()
    waveform = np.pad(audio.astype(np.float32) / 32767, (384, 384), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(waveform, 1024)[::256]
    spectrum = np.fft.rfft(frames * np.hanning(1025)[:-1], axis=1).T
    magnitude = np.sqrt(np.abs(spectrum) ** 2 + 1e-9)
    filters = librosa.filters.mel(sr=24000, n_fft=1024, n_mels=100,
                                 fmin=0, fmax=None, htk=False, norm="slaney")
    expected = np.log(np.maximum(filters @ magnitude, 1e-5))
    np.testing.assert_allclose(mel, expected, atol=5e-5)
    assert np.any(mel[:, -1] > np.log(1e-5))


def test_stereo_integer_pcm_uses_original_amplitude_scale(tmp_path):
    mono = (np.sin(2 * np.pi * 440 * np.arange(6000) / 24000) * 16000).astype(np.int16)
    wavfile.write(tmp_path / "mono.wav", 24000, mono)
    wavfile.write(tmp_path / "stereo.wav", 24000, np.stack([mono, mono], axis=1))
    np.testing.assert_array_equal(canonical_log_mel(tmp_path / "mono.wav"),
                                  canonical_log_mel(tmp_path / "stereo.wav"))


def test_missing_raw_input_fails_before_outputs(tmp_path):
    destination = tmp_path / "processed"
    with pytest.raises(FileNotFoundError, match="S1/T2"):
        prepare(tmp_path / "raw", destination, task="tone")
    assert not destination.exists()


def test_missing_event_and_invalid_event_counts_fail(tmp_path):
    with pytest.raises(FileNotFoundError, match="evtvar"):
        event_groups(tmp_path)
    (tmp_path / "trial_evtvar.txt").write_text("1\t0\t2000\tTime-group-tone-times\ttime\t1\t1\t1\n")
    with pytest.raises(ValueError, match="120 events"):
        event_groups(tmp_path)
