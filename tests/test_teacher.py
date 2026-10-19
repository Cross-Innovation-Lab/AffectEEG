import numpy as np
import torch

from affecteeg.mdjpt import FrozenMdJPTEncoder, per_channel_zscore, resample_poly_windows


def test_bundled_checkpoint_is_strictly_loaded_frozen_and_batch_independent():
    model = FrozenMdJPTEncoder()
    model.train()
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    generator = torch.Generator().manual_seed(71)
    eeg = torch.randn(3, 500, 60, generator=generator)
    with torch.no_grad():
        together = model(eeg, model.uni_channels)
        separate = torch.cat([model(x[None], model.uni_channels) for x in eeg])
    assert together.shape == (3, 512)
    assert torch.isfinite(together).all()
    torch.testing.assert_close(together, separate, atol=2e-5, rtol=2e-5)


def test_channel_order_is_explicit_and_missing_channels_are_interpolated():
    model = FrozenMdJPTEncoder()
    eeg = torch.randn(1, 500, 59)
    names = model.uni_channels[:-1]
    with torch.no_grad():
        expected = model(eeg, names)
        reordered = model(eeg.flip(-1), names[::-1])
    torch.testing.assert_close(expected, reordered, rtol=0, atol=0)


def test_reconstruction_teacher_preprocessing_preserves_four_seconds():
    eeg = np.random.default_rng(8).normal(size=(2, 800, 59)).astype(np.float32)
    normalized = per_channel_zscore(eeg)
    np.testing.assert_allclose(normalized.mean(axis=1), 0, atol=1e-6)
    assert resample_poly_windows(normalized, source_rate=200).shape == (2, 500, 59)
