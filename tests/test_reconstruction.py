import os
import numpy as np
import pytest
import torch
from affecteeg.reconstruction.models import BACKBONES, ReconstructionModel
from affecteeg.reconstruction.arra import POOL_SCHEMA, fit_targets, grouped_covariance, covariance_correction
from affecteeg.reconstruction.losses import MelStandardizer, corrected_shape_loss, reconstruction_loss, item_metrics, Selector, equal_metrics
from affecteeg.reconstruction.runner import fit_initial_statistics, normalize_content


def statistics():
    return ({'mean': np.zeros((800, 256), np.float32), 'sigma': np.ones((800, 256), np.float32)},
            {'schema': POOL_SCHEMA, 'statistics_split': 'train', 'mean': np.zeros(2048, np.float32),
             'sigma': np.ones(2048, np.float32), 'active': np.ones(2048, bool)})


@pytest.mark.parametrize('backbone', BACKBONES)
@pytest.mark.parametrize('variant', ('base', 'affecteeg'))
def test_each_backbone_forward_update_and_checkpoint(backbone, variant, tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    device = torch.device(os.environ.get('AFFECTEEG_TEST_DEVICE', 'cpu'))
    stats, pool = statistics()
    model = ReconstructionModel(backbone, variant, stats, pool).to(device)
    x, token = torch.randn(2, 800, 59, device=device), torch.randn(2, 128, device=device)
    target = torch.randn(2, 100, 188, device=device)
    scaler = MelStandardizer(torch.zeros(1, 100, 1), torch.ones(1, 100, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = next(model.content_encoder.parameters()).detach().clone()
    result = model(x, token if variant == 'affecteeg' else None)
    assert result.mel.shape == (2, 100, 188)
    loss = reconstruction_loss(result.mel, target, scaler)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.content_encoder.parameters()))
    if variant == 'affecteeg':
        assert torch.count_nonzero(model.adapter.up.weight) > 0
        assert torch.count_nonzero(model.fusion.up.weight) > 0
    model.eval()
    with torch.no_grad():
        expected = model(x, token if variant == 'affecteeg' else None).mel
    torch.save(model.state_dict(), tmp_path/'state.pt')
    restored = ReconstructionModel(backbone, variant, stats, pool).to(device)
    restored.load_state_dict(torch.load(tmp_path/'state.pt', map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(x, token if variant == 'affecteeg' else None).mel
    torch.testing.assert_close(actual, expected)


def test_identity_auxiliary_gradients_modes_rng_and_preservation():
    torch.set_num_threads(2)
    stats, pool = statistics()
    model = ReconstructionModel('convconcatnet', 'affecteeg', stats, pool)
    model.train()
    memory = torch.randn(8, 800, 256)
    token = torch.randn(8, 128)
    assert torch.equal(model.adapter(memory), memory)
    assert torch.equal(model.fusion(memory, token)[0], memory)
    before_rng = torch.get_rng_state().clone()
    before_buffers = {name: b.clone() for name, b in model.content_encoder.named_buffers()}
    loss = model.auxiliary_covariance(torch.randn(8, 800, 59), torch.randn(8, 32), np.zeros(8, dtype=int))
    # Inputs above intentionally draw RNG; isolate the encoding itself separately.
    input_value = torch.randn(2, 800, 59)
    before_rng = torch.get_rng_state().clone()
    model.encode_auxiliary(input_value)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert model.content_encoder.training
    assert all(torch.equal(before_buffers[n], b) for n, b in model.content_encoder.named_buffers())
    loss.backward()
    assert all(p.grad is None for p in model.content_encoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    assert model.adapter.up.weight.grad.norm() > 0
    model.zero_grad(set_to_none=True)
    model.adapter.up.bias.data.fill_(.1)
    before = model.content_encoder(torch.randn(2, 800, 59))
    preservation = model.adapter.preservation(before.detach(), model.adapter(before.detach()))
    preservation.backward()
    assert all(p.grad is None for p in model.content_encoder.parameters())


def test_conflicting_covariance_gradient_is_projected():
    task = (torch.tensor([1., 0.]),)
    cov = (torch.tensor([-2., 3.]),)
    correction, geometry = covariance_correction(task, cov)
    assert geometry['projection_applied']
    torch.testing.assert_close(cov[0]+correction[0], torch.tensor([0., 3.]))


def test_teacher_pca_uses_training_only():
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(160, 48))
    groups = np.repeat(np.arange(4), 40)
    splits = np.tile(['train']*32+['val']*4+['test']*4, 4)
    a, stats_a = fit_targets(embeddings, groups, splits)
    embeddings[splits != 'train'] += 1e5
    b, stats_b = fit_targets(embeddings, groups, splits)
    np.testing.assert_array_equal(a[splits == 'train'], b[splits == 'train'])
    for key in stats_a:
        np.testing.assert_array_equal(stats_a[key], stats_b[key])


def test_initial_memory_statistics_ignore_heldout_sources():
    torch.set_num_threads(2)
    rng = np.random.default_rng(5)
    eeg = rng.normal(size=(6, 800, 59)).astype('float32')
    data = {'content_eeg': eeg, 'manifest': {'content': [{'split': 'train'}]*4+[{'split': 'val'}, {'split': 'test'}]}}
    model = ReconstructionModel('convconcatnet')
    a, pooled_a = fit_initial_statistics(model, data, torch.device('cpu'))
    eeg[4:] = 1e10
    b, pooled_b = fit_initial_statistics(model, data, torch.device('cpu'))
    np.testing.assert_array_equal(a['mean'], b['mean'])
    np.testing.assert_array_equal(pooled_a['sigma'], pooled_b['sigma'])


def test_corrected_loss_is_normalized_per_frequency_element():
    torch.manual_seed(4)
    p = torch.randn(3, 1, 188)
    y = torch.randn(3, 1, 188)
    # Identical frequency copies must not multiply the SmoothL1 term by 100.
    torch.testing.assert_close(corrected_shape_loss(p.repeat(1, 3, 1), y.repeat(1, 3, 1)),
                               corrected_shape_loss(p.repeat(1, 100, 1), y.repeat(1, 100, 1)))


def test_metrics_ignore_padded_frame_and_selector_order():
    torch.manual_seed(5)
    target = torch.randn(3, 100, 188)
    prediction = target.clone()
    prediction[..., -1] += 1e6
    np.testing.assert_allclose(item_metrics(prediction, target), [[1, 0, 0]]*3, atol=1e-12)
    selector = Selector()
    assert selector.consider({'pcc': .6, 'l1': 2}, 1)
    assert not selector.consider({'pcc': .6, 'l1': 2}, 2)
    assert selector.consider({'pcc': .6, 'l1': 1}, 2)
    assert not selector.consider({'pcc': .5, 'l1': 0}, 3)
    rows = [dict(key='a|1|0', pcc=1, l1=1, rfe=1)]*5 + [dict(key='b|1|0', pcc=0, l1=0, rfe=0), dict(key='a|1|1', pcc=0, l1=0, rfe=0)]
    assert equal_metrics(rows)['pcc'] == .25


def test_complete_joint_update_changes_encoder_and_adapter():
    from affecteeg.reconstruction.runner import update, make_optimizer
    torch.set_num_threads(2)
    stats, pool = statistics()
    model = ReconstructionModel('convconcatnet', 'affecteeg', stats, pool)
    optimizer, _ = make_optimizer(model, 'seediv')
    data = {'content_eeg': np.random.default_rng(42).normal(size=(8, 800, 59)).astype('float32')}
    teacher = {'targets': np.random.default_rng(43).normal(size=(8, 32)).astype('float32')}
    class Sampler:
        def sample(self):
            return np.arange(8), np.zeros(8, dtype=int)
    batch = {'eeg': torch.randn(2, 800, 59), 'emotion': torch.arange(2), 'mel': torch.randn(2, 100, 188)}
    def resolver(rows):
        return torch.ones(len(rows), 128)
    scaler = MelStandardizer(torch.zeros(1, 100, 1), torch.ones(1, 100, 1))
    before = next(model.content_encoder.parameters()).detach().clone()
    rec, objective = update(model, optimizer, data, batch, resolver, scaler, teacher, Sampler(), 1., 1, torch.device('cpu'))
    assert np.isfinite([rec, objective]).all()
    assert not torch.equal(before, next(model.content_encoder.parameters()))
    assert model.adapter.up.weight.detach().norm() > 0
