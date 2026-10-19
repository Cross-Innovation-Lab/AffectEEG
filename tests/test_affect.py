import numpy as np
import pytest
import torch
from affecteeg.affect import extract_de, inner_split, fit_de_stats, MAETProjector, subject_balanced_prototypes


def test_training_coordinates_follow_task_specific_manuscript_seeds():
    from affecteeg.__main__ import parser
    from affecteeg.affect import coordinates
    from affecteeg.reconstruction.runner import selections
    from affecteeg.tone.training import _coordinates

    tone = _coordinates(parser().parse_args(['train-tone']))
    assert len(tone) == 300
    assert {seed for _, seed, _ in tone} == {42, 123, 456}
    cat = _coordinates(parser().parse_args(['train-tone', '--backbone', 'cat_net']))
    assert len(cat) == 30
    assert {seed for _, seed, _ in cat} == {42, 123, 456}
    tracks, seeds = coordinates(parser().parse_args(['train-affect']))
    assert set(seeds) == {42, 123, 456, 789, 1024}
    assert len(tracks) * len(seeds) == 10
    tracks, seeds, backbones, variants = selections(parser().parse_args(['train-reconstruction']))
    assert set(seeds) == {42, 123, 456, 789, 1024}
    assert len(tracks) * len(seeds) * len(backbones) * len(variants) == 140


def test_cli_rejects_wrong_task_seeds_before_data_loading():
    from affecteeg.__main__ import main, parser
    from affecteeg.affect import coordinates
    from affecteeg.reconstruction.runner import selections

    for seed in ('43', '44', '789', '1024'):
        with pytest.raises(SystemExit) as error:
            main(['train-tone', '--backbone', 'cat_net', '--seed', seed])
        assert error.value.code == 2
    for command in ('train-affect', 'evaluate'):
        with pytest.raises(SystemExit) as error:
            main([command, '--task', 'tone', '--seed', '789'])
        assert error.value.code == 2
    for seed in ('789', '1024'):
        args = parser().parse_args(['train-affect', '--task', 'reconstruction', '--seed', seed])
        assert coordinates(args)[1] == (int(seed),)
        args = parser().parse_args(['train-reconstruction', '--seed', seed])
        assert selections(args)[1] == (int(seed),)
        args = parser().parse_args(['evaluate', '--task', 'reconstruction', '--seed', seed])
        assert selections(args)[1] == (int(seed),)


def test_de_matches_explicit_1024_fft_band_reference():
    raw = np.random.default_rng(1).normal(size=(800, 62))
    spectrum = np.abs(np.fft.fft(raw.T*np.hanning(800), n=1024)[:, :512])
    expected = np.concatenate([np.log2(100*np.maximum((spectrum[:, max(int(low/200*1024), 1)-1:int(high/200*1024)]**2).mean(1), 1e-12))
                               for low, high in [(1,4), (4,8), (8,14), (14,31), (31,49)]])
    np.testing.assert_allclose(extract_de(raw), expected, rtol=1e-6)


def test_inner_trials_and_normalization_use_training_only():
    rows = [{'subject': 1, 'session': 1, 'trial': t, 'split': 'train' if t < 10 else 'test'} for t in range(12) for _ in range(3)]
    train, val = inner_split(rows, 42)
    assert len(train) == 24 and len(val) == 6
    assert not ({rows[i]['trial'] for i in train} & {rows[i]['trial'] for i in val})
    assert all(rows[i]['split'] == 'train' for i in np.r_[train,val])
    values = np.random.default_rng(4).normal(size=(36,310)).astype('float32')
    a = fit_de_stats(values, train)
    values[np.setdiff1d(np.arange(36), train)] = 1e10
    b = fit_de_stats(values, train)
    np.testing.assert_array_equal(a['mean'], b['mean'])
    np.testing.assert_array_equal(a['std'], b['std'])


def test_maet_update_and_subject_balanced_prototypes():
    torch.set_num_threads(2)
    torch.manual_seed(10)
    model = MAETProjector(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    x = torch.randn(6,310)
    labels = torch.tensor([0,0,0,1,1,1])
    before = model.project.weight.detach().clone()
    torch.nn.functional.cross_entropy(model(x)[1], labels).backward()
    optimizer.step()
    assert not torch.equal(before, model.project.weight)
    model.eval()
    subjects = np.array([1,1,2,1,2,2])
    prototypes = subject_balanced_prototypes(model, x.numpy(), labels.numpy(), subjects, np.arange(6), torch.device('cpu'))
    with torch.no_grad():
        embedding = model.encode(x)
        reference = torch.stack([(embedding[:2].mean(0)+embedding[2])/2, (embedding[3]+embedding[4:].mean(0))/2])
        reference = torch.nn.functional.layer_norm(reference, (128,))
    torch.testing.assert_close(prototypes, reference)


def test_de_cache_tracks_input_version_and_source_order(tmp_path):
    from affecteeg.affect import prepare_features
    raw_path = tmp_path/'emotion.npy'
    raw = np.random.default_rng(17).normal(size=(2,800,62)).astype('float32')
    np.save(raw_path, raw)
    data = {'emotion_eeg': np.load(raw_path, mmap_mode='r'),
            'manifest': {'emotion': [{'source_id': 'a'}, {'source_id': 'b'}]}}
    folder = tmp_path/'features'
    first = np.asarray(prepare_features(data, folder)).copy()
    np.save(raw_path, raw*3)
    data['emotion_eeg'] = np.load(raw_path, mmap_mode='r')
    second = np.asarray(prepare_features(data, folder)).copy()
    np.testing.assert_allclose(second-first, np.log2(9), atol=3e-6)
