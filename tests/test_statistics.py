import numpy as np
from affecteeg.statistics import holm_adjust, centered_bootstrap_p, stratified_bootstrap, summarize_tone


def test_holm_preserves_comparison_order_and_monotonicity():
    np.testing.assert_allclose(holm_adjust([.03, .001, .02]), [.04, .003, .04])


def test_stratification_weights_emotions_equally_and_shares_draws():
    # Equal weighting of emotions gives a mean of 0.5 for these four keys.
    values = np.array([[0, 0, 0, 1], [0, 0, 0, 2]])
    boot = stratified_bootstrap(values, [0, 0, 0, 1], draws=200)
    np.testing.assert_allclose(boot[:, 0], .5)
    np.testing.assert_allclose(boot[:, 1], 1.)
    assert centered_bootstrap_p(.5, boot[:, 0]) == 1/201


def test_tone_pairs_seeds_within_participants_and_refuses_incomplete_inference(tmp_path):
    rows = []
    for seed in (42, 123, 456):
        for fold in range(1, 11):
            for variant, delta in [('base', 0), ('affecteeg', .02)]:
                rows.append(dict(backbone='eegnet', seed=seed, fold=fold, variant=variant,
                                 balanced_accuracy=.5+fold*.01+delta, macro_f1=.4+delta))
    summary = summarize_tone(rows, tmp_path)
    assert summary['status'] == 'partial'  # Other nine models are absent.
    model = summary['models'][0]
    assert model['status'] == 'complete'
    np.testing.assert_allclose(model['delta_ba_ci95'], [.02, .02])
    incomplete = summarize_tone(rows[:-1], tmp_path)
    assert incomplete['models'][0]['status'] == 'partial'
    assert 'delta_ba_ci95' not in incomplete['models'][0]


def test_five_seed_reconstruction_matrix_and_incomplete_inference(tmp_path, monkeypatch):
    from affecteeg import statistics as s
    def read_metrics(path):
        relative = path.relative_to(tmp_path / 'reconstruction')
        track, model, variant, seed = relative.parts[:4]
        n, classes = (37, 4) if track == 'seediv' else (71, 7)
        delta = {42: .01, 123: .02, 456: .03, 789: .04, 1024: .05}[int(seed.removeprefix('seed_'))] if variant == 'affecteeg' else 0
        keys = {f'v{i}|0|{i % classes}': np.array([.5+delta, 1.-delta, .2-delta]) for i in range(n)}
        return keys, {0: ('shared-key', 1, 2)}
    monkeypatch.setattr(s, '_read_key_metrics', read_metrics)
    for track in ('seediv', 'seedvii'):
        for model in s.GENERATION_MODELS:
            for variant in ('base', 'affecteeg'):
                for seed in (42, 123, 456, 789, 1024):
                    path = tmp_path/'reconstruction'/track/model/variant/f'seed_{seed}'/'test_metrics.csv'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
    result = s.summarize_reconstruction(tmp_path)
    assert result['status'] == 'complete'
    assert result['completed_fits'] == result['expected_fits'] == 140
    assert result['missing'] == []
    assert len(result['paired_pcc']) == 16
    for row in result['paired_pcc']:
        np.testing.assert_allclose(row['delta_pcc'], .03)
        np.testing.assert_allclose(row['ci95'], [.03, .03])
        assert row['significant_positive']
    path.unlink()
    incomplete = s.summarize_reconstruction(tmp_path)
    assert incomplete['status'] == 'partial'
    assert incomplete['completed_fits'] == 139
    assert incomplete['expected_fits'] == 140
    assert incomplete['missing'] == ['seedvii/cat_wavenet/affecteeg/1024']
    assert incomplete['paired_pcc'] == []
