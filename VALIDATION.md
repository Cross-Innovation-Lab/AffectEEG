# Release validation

Validated on 2026-10-19. Full main-experiment training was not launched.

## Independent package checks

The release file set was copied to a new temporary directory outside the source repository. The original repository was removed from the Python import path, user site-packages were disabled, and all tests ran against the copied package and its bundled assets.

The initial release passed **64 tests in 37.42 seconds**, including the EEGMamba CUDA test. All five CLI subcommands (`prepare`, `train-affect`, `train-tone`, `train-reconstruction`, `evaluate`) successfully displayed their help from the copied directory.

The test coverage includes:

- All ten tone backbone architectures: native/frozen-interface equivalence, forward passes, gradients, and optimizer updates at the manuscript's input lengths.
- The four alternate tone adaptation interfaces, frozen downstream layers, candidate calibration/selection, training-only statistics, participant splits, and checkpoint coordinate checks.
- All seven reconstruction backbones in both Base and full AffectEEG variants: forward passes, optimizer updates, and checkpoint round trips.
- ARRA/FiLM identity initialization, auxiliary-gradient isolation, batch-normalization/RNG preservation, covariance-gradient projection, training-only teacher PCA and mel statistics, corrected loss normalization, and the 187-frame metric boundary.
- MAET original-trial inner splits, DE features, participant-balanced prototypes, and feature-cache invalidation.
- Strict loading of all 82 MD-JPT checkpoint state entries, frozen inference, channel alignment, and batch-independent embeddings.
- Fixed CEVB metadata, source separation, donor cycling, validation/test pairing, canonical actor aggregation, stereo PCM handling, and missing-input errors.
- Participant bootstrap aggregation, emotion-stratified key bootstrap, the complete 16-comparison Holm family, and partial-matrix handling.

PyTorch emitted informational convolution/Transformer warnings and a Mamba autocast deprecation warning; no test failed or was skipped in this CUDA run.

## Manuscript seed alignment

CAT-Net and the other nine tone backbones use `42, 123, 456`, matching the main manuscript's tone table and supplementary participant-split protocol. Reconstruction and MAET-DE use `42, 123, 456, 789, 1024`, matching the manuscript's reconstruction protocol. The two task-specific seed lists are defined in `affecteeg/common.py`. The experiment matrix contains 300 tone backbone/seed/fold coordinates, ten MAET-DE models, and 140 reconstruction fits.

After this alignment, **22 focused tests passed in 4.67 seconds**. They verify CAT-Net's 30 participant/seed coordinates, all task-specific experiment counts, five-seed effect averaging, a complete 140-fit reconstruction family, and withheld inference when one of those fits is missing. CLI checks reject `43/44/789/1024` for CAT-Net, reject reconstruction-only seeds for tone evaluation and teacher preparation before loading data, and accept `789/1024` for reconstruction training, evaluation, and MAET-DE training. Data-protocol and tone-participant-split checks also passed.

The focused checks can be repeated with:

```bash
python -m pytest -q tests/test_statistics.py tests/test_affect.py tests/test_data.py tests/test_tone.py::test_seed_specific_participant_splits
```

## End-to-end diagnostic runs

Two short pipelines used synthetic arrays in temporary directories and the actual bundled MD-JPT checkpoint:

1. **Tone:** EEGNet scratch training → MD-JPT feature extraction → reference/adaptation/control fits → selection → checkpoint reuse → paired Native/+AffectEEG test predictions.
2. **Reconstruction:** MD-JPT feature extraction → MAET-DE training and prototypes → ConvConcatNet Base/full training, including eight-batch calibration and a joint update → selected checkpoint loading → paired test metrics.

These runs exercised the public CLI with `--smoke`. Their data, features, predictions, logs, and trained weights are not included in the release.

## Checks against locally available original data

- The official MCAE spreadsheet/vocabulary join yielded **1,307 distinct recordings and 357 total condition keys**.
- All **1,080 SEED-IV trials** yielded **7,035 historical windows**. A raw 4,000-point window resampled to 800 points matched the corresponding existing array exactly (maximum absolute difference 0).
- All **1,600 SEED-VII trials** yielded **69,742 four-second windows**.
- A Mandarin file pair contained **120 × 2,000 × 59 EEG values and five EMG channels**; the 500-sample preprocessing outputs were finite. EEG-only loading was also checked.
- A real MCAE clip's canonical log-mel calculation matched the previous implementation exactly (maximum absolute difference 0).
- Fixed metadata retained the expected key and pair counts, with no recorded EEG/audio payloads or absolute source paths.

Full prepared dataset arrays were not regenerated during release validation. The SEED-IV check validates the explicitly retained historical 20-second-to-800-sample conversion described in the README.

## Environment

| Component | Verified version |
|---|---|
| Python | 3.10.19 |
| PyTorch | 2.6.0+cu124 |
| NumPy / SciPy | 2.2.6 / 1.15.3 |
| MNE | 1.11.0 |
| timm / Transformers | 1.0.27 / 5.0.0 |
| librosa | 0.11.0 |
| mamba-ssm / causal-conv1d | 2.3.0 / 1.6.0 |
| pytest | 8.4.2 |

Re-run the suite with:

```bash
AFFECTEEG_TEST_DEVICE=cuda:0 python -m pytest -q
```

The release packager traverses only source/asset/test directories and explicit documentation files. It includes only empty data-directory placeholders and the required MD-JPT weight; downloaded data, runtime outputs, bytecode, caches, and main-experiment checkpoints are excluded.
