# AffectEEG: standalone main experiments

This package contains the main tone-decoding and canonical-mel reconstruction experiments for AffectEEG. It provides raw-data preparation, CEVB construction, scratch backbone training, AffectEEG adaptation, affect-encoder training, and evaluation. All runtime paths are relative to this package or supplied on the command line.

The release includes code, fixed protocol indices, and the frozen MD-JPT checkpoint. Acquire the four source datasets separately. Generated data, features, checkpoints, predictions, and logs are written to your local `data/processed/` and `outputs/` directories and are excluded by the release packager.

## Installation

Use Python 3.10 or 3.11. Install a PyTorch 2.6 build appropriate for your machine before the remaining requirements. The verification environment used Python 3.10, PyTorch 2.6.0 with CUDA 12.4, and an NVIDIA GPU. CPU execution is supported by the other backbones; EEGMamba requires CUDA and its extensions.

Clone the repository with the command below, or extract the standalone archive into `affecteeg_main_experiments/`:

```bash
git clone https://github.com/Cross-Innovation-Lab/AffectEEG.git affecteeg_main_experiments
```

```bash
cd affecteeg_main_experiments
python -m pip install -r requirements.txt
# Required for EEGMamba, after installing matching PyTorch/CUDA:
python -m pip install --no-build-isolation -r requirements-mamba.txt
# Needed only to run the tests:
python -m pip install -r requirements-test.txt
python -m affecteeg --help
```

Keep the complete extracted directory, including `assets/` and `vendor/`. Run the commands below from that directory. Optional editable installation is supported with `python -m pip install --no-deps -e .`; a relocated wheel without the external asset directories is not the supported distribution format. No command downloads datasets, architectures, or pretrained weights automatically.

## Dataset index and layout

| Source | Role | Acquisition |
|---|---|---|
| Mandarin EEG–EMG, silent articulation (T2) | Tone labels, 59-channel content EEG, five-channel EMG | [Ju et al., dataset paper](https://doi.org/10.1109/TNSRE.2025.3616276). A public download URL has not been verified; contact the dataset authors for access. |
| SEED-IV | Four-class affective EEG | [Official dataset page](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) and [access application](https://bcmi.sjtu.edu.cn/home/seed/downloads.html). Acquire the EEG files and channel-order spreadsheet. |
| SEED-VII | Seven-class affective EEG | [Official dataset page](https://bcmi.sjtu.edu.cn/home/seed/seed-vii.html) and [access application](https://bcmi.sjtu.edu.cn/home/seed/downloads.html). Acquire `EEG_preprocessed` and the channel-order and emotion-label spreadsheets. |
| MCAE-Monosyllable | Actor-aggregated acoustic targets | [Official OSF project](https://osf.io/h3uem/) and [dataset paper](https://doi.org/10.3758/s13428-025-02607-4). Acquire the audio files and `corpus information.xlsx`. |

Arrange extracted files as follows. Preserve any `.fdt` files referenced by the Mandarin `.set` files.

```text
data/raw/
  mandarin/
    S1/ ... S10/
      T2/
        <original-name>evtvar.txt
        All/
          EEGnosoundstim1.set ... EEGnosoundstim4.set
          EMGnosoundstim1.set ... EMGnosoundstim4.set
          <referenced .fdt files, if present>
  seed_iv/
    Channel Order.xlsx
    eeg_raw_data/
      1/<subject>_<date>.mat
      2/<subject>_<date>.mat
      3/<subject>_<date>.mat
  seed_vii/
    Channel Order.xlsx
    emotion_label_and_stimuli_order.xlsx
    EEG_preprocessed/1.mat ... 20.mat
  mcae/
    corpus information.xlsx
    Audio files/<emotion>/<six-digit-filename>.wav
```

`assets/protocol/vocabulary.csv` contains the 120-row `(group, tone) -> syllable` index. Labels are resolved using both group and tone. The two compressed protocol files contain source identities, partitions, condition keys, and fixed evaluation pair indices; they contain no EEG or audio signals. Mandarin event files are checked against this mapping before output arrays are allocated.

```bash
# All data needed for both main tasks:
python -m affecteeg prepare --task all
# Or prepare only one task/track:
python -m affecteeg prepare --task tone
python -m affecteeg prepare --task cevb --track seediv
# A separately mounted raw-data directory is also supported:
python -m affecteeg prepare --raw-root /path/to/raw --processed-root /path/to/processed
```

Use one preparation command for each destination. Completed prepared directories are not overwritten; choose a fresh `--processed-root` to rebuild. Missing files, mismatched labels/channels, incomplete trials, and unexpected source counts raise an error.

## CEVB construction

CEVB aligns independently recorded content EEG, affective EEG, and canonical acoustic references by `(syllable, tone, emotion)`. Content EEG consists of 4,800 Mandarin T2 trials, each with 59 channels and 800 samples at 200 Hz. The source partitions contain 2,400/1,200/1,200 content trials. Condition keys and exact source instances are disjoint across training, validation, and test; participants and original affective trials may occur in multiple outer partitions.

**SEED-IV uses the explicitly retained historical preprocessing.** Each window consumes 4,000 samples from the actual 200 Hz source recording, covering **20 seconds**, then Fourier-resamples it to 800 samples. This produces 7,035 windows. The downstream model/DE preprocessing retains the historical 200 Hz processing convention after temporal compression. These are not physical four-second windows. This choice preserves the agreed historical sample pool and is an explicit exception to the manuscript's four-second wording. SEED-VII uses genuine non-overlapping four-second, 800-sample windows at 200 Hz, producing 69,742 windows. Incomplete tails are discarded.

The official MCAE spreadsheet is joined to the 120-row vocabulary, retaining 1,307 recordings. Waveforms become mono, are linearly resampled to 24 kHz, and are truncated or right-zero-padded to two seconds. Mel extraction uses FFT/window 1,024, hop 256, a periodic Hann window, 384-sample reflection padding, `center=False`, and 100 Slaney mel filters. Magnitudes use epsilon `1e-9`, and the natural logarithm has a `1e-5` floor. The 187 valid frames are padded to 188 with `log(1e-5)`.

Repeated recordings are averaged within each actor. At each time–frequency location, the highest and lowest actor means are removed when at least three actors contribute; the remaining actor means are averaged. With one or two actors, all actor means are averaged. Every item with the same condition key shares this canonical target.

| Track | Affective windows | Train/validation/test keys | Train/validation/test pairs |
|---|---:|---:|---:|
| SEED-IV | 7,035 historical compressed windows | 127 / 38 / 37 | 12,700 / 3,800 / 3,700 |
| SEED-VII | 69,742 four-second windows | 215 / 71 / 71 | 21,500 / 7,100 / 7,100 |

Only keys with acoustic references are included: 202 for SEED-IV and 357 for SEED-VII. Each key contributes 100 pairs. Training pairs are deterministically regenerated for each training seed and epoch, exhausting emotion donors before reuse. Validation and test pairs are fixed and shared across all backbones and seeds. Training-only keys determine per-mel-bin target normalization.

## Tone decoding: Native and +AffectEEG

The ten backbones are CAT-Net, EEGNet, ATCNet, EEG-Conformer, RGNN, CBraMod, ST-EEGFormer, EEG-Deformer, EEGMamba, and REVE. CLI identifiers are `cat_net`, `eegnet`, `atcnet`, `eeg_conformer`, `rgnn`, `cbramod`, `st_eegformer`, `eeg_deformer`, `eegmamba`, and `reve`.

All tone backbones start from random initialization. Three seeds (`42, 123, 456`) and ten held-out participants define 300 backbone/fold/seed coordinates. Each fold has eight training, one validation, and one test participant (3,840/480/480 trials). Validation offsets from the test participant are `+1/-1/+2`, respectively, modulo ten.

EEG uses the pre-cue interval `[-1.5, 0]` seconds for baseline standardization, followed by trial/channel standardization. EMG is rectified, smoothed with a 50-sample envelope, and standardized. Four-second trials are resampled to 500 samples at 125 Hz. Nine EEG-only models receive the fixed 20 electrodes:

```text
AF7 PO3 PO6 F7 FC6 Fp1 CP6 FC4 F1 PO4 O1 O2 FT7 CP4 FT8 CP3 PO7 C3 PO5 F6
```

CAT-Net receives `T7 T8 Fp1 Fpz Fp2` and five EMG channels. Each modality concatenates the signal from sample 1 onward with first differences, yielding 499 samples and ten input features. The frozen MD-JPT teacher uses all 59 EEG channels and all 500 samples.

| Backbone | Batch | Backbone/head learning rate | Precision |
|---|---:|---|---|
| CAT-Net | 64 | Validation pilot selects `1e-4`, `3e-4`, or `1e-3` for all parameters | FP32 |
| EEGNet | 128 | `1e-3 / 1e-3` | AMP on CUDA |
| ATCNet, EEG-Conformer, RGNN, EEG-Deformer | 256 | `1e-3 / 1e-3` | AMP on CUDA |
| CBraMod | 16 | `1e-4 / 1e-3` | AMP on CUDA |
| ST-EEGFormer | 8 | `1e-4 / 1e-3` | AMP on CUDA |
| EEGMamba | 16 | `1e-4 / 1e-3` | AMP on CUDA |
| REVE | 8 | `2e-5 / 1e-3` | AMP on CUDA |

Native training uses AdamW, weight decay `1e-4`, at most 50 epochs, and early stopping after ten non-improving epochs. CAT-Net compares three independently initialized 15-epoch learning-rate pilots, then trains a new model at the selected rate; its plateau scheduler has factor 0.5, patience five, and minimum rate `1e-5`. Other models select validation BA, then macro-F1. Native regularization and applicable domain loss (weight 0.05) are retained. The four foundation-model rows above are retained project implementation defaults for details not specified in the manuscript; they have no pretrained loading or head-only warmup.

AffectEEG freezes the content backbone and original downstream layers. It extracts MD-JPT embeddings, centers them within training-fold tone classes, and fits normalization/PCA using training trials only. A reference classifier is fitted on frozen features at learning rate `1e-3`. Reference and adaptation fits use FP32, AdamW, batch 128, weight decay `1e-4`, and 30 epochs (900 updates on a full training fold).

The main candidate search includes the eight supplementary configurations, six fixed learning-rate/covariance grid candidates, and validation-triggered alternative interfaces for CAT-Net, EEG-Conformer, REVE, and EEG-Deformer. Head-only and zero-covariance fits are retained internally where required for interface activation. Candidate calibration, distillation, feature preservation, and validation tie-breaking are implemented in `affecteeg/tone/`; the candidate definitions are in `candidate_templates.json`. Test trials are evaluated after selection.

```bash
python -m affecteeg train-affect --task tone --device cuda:0
python -m affecteeg train-tone --device cuda:0
python -m affecteeg evaluate --task tone --device cuda:0
# One coordinate:
python -m affecteeg train-tone --backbone eegnet --seed 42 --fold 1 --variant both --device cuda:0
```

`--variant base` means the Native classifier; `affecteeg` trains its adaptation, obtaining the scratch base first if needed. The two variants share the selected base checkpoint and held-out trials.

## Reconstruction and affect-encoder training

The seven reconstruction backbones are NeuroTalk (`neurotalk`), ConvConcatNet (`convconcatnet`), FESDE (`fesde`), SSM2Mel (`ssm2mel`), DMF2Mel (`dmf2mel`), NeuroSonic (`neurosonic`), and CAT-WaveNet (`cat_wavenet`). These are the task-adapted implementations used by the project with a common `800×256` content memory and `100×188` mel output. The names identify their architectural sources; they do not imply exact official speech-system reproductions or pretrained speech weights.

For each track and seed, MAET-DE is trained separately on the outer-training affective EEG pool. Its 80/20 inner split groups windows by participant/session/original trial. Five-band DE features have width 310; the MAET EEG embedding has width 32 and a learned projection has width 128. Training uses plain cross-entropy, inverse-class-frequency replacement sampling, Adam `5e-4`, batch 256, 30 epochs, and gradient clipping at one. The selected checkpoint maximizes inner-validation macro-F1, then BA, retaining the earlier tied epoch. DE statistics and participant-balanced class prototypes use only inner-training data. The encoder, classifier, and prototypes remain frozen during reconstruction.

Base and +AffectEEG are trained independently for **50 epochs** on each track with five seeds (`42, 123, 456, 789, 1024`): 140 fits in total. Both use Adam, FP32, batch 32, gradient clipping at one, and the same reconstruction loss and validation selection. Learning rates are `1e-3` for SEED-IV and `3e-4` for SEED-VII. The ARRA adapter rate is 0.3 times the main rate. A plateau scheduler monitors reconstruction training loss (factor 0.5, patience five). Validation PCC selects the checkpoint, with lower validation L1 and then earlier epoch breaking ties; epoch zero is eligible.

Full AffectEEG combines bounded ARRA (rank 16, bound 0.1, zero-initialized output) and rank-four FiLM. The content encoder, ARRA, FiLM, and mel decoder receive reconstruction gradients. Covariance and preservation gradients update only ARRA. Covariance uses training-only MD-JPT targets, within-content/tone centering, PCA32 without whitening, and auxiliary batches of four groups with eight distinct sources each. Eight training batches calibrate the covariance coefficient to a gradient-norm ratio of 0.03, clipped to `[1e-4, 300]`; the coefficient ramps over five epochs. Conflicting covariance gradients are projected against the adapter's reconstruction gradient. Feature-preservation weight is 0.01.

The shared reconstruction objective uses masked Smooth-L1 with beta 0.5, spectral convergence (0.5), temporal differences (0.2), frequency differences (0.1), masked PCC (0.5), standard-deviation matching (0.05), and original-log-mel PCC over 187 valid frames (0.5). **Smooth-L1 divides by retained time–frequency elements**, following the corrected full-model implementation; it does not use the historical frame-only denominator. There is no additional raw-log-mel L1 term.

```bash
python -m affecteeg train-affect --task reconstruction --device cuda:0
python -m affecteeg train-reconstruction --device cuda:0
python -m affecteeg evaluate --task reconstruction --device cuda:0
# One track/backbone/seed:
python -m affecteeg train-reconstruction --track seediv --backbone cat_wavenet --seed 42 --variant both --device cuda:0
```

Train the corresponding affect models before full reconstruction; Base alone does not require them. `train-affect` also extracts frozen content-teacher features. This trains ten MAET-DE models, not MD-JPT pretraining. MD-JPT's upstream checkpoint and inference configuration are already bundled.

## Outputs, metrics, and complete runs

All training/evaluation commands accept `--processed-root`, `--output-root`, and `--device`; individual coordinates can be selected with `--seed`, `--backbone`, `--fold` (tone), or `--track` (reconstruction). `--smoke` performs shortened diagnostics in a separate `outputs/smoke/` tree. These diagnostics are not paper training runs.

```text
outputs/
  teacher/{tone,seediv,seedvii}/features.npy
  affect/<track>/seed_<seed>/selected.pt
  tone/<backbone>/seed_<seed>/fold_<01..10>/
    base.pt, reference.pt, affecteeg.pt, selection.json
    predictions_base.npz, predictions_affecteeg.npz
  tone/paper_summary.json
  reconstruction/<track>/<backbone>/<variant>/seed_<seed>/
    selected.pt, last.pt, history.json, evaluation.json, test_metrics.csv
  reconstruction/paper_summary.json
```

Tone metrics are BA and macro-F1. Paired effects average seeds within participant, then weight the ten participants equally. Per-backbone 95% intervals use 50,000 participant bootstrap draws (seed 17003), without multiplicity correction. Nine-EEG and ten-backbone averages are descriptive.

Reconstruction metrics reverse target standardization and use only the valid `100×187` region: flattened PCC, mean absolute log-mel error, and relative Frobenius error. Aggregation averages items within key, five seeds, keys within emotion, then emotions equally; backbone means weight all seven models equally. Paired PCC inference uses 10,000 emotion-stratified key bootstrap draws shared across models (track seeds 20261010/20261011), centered two-sided bootstrap p-values, and one Holm family of 16 comparisons. The complete family is produced only after all 140 evaluations exist. Incomplete outputs are labelled partial.

After preparing data, the full sequential training/evaluation matrix can be launched with:

```bash
AFFECTEEG_DEVICE=cuda:0 bash scripts/run_main.sh
```

For validation without full experiment training:

```bash
python -m pytest -q
# Include EEGMamba and run architecture checks on a selected CUDA device:
AFFECTEEG_TEST_DEVICE=cuda:0 python -m pytest -q
python scripts/package_release.py
```

See [VALIDATION.md](VALIDATION.md) for the checks performed on this release.

## Protocol provenance

The package follows the current main manuscript for backbone scope, scratch tone initialization, `42/123/456` tone seeds, 500-sample EEG-only inputs, and 50-epoch Base/full reconstruction. Supplementary settings supply details not expanded in the main text. Four foundation-model optimizer defaults and the corrected reconstruction loss come from the existing project implementation. The historical SEED-IV compression is retained by explicit project decision and described above with its actual duration.

These choices change parts of the historical training recipe: earlier runs included pretrained tone backbones, CAT-Net seeds `42/43/44`, 499-sample EEG-only caches, and 60-epoch reconstruction Base fits. The new package therefore reproduces the specified methods and protocol; old manuscript table numbers are not acceptance targets for numerical replay. No historical predictions or trained main-experiment models are distributed.

Third-party architecture and checkpoint sources, together with included license files, are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
