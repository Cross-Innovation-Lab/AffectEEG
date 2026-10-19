# Third-party sources and implementation provenance

This release preserves upstream attribution and the license files present in the source snapshots. No blanket license is assigned to third-party code or weights.

## Tone architectures

See [the tone source inventory](vendor/tone/PROVENANCE.md) for the upstream repositories, source commits, included licenses, and the distinction between architecture copies and project PyTorch ports. The included `vendor/tone/LICENSE_*.txt` files retain the applicable terms. The REVE electrode-position table is architecture metadata; no pretrained REVE, CBraMod, ST-EEGFormer, or EEGMamba weights are distributed or loaded.

CAT-Net uses the project's five-EEG/five-EMG `B3ModelNoAdapter` implementation. Main tone adaptation was extracted from the project's journal/v3 adaptation and v6 CAT-Net implementations. The release replaces historical cache dependencies with scratch training, uses the manuscript's seeds and signal lengths, and refits all selection candidates on the current base checkpoint.

## MD-JPT

- Source: [ncclab-sustech/mdJPT_nips2025](https://github.com/ncclab-sustech/mdJPT_nips2025), local snapshot `bb84e89df5d05928ef3aa8645cbc339b59ed59de`.
- Bundled weight: [`log/pretrain/ckpt/epoch=19.ckpt`](https://github.com/ncclab-sustech/mdJPT_nips2025/blob/main/log/pretrain/ckpt/epoch%3D19.ckpt), 6,453,928 bytes.
- Included support assets: the original channel interpolation table and a JSON extraction of the checkpoint's inference architecture configuration.
- Adaptations: the inference-bearing MLLA, channel projection, and CNN modules replace the Lightning training wrapper; all 82 checkpoint state entries load strictly. The project's row-wise normalization and physical-channel interpolation are retained. Pretraining, loss modules, logging, data paths, and unused models are omitted.

The upstream snapshot does not include an explicit license file. The bundled checkpoint and adapted source retain their upstream attribution; this package does not relabel them as MIT or Apache licensed.

## MAET-DE

The EEG architecture is adapted from the project's timm-free MAET implementation, referencing [935963004/MAET](https://github.com/935963004/MAET) and Jiang et al., *SEED-VII*, IEEE Transactions on Affective Computing (2025). This package trains the EEG branch, learned projection, and classifier locally for each track and seed. No trained MAET checkpoints are included. The local architecture source used here did not carry a separate license file.

## Reconstruction architectures and losses

The seven model names refer to the task-adapted implementations already used in the project: NeuroTalk, ConvConcatNet, FESDE, SSM2Mel, DMF2Mel, NeuroSonic, and CAT-WaveNet. Their project content cores and mel decoders were extracted from `Codes/model/part2_class_style.py`. They expose the common content-memory interface needed by the manuscript and are not claimed to reproduce the complete official speech systems.

The full training behavior comes from the project's `part3_paper_followup_v1` end-to-end experiment. ARRA, its auxiliary-gradient projection, and corrected loss calculations come from the corresponding joint/iterative modules. Historical experiment admission checks, result receipts, machine resource queues, absolute paths, and unrelated variants are omitted. The new Base training uses the same 50-epoch objective and selection protocol as the full model.

## Data and fixed protocol indices

Dataset ownership remains with the respective dataset authors; no recorded EEG, EMG, audio, canonical spectrograms, or extracted trial features are distributed. The protocol assets contain a vocabulary index, condition splits, source identities, and pairing indices extracted from the existing T2 main-experiment protocol. These metadata are sufficient to rebuild matching arrays from separately acquired datasets. Dataset acquisition links are in the [README](README.md#dataset-index-and-layout).
