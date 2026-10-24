# Tone architecture sources

This directory contains architecture source files and project PyTorch ports
with package-relative imports. The four foundation architectures use random
initialization and retain their official model dimensions.

| Model / asset | Source snapshot used by the project | Included terms |
| --- | --- | --- |
| EEGNet and EEG-Conformer PyTorch port | [TorchEEG](https://github.com/torcheeg/torcheeg), `6a6daed43e765b2da6fe65060fa1222ee3c9b5b1` | `LICENSE_torcheeg.txt` |
| ATCNet local PyTorch port | [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet), `f01c80c1ca86ae7708c5075eb4d9663284d323b4` | `LICENSE_atcnet.txt` |
| RGNN signed graph convolution local dense port | [RGNN](https://github.com/zhongpeixiang/RGNN), `685b1a2645185bd8129c04a789dbfde8ad896a59` | License file absent from the source snapshot |
| RGNN 10-05 coordinates | [LibEER](https://github.com/XJTU-EEG/LibEER), local project snapshot | `LICENSE_rgnn.txt` applies to these coordinates |
| CBraMod | [CBraMod](https://github.com/wjq-learning/CBraMod), `0ff6be918985689e7df679bc731ffb70e6c6224f` | `LICENSE_cbramod.txt` |
| ST-EEGFormer | [STEEGFormer](https://github.com/LiuyinYang1101/STEEGFormer), `542ee17918c3c2c36ba1d4ea02bedff5eb149370` | `LICENSE_st_eegformer.txt` |
| EEG-Deformer | [EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer), `47df1c4d8ce375ae21117f84e81e969041d907f6` | `LICENSE_eeg_deformer.txt` (CBCR terms) |
| EEGMamba | [EEGMamba](https://github.com/wjq-learning/EEGMamba), `dbc83fa072744201e8897aeb9f65007b952ad323` | `LICENSE_eegmamba.txt` |
| REVE architecture and electrode positions | [reve_eeg](https://github.com/elouayas/reve_eeg), `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7` | `LICENSE_reve.txt` |

CAT-Net uses the project's `B3ModelNoAdapter` architecture (711,150 parameters).
The native feature boundaries, cloned classifier heads, ARRA relative adapter,
training-only statistics, candidate search, and optional structural interfaces
come from the project's tone main-experiment implementations.
