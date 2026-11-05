# MossFormerGAN VoiceBank+DEMAND lists

These files are the public training lists from
<https://github.com/modelscope/ClearerVoice-Studio> commit
`6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` (`6b3774d`).

| File | Upstream path | Lines | SHA-256 |
|---|---|---:|---|
| `tr_demand_28_spks_16k.scp` | `train/speech_enhancement/data/tr_demand_28_spks_16k.scp` | 11,572 | `52e824ec9a44111e2d75333834ac99642f86a06cd2d6467ccdc6ad023d166e3f` |
| `cv_demand_testset_16k.scp` | `train/speech_enhancement/data/cv_demand_testset_16k.scp` | 824 | `4fc7acfe06a5b8f6cfafe2e0a975e04640db7f2a595b94669e4ac537e9c4d92b` |

`MossFormerGAN_SE_16K.yaml` points `tr_list` and `cv_list` at these files.
Each line is `noisy_path clean_path` on the author's NAS. Both stems are the
VoiceBank utterance id. The NAS path is not a local path and is not rewritten.

The utterance ids and their order match MP-SENet `training.txt` and `test.txt`
at `89932cf`. The file bytes differ because the line format differs.
`cv_demand_testset_16k.scp` is the official test set and is also the official
validation list.

The wav files are not included. The lists describe a 16 kHz copy whose
resampler is unknown. This repository keeps the 48 kHz corpus and records its
own 16 kHz derivation separately.

License: Apache-2.0. The upstream `LICENSE` is copied next to these lists.
Copyright notices in that file stay with the copy.
