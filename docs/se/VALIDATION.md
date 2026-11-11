# 公式出力との数値比較

このファイルは、独立実装と公式処理を同じ入力で比べた最大絶対誤差の記録である。Mock のテスト、実モデルのスモーク、この表は混ぜない。区別は [.agents/OPERATIONS.md](../../.agents/OPERATIONS.md) の「検証の区別」。

値は記録されたものをそのまま載せている。条件が記録に無い欄は「未記載」。新しく測り直してはいない。

## 測定時の環境

数値は次の仮想環境で取った。現行の役割名（`.venvs/se`、`.venvs/official`、`.venvs/semambapp`、`.venvs/deepfilternet`）で取り直してはいない。

| 測定時 | 内容 |
|---|---|
| `.venvs/lab` | Python 3.13.9、PyTorch 2.6.0+cu124、torchaudio 2.6.0+cu124、NumPy 2.5.4。ホスト turing。CPU の初期比較と、後述の CPU/CUDA 自己差の一部 |
| `.venvs/ref` | Python 3.10.12、PyTorch 2.6.0+cu124、torchaudio 2.6.0+cu124、NumPy 1.26.4。追加は `einops==0.8.2`、`rotary-embedding-torch==0.8.3`、`torchdiffeq==0.2.5`、`x-transformers==2.14.2`、`omegaconf==2.3.0`、`librosa==0.10.2.post1`、`yamlargparse==1.31.1`、`pydub==0.25.1`、`maturin==1.4.0`。`libdf` は Rust 1.76.0 で `pyDF` をビルドした wheel `DeepFilterLib-0.5.7rc0`（`CARGO_TARGET_DIR=data/results/se-repro/cargo-target`）。公式ツリーは編集していない。GPU は `CUDA_VISIBLE_DEVICES=1` の Quadro RTX 5000 |
| `.venvs/semamba` | スクリプトの注記は Python 3.10、PyTorch 2.6.0+cu124、`causal-conv1d==1.7.0`、`mamba-ssm==2.3.2.post1`。公開 wheel `mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE` は `std::__cxx11::basic_string` を参照し、この PyTorch（CXX11 ABI false）では `selective_scan_cuda` がロードできなかった。`.venvs/cuda-nvcc`（conda-forge `cuda-nvcc=12.9`）で sm_75 だけをソースビルドした。`src/se` はこの環境を import しない |

2026-10-31 の最終比較の作業開始 HEAD は `6ea86f3`。ホスト turing。GPU 0 は他ユーザーが使用中のため、比較は GPU 1・2・3（Quadro RTX 5000、ドライバ 535.274.02）。

無音の MP-SENet は、公式のエネルギー正規化がゼロ除算になる。API は無音を無音のまま返す。この差は下の表に入れていない。

## 公式との最大絶対誤差

比較相手が「公式同士」の行は、独立実装の誤差ではない。

| モデル | device | sample rate | 長さ | seed | 比較相手 | 最大絶対誤差 |
|---|---|---|---|---|---|---:|
| MP-SENet VB / DNS | CPU | 16 kHz | 1 秒・5 秒・20 秒 | 11 | 公式 `mag_pha_stft` と独立 `enhance` | 0 |
| MP-SENet VB | GPU（Quadro RTX 5000） | 16 kHz | 1 秒 | 7 | 公式 GPU と独立 GPU | 3.73e-9 |
| MP-SENet DNS | GPU（Quadro RTX 5000） | 16 kHz | 1 秒 | 7 | 公式 GPU と独立 GPU | 2.33e-10 |
| MP-SENet VB | CPU 対 GPU | 16 kHz | 同じ入力（長さは未記載） | 未記載 | 公式同士 | 4.16e-3 |
| MP-SENet DNS | CPU 対 GPU | 16 kHz | 同じ入力（長さは未記載） | 未記載 | 公式同士 | 1.29e-4 |
| MP-SENet VB | CPU float32 | 波形ではない | 複素でない入力 `(1, 201, 8)` | 7 | 公式 Generator | 0 |
| TF-GridNet urgent24 | CPU | 8 kHz と 16 kHz | 1 秒と 20 秒 | 未記載 | 公式 separator | 0 |
| TF-GridNet ms-snsd | CPU | 16 kHz | 1 秒と 20 秒 | 未記載 | 公式 separator | 0 |
| MossFormerGAN | CPU | 16 kHz | 1 秒 | 3 | 公式 decode | 0 |
| MossFormerGAN | CPU | 16 kHz | 11 秒 | 未記載 | 公式 decode（10 秒窓の修正後） | 0 |
| MossFormerGAN | CPU | 16 kHz | 20 秒 | 3 | 公式 decode | 0 |
| MossFormerGAN | GPU（`CUDA_VISIBLE_DEVICES=3`、Quadro RTX 5000） | 16 kHz | 15 秒 / 20 秒 / 25 秒 | 3 | 公式 decode | 1.40e-6 / 2.09e-6 / 2.33e-6 |
| DeepFilterNet3 | CPU | 48 kHz | 0.1 秒 / 1 秒 / 20 秒 / 正弦 / 小振幅 | 0.1 秒と 1 秒は現行スクリプトが 12。20 秒・正弦・小振幅は未記載 | `libdf` の `enhance` | 1.19e-7 / 4.10e-8 / 6.52e-8 / 8.94e-7 / 1.62e-8 |
| DeepFilterNet3 | CPU | 入力 16 kHz | 現行スクリプトは 16000 サンプル | 12 | torchaudio で 48 kHz にして `libdf` `enhance` し、同じ torchaudio で 16 kHz へ戻す | 3.86e-6 |
| SEMamba++ 波形 | CPU | 16 kHz | 1 秒 | 9 | 公式グラフと参照スキャン。融合カーネルではない | 2.38e-8 |
| SEMamba++ 位相 | CPU | 16 kHz | 1 秒 | 9 | 同上 | 6.02e-4 |
| SEMamba++ 単一 Mamba | GPU | — | 入力 `(1, 64, d_model)` | 9 | 融合カーネルと CUDA scan | 0 |
| SEMamba++ 単一 Mamba | GPU | — | 入力 `(1, 64, d_model)` | 9 | 融合カーネルと PyTorch 漸化式 | 3.28e-7 |
| SEMamba++ 波形 | GPU（`CUDA_VISIBLE_DEVICES=1`） | 16 kHz | 1 秒 / 5 秒 / 20 秒 | 9 | 公式 `mamba_inner_fn`。missing=unexpected=0 | 3.97e-5 / 1.56e-5 / 1.80e-5 |
| SEMamba++ 位相 | GPU（同上） | 16 kHz | 1 秒 / 5 秒 / 20 秒 | 9 | 同上。差の最大は約 π で、振幅の小さいビンの折り返し | 6.28 |

MP-SENet の GPU 行に sample rate の再掲は無い。処理レートは 16 kHz である。TF-GridNet の 1 秒を再現する現行スクリプトは seed 11 である。20 秒の seed は記録に無い。

SEMamba++ の 20 秒は、公式 0.79 秒（RTF 0.039、peak VRAM 1.3 GiB）、独立実装 6.57 秒（RTF 0.33、peak VRAM 6.5 GiB）。再現スクリプトが peak として出すのは `torch.cuda.max_memory_allocated` である。

20 秒の MP-SENet、TF-GridNet、DeepFilterNet3 は、下の現行スクリプトの引数には含まれない。値は当時の記録である。

## チェックポイントと参照 commit

参照 commit と SHA-256 の正本は `requirements/official-assets.toml`。この表はそこから写した。

| 対象 | commit または SHA-256 |
|---|---|
| MP-SENet | `89932cfe90d1dacb8e170e4a331d762462c21792` |
| MP-SENet VB `g_best_vb` | `aedfb1aa549159f71b39613d94db831dfa983b3a68b0deea02e84e0ad563f4f9` |
| MP-SENet DNS `g_best_dns` | `97a77ba67c5c484c65363bb703ea85962f773ca0819e22ce81b4ec33db5e7206` |
| MP-SENet `config.json` | `339cb60721e4ab0ee3ed4825ca9dfc981c7554c7b5e87f043e454e5a17b58f92` |
| TF-GridNet（ESPnet） | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` |
| TF-GridNet urgent24 `21epoch.pth` | `51c500e0368c9cfc469e5d83a6cabf35c0de9bbcc027ab633aefcbc2a3e3cb91` |
| TF-GridNet ms-snsd `valid.loss.best.pth` | `a680263b1339d86ddc5d168a5f985fbd0b3e658d7711971f3c03fe2d23af7bac` |
| TF-Locoformer tag `v1.1.0` | `7a615460d347ff7334a13dbb831d16280da72cdc` |
| TF-Locoformer DNS `valid.loss.ave_5best.pth` | `6ffa518f2b73bed289e38349366eb777c94b9130acc32bf4c085cb049d9d1666` |
| DeepFilterNet | `d375b2d8309e0935d165700c91da9de862a99c31` |
| DeepFilterNet3 zip | `49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2` |
| `model_120.ckpt.best` | `23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003` |
| `config.ini` | `415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290` |
| MossFormerGAN（ClearerVoice-Studio） | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` |
| `last_best_checkpoint.pt` | `04bf00db575389e1ea8b1fcca48d9efcf073e50ad170e6dd5230df3bac98be27` |
| SEMamba++ | `cd369c89a8106072c077b468417b765b227afd74` |
| `semambapp.pth` | `ccb9d1c49146e915c427e5724fec36825196f6b7a8fe23414c26a5224b39af72` |

以前のこのファイルは `config.json` の SHA-256 を `339cb60721e4ab0ee3ed4825c7e87f043e454e5a17b58f92` と書いていた。桁が足りず、上のピンと途中が違う。

## 再現コマンド

現行の仮想環境名で、実在するスクリプトに対して書けるものだけを載せる。測定値そのものは上の旧環境の記録である。

```bash
# MP-SENet の 1 秒・5 秒（seed 11）と TF-GridNet の 1 秒（seed 11）。20 秒は含まない。
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python tests/se/reference/run_lab.py

# MossFormerGAN の 1 秒、seed 3、CPU。測定時は .venvs/ref。
CUDA_VISIBLE_DEVICES="" .venvs/official/bin/python tests/se/reference/run_ref.py

# MossFormerGAN。--seconds と --seed と --device を変える。測定時は .venvs/ref。
CUDA_VISIBLE_DEVICES="" .venvs/official/bin/python tests/se/reference/run_long.py mossformer --seconds 20 --seed 3 --device cpu
CUDA_VISIBLE_DEVICES=3 .venvs/official/bin/python tests/se/reference/run_long.py mossformer --seconds 20 --seed 3 --device cuda

# SEMamba++ の CPU。参照スキャン。融合カーネルは要らない。
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python tests/se/reference/run_semamba.py

# SEMamba++ の GPU。測定時は .venvs/semamba。fused 用の現行環境は .venvs/semambapp。
CUDA_VISIBLE_DEVICES=1 .venvs/semambapp/bin/python tests/se/reference/run_semamba_cuda.py 1 5 20
```

`tests/se/reference/run_df.py` と `run_df_api.py` は `libdf` と `src.se` の両方を読む。`src.se` は TF-Locoformer 経由で `rotary_embedding_torch` を import する。`.venvs/deepfilternet` は `libdf` 用で、`requirements/deepfilternet.in` に `rotary-embedding-torch` が無い。`.venvs/se` と `.venvs/official` は NumPy 2.5.4 で、`libdf` は NumPy 1.26.4 の `.venvs/deepfilternet` に分けている。測定時の `.venvs/ref` は NumPy 1.26.4 に両方を入れていた。現行の役割名では同じコマンドを書けない。

通常の CPU テストは次である。この実行結果を上の表の一致とは呼ばない。

```bash
CUDA_VISIBLE_DEVICES="" .venvs/se/bin/python -m unittest discover -s tests/se -v
```

## CPU と CUDA の自己差

独立実装の CPU 出力と CUDA 出力の差である。公式一致の証拠にしない。

ampere の RTX 3090 GPU 0。Python 3.13.9、PyTorch 2.6.0+cu124、seed 11、1000 サンプルのランダム入力。長さは 1000。出力は有限。

| モデル | 最大絶対差 | MAE | MSE |
|---|---:|---:|---:|
| MP-SENet VB 16 kHz | 0.0044227270 | 0.0009261760 | 1.4355676e-6 |
| TF-GridNet urgent24 8 kHz | 0.0000648554 | 0.0000079845 | 1.0956657e-10 |
| TF-GridNet ms-snsd 16 kHz | 0.0003389884 | 0.0000153409 | 8.3617518e-10 |

同じ MP-SENet の入力で cuDNN を無効化すると、最大差 0.00438077、MAE 0.000922169 だった。cuDNN の有無だけでは説明できず、原因は未特定。許容誤差は決めていない。

別の測定。`CUDA_VISIBLE_DEVICES=1` の Quadro RTX 5000。公式実装との比較ではない。

| モデル | 入力 | 最大絶対差 | 平均絶対差 |
|---|---|---:|---:|
| DeepFilterNet3 | 48 kHz、4800 サンプル | 1.28e-9 | 2.20e-10 |
| MossFormerGAN | 16 kHz、1600 サンプル | 1.91e-6 | 4.37e-7 |

SEMamba++ の独立実装について、CPU と CUDA の自己差は記録が無い。上の GPU 行は公式の融合カーネルとの差である。

## 未検証のまま残っているもの

- TF-GridNet の ESPnet 推論コマンドは起動していない。比較は公式 separator と、ESPnet の `Stft` が呼ぶ `torch.stft` の条件である。公式のリサンプリング実装との数値比較は記録が無い。
- MP-SENet の公式 CLI のファイル入出力と PCM 16 書き出し。波形の `enhance` とは別である。
- DeepFilterNet3 の 16 kHz 入力は torchaudio のリサンプルであり、`libdf` の sinc ではない。sinc との比較はしていない。上の 3.86e-6 は torchaudio 往復である。
- SEMamba++ の既定スキャンは PyTorch の漸化式のままである。`scan_backend: fused` は任意指定で、融合カーネルへの置き換えはしていない。
- TF-Locoformer の公開重みについて、以前の `docs/se/MODELS.md` は「CPU の最大絶対誤差 0、空の Quadro RTX 5000 では 20 秒まで `1e-5` 未満」と書いていた。device 以外の sample rate、長さの一覧、seed、比較相手、チェックポイントをこの記録は特定できない。`tests/se/test_tf_locoformer.py` が公式 standalone separator と比べて最大絶対誤差 0 を要求するのは、公開重みではなく小さいランダム重み（`n_layers=1`、`emb_dim=16`、`n_fft=64`）の CPU で、スペクトル `(2, 12, 33)` と長さ 512 の波形、seed 3 である。同じファイルの CUDA 側は、空いている GPU での長さ 1600 の有限値であり、20 秒の公式比較ではない。
- SI-SDR、相関、時間ずれの数値は残っていない。判定に使ったのは最大絶対誤差である。SI-SDR は誤差 0 でも分母を 1e-12 で下駄上げするため、完全一致でも有限値になる、という記録がある。
- 設定ローダは YAML を読める（`src/se/core/config.py`。PyYAML は `requirements/se.in` の `pyyaml==6.0.3`）。`tests/se` に YAML を読むテストは無い。
- `tests/se/test_independence.py` が公式ツリー無しの一時コピーで見るのは MP-SENet と TF-GridNet だけである。DeepFilterNet3、MossFormerGAN、SEMamba++ については、当時の一時コピーで import、チェックポイント読み込み、短音声 `enhance` が完了したという記録がある。TF-Locoformer の同様の記録は無い。
- 初期の MP-SENet 記録にある元リポジトリ commit `d91b88c2fc5469597ba0114f1c5b759c3ebac28d` と、現行ピン `89932cfe90d1dacb8e170e4a331d762462c21792` の関係は未確認である。
- DeepFilterNet3 の分析・合成だけを通した 48 kHz、長さ 4800 の正弦（遅延 480 サンプルを除く）の最大絶対誤差は 4.17e-7。公式 `libdf` との波形差ではない。同一発話の再実行差は 0、という記録がある。
