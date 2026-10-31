# 検証（途中）

`lab`: Python 3.13.9、PyTorch 2.6.0+cu124、torchaudio 2.6.0+cu124、NumPy 2.5.4、ホスト turing。GPU は使用中につき CPU のみ。環境構築は `docs/envs/LAB.md`。追加インストールなし。

実行: `.venvs/lab/bin/python -m unittest discover -s tests/se -v`。公式比較テストは参照コードとチェックポイントがある場合のみ実行し、なければ skip される。公式コードを含まない一時コピーでロードと推論を行う独立性テストも含む。

MP-SENet の公式 Generator と移植版は VB 重み、固定乱数 seed 7、入力 `(1,201,8)`、CPU float32 で完全一致（atol=rtol=0）。公式推論式との波形比較は VB・DNS、16 kHz、seed 7 のランダム音声 1000 samples、CPU float32、atol=rtol=0。これらは小さい入力の検証に限る。波形式の参照は公式 inference.py と dataset.py からテスト内に再現したもので、公式スクリプトのファイル入出力を実行していない。最初に振幅計算の演算順序の違いで最大 1.12e-8 の差が出たため公式の `view_as_real().pow(2).sum()` に合わせた。

チェックポイント SHA256:

- VB: `aedfb1aa549159f71b39613d94db831dfa983b3a68b0deea02e84e0ad563f4f9`
- DNS: `97a77ba67c5c484c65363bb703ea85962f773ca0819e22ce81b4ec33db5e7206`
- config.json: `339cb60721e4ab0ee3ed4825c7e87f043e454e5a17b58f92`

元リポジトリ commit `d91b88c2fc5469597ba0114f1c5b759c3ebac28d`。長時間音声、他サンプルレートでの公式比較、GPU、残り6モデルは未検証。独立性検証の対象は MP-SENet のみ。全体 PASS ではない。

## 継続作業の結果

`.venvs/lab/bin/python -m unittest discover -s tests/se -v`: 13 tests OK。turing の GPU はすべて使用中、CPU のみ。

TF-GridNet urgent24: 公式 Separator と Generator のスペクトル出力を CPU float32、seed=11、複素入力 `(1,8,129)`、公式重みで比較し、最大絶対誤差 0（atol=rtol=0）。STFT と最終波形の公式 CLI 比較は未実施。8 kHz で短音声と可変長バッチの出力形状・有限値を確認した。

TF-GridNet ms-snsd: 公式 Separator と移植版を CPU float32、seed=11、1001 サンプル、公式重みで比較し、最大絶対誤差 0（atol=rtol=0）。テストでは公式 STFT/iSTFT の演算を最小スタブで再現しており、公式 CLI 全体は実行していない。両 variant は `official/` を含まない一時コピーで重みロードと短音声推論を実施。長時間、GPU、複数レートでの公式比較は未実施。SI-SDR、相関、時間ずれの計測は未実施。

SHA256: urgent24 `51c500e0368c9cfc469e5d83a6cabf35c0de9bbcc027ab633aefcbc2a3e3cb91`、ms-snsd `a680263b1339d86ddc5d168a5f985fbd0b3e658d7711971f3c03fe2d23af7bac`。

YAML 読み込み用の PyYAML は lab に存在せず、YAML 実行テスト未実施。残りの5モデルは未実装。SEMamba++ はソース許諾未確認のため参照コード複製を見送り、仕様に基づく独立実装は未着手。研究上の完全再現や7モデルの受入は未達。

### ampere の CUDA スモークテスト

空き GPU を `nvidia-smi` で確認し、RTX 3090 GPU 0 のみを短時間使用。Python 3.13.9、PyTorch 2.6.0+cu124、seed 11、1000 samples のランダム入力。公式との GPU 比較ではなく、独立実装の CPU と CUDA の比較である。全出力は有限、長さ 1000。

| モデル | 最大絶対差 | MAE | MSE |
|---|---:|---:|---:|
| MP-SENet VB 16 kHz | 0.0044227270 | 0.0009261760 | 1.4355676e-6 |
| TF-GridNet urgent24 8 kHz | 0.0000648554 | 0.0000079845 | 1.0956657e-10 |
| TF-GridNet ms-snsd 16 kHz | 0.0003389884 | 0.0000153409 | 8.3617518e-10 |

MP-SENet の CPU/CUDA 差は原因未調査で、許容誤差は未設定。CUDA 上の公式実装比較と再現コマンドの自動化は必要。上記の数値を公式一致の証拠に使わない。

MP-SENet の差の切り分けとして同じ入力で cuDNN を無効化して再実行したところ、最大差 0.00438077、MAE 0.000922169 だった。cuDNN の有無だけでは説明できず、原因は未特定。

## 5モデル追加後

環境は同じ `lab`（Python 3.13.9、PyTorch 2.6.0+cu124）。追加パッケージは入れていない。`libdf`、`mamba_ssm`、`einops`、`omegaconf`、`espnet2`、`torchdiffeq`、`x_transformers` は未インストール。

```bash
.venvs/lab/bin/python -m unittest discover -s tests/se -v
```

19 tests OK（既存 13 に DeepFilterNet3、MossFormerGAN、SEMamba++、UniPASE、StuPASE の短音声テストを追加）。リポジトリ内の他の自動テストディレクトリは `tests/se` だけだった。

`src/se/official` を除いた一時コピーで、5モデルとも import、チェックポイント読み込み、短音声 `enhance` が完了した。

DeepFilterNet3 の分析・合成だけを通した 48 kHz・長さ 4800 の正弦（遅延 480 サンプルを除いた比較）の最大絶対誤差は 4.17e-7。これは公式 `libdf` との波形差ではない。同一発話の再実行差は 0。発話ごとに状態をリセットする。

CUDA は `CUDA_VISIBLE_DEVICES=1` の Quadro RTX 5000。公式実装との GPU 比較ではなく、独立実装の CPU と CUDA の差。

| モデル | 入力 | 最大絶対差 | 平均絶対差 |
|---|---|---:|---:|
| DeepFilterNet3 | 48 kHz、4800 samples | 1.28e-9 | 2.20e-10 |
| MossFormerGAN | 16 kHz、1600 samples | 1.91e-6 | 4.37e-7 |

SEMamba++、UniPASE、StuPASE の CUDA 比較は未実施。StuPASE はサンプリングノイズを含むため、比較には呼び出し前の乱数種が必要。種 5 の再実行差は 0。

公式出力との数値差は、今回追加した5モデルについては未計測。DeepFilterNet3 は `libdf` が無く、SEMamba++ は `mamba_ssm` が無く、MossFormerGAN の公式モジュールは `einops` と `rotary-embedding-torch` が無く、StuPASE は `torchdiffeq` と `x-transformers` が無い。UniPASE の 16 kHz 経路は上流推論クラスそのものだが、別プロセスの公式出力とは突き合わせていない。これらを公式一致とは記録しない。

## 公式参照環境での波形比較

turing。`.venvs/ref` は Python 3.10.12、PyTorch 2.6.0+cu124、torchaudio 2.6.0+cu124、NumPy 1.26.4。追加は `einops==0.8.2`、`rotary-embedding-torch==0.8.3`、`torchdiffeq==0.2.5`、`x-transformers==2.14.2`、`omegaconf==2.3.0`、`librosa==0.10.2.post1`、`yamlargparse==1.31.1`、`pydub==0.25.1`、`maturin==1.4.0`。`libdf` は Rust 1.76.0 で `pyDF` を `CARGO_TARGET_DIR=data/results/se-repro/cargo-target` にビルドした wheel `DeepFilterLib-0.5.7rc0`。公式ツリーは編集していない。GPU は `CUDA_VISIBLE_DEVICES=1` の Quadro RTX 5000。

入力は `torch.manual_seed` の正規乱数。比較は最大絶対誤差。SI-SDR は誤差 0 でも分母を 1e-12 で下駄上げしているため、完全一致でも有限値になる。判定は最大絶対誤差を見る。

| 比較 | 条件 | 最大絶対誤差 |
|---|---|---:|
| MP-SENet VB/DNS 対公式 `mag_pha_stft` | CPU、16 kHz、1 秒・5 秒・20 秒、seed 11 | 0 |
| MP-SENet VB GPU公式 対 GPU独立 | 1 秒、seed 7 | 3.73e-9 |
| MP-SENet DNS GPU公式 対 GPU独立 | 1 秒、seed 7 | 2.33e-10 |
| MP-SENet VB CPU公式 対 GPU公式 | 同じ入力 | 4.16e-3 |
| MP-SENet DNS CPU公式 対 GPU公式 | 同じ入力 | 1.29e-4 |
| TF-GridNet urgent24 | CPU、8 kHz と 16 kHz、1 秒と 20 秒 | 0 |
| TF-GridNet ms-snsd | CPU、16 kHz、1 秒と 20 秒 | 0 |
| MossFormerGAN | CPU、16 kHz、1 秒。修正前の 11 秒は 0.151 | 1 秒は 0 |
| MossFormerGAN 分割修正後 | CPU、16 kHz、11 秒 | 0 |
| DeepFilterNet3 対 `libdf` `enhance` | CPU、48 kHz、0.1 秒 / 1 秒 / 20 秒 / 正弦 / 小振幅 | 1.19e-7 / 4.10e-8 / 6.52e-8 / 8.94e-7 / 1.62e-8 |
| SEMamba++ 対公式グラフ＋参照スキャン | CPU、16 kHz、1 秒波形 | 2.38e-8 |
| UniPASE `forward` | CPU、16 kHz・0.5 秒、および 48 kHz 指定の 16000 samples | 0 |
| StuPASE rotary | x-transformers 2.14.2、dim 64、長さ 17 | 0 |
| StuPASE `forward` | CPU、16 kHz、0.5 秒、seed 5、steps 8、CFG 0.5、sway -1 | 0 |

無音の MP-SENet は公式のエネルギー正規化がゼロ除算になる。API は無音を無音のまま返す。これは公式波形との不一致として数えない。

SEMamba++ の位相の最大絶対誤差は 6.02e-4 だが、波形は 2.38e-8。位相の差は振幅が小さいビンに寄っている。`mamba_ssm` の CUDA カーネルとは比べていない。

再現は `tests/se/reference/run_lab.py`（lab）、`run_ref.py`、`run_semamba.py`、`run_df.py`（ref）。20 秒は MP-SENet、TF-GridNet、DeepFilterNet3 で計測済み。MossFormerGAN の 20 秒は未計測で、11 秒の窓境界は 0。UniPASE と StuPASE の 20 秒は未実施。

`.venvs/lab/bin/python -m unittest discover -s tests/se -v` は 21 tests OK（既存 19 に rotary の 2 件）。

## 最終検証（2026-10-31）

作業開始 HEAD は `6ea86f3`（`[WIP]7モデルの動作を確認`）。ホスト turing。GPU 0 は他ユーザーが使用中のため、比較は GPU 1・2・3（Quadro RTX 5000、ドライバ 535.274.02）。通常テストは `.venvs/lab`。UniPASE、StuPASE、MossFormerGAN、DeepFilterNet3 の公式比較は `.venvs/ref`。SEMamba++ の CUDA 比較は `.venvs/semamba`。

公開 wheel `mamba_ssm-2.3.2.post1+cu12torch2.6cxx11abiFALSE` は `std::__cxx11::basic_string` を参照し、この PyTorch（CXX11 ABI false）では `selective_scan_cuda` がロードできなかった。`.venvs/cuda-nvcc`（conda-forge `cuda-nvcc=12.9`）で sm_75 だけをソースビルドした。ライブラリ本体はこの環境を import しない。

再現:

```bash
CUDA_VISIBLE_DEVICES=1 .venvs/ref/bin/python tests/se/reference/run_long.py unipase --seconds 20 --sr 48000 --seed 4 --device cuda
CUDA_VISIBLE_DEVICES=1 .venvs/ref/bin/python tests/se/reference/run_long.py unipase-long --seconds 24 --sr 16000 --seed 4 --device cuda
CUDA_VISIBLE_DEVICES=2 .venvs/ref/bin/python tests/se/reference/run_long.py stupase --seconds 20 --seed 5 --device cuda
CUDA_VISIBLE_DEVICES=3 .venvs/ref/bin/python tests/se/reference/run_long.py mossformer --seconds 20 --seed 3 --device cuda
CUDA_VISIBLE_DEVICES="" .venvs/ref/bin/python tests/se/reference/run_long.py mossformer --seconds 20 --seed 3 --device cpu
CUDA_VISIBLE_DEVICES="" .venvs/ref/bin/python tests/se/reference/run_df_api.py
CUDA_VISIBLE_DEVICES=1 .venvs/semamba/bin/python tests/se/reference/run_semamba_cuda.py 1 5 20
```

入力は `torch.Generator.manual_seed` の正規乱数に 0.05 を掛けたもの。UniPASE と MossFormerGAN の seed は 4 と 3。StuPASE は呼び出し直前の `torch.manual_seed`。デバイスは各行の CUDA、dtype は float32。許容の見方は、決定的な処理は最大絶対誤差 0、CUDA カーネルと PyTorch 漸化式の差は測定値をそのまま記録する。

| 比較 | 条件 | 最大絶対誤差 |
|---|---|---:|
| UniPASE `forward` | GPU、16 kHz、1・5・20 秒、seed 4。ピーク合わせ後と API も同じ | 0 |
| UniPASE `forward` と PostNet 入出力 | GPU、48 kHz、1・5・20 秒、seed 4 | 0 |
| UniPASE `inference_long` | GPU、16 kHz と 48 kHz、24 秒、seed 4。API も同じ | 0 |
| StuPASE `forward` | GPU、16 kHz、1 秒は seed 5・7・11。5 秒と 20 秒は seed 5 | 0 |
| StuPASE ODE 軌跡と API | GPU、16 kHz、1・5・20 秒、seed 5。API は `enhance` の直前に seed を再設定 | 0 |
| StuPASE CFM / Vocoder | `strict=False` の missing と unexpected | 0 |
| MossFormerGAN decode | CPU、16 kHz、20 秒、seed 3 | 0 |
| MossFormerGAN decode | GPU、16 kHz、15・20・25 秒、seed 3 | 1.40e-6 / 2.09e-6 / 2.33e-6 |
| DeepFilterNet3 16 kHz API | CPU。torchaudio で 48 kHz にして `libdf` `enhance` し、同じ torchaudio で 16 kHz へ戻す | 3.86e-6 |
| SEMamba++ 単一 Mamba | GPU。融合カーネルと CUDA scan は 0。融合カーネルと PyTorch 漸化式は 3.28e-7 | 3.28e-7 |
| SEMamba++ 波形 | GPU、16 kHz、1・5・20 秒、seed 9。公式は `mamba_inner_fn`。missing=unexpected=0 | 3.97e-5 / 1.56e-5 / 1.80e-5 |
| SEMamba++ 位相 | 同上。差の最大は約 π で、振幅の小さいビンの折り返し | 6.28 |

SEMamba++ の 20 秒は公式 0.79 秒（RTF 0.039、peak VRAM 1.3 GiB）、独立実装 6.57 秒（RTF 0.33、peak VRAM 6.5 GiB）。融合カーネルへの置き換えはしていない。

UniPASE の 20 秒全文は公式 `inference.py`、24 秒は `inference_long.py`。両方とも独立実装と一致した。エンハンサは 20 秒ちょうどまで全文、それを超えると長時間経路に入る。

共通 API の形状、dtype、8/16/48 kHz の長さ、無音、NaN 拒否、可変長バッチ、チェックポイントのシンボリックリンク、DeepFilterNet3 の 16 kHz が torchaudio 往復と 1e-6 未満で一致することは `tests/se/test_api_contract.py`。UniPASE の窓が入力を欠かさず覆うことは、恒等モデルで 21・24・30 秒を確認した。

`.venvs/lab/bin/python -m unittest discover -s tests/se -v` は 28 tests OK（従来 21 に、共通 API 契約 6 と UniPASE 窓の恒等テスト 1 を加えた）。公式比較スクリプトは lab では走らない。SKIP は PASS に数えない。

L2 は GO。残る未検証は、TF-GridNet の ESPnet CLI、MP-SENet の PCM 16 と公式 CLI、DeepFilterNet3 の sinc リサンプラ、SEMamba++ の速度最適化。いずれも今回の波形不一致ではない。
