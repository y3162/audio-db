# クリーン環境の再構築

既存の `.venvs/tsubame` や `.venvs/tsubame-h100-build` を更新しない。新しい仮想環境と `build/clean-20261104/` にソースとビルド成果物を置く。`build/` は Git 管理外である。

2026-11-04 に TSUBAME の計算ノード `r4n11`（対話ジョブ `115437`、H100 MIG 4g.47gb）で通した手順である。別ノードでの再実行は未実施。

## システム

- Python 3.12.13（`uv`）
- rustc / cargo 1.99.1
- CUDA 12.3.2（`module load cuda/12.3.2`）。PyTorch は cu124 の wheel
- HDF5 1.14.3（`module load hdf5-parallel/1.14.3/gcc11.4.1`）。公式 DataLoader のビルドだけに使う
- `patchelf` は DataLoader の wheel 化に必要

## Python

```bash
uv venv .venvs/clean-20261104 --python 3.12
uv pip install --python .venvs/clean-20261104/bin/python torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venvs/clean-20261104/bin/python \
  -r requirements/clean-rebuild.txt
```

DeepFilterNet の公式 `libdf` は NumPy 1.26 向けの別環境 `.venvs/clean-20261104-df` に置く。PyO3 0.20 / `numpy` crate 0.20 は、この再構築では NumPy 2.5 の主環境には載せていない。

直接依存の一覧は `requirements/clean-rebuild.txt`。導入後の全パッケージは `uv pip freeze` で `build/clean-20261104/requirements-freeze.txt` に残す。

## 公式ソース

次の commit を `git clone --filter=blob:none` して `git checkout --detach` する。既存の `src/se/official/` は上書きしない。

| 実装 | URL | commit |
|---|---|---|
| MP-SENet | https://github.com/yxlu-0102/MP-SENet.git | `89932cfe90d1dacb8e170e4a331d762462c21792` |
| SEMamba++ | https://github.com/infected4098/SEMambapp.git | `cd369c89a8106072c077b468417b765b227afd74` |
| DeepFilterNet | https://github.com/Rikorose/DeepFilterNet.git | `d375b2d8309e0935d165700c91da9de862a99c31` |
| ClearerVoice-Studio | https://github.com/modelscope/ClearerVoice-Studio.git | `6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61` |
| ESPnet | https://github.com/espnet/espnet.git | `0d136c5456d403cb2c0a4c13e76e8136e32e9b7a` |

Mamba 拡張のソースはタグの tarball である。

- https://github.com/Dao-AILab/causal-conv1d/archive/refs/tags/v1.7.0.tar.gz
- https://github.com/state-spaces/mamba/archive/refs/tags/v2.3.2.post1.tar.gz

`mamba-ssm` 2.3.2.post1 の `mamba_ssm/__init__.py` は `Mamba2`、`Mamba3`、言語モデルヘッドを import する。Triton 3.2.0 には `Mamba3` が要求する API が無く、ヘッドは `transformers` を要求する。インストール後に `scripts/apply_mamba_init_patch.py --init <venv>/lib/python3.12/site-packages/mamba_ssm/__init__.py` を実行する。差分の正本は `patches/mamba_ssm-2.3.2.post1-init.patch` で、対象は import 3 行だけである。再実行は何もしない。selective scan のソースや CUDA 拡張はこのスクリプトが書き換えない。公式 SEMamba++ は `mamba_ssm.models.mixer_seq_simple` を直接 import するため、学習エントリーポイントには別途 `transformers==4.46.3` が要る。

## CUDA 拡張

```bash
module load cuda/12.3.2
export TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
export PATH="${CUDA_HOME}/bin:$PATH"
# 展開した各ディレクトリで
uv pip install --python .venvs/clean-20261104/bin/python --no-build-isolation --no-deps .
```

`PYTHONPATH` に既存の `.venvs/tsubame-h100-build` を入れない。import 元が `.venvs/clean-20261104` であることを確認する。

## Rust FFT と libdf

解析 FFT は既存の共有ライブラリを置き換えない。

```bash
cargo build --release --locked \
  --manifest-path src/se/models/deepfilternet3/rfft_kernel/Cargo.toml \
  --target-dir build/clean-20261104/cargo-rfft
export DF_RFFT_LIBRARY="$PWD/build/clean-20261104/cargo-rfft/release/libdf_rfft.so"
```

`DF_RFFT_REQUIRED=1` のとき、ライブラリが無いと `FileNotFoundError` になる。未設定なら `torch.fft.rfft` に戻る。このフォールバックはビルド成功の証拠にしない。

公式 `libdf` は取得した DeepFilterNet の `pyDF` を、Python 3.12 を `--interpreter` に指定して `maturin build --release` する。`--interpreter` を省くと cp39 の wheel になる。`pyDF-data` は `HDF5_DIR` と PATH 上の `patchelf` が要る。できた wheel は `DeepFilterLib-0.5.7rc0-cp312` と `DeepFilterDataLoader-0.5.7rc0-cp312` で、`.venvs/clean-20261104-df` に入れる。ソースツリーをカレントにして import すると、インストール先ではなくソースが読まれる。`/tmp` から import し、`libdf` と `libdfdata` の `__file__` が site-packages であることを確認する。実行時は `module load hdf5-parallel/1.14.3/gcc11.4.1` を維持する。公式 `df/train.py` は `icecream` も import する。

## スモーク

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  .venvs/clean-20261104/bin/python tests/training/gpu_smoke.py
```

公式学習エントリーポイントの短時間実行は `tests/training/reference/official_entry_smoke.py`。出力先は存在しないディレクトリを指定する。

## 既知の制約

- 別ノード、別 GPU アーキテクチャでの再ビルドは未実施。今回の cubin は `sm_90` だけである。未実施の理由はジョブを増やさないためであり、依存関係やコンパイラの失敗ではない。
- 公式 DeepFilterNet の学習は、HDF5 を `df/scripts/prepare_data.py` で作り、`libdfdata` の DataLoader で読む。`num_workers = 0` では epoch 開始直後に `TimeoutError` になり、プロセスは戻らなかった。`num_workers = 2` と `overfit = true` では 2 更新が完了した。記録は `docs/training/VERIFICATION.md` の 2026-11-04 節。
- ESPnet の `espnet2/bin/enh_train.py` は、`typeguard==4.4.4`、`configargparse`、`humanfriendly`、`torch-complex`、`omniio`、`sentencepiece`、`fast-bss-eval` を入れると起動する。`flash_attn` は無く、ESPnet は自前の注意実装へ戻る。MS-SNSD の公式設定は 16 kHz、`chunk_length` 48000、`n_fft` 512、`stride` 256、Adam、`mr_l1_tfd` である。2026-11-04 の短時間実行は batch 1、worker 0、更新 2、chunk 16000、1 epoch で、VoiceBank を 16 kHz に変換して公式 DataLoader に渡した。記録は `docs/training/VERIFICATION.md`。
- SEMamba++ の `train.py` は epoch 数を引数にしない。短時間実行は出力ディレクトリの外部 launcher が `main` のソース文字列だけを差し替えて `exec` する。ピン留めツリーは `utils.py` しか無いのに `dataset.py` が未使用の `utils.util` を import するので、launcher が同じ関数を `sys.modules` に登録する。公式ファイルは編集しない。
