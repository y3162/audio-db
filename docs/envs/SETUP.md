# セットアップ

利用者が編集するのは `.env` だけである。プラットフォームは構築コマンドの引数で、仮想環境名には入らない。

```bash
cp .env.example .env
# AUDIO_DB_DATASET_ROOT と、必要なら UV_CACHE_DIR を書く
make setup PLATFORM=tsubame
make doctor
```

SHINODA-LAB では最後の構築コマンドだけを変える。ホスト差は [TSUBAME](TSUBAME.md) と [SHINODA-LAB](SHINODA_LAB.md)、ASR 用の環境は [ASR](ASR.md)。作り直しは同じ `make setup` の再実行である。2026-11-04 の一回の記録は [CLEAN_REBUILD](CLEAN_REBUILD.md)。

```bash
make setup PLATFORM=shinoda_lab
```

`source .env` は任意である。既に export した変数は `.env` より優先される。`make` は `.env` を、未設定の変数に限って読む。

## 役割

| パス | 用途 |
|---|---|
| `.venvs/se` | 通常の唯一の環境。6 モデル、ASR、DB、評価、テスト、fused scan |
| `.venvs/official` | 公式実装との比較。NumPy 2.5。通常の学習には使わない |
| `.venvs/deepfilternet` | 公式 `libdf`。PyO3 が NumPy 1.26 を要求し、`.venvs/se` の 2.1.3 とも `.venvs/official` の 2.5.4 とも共存しない |

依存の正本は `requirements/se.in`、`requirements/official.in`、`requirements/deepfilternet.in` と、それぞれの lock である。通常の `make setup` は lock を更新しない。lock を作り直すときだけ `scripts/setup/compile_locks.sh` を実行する。

`.venvs/se` は NumPy 2.1.3、numba 0.61.0、llvmlite 0.44.0 である。NeMo 2.2.1 がこの Numba を要求し、その Numba は NumPy 2.2 以降を拒む。scipy 1.18.1、librosa 1.0.0、nnAudio 0.3.3 はこの NumPy に合わせる。以前の NumPy 2.5.4 と numba 0.68.1 は、NeMo と同じプロセスでは使わない。`mamba-ssm` と `causal-conv1d` は lock に入らない。`openai-whisper==20240930` も lock の外で、sync のあと `--no-build-isolation` で入れる。

```bash
make setup-env PLATFORM=shinoda_lab ROLES=se
```

`scripts/setup/with_env.sh` は `.venvs/se` の Python を起動するときだけ、未設定の `CUDA_HOME` をその環境の `nvidia/cuda_nvcc` にする。`nvvm/lib64`、`lib64`、`nvidia/cuda_runtime/lib` を `LD_LIBRARY_PATH` の先頭に置く。呼び出し側が `CUDA_HOME` を置いていればそれを使う。公式役割には渡さない。この wheel は `libnvvm.so` と `ptxas` を含み、`nvcc` 本体は含まない。Numba は版付きの `libnvvm.so.*` を探すので wheel を見つけられないことがあり、そのときはこの `LD_LIBRARY_PATH` の `libnvvm.so` を読む。手で同じ変数を置く手順は [ASR](ASR.md)。

SE 学習ログを TensorBoard で確認できるよう、`.venvs/se` に `tensorboard==2.20.0` を含める。TSUBAME と SHINODA-LAB のどちらも `make setup PLATFORM=...` で構築される同じ lock を使う。

## 部分実行

```bash
make setup-env PLATFORM=tsubame ROLES=se
make assets MODELS=mp_senet
make assets-check
make db-link DRY_RUN=1
make db-link ONLY=LibriSpeech ALLOW_PARTIAL=1
make test-setup
```

`make setup` は `db-init`、`db-ingest`、`db-bootstrap` を呼ばない。役割の導入のあと `build_df_rfft` が `make df-rfft` と同じ Cargo ロックで `build/df-rfft/release/libdf_rfft.so` を作り、続けて `python -m src.data.link_corpora --allow-partial` と `scripts/setup/doctor.py` を実行する。`link_corpora` は `data/corpora` へのシンボリックリンクだけで、無いソースは記録して残りを結び、既存のリンクは上書きしない。DuckDB は作らない。最後の診断を、別の `make doctor` で再実行してよい。

追跡中の `src/models/deepfilternet3/libdf_rfft.so` と内容が違うときは置き換えない。環境変数 `DF_RFFT_LIBRARY` が空で追跡ファイルがこのホストで読めないときは、そのローカル成果物が ABI 1 なら実行時にそれを使う。ファイル自体が無いときの戻り先は [DeepFilterNet3](../se/MODELS.md)。`make doctor` の `df_rfft` は、選ばれたファイルの source、ABI、SHA-256 を出す。

`scripts/setup/platforms/tsubame.sh` は cargo が無いと準備を打ち切る。両プラットフォームのメッセージが求めるのは Rust 1.76 以上である。`make df-rfft` が通った記録は rustc 1.99.1 で、他の版での再コンパイルは記録が無い。`libdf` の wheel を Rust 1.76.0 で作った記録は [VALIDATION.md](../se/VALIDATION.md)。依存の版は `src/models/deepfilternet3/rfft_kernel/Cargo.lock` の `realfft` 3.3.0 と `rustfft` 6.4.1。

## 融合カーネル

`.venvs/se` の `mamba-ssm` と `causal-conv1d` は、`requirements/official-assets.toml` のピンしたソースからビルドする。lock には入っていない。インストール後、`scripts/apply_mamba_init_patch.py` が `patches/mamba_ssm-2.3.2.post1-init.patch` を当てる。対象は `mamba_ssm/__init__.py` の Mamba2、Mamba3、言語モデルヘッドの import だけである。selective scan のソースと CUDA 拡張は変えない。再実行は何もしない。

`AUDIO_DB_REBUILD_MAMBA=1` は、入っている拡張と `build/native-src` の中間ビルドを消してから作り直す。`TORCH_CUDA_ARCH_LIST` はプラットフォームスクリプトが設定する。ピンした causal-conv1d 1.7.0 と mamba-ssm 2.3.2.post1 の `setup.py` は CUDA 12.1 で `sm_75`、`sm_80`、`sm_87`、`sm_90` を `-gencode` に固定し、この環境変数では cubin の一覧を変えない。`sm_86` は含まれない。あるホストでビルドした `.so` を、別のアーキテクチャのホストへコピーしない。`uv` がホームのキャッシュへ `.so` を書けないときは、`UV_CACHE_DIR` を書けるローカルディレクトリにしてから同じビルドをやり直す。`build_mamba` に渡す Python は仮想環境の `bin/python` を絶対パスにする。そのシンボリックリンクを辿った先の uv 管理 Python へは入れない。

## 公式アセット

取得元・commit・SHA-256 は `requirements/official-assets.toml`。配置先は既存テストが読むパスである。

| モデル | ソース | 重み |
|---|---|---|
| MP-SENet | `src/official/mp-senet` @ `89932cfe` | `data/checkpoints/mp-senet/g_best_vb` と `g_best_dns`（リポジトリ内の `best_ckpt`） |
| DeepFilterNet3 | `src/official/deepfilternet` @ `d375b2d83` | `data/checkpoints/deepfilternet3/DeepFilterNet3/`（ピンした commit の公式 zip） |
| SEMamba++ | `src/official/semambapp` @ `cd369c89` | `data/checkpoints/semambapp/semambapp.pth` |
| MossFormerGAN | `src/official/clearervoice-studio` @ `6b3774dc` | `data/checkpoints/mossformergan/last_best_checkpoint.pt` |
| TF-GridNet | `src/official/espnet` @ `0d136c54`（新規 clone は `espnet2` のみ） | `data/checkpoints/tf-gridnet/urgent24/.../21epoch.pth` と `ms-snsd/valid.loss.best.pth` |
| TF-Locoformer | `src/official/tf-locoformer` @ `7a615460` | `data/checkpoints/tf-locoformer/dns/valid.loss.ave_5best.pth` |

既にピンした commit のツリーがあれば再取得しない。別 commit や手元の変更は上書きしない。チェックサムが違う既存ファイルも置き換えない。

TF-GridNet の重みカードは、コードの Apache-2.0 とは別に利用条件を書いていない。SEMamba++ の重みカードは GPL-2.0 と表示されている。ソースツリーにライセンスファイルは無い。

## 失敗したとき

同じ `make setup PLATFORM=...` を再実行する。成功済みの仮想環境（lock のハッシュが一致し、import できるもの）と、チェックサムが一致するファイルはやり直さない。`FORCE_SETUP=1` は lock の同期からやり直すが、仮想環境ディレクトリは消さない。

公式 `libdfdata` は HDF5 が必要である。TSUBAME では `hdf5-parallel/1.14.3/gcc11.4.1` を読む。モジュールが無い場合、`libdf` 本体の成否と `libdfdata` の未構築は分けて表示する。

`libdf` のビルドは `src/official/deepfilternet` を書き換えない。Cargo 1.99 はピンした `Cargo.lock`（version 3）を `--locked` のまま読めず、ロックの更新を拒否する。再ビルドが必要なときは `build/libdf-src` の共有 clone だけでロックを更新し、`--offline` で maturin する。既に `build/libdf-cargo/wheels` にある wheel が import できるときは、それを入れ直して再コンパイルしない。
