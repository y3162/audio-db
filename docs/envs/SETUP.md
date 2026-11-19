# セットアップ

初回はリポジトリのルートで作業する。

## 1. ツールを用意する

- Git、Git LFS、GNU Make、uv、Python 3 を PATH に置く。仮想環境で使う Python 3.12 は `uv python install 3.12` で用意する。
- 通常環境のビルドには C/C++ コンパイラ、Rust 1.76 以降と Cargo、CUDA 12 系の Toolkit が要る。`nvcc --version` が表示できることを確認する。Toolkit が PATH に無い場合は、そのルートを `CUDA_HOME` に指定する。
- 公式 DeepFilterNet の比較も行う場合は、HDF5 の開発用ヘッダーを用意する。`pkg-config --modversion hdf5` で確認する。別の場所にある HDF5 を使う場合は `HDF5_DIR` にそのルートを指定する。
- GPU を使う前に `nvidia-smi` で空きを確認する。使用する番号を `CUDA_VISIBLE_DEVICES` に指定できる。

ホスト上でツールを有効にする手順は、[SHINODA_LAB.md](SHINODA_LAB.md) または [TSUBAME.md](TSUBAME.md) を参照する。

## 2. データの置き場を指定する

まだ `.env` が無ければ `cp .env.example .env` で作る。`.env` の `AUDIO_DB_DATASET_ROOT` に、`LibriSpeech` や `VoiceBank+DEMAND` などを直接含むディレクトリの絶対パスを書く。

```bash
export AUDIO_DB_DATASET_ROOT="/path/to/datasets"
```

コーパスごとに別の場所を使うときは、対応する `AUDIO_DB_CORPUS_...` にそのコーパス自体の絶対パスを書く。いずれも `.env.example` にある。シェルから export した値は `.env` より優先される。

## 3. 構築してテストする

```bash
make setup
make test
```

`make setup` は、バージョンを固定した公式アセットと仮想環境を用意し、コーパスへのリンクを `data/corpora/` に作り、最後に `doctor` を実行する。リンク元が無いコーパスは表示して残りを結ぶ。既存リンクが指定先と違う場合はエラーになる。DB の準備が必要な場合は、別途 [DB の手順](../db/SETUP.md) を参照する。

`make test` はデータを読み取り専用にして、CPU テストと、空き GPU がある場合の GPU テストを実行する。CPU だけなら `make test-cpu` を使う。結果とスキップの読み方は [テスト](../TESTING.md) にある。

## 部分実行と再実行

| コマンド | 用途 |
|---|---|
| `make doctor` | 構築済みの環境、アセット、コーパスリンクを診断する |
| `make setup-env ROLES=se` | 通常環境だけを構築する。初回は先に `make assets MODELS=causal_conv1d,mamba` を実行する |
| `make setup-env ROLES=official,deepfilternet` | 公式比較用の環境を構築する。初回は先に `make assets MODELS=deepfilternet` を実行する |
| `make assets MODELS=mp_senet` / `make assets-check` | アセットを取得する / 既存のアセットを検証する |
| `make db-link DRY_RUN=1 ALLOW_PARTIAL=1` | コーパスリンクの作成予定を確認する |
| `make test-setup` | セットアップ用のテストだけを実行する |

失敗したら原因を直し、同じコマンドを再実行する。lock のハッシュが一致して import できる仮想環境と、検証済みのアセットは再利用される。`FORCE_SETUP=1` を付けると、仮想環境の依存を再同期する。キャッシュの書き込みに失敗する場合は `UV_CACHE_DIR` を書き込み可能なディレクトリに設定する。

## 仮想環境とビルド

| パス | 用途 |
|---|---|
| `.venvs/se` | 通常の SE、ASR、評価、学習、テスト。PyTorch 2.6.0+cu124。ASR の依存は [ASR](ASR.md) を参照 |
| `.venvs/official` | 公式実装との比較。NumPy 2.5.4 |
| `.venvs/deepfilternet` | 公式 `libdf` と `libdfdata` との比較。これらの拡張は NumPy 1.26.4 を使う |

Python パッケージのバージョンは役割ごとの `requirements/<role>.lock.txt` で固定している。`setup` は lock を更新しない。依存を変更するときは `requirements/<role>.in` を編集し、`scripts/setup/compile_locks.sh` で lock を作り直す。

- 融合カーネルの `mamba-ssm` と `causal-conv1d` は、`requirements/official-assets.toml` が指定するソースからビルドする。`mamba-ssm` には import 用のパッチも当てる。Toolkit と各ソースのビルド設定によって対応する GPU が決まる。別ホストで共有するときは GPU と glibc の互換性を確認する。作り直すには `AUDIO_DB_REBUILD_MAMBA=1 make setup-env ROLES=se` を使う。
- 通常環境で使う FFT は `build/df-rfft/release/libdf_rfft.so` にビルドする。追跡中のライブラリをロードできないときに、この成果物が使われる。選ばれたファイルと ABI は `make doctor` の `df_rfft` に出る。実行時に指定するなら `DF_RFFT_LIBRARY` を使う。
- 公式 DeepFilterNet の `libdf` と `libdfdata` は、構築したホストで使う Linux wheel としてビルドする。`build/libdf-src` に作るビルド用コピーではデモとプラグインを対象から外す。残りの Rust 依存のバージョンとチェックサムは、公式の `Cargo.lock` で検証する。

公式コードと重みの配置先、commit、SHA-256 は [アセット一覧](../../requirements/official-assets.toml) にある。取得時に、編集済みの checkout や検証に失敗した既存ファイルが見つかるとエラーになる。
