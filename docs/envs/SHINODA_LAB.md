# SHINODA-LAB

同じ研究室の ampere / tensor / turing / shannon を指す。構築後の仮想環境名は TSUBAME と共通で、`.venvs/se` である。手順の本体は [セットアップ](SETUP.md)。

| host | GPU |
|---|---|
| tensor | A100 40GB x7 / A100 80GB x1 |
| turing | Quadro RTX 5000 x4 |
| shannon | A100 80GB x8 |
| ampere | RTX 3090 x4 |

ドライバ 535 系で動かした記録がある。PyTorch は CUDA 12.4 の wheel を使う。Numba の ptxas は環境内の `nvidia-cuda-nvcc-cu12==12.2.140`（`requirements/se.in`）で、システムの `/usr/local/cuda` は使わない。

```bash
cp .env.example .env
make setup PLATFORM=shinoda_lab
make doctor
```

`scripts/setup/platforms/shinoda_lab.sh` は Environment Modules から `cuda/12.1`、無ければ `cuda/12.8` を読む。`module load cuda/12.1` の実体は `nvcc --version` と `readlink -f "$(command -v nvcc)"` で確認する。`nvidia-smi` が GPU を報告しないホストでは融合拡張をビルドせず、その役割は失敗として残す。

融合カーネルの `TORCH_CUDA_ARCH_LIST` の既定は `7.5;8.0;8.6`（Turing 7.5、A100 8.0、RTX 3090 8.6）である。既に環境変数があれば上書きしない。以前の記録では、causal-conv1d 1.7.0 と mamba-ssm 2.3.2.post1 の `setup.py` は CUDA 12.1 のとき `sm_75`、`sm_80`、`sm_87`、`sm_90` を明示し、この環境変数だけでは一覧を変えられない、とある。ピンした `setup.py` はこの作業ツリーに無く、その記録は再確認していない。RTX 3090 の sm_86 を sm_80 の cubin で動かす、という記述も同じ記録である。実際の `-gencode` はビルドログで確認する。

共有の `.venvs/semambapp` を複数ホストで使うときは、最も古い glibc のホストでビルドする。`AUDIO_DB_REBUILD_MAMBA=1` は共有の拡張と `build/native-src` の中間ディレクトリを削除して再構築する。新しい glibc でコンパイルした中間ファイルを残すと、Shannon が `GLIBC_2.32` を読めない、という理由で `scripts/setup/env_roles.sh` が中間ディレクトリを消す。

公式 `libdfdata` は Debian/Ubuntu の `libhdf5-dev`（`/usr/lib/x86_64-linux-gnu/hdf5/serial`）を自動検出する。別の HDF5 だけ使うときは `HDF5_DIR` を export してから `make setup-env PLATFORM=shinoda_lab ROLES=deepfilternet` を再実行する。

以前の手順書の名前は `LAB.md`、仮想環境名は `.venvs/lab` だった。`.venvs/lab` が残っていても削除しない。役割の環境とはパスが違う。
