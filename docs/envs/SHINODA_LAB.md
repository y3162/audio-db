# SHINODA-LAB

以前の手順書の名前は `LAB.md`、仮想環境名は `.venvs/lab` だった。同じ研究室の ampere / tensor / turing / shannon を指す。構築後の環境名は TSUBAME と共通で、`.venvs/se` である。

| host | GPU |
|---|---|
| tensor | A100 40GB x7 / A100 80GB x1 |
| turing | Quadro RTX 5000 x4 |
| shannon | A100 80GB x8 |
| ampere | RTX 3090 x4 |

ドライバは 535 系の記録がある。PyTorch は CUDA 12.4 の wheel を使う。Numba の ptxas は環境内の `nvidia-cuda-nvcc-cu12==12.2.140` で、システムの `/usr/local/cuda` は使わない。

```bash
cp .env.example .env
make setup PLATFORM=shinoda_lab
make doctor
```

GPU が見えないホストでは fused 拡張をビルドせず、失敗として残す。`make setup` は Environment Modules から `cuda/12.1`（無ければ `cuda/12.8`）を読み、nvcc で融合カーネルをビルドする。この環境では `module load cuda/12.1` の実体を `nvcc --version` と `readlink -f "$(command -v nvcc)"` で確認する。CUDA 12.1 の実体は `/net/ox/data/.apps/modules/binaries/cuda/12.1` だった。

固定済み causal-conv1d 1.7.0 / mamba-ssm 2.3.2.post1 の `setup.py` は CUDA 12.1 のとき `sm_75`、`sm_80`、`sm_87`、`sm_90` を明示し、`TORCH_CUDA_ARCH_LIST` だけではこの一覧を変更できない。RTX 3090 の sm_86 は sm_80 cubin の互換実行を使う。実際の `-gencode` はビルドログで確認する。共有の `.venvs/semambapp` を複数ホストで使うときは最も古い glibc のホストでビルドする。`AUDIO_DB_REBUILD_MAMBA=1` は共有の拡張と中間 build ディレクトリを削除して再構築するため、既存環境を残す試験では別ディレクトリにソースを複製する。Numba の ptxas は引き続き wheel の `nvidia-cuda-nvcc-cu12` を使う。公式 `libdfdata` は Debian/Ubuntu の `libhdf5-dev`（`/usr/lib/x86_64-linux-gnu/hdf5/serial`）を自動検出する。別の HDF5 だけ使うときは `HDF5_DIR` を export してから `make setup-env PLATFORM=shinoda_lab ROLES=deepfilternet` を再実行する。

歴史的な `.venvs/lab` が残っていても削除しない。新しい役割の環境とはパスが違う。

詳細は [セットアップ](SETUP.md)。

共有インストールを変更しない検証用ビルドは、各ホストで新しい出力ディレクトリを指定する。依存環境 `.venvs/semambapp` と `make assets` で取得した固定ソースが必要。ビルドはソースと中間成果物を出力先へ複製し、`.so` を `lib/` に置く。`PYTHONPATH` の先頭にその `lib/` を置いて、どの `.so` が読み込まれたか確認する。

```bash
CUDA_HOME=/net/ox/data/.apps/modules/binaries/cuda/12.1 \
  scripts/training/build_mamba_isolated.sh <新規ディレクトリ> > <新規ログ> 2>&1
# 使用前に nvidia-smi で空き GPU を確認
CUDA_VISIBLE_DEVICES=<空きGPU> OMP_NUM_THREADS=1 \
  PYTHONPATH=<新規ディレクトリ>/lib:. \
  .venvs/semambapp/bin/python scripts/training/probe_semamba_fused.py
```

この短い probe は 0.5 秒の合成音声で公式幅 Generator と 2 Discriminator の forward・backward・optimizer 更新を実行し、損失・重みの有限性と各ネットワークの更新を確認する。本格学習や音質の評価ではない。
