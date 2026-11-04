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

融合カーネルのビルドは、そのマシンの `nvidia-smi` が報告する compute capability を `TORCH_CUDA_ARCH_LIST` にする。GPU が見えないホストでは fused 拡張をビルドせず、失敗として残す。公式 `libdfdata` は HDF5 の場所を `HDF5_DIR` で渡してから `make setup-env PLATFORM=shinoda_lab ROLES=deepfilternet` を再実行する。

歴史的な `.venvs/lab` が残っていても削除しない。新しい役割の環境とはパスが違う。

詳細は [セットアップ](SETUP.md)。
