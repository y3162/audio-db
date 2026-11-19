# SHINODA_LAB

[共通セットアップ](SETUP.md)に沿って、GPU を使うホストで作業する。データの場所は自分の `.env` に設定する。

| ホスト | GPU |
|---|---|
| tensor | A100 40 GB × 7、A100 80 GB × 1 |
| turing | Quadro RTX 5000 × 4 |
| shannon | A100 80 GB × 8 |
| ampere | RTX 3090 × 4 |

`nvidia-smi` で空きを確認し、使う GPU の番号を `GPU_ID` に設定する。CUDA Toolkit は構築時にだけモジュールで読み込む。`module` が使えない非ログインシェルでは、先に `source /usr/share/Modules/init/bash` を実行する。

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
(
  module load cuda/12.1 &&
  nvcc --version &&
  make setup
) &&
make test
```

通常の実行時には、Numba 用の CUDA ライブラリを `.venvs/se` から読み込む。上の例では Toolkit の設定が構築用のサブシェルに留まる。

CUDA 12.1 でピンした causal-conv1d と mamba-ssm をビルドすると、ソースの設定で `sm_75`、`sm_80`、`sm_87`、`sm_90` が指定される。RTX 3090 用の `sm_86` cubin はこの指定に含まれない。2026-11-17 に、ビルド設定と拡張に入った cubin を確認した。

共有の `.venvs/se` を複数ホストで使う場合は、最も古い glibc のホストでビルドする。tensor と shannon は glibc 2.31 である。新しい glibc で作った拡張を作り直す場合は、Toolkit を読み込んでから `AUDIO_DB_REBUILD_MAMBA=1 make setup-env ROLES=se` を実行する。
