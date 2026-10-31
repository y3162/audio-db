# lab

tensor / turing / shannon / ampere で共通。仮想環境は `.venvs/lab` 一つ。

| host | GPU |
|---|---|
| tensor | A100 40GB x7 / A100 80GB x1 |
| turing | Quadro RTX 5000 x4 |
| shannon | A100 80GB x8 |
| ampere | RTX 3090 x4 |

ドライバは 535。PyTorch は CUDA 12.4 の wheel、Numba の ptxas は 12.2.140。12.4 の NVVM が出す PTX はドライバがロードできない。システムの `/usr/local/cuda` は使わない（tensor 11.0、turing 11.2、shannon 11.3、ampere 10.2）。

### CLI

`~/.local/bin/duckdb` が無いときだけ入れる。

```bash
export PATH="$HOME/.local/bin:$PATH"
curl -fsSL https://install.duckdb.org | DUCKDB_VERSION=2.0.0 bash
```

### Python

`duckdb` 2.0.0 は Python 3.11 以降が必要。共有の Miniconda 3.13.9 を使う。

```bash
cd /net/spring/work/seki/exp/audio-db
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=./.venvs/.uv
uv venv .venvs/lab --python "$HOME/miniconda3/bin/python3"
uv pip install --python .venvs/lab/bin/python --only-binary :all: \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    duckdb==2.0.0 \
    "numba==0.68.1" \
    "numpy==2.5.4" \
    "llvmlite==0.50.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
uv pip install --python .venvs/lab/bin/python --only-binary :all: \
    "nvidia-cuda-nvcc-cu12==12.2.140"
CUDA_HOME="$PWD/.venvs/lab/lib/python3.13/site-packages/nvidia/cuda_nvcc"
CUDA_RUNTIME="$PWD/.venvs/lab/lib/python3.13/site-packages/nvidia/cuda_runtime/lib"
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$CUDA_RUNTIME/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"
```

Numba は `CUDA_HOME/lib64/libcudart.so.12` を読む。実体は `nvidia/cuda_runtime` にあるのでリンクする。

### 実行

`nvidia-smi` で空き GPU を確認し、その番号だけを渡す。

```bash
source .venvs/lab/bin/activate
export CUDA_HOME="$PWD/.venvs/lab/lib/python3.13/site-packages/nvidia/cuda_nvcc"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=<空きGPU>
```

| | |
|---|---|
| Python | 3.13.9 |
| torch / torchaudio | 2.6.0+cu124 / 2.6.0 |
| numba / llvmlite / numpy | 0.68.1 / 0.50.0 / 2.5.4 |
| duckdb | 2.0.0 |
| ptxas | 12.2.140 |
| pyyaml / scipy / pystoi / pesq | 6.0.3 / 1.18.1 / 0.4.1 / 0.0.4 |

評価パイプラインの追加パッケージは、既存の numpy を上げずに入れる。

```bash
uv pip install --python .venvs/lab/bin/python "numpy==2.5.4" -r requirements/evaluation.txt
```
