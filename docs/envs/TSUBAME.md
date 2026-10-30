# TSUBAME 4.0

### CLI

`~/.local/bin/duckdb` が無いときだけ入れる。

```bash
export PATH="$HOME/.local/bin:$PATH"
curl -fsSL https://install.duckdb.org | DUCKDB_VERSION=2.0.0 bash
```

### Python

`duckdb` 2.0.0 は Python 3.11 以降が必要。

```bash
cd /gs/bs/tga-shinoda/seki/audio-db
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=./.venvs/.uv
uv venv .venvs/tsubame --python 3.13
uv pip install --python .venvs/tsubame/bin/python --only-binary :all: \
    torch==2.6.0 \
    torchaudio==2.6.0 \
    duckdb==2.0.0 \
    "numba==0.68.1" \
    "numpy==2.5.4" \
    "llvmlite==0.50.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple
uv pip install --python .venvs/tsubame/bin/python --only-binary :all: \
    "nvidia-cuda-nvcc-cu12==12.2.140"
CUDA_HOME="$PWD/.venvs/tsubame/lib/python3.13/site-packages/nvidia/cuda_nvcc"
CUDA_RUNTIME="$PWD/.venvs/tsubame/lib/python3.13/site-packages/nvidia/cuda_runtime/lib"
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$CUDA_RUNTIME/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"
```

Numba は `CUDA_HOME/lib64/libcudart.so.12` を読む。実体は `nvidia/cuda_runtime` にあるのでリンクする。

### 実行

`nvidia-smi` で空き GPU を確認し、その番号だけを渡す。

```bash
source .venvs/tsubame/bin/activate
export CUDA_HOME="$PWD/.venvs/tsubame/lib/python3.13/site-packages/nvidia/cuda_nvcc"
export PATH="$HOME/.local/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=<空きGPU>
```

| | |
|---|---|
| Python | 3.13.14 |
| torch / torchaudio | 2.6.0 / 2.6.0 |
| numba / llvmlite / numpy | 0.68.1 / 0.50.0 / 2.5.4 |
| duckdb | 2.0.0 |
| ptxas | 12.2.140 |
