# ASR 用の仮想環境

NeMo は `.venvs/lab`（Python 3.13）と依存関係が合わないため、ASR は別の仮想環境に入れます。lab のパッケージは変更しません。

| | |
|---|---|
| 場所 | `.venvs/asr` |
| Python | 3.12.13（uv の CPython 3.12） |
| 用途 | Parakeet（NeMo）、Whisper、Transformers CTC |

Whisper と Transformers もこの環境に入っています。評価の DuckDB と PESQ / STOI も、SE と ASR を同じプロセスでつなぐために入れます。

### 作成

```bash
cd /net/spring/work/seki/exp/audio-db
export UV_CACHE_DIR="$PWD/.venvs/.uv"
uv venv .venvs/asr --python 3.12
uv pip install --python .venvs/asr/bin/python --only-binary :all: \
  torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venvs/asr/bin/python -r requirements/asr.txt
uv pip install --python .venvs/asr/bin/python --no-build-isolation openai-whisper==20240930
CUDA_HOME="$PWD/.venvs/asr/lib/python3.12/site-packages/nvidia/cuda_nvcc"
CUDA_RUNTIME="$PWD/.venvs/asr/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$CUDA_RUNTIME/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"
```

Numba を使う NeMo のデコードは、lab と同じ理由で ptxas 12.2 を見ます。実行前に次を渡します。

```bash
export CUDA_HOME="$PWD/.venvs/asr/lib/python3.12/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=<空きGPU>
```

`nvidia-smi` で使用メモリが 1024 MiB 未満の GPU だけを使います。評価設定の `asr.device` は、その GPU 番号です。`CUDA_VISIBLE_DEVICES` で隠した場合、見える番号は 0 から振り直されます。

固定バージョンは `requirements/asr.txt` です。torch は CUDA 12.4 の wheel を先に入れるため、このファイルには入れていません。`openai-whisper` は wheel が無く、setuptools を入れたあとに `--no-build-isolation` でビルドします。この環境の NumPy は 2.1.3 です。lab の 2.5.4 とは別です。
