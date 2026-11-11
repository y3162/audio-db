# ASR 用の仮想環境

NeMo は `.venvs/se` と依存関係が合わないため、ASR は `.venvs/asr` に入れる。構築は `make setup` が `requirements/asr.lock.txt` から行う。

| | |
|---|---|
| 場所 | `.venvs/asr` |
| Python | 3.12.13（uv の CPython 3.12） |
| 用途 | Parakeet（NeMo）、Whisper、Transformers CTC |

Whisper と Transformers もこの環境に入っています。評価の DuckDB と PESQ / STOI も、SE と ASR を同じプロセスでつなぐために入れます。

### 作成

```bash
cd /path/to/audio-db
make setup-env PLATFORM=<tsubame|shinoda_lab> ROLES=asr
```

`make setup` は同じ処理を全役割に行います。`requirements/asr.lock.txt` を `uv pip sync` で入れ、wheel の無い `openai-whisper==20240930` を `--no-build-isolation` で足し、`nvidia/cuda_nvcc/lib64` に `libcudart.so.12` へのリンクを置きます。

Numba を使う NeMo のデコードは、lab と同じ理由で ptxas 12.2 を見ます。実行前に次を渡します。

```bash
export CUDA_HOME="$PWD/.venvs/asr/lib/python3.12/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=<空きGPU>
```

`nvidia-smi` で使用メモリが 1024 MiB 未満の GPU だけを使います。評価設定の `asr.device` は、その GPU 番号です。`CUDA_VISIBLE_DEVICES` で隠した場合、見える番号は 0 から振り直されます。

直接依存は `requirements/asr.in`、固定バージョンは `requirements/asr.lock.txt` です。torch は CUDA 12.4 の wheel（`2.6.0+cu124`）です。`openai-whisper` は wheel が無いので lock には入れず、同期のあとに `--no-build-isolation` でビルドします。この環境の NumPy は 2.1.3 で、`.venvs/se` とは別です。
