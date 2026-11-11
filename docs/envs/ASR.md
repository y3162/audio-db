# ASR 用の仮想環境

NeMo は `.venvs/se` と依存が合わないため、ASR は `.venvs/asr` に入れる。役割の一覧と lock の扱いの本体は [セットアップ](SETUP.md)。モデルの入口は [ASR](../asr/MODELS.md)。

| | |
|---|---|
| 場所 | `.venvs/asr` |
| Python | 3.12（`uv venv --python 3.12`。パッチ版は lock に書いていない） |
| 用途 | Parakeet（NeMo）、Whisper、Transformers CTC。評価の DuckDB と PESQ / STOI も、SE と ASR を同じプロセスでつなぐために入る |

```bash
make setup-env PLATFORM=<tsubame|shinoda_lab> ROLES=asr
```

`make setup` は全役割に同じ導入を行う。`requirements/asr.lock.txt` を `uv pip sync` で入れ、wheel の無い `openai-whisper==20240930` を `--no-build-isolation` で足す。`openai-whisper` は lock に入っていない。torch は `2.6.0+cu124`。この環境の NumPy は 2.1.3 で、`.venvs/se` の 2.5.4 とは別である。直接依存は `requirements/asr.in`。

`scripts/setup/env_roles.sh` は、wheel の `nvidia/cuda_nvcc` から `libcudart.so.12` へのリンクを置く。Numba を使う NeMo のデコードが ptxas を見る先は、`requirements/asr.in` の `nvidia-cuda-nvcc-cu12==12.2.140` である。実行前に次を渡す。

```bash
export CUDA_HOME="$PWD/.venvs/asr/lib/python3.12/site-packages/nvidia/cuda_nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=<空きGPU>
```

使う前に `nvidia-smi` を見る。評価が使う GPU は、使用メモリが 1024 MiB 未満のものに限る。評価設定の `asr.device` はその番号である。`CUDA_VISIBLE_DEVICES` で隠した場合、見える番号は 0 から振り直される。
