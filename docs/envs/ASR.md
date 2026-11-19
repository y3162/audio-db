# ASR

ASR は通常環境 `.venvs/se` で動く。役割の一覧と lock の扱いの本体は [セットアップ](SETUP.md)。モデルの入口は [ASR](../asr/MODELS.md)。

| | |
|---|---|
| 場所 | `.venvs/se` |
| Python | 3.12（`uv venv --python 3.12`。パッチ版は lock に書いていない） |
| 用途 | Parakeet（NeMo）、Whisper、Transformers CTC。SE と同じプロセスで評価と学習に使う |

```bash
make setup-env ROLES=se
```

`make setup` は `requirements/se.lock.txt` を `uv pip sync` で入れ、wheel の無い `openai-whisper==20240930` を `--no-build-isolation` で足す。`openai-whisper` は lock に入っていない。torch は `2.6.0+cu124`。NumPy は 2.1.3、Numba は 0.61.0 である。NeMo 2.2.1 はこの Numba を要求し、NumPy 2.2 以降を拒む。直接依存は `requirements/se.in`。

`scripts/setup/env_roles.sh` は、wheel の `nvidia/cuda_nvcc/lib64/libcudart.so.12` を `nvidia/cuda_runtime` の実体へリンクする。`uv pip sync` はこのディレクトリを消しても lock の stamp は変わらない。`scripts/setup/with_env.sh` は未設定の `CUDA_HOME` をその wheel に向ける前に、同じリンクを張り直す。Numba を使う NeMo のデコードが ptxas を見る先は、`requirements/se.in` の `nvidia-cuda-nvcc-cu12==12.2.140` である。手で渡すときは、空き GPU の番号を `GPU_ID` に設定してから次を実行する。

```bash
export CUDA_HOME="$PWD/.venvs/se/lib/python3.12/site-packages/nvidia/cuda_nvcc"
mkdir -p "$CUDA_HOME/lib64"
ln -sfn "$PWD/.venvs/se/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/nvvm/lib64:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
```

使う前に `nvidia-smi` を見る。評価が使う GPU は、使用メモリが 1024 MiB 未満のものに限る。評価設定の `asr.device` はその番号である。`CUDA_VISIBLE_DEVICES` で隠した場合、見える番号は 0 から振り直される。

決定的アルゴリズムが有効な CUDA では、featurizer の `center=True` STFT を index reflection と `center=False` に替える。スペクトルは centered STFT と一致し、`reflection_pad1d` の CUDA backward は使わない。同じプロセスで続けて計算した TDT 損失は omega 0 でも一致しない。別プロセスで同じ初期状態から始めた step 1 は、このホストでは preflight の有無で一致した。

以前の独立した役割 `.venvs/asr`、`.venvs/semambapp`、`.venvs/asr_guided_se` は使わない。ディレクトリがディスクに残っていても、現行の手順ではない。
