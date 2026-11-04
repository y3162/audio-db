#!/usr/bin/env bash
# Build the pinned extensions without changing the shared virtualenv or sources.
set -euo pipefail
if [[ $# != 1 || -z ${CUDA_HOME:-} || ! -x ${CUDA_HOME}/bin/nvcc ]]; then
  echo "Usage: CUDA_HOME=<toolkit> $0 <new-output-directory>" >&2
  exit 2
fi
cd "$(dirname "$0")/../.."
root=$PWD
python=${AUDIO_DB_MAMBA_PYTHON:-$root/.venvs/semambapp/bin/python}
mkdir "$1"
out=$(cd "$1" && pwd)
for pkg in causal-conv1d mamba; do
  mkdir "$out/$pkg"
  tar -C "$root/build/native-src/$pkg" --exclude='./build' --exclude='./.git' --exclude='*.so' -cf - . | tar -C "$out/$pkg" -xf -
done
export PATH="$CUDA_HOME/bin:$PATH" MAX_JOBS="${MAX_JOBS:-2}"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE
for pkg in causal-conv1d mamba; do
  (cd "$out/$pkg" && "$python" setup.py build_ext --inplace)
done
mkdir "$out/lib"
find "$out/causal-conv1d" "$out/mamba" -maxdepth 1 -name '*cuda*.so' -exec cp -t "$out/lib" {} +
