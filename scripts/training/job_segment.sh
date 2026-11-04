#!/bin/bash
set -eo pipefail
source /etc/profile.d/modules.sh
set -u
cd "${SGE_O_WORKDIR:?submit from the repository root}"
module load cuda/12.3.2
export PYTHONPATH="$PWD/.venvs/tsubame-h100-causal:$PWD/.venvs/tsubame-h100-build:$PWD/.venvs/tsubame-ref-extra:$PWD"
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'host=%s job=%s CUDA_VISIBLE_DEVICES=%s\n' "$(hostname)" "${JOB_ID:-unknown}" "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi
nvidia-smi -L
exec .venvs/tsubame/bin/python -m src.training.se.job_chain run "$@"
