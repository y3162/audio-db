#!/bin/bash
#$ -cwd
#$ -j y
# Run one experiment under the audio-db launcher.
#
#   scripts/experiments/job.sh CONFIG [NPROC]
#
# TSUBAME 4.0, from the repository root (the scheduler runs a copy of this file,
# so the working directory is how it finds the repository):
#
#   qsub -g GROUP -p -5 -l node_h=1 -l h_rt=0:20:00 scripts/experiments/job.sh CONFIG 2
#
# The launcher runs on one node. Ask for one unit of a resource type that holds
# the GPUs (gpu_1: 1 GPU, node_h: 2, node_f: 4); "gpu_1=2" would be two nodes.
#
# The same line without qsub works inside qrsh, on a lab GPU server, and on a
# workstation. The scheduler or the caller chooses the GPUs: NPROC must not
# exceed the devices in CUDA_VISIBLE_DEVICES. This script does not submit,
# retry, or resubmit.
#
#   AUDIO_DB_RESUME      checkpoint, or experiment directory, to resume
#   AUDIO_DB_STOP_AFTER  seconds after which the trainer stops at a safe point.
#                        Keep AUDIO_DB_STOP_AFTER + AUDIO_DB_GRACE below h_rt.
#   AUDIO_DB_GRACE       seconds that stop may take (launcher default: 300)
#   AUDIO_DB_GPU_MEMORY_INTERVAL
#                        seconds between nvidia-smi samples (default 10; 0 disables)
#   AUDIO_DB_PYTHON      interpreter (default: .venvs/se/bin/python)
#
# Exit code: 0 completed, 3 stopped at a safe point (resume it), 1 failed, 2 usage.
set -u
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/experiments/job.sh CONFIG [NPROC]" >&2
  exit 2
fi
if [[ -f "$PWD/src/experiments/launcher.py" ]]; then
  root="$PWD"
else
  root="$(cd "$(dirname "$0")/../.." && pwd)"
fi
if [[ ! -f "$root/src/experiments/launcher.py" ]]; then
  echo "job.sh: run it from the audio-db repository root" >&2
  exit 2
fi
config="$1"
[[ "$config" = /* ]] || config="$PWD/$config"
cd "$root" || exit 2
python="${AUDIO_DB_PYTHON:-$root/.venvs/se/bin/python}"
arguments=(--config "$config" --nproc "${2:-1}")
[[ -n "${AUDIO_DB_RESUME:-}" ]] && arguments+=(--resume "$AUDIO_DB_RESUME")
[[ -n "${AUDIO_DB_STOP_AFTER:-}" ]] && arguments+=(--stop-after "$AUDIO_DB_STOP_AFTER")
[[ -n "${AUDIO_DB_GRACE:-}" ]] && arguments+=(--grace "$AUDIO_DB_GRACE")
echo "job.sh: host=$(hostname) job=${JOB_ID:-none} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"
# exec: the scheduler's SIGTERM reaches the launcher, which asks the trainer for a safe stop.
exec "$root/scripts/setup/with_env.sh" "$python" -m src.experiments.launcher "${arguments[@]}"
