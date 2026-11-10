#!/bin/bash
# One P5 DDP smoke. Arguments: profile (gridnet|locoformer) and GPU count.
set -u
cd /gs/bs/tga-shinoda/seki/exp/audio-db
PROFILE="${1:?profile}"
NEED="${2:?gpu count}"
LOG="data/results/training/p5-smoke/${PROFILE}-job.log"
mkdir -p data/results/training/p5-smoke
{
  echo "START $(date -Is)"
  echo "HOST $(hostname)"
  echo "JOB_ID ${JOB_ID:-unset}"
  echo "PROFILE ${PROFILE} NEED ${NEED}"
  nvidia-smi -L
  nvidia-smi
} >> "$LOG" 2>&1
COUNT=$(nvidia-smi -L | grep -c '^GPU ' || true)
echo "GPU_COUNT ${COUNT}" >> "$LOG"
if [ "$COUNT" -lt "$NEED" ]; then
  echo "NOT_ENOUGH_GPUS ${COUNT} < ${NEED}" >> "$LOG"
  echo "END $(date -Is)" >> "$LOG"
  exit 3
fi
export PYTHONPATH="$PWD"
.venvs/se/bin/torchrun --standalone --nproc_per_node="$NEED" tests/numerics/p5_gpu_smoke.py "$PROFILE" >> "$LOG" 2>&1
STATUS=$?
echo "EXIT ${STATUS}" >> "$LOG"
echo "END $(date -Is)" >> "$LOG"
exit "$STATUS"
