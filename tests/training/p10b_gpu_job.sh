#!/bin/bash
# P10-B official-width GPU smoke. No interactive input.
# Usage: p10b_gpu_job.sh MODEL GPU_COUNT RUN_ID
set -u
cd /gs/bs/tga-shinoda/seki/exp/audio-db
MODEL="${1:?model}"
GPUS="${2:?gpu count}"
RUN_ID="${3:?run id}"
ROOT="data/results/training/p10b/${RUN_ID}/${MODEL}"
mkdir -p "$ROOT"
LOG="$ROOT/job.log"
exec > >(tee -a "$LOG") 2>&1
START_EPOCH=$(date +%s)
stop() {
  local code="$1"
  local end
  end=$(date +%s)
  echo "END $(date -Is)"
  echo "END_EPOCH ${end}"
  echo "ELAPSED_SEC $((end - START_EPOCH))"
  exit "$code"
}
echo "START $(date -Is)"
echo "START_EPOCH ${START_EPOCH}"
echo "HOST $(hostname)"
echo "JOB_ID ${JOB_ID:-unset}"
echo "MODEL ${MODEL} GPUS ${GPUS} RUN_ID ${RUN_ID}"
echo "HEAD $(git rev-parse HEAD)"
echo "DIRTY $(git status --porcelain | wc -l)"
echo "IMPL_SHA256 $(
  {
    git diff HEAD
    git ls-files --others --exclude-standard | sort | while IFS= read -r file; do
      printf 'UNTRACKED %s\n' "$file"
      cat "$file"
    done
  } | sha256sum | awk '{print $1}'
)"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,memory.total,mig.mode.current --format=csv || true
nvidia-smi || true
COUNT=$(nvidia-smi -L | grep -c '^GPU ' || true)
echo "GPU_COUNT ${COUNT}"
if [ "$COUNT" -lt "$GPUS" ]; then
  echo "NOT_ENOUGH_GPUS ${COUNT} < ${GPUS}"
  stop 3
fi
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
TEMPLATE="configs/training/reproduction/p10b/${MODEL}.yaml"
if [ ! -f "$TEMPLATE" ]; then
  echo "MISSING_TEMPLATE ${TEMPLATE}"
  stop 2
fi

render() {
  local steps="$1"
  local dest="$2"
  local out="$3"
  .venvs/se/bin/python - "$TEMPLATE" "$steps" "$dest" "$out" <<'PY'
import sys
from pathlib import Path
import yaml
template, steps, dest, out = sys.argv[1:]
raw = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
raw["max_steps"] = int(steps)
raw["checkpoint_dir"] = dest
Path(out).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
PY
}

run_train() {
  local config="$1"
  local resume="${2:-}"
  if [ -n "$resume" ]; then
    .venvs/se/bin/torchrun --standalone --nproc_per_node="$GPUS" -m src.training.se.cli --config "$config" --resume "$resume"
  else
    .venvs/se/bin/torchrun --standalone --nproc_per_node="$GPUS" -m src.training.se.cli --config "$config"
  fi
}

WORK="$ROOT/work"
CONT="$ROOT/continuous"
mkdir -p "$WORK" "$CONT"
render 2 "$WORK" "$ROOT/train.yaml"
echo "PHASE train $(date -Is)"
run_train "$ROOT/train.yaml"
STATUS=$?
echo "TRAIN_EXIT ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
  stop "$STATUS"
fi
CKPT=$(ls "$WORK"/step_*.pt 2>/dev/null | sort | tail -n 1 || true)
echo "CHECKPOINT ${CKPT:-missing}"
if [ -z "$CKPT" ]; then
  stop 4
fi
if [ -f "$WORK/complete.json" ]; then
  # max_steps writes complete.json. Resume is a new process with a higher
  # max_steps, so the finished-run marker has to move first.
  mv "$WORK/complete.json" "$WORK/complete.phase-train.json"
  echo "MOVED_COMPLETE complete.phase-train.json"
fi
render 3 "$WORK" "$ROOT/resume.yaml"
echo "PHASE resume $(date -Is)"
run_train "$ROOT/resume.yaml" "$CKPT"
STATUS=$?
echo "RESUME_EXIT ${STATUS}"
if [ "$STATUS" -ne 0 ]; then
  stop "$STATUS"
fi
render 3 "$CONT" "$ROOT/continuous.yaml"
echo "PHASE continuous $(date -Is)"
run_train "$ROOT/continuous.yaml"
STATUS=$?
echo "CONTINUOUS_EXIT ${STATUS}"
.venvs/se/bin/python - "$WORK" "$CONT" "$ROOT/compare.json" <<'PY'
import json, sys
from pathlib import Path
import torch
work, cont, out = map(Path, sys.argv[1:])

def rows(directory):
    path = directory / "metrics.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def by_step(records, kind):
    return {int(row["global_step"]): row for row in records if row.get("type") == kind}

def state_of(directory):
    found = {}
    for path in sorted(directory.glob("step_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        saved = payload.get("data_state") or {}
        found[path.name] = {
            "global_step": payload.get("global_step"),
            "epoch": payload.get("epoch"),
            "micro_batches_consumed": payload.get("micro_batches_consumed"),
            "data_epoch": saved.get("epoch"),
            "update_in_epoch": saved.get("update_in_epoch"),
            "plan_hash": saved.get("plan_hash"),
            "dataset_id": saved.get("dataset_id"),
            "snapshot_hash": saved.get("snapshot_hash"),
            "world_size": saved.get("world_size"),
        }
    return found

left, right = rows(work), rows(cont)
train_l, train_r = by_step(left, "train"), by_step(right, "train")
report = {"work_steps": sorted(train_l), "continuous_steps": sorted(train_r)}
for step in (1, 2, 3):
    if step in train_l and step in train_r:
        report[f"step{step}_case_ids_equal"] = train_l[step].get("case_ids") == train_r[step].get("case_ids")
        report[f"step{step}_work_loss"] = train_l[step].get("losses")
        report[f"step{step}_continuous_loss"] = train_r[step].get("losses")
        report[f"step{step}_work_lr"] = train_l[step].get("learning_rate")
        report[f"step{step}_continuous_lr"] = train_r[step].get("learning_rate")
        report[f"step{step}_work_epoch"] = train_l[step].get("epoch")
        report[f"step{step}_continuous_epoch"] = train_r[step].get("epoch")
report["work_types"] = sorted({row.get("type") for row in left})
report["continuous_types"] = sorted({row.get("type") for row in right})
report["work_validation"] = [row for row in left if row.get("type") == "validation"]
report["continuous_validation"] = [row for row in right if row.get("type") == "validation"]
report["provenance_work"] = (work / "data_provenance.json").is_file()
report["provenance_continuous"] = (cont / "data_provenance.json").is_file()
report["tensorboard_work"] = (work / "tensorboard").is_dir()
report["tensorboard_continuous"] = (cont / "tensorboard").is_dir()
report["work_checkpoints"] = state_of(work)
report["continuous_checkpoints"] = state_of(cont)
out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("COMPARE_WROTE", out)
PY
stop "$STATUS"
