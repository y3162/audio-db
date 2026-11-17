#!/usr/bin/env bash
# Run a command with .env applied only for variables that are not already set.
# Explicit arguments and the caller's environment win over .env.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
if [[ -f .env ]]; then
  eval "$(python3 "$root/scripts/setup/dotenv.py" --emit-unset "$root/.env")"
fi
if [[ $# -eq 0 ]]; then
  echo "usage: scripts/setup/with_env.sh command..." >&2
  exit 2
fi
# Numba 0.61 in .venvs/se must see that venv's CUDA 12 NVVM, not /usr/local/cuda.
# Other interpreters, including the official roles, are not modified.
for _audio_db_arg in "$@"; do
  case "$_audio_db_arg" in
    */.venvs/se/bin/python|.venvs/se/bin/python)
      # shellcheck disable=SC1091
      source "$root/scripts/setup/cuda_wheel.sh"
      apply_cuda_wheel "$_audio_db_arg" || exit 1
      break
      ;;
  esac
done
unset _audio_db_arg
exec "$@"
