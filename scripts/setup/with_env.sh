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
exec "$@"
