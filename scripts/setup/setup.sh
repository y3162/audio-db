#!/usr/bin/env bash
# Common setup entry. Platform scripts only prepare the toolchain.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
platform=${1:-local}
case "$platform" in
  local|tsubame) ;;
  *)
    echo "Unknown setup platform '$platform'. See docs/envs/SETUP.md." >&2
    exit 2
    ;;
esac
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not on PATH. Install it before running setup." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$root/scripts/setup/platforms/${platform}.sh"
platform_prepare
# shellcheck disable=SC1091
source "$root/scripts/setup/env_roles.sh"

if [[ "${2:-}" == --env-only ]]; then
  install_roles "${ROLES:-se,official,deepfilternet}"
  exit
fi

status=0
py=".venvs/se/bin/python"
if [[ ! -x "$py" ]]; then
  py=$(uv python find 3.12)
fi
"$py" "$root/scripts/setup/assets.py" fetch --models "${MODELS:-}" || status=1
install_roles "${ROLES:-se,official,deepfilternet}" || status=1
if [[ -x .venvs/se/bin/python ]]; then
  py=".venvs/se/bin/python"
fi
"$py" -m src.data.link_corpora --allow-partial || status=1
"$py" "$root/scripts/setup/doctor.py" || status=1
exit "$status"
