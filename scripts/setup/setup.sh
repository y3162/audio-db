#!/usr/bin/env bash
# Common setup entry. Platform scripts only prepare the toolchain.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
platform=${1:-}
case "$platform" in
  tsubame|shinoda_lab) ;;
  *)
    echo "PLATFORM must be tsubame or shinoda_lab (got '${platform:-}')." >&2
    exit 2
    ;;
esac
# shellcheck disable=SC1091
source "$root/scripts/setup/platforms/${platform}.sh"
platform_prepare
# shellcheck disable=SC1091
source "$root/scripts/setup/env_roles.sh"

status=0
py=$(uv python find 3.12)
if [[ -x .venvs/se/bin/python ]]; then
  py=".venvs/se/bin/python"
fi
"$py" "$root/scripts/setup/assets.py" fetch --models "${MODELS:-}" || status=1
if [[ -x .venvs/se/bin/python ]]; then
  py=".venvs/se/bin/python"
fi
install_roles "${ROLES:-se,asr,official,semambapp,deepfilternet}" || status=1
if [[ -x .venvs/se/bin/python ]]; then
  py=".venvs/se/bin/python"
fi
"$py" -m src.data.link_corpora --allow-partial || status=1
"$py" "$root/scripts/setup/doctor.py" || status=1
exit "$status"
