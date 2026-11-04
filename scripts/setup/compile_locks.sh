#!/usr/bin/env bash
# Rebuild requirements/*.lock.txt from the .in files. Setup does not call this.
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$root"
args=(
  --python-version 3.12
  --index-url https://download.pytorch.org/whl/cu124
  --extra-index-url https://pypi.org/simple
  --index-strategy unsafe-best-match
  --emit-index-url
)
compile() {
  uv pip compile "${args[@]}" "$1" -o "$2"
}
compile requirements/se.in requirements/se.lock.txt
compile requirements/semambapp.in requirements/semambapp.lock.txt
compile requirements/asr.in requirements/asr.lock.txt
compile requirements/official.in requirements/official.lock.txt
compile requirements/deepfilternet.in requirements/deepfilternet.lock.txt
