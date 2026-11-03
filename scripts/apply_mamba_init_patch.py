"""Apply the Mamba 2.3.2.post1 import shim and check the scan code is untouched.

The patch comments three unused imports in ``mamba_ssm/__init__.py``. It does
not edit selective scan, causal conv, or any CUDA source. Re-running on an
already patched file is a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "mamba_ssm-2.3.2.post1-init.patch"
VERSION = "2.3.2.post1"
REMOVED = (
    "from mamba_ssm.modules.mamba2 import Mamba2\n",
    "from mamba_ssm.modules.mamba3 import Mamba3\n",
    "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel\n",
)
KEPT = (
    "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn\n",
    "from mamba_ssm.modules.mamba_simple import Mamba\n",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _active(text: str, line: str) -> bool:
    target = line.rstrip("\n")
    return any(row == target for row in text.splitlines())


def apply_init(path: Path) -> str:
    text = path.read_text()
    if f'__version__ = "{VERSION}"' not in text:
        raise SystemExit(f"{path} is not mamba_ssm {VERSION}")
    if not any(_active(text, line) for line in REMOVED):
        if "clean-rebuild:" not in text or any(not _active(text, line) for line in KEPT):
            raise SystemExit(f"{path} is not the recorded Mamba init patch")
        return "already-applied"
    missing = [line.strip() for line in REMOVED if not _active(text, line)]
    if missing:
        raise SystemExit(f"{path} does not contain the expected imports: {missing}")
    lines = text.splitlines(keepends=True)
    rewritten = []
    inserted = False
    for row in lines:
        if any(row == line for line in REMOVED):
            if not inserted:
                rewritten.append("# clean-rebuild: Mamba2, Mamba3, and the language-model head are unused by\n")
                rewritten.append("# SEMamba++ and import transformers. The Mamba1 CUDA kernels are unchanged.\n")
                inserted = True
            rewritten.append("# " + row)
            continue
        rewritten.append(row)
    path.write_text("".join(rewritten))
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=Path, required=True, help="path to mamba_ssm/__init__.py")
    parser.add_argument("--scan", type=Path, action="append", default=[], help="selective-scan file that must stay unchanged")
    parser.add_argument("--scan-digest", action="append", default=[], help="expected sha256 for each --scan, same order")
    args = parser.parse_args()
    if len(args.scan) != len(args.scan_digest):
        raise SystemExit("--scan and --scan-digest must be paired")
    before = {path: _sha256(path) for path in args.scan}
    status = apply_init(args.init)
    after = {path: _sha256(path) for path in args.scan}
    changed = [str(path) for path in args.scan if before[path] != after[path]]
    if changed:
        raise SystemExit(f"selective scan files changed: {changed}")
    mismatches = [f"{path} {after[path]}" for path, digest in zip(args.scan, args.scan_digest) if after[path] != digest]
    if mismatches:
        raise SystemExit("selective scan digest mismatch:\n" + "\n".join(mismatches))
    patched = args.init.read_text()
    for line in KEPT:
        if line not in patched:
            raise SystemExit(f"patched init dropped {line.strip()}")
    for line in REMOVED:
        if f"# {line}" not in patched:
            raise SystemExit(f"patched init still imports {line.strip()}")
    print(status)
    print(PATCH)


if __name__ == "__main__":
    main()
