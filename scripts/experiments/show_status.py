"""Print the status of one experiment directory.

This does not start training and does not rewrite the directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: .venvs/se/bin/python scripts/experiments/show_status.py DIRECTORY", file=sys.stderr)
        return 2
    directory = Path(args[0])
    if not directory.is_dir():
        print(f"missing directory: {directory}", file=sys.stderr)
        return 1
    print(f"directory\t{directory.resolve()}")
    for name in ("status.json", "complete.json", "data_provenance.json", "rfft.json"):
        path = directory / name
        if not path.is_file():
            print(f"{name}\tabsent")
            continue
        text = path.read_text(encoding="utf-8").strip()
        print(f"{name}\t{text.replace(chr(10), ' ')}")
    steps = sorted(path for path in directory.glob("step_*.pt") if path.is_file() and not path.name.startswith("."))
    print(f"checkpoints\t{len(steps)}")
    if steps:
        latest = steps[-1]
        print(f"latest_checkpoint\t{latest.name}")
        sidecar = latest.with_suffix(".data_provenance.json")
        print(f"latest_sidecar\t{'present' if sidecar.is_file() else 'absent'}")
    metrics = directory / "metrics.jsonl"
    if metrics.is_file():
        lines = [line for line in metrics.read_text(encoding="utf-8").splitlines() if line]
        print(f"metrics\t{len(lines)}")
        if lines:
            try:
                last = json.loads(lines[-1])
            except json.JSONDecodeError:
                last = {"unreadable": True}
            print(f"metrics_last\t{json.dumps(last, sort_keys=True)}")
    else:
        print("metrics\tabsent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
