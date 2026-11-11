"""Print the status of one experiment directory.

This does not start training and does not rewrite the directory.
``liveness`` tells a run that is still going from a ``RUNNING`` record left
by a process that died: ``alive``, ``stale``, ``finished``, ``absent``, or
``unknown`` (started on another host without the launcher).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.launcher import liveness  # noqa: E402


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
    alive = liveness(directory)
    print(f"liveness\t{alive['state']}\t{alive['reason']}")
    for name in (
        "status.json",
        "launch.json",
        "complete.json",
        "initialization.json",
        "data_provenance.json",
        "rfft.json",
    ):
        path = directory / name
        if not path.is_file():
            print(f"{name}\tabsent")
            continue
        text = path.read_text(encoding="utf-8").strip()
        print(f"{name}\t{' '.join(text.split())}")
    for path in sorted(directory.glob("failure.rank*.json")):
        print(f"{path.name}\t{' '.join(path.read_text(encoding='utf-8').split())}")
    latest = directory / "latest.pt"
    print(f"latest\t{'present' if latest.is_file() else 'absent'}")
    if latest.is_file():
        sidecar = latest.with_suffix(".data_provenance.json")
        print(f"latest_sidecar\t{'present' if sidecar.is_file() else 'absent'}")
    archives = sorted(path.name for path in (directory / "archive").glob("*.pt") if path.is_file())
    print(f"archives\t{len(archives)}")
    if archives:
        print(f"archive_last\t{archives[-1]}")
    # Written before Latest and Archive existed. Each one can still be resumed.
    legacy = sorted(path.name for path in directory.glob("step_*.pt") if path.is_file())
    if legacy:
        print(f"legacy_checkpoints\t{len(legacy)}\t{legacy[-1]}")
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
    _gpu_memory(directory)
    return 0


def _gpu_memory(directory: Path) -> None:
    """Print the launcher's whole-device series, when this run recorded one."""

    summary_path = directory / "gpu_memory.json"
    series = directory / "gpu_memory.jsonl"
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("gpu_memory\tunreadable")
            return
        peaks = payload.get("peaks") if isinstance(payload, dict) else None
        shown = ",".join(
            f"{item.get('index')}:{item.get('memory_used_mib')}MiB"
            for item in peaks or []
            if isinstance(item, dict)
        )
        count = payload.get("sample_count") if isinstance(payload, dict) else None
        print(f"gpu_memory\tsamples={count}\tpeak={shown or 'none'}")
        if isinstance(payload, dict) and payload.get("last_error"):
            print(f"gpu_memory_error\t{payload['last_error']}")
        return
    if series.is_file():
        lines = [line for line in series.read_text(encoding="utf-8").splitlines() if line]
        print(f"gpu_memory\tlines={len(lines)}\tsummary=absent")
        return
    print("gpu_memory\tabsent")


if __name__ == "__main__":
    raise SystemExit(main())
