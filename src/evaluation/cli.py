"""Command line for the evaluation pipeline.

    python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --dry-run
    python -m src.evaluation.cli --config configs/evaluation/smoke.yaml --limit 2
    python -m src.evaluation.cli --summarize data/results/se-evaluation/<run_id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.evaluation.config import apply_overrides, load_config
from src.evaluation.errors import ConfigError, DatasetError
from src.evaluation.pipeline import execute, format_plan
from src.evaluation.results.summary import format_summary, summarize_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Speech enhancement evaluation pipeline")
    parser.add_argument("--config", help="YAML or JSON experiment file")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without loading models")
    parser.add_argument("--limit", type=int, help="Process at most this many utterances")
    parser.add_argument("--device", help="Runtime device, for example cpu or cuda:1")
    parser.add_argument("--seed", type=int, help="Experiment seed")
    parser.add_argument("--output", help="Directory that will contain the run")
    parser.add_argument("--run-id", help="Stable run directory name")
    parser.add_argument("--resume", action="store_true", help="Continue an existing run_id")
    parser.add_argument("--models", help="Comma-separated model names to keep")
    parser.add_argument("--summarize", metavar="RUN_DIR", help="Rebuild summary files for a run")
    args = parser.parse_args(argv)
    try:
        if args.summarize:
            summary = summarize_directory(Path(args.summarize))
            print(format_summary(summary))
            return 0
        if not args.config:
            parser.error("--config is required unless --summarize is set")
        config = load_config(args.config)
        apply_overrides(
            config,
            limit=args.limit,
            device=args.device,
            seed=args.seed,
            output=args.output,
            run_id=args.run_id,
            resume=args.resume,
            models=args.models,
        )
        result = execute(config, dry_run=args.dry_run)
    except (ConfigError, DatasetError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if result.get("dry_run"):
        print(format_plan(result["plan"]))
        return 0 if result["plan"]["ready"] else 2
    print(format_summary(result["summary"]))
    print(f"run_dir: {result['run_dir']}")
    print(f"reused: {result['reused']} written: {result['written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
