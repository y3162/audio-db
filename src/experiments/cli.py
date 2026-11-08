"""Command line entry for one experiment YAML.

``--preflight`` builds and validates the graph, then stops before any
optimizer update.
"""

from __future__ import annotations

import argparse
import sys

from src.experiments.api import Experiment, build_experiment, preflight_experiment
from src.experiments.config import load_experiment
from src.experiments.errors import CheckpointError, ConfigError, ExperimentError


def main(argv: list[str] | None = None) -> int:
    """Return 0 on success and 1 on a known experiment error."""

    parser = argparse.ArgumentParser(description="Run an audio-db experiment")
    parser.add_argument("--config", required=True, help="audio-db-experiment/v1 YAML")
    parser.add_argument("--resume", default=None, help="audio-db-experiment checkpoint to continue")
    parser.add_argument("--initialize-from", default=None, help="checkpoint whose selected weights initialize a new run")
    parser.add_argument("--preflight", action="store_true", help="build and validate without training")
    parser.add_argument("--preflight-dir", default=None, help="directory for preflight.json")
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except ExperimentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _execute(args: argparse.Namespace) -> None:
    if args.resume is not None and args.initialize_from is not None:
        raise ConfigError("resume and initialize_from cannot both be set")
    config = load_experiment(args.config)
    if args.preflight:
        experiment = build_experiment(config)
        try:
            if args.resume is not None or args.initialize_from is not None:
                raise CheckpointError("preflight does not restore a checkpoint")
            preflight_experiment(experiment, args.preflight_dir)
        finally:
            experiment.close()
        return
    Experiment(config).run(resume=args.resume, initialize_from=args.initialize_from)


if __name__ == "__main__":
    raise SystemExit(main())
