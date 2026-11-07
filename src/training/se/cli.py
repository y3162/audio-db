"""Train MP-SENet.

GPU selection is the process environment (``CUDA_VISIBLE_DEVICES`` and
``torchrun``), not a physical index in this file. CPU runs do not start a
process group.

``CUBLAS_WORKSPACE_CONFIG`` is set before PyTorch is imported so deterministic
CUDA matmul can be enabled later in the same process.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train MP-SENet from a named recipe set")
    parser.add_argument("--config", required=True, help="YAML training config")
    parser.add_argument("--resume", default=None, help="audio-db training checkpoint to continue")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="check the config and write preflight.json and preflight.md without training",
    )
    parser.add_argument("--world-size", type=int, default=None, help="process world size to check in preflight")
    parser.add_argument("--expect-dataset", default=None, help="dataset name@version that preflight must see")
    parser.add_argument("--expect-snapshot", default=None, help="snapshot hash that preflight must see")
    parser.add_argument("--preflight-dir", default=None, help="directory for preflight.json and preflight.md")
    args = parser.parse_args(argv)
    if args.preflight:
        from src.training.se.preflight import run_preflight

        code = run_preflight(
            args.config,
            world_size=args.world_size,
            expect_dataset=args.expect_dataset,
            expect_snapshot=args.expect_snapshot,
            resume=args.resume,
            output_dir=args.preflight_dir,
        )
        raise SystemExit(code)
    from src.training.se.run import execute

    execute(args.config, resume=args.resume, strict_resume=bool(args.resume))


if __name__ == "__main__":
    main()
