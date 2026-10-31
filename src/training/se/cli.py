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
    args = parser.parse_args(argv)
    from src.training.se.run import execute

    execute(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
