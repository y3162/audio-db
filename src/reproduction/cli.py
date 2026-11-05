"""Command line for the reproduction database.

``python -m src.reproduction.cli doctor|init-db|snapshot|rebuild-db|verify|prepare``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.reproduction.doctor import run_doctor
from src.reproduction.errors import PreparationBlocked, ReproductionError
from src.reproduction.paths import (
    DEFAULT_CORPORA,
    DEFAULT_DB,
    DEFAULT_DERIVED,
    DEFAULT_GENERATED,
    DEFAULT_RESOURCE_DB,
    DEFAULT_SNAPSHOTS,
    DEFAULT_STAGING,
    ROOT,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.reproduction.cli")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--database", default=str(DEFAULT_DB))
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", parents=[shared])
    doctor.add_argument("--resource-db", default=str(DEFAULT_RESOURCE_DB))
    doctor.add_argument("--models", default="")
    doctor.add_argument("--corpora", default=str(DEFAULT_CORPORA))
    doctor.add_argument("--generated", default=str(DEFAULT_GENERATED))

    commands.add_parser("init-db", parents=[shared])

    snapshot = commands.add_parser("snapshot", parents=[shared])
    snapshot.add_argument("--dataset", default="")
    snapshot.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    snapshot.add_argument("--allow-unknown-leak-check", action="store_true")
    snapshot.add_argument("--corpora", default=str(DEFAULT_CORPORA))
    snapshot.add_argument("--generated", default=str(DEFAULT_GENERATED))
    snapshot.add_argument("--derived", default=str(DEFAULT_DERIVED))

    rebuild = commands.add_parser("rebuild-db", parents=[shared])
    rebuild.add_argument("--out", default="")
    rebuild.add_argument("--staging", default=str(DEFAULT_STAGING))
    rebuild.add_argument("--reference", default="")
    rebuild.add_argument("--resource-db", default="")

    verify = commands.add_parser("verify", parents=[shared])
    verify.add_argument("--dataset", default="")
    verify.add_argument("--build", default="")
    verify.add_argument("--hash", action="store_true")
    verify.add_argument("--staging", default=str(DEFAULT_STAGING))
    verify.add_argument("--corpora", default=str(DEFAULT_CORPORA))
    verify.add_argument("--generated", default=str(DEFAULT_GENERATED))
    verify.add_argument("--derived", default=str(DEFAULT_DERIVED))

    prepare = commands.add_parser("prepare", parents=[shared])
    prepare.add_argument("--step", default="all")
    prepare.add_argument("--staging", default=str(DEFAULT_STAGING))
    prepare.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    prepare.add_argument("--resource-db", default=str(DEFAULT_RESOURCE_DB))
    prepare.add_argument("--corpora", default=str(DEFAULT_CORPORA))
    prepare.add_argument("--derived", default=str(DEFAULT_DERIVED))
    prepare.add_argument("--repo-root", default=str(ROOT))
    prepare.add_argument("--config-root", default=str(ROOT / "configs" / "data"))
    prepare.add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)
    try:
        return _run(args)
    except PreparationBlocked as exc:
        print(f"BLOCKED\t{exc}", file=sys.stderr)
        return 3
    except ReproductionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    if args.command == "doctor":
        return run_doctor(
            database=args.database,
            resource_db=args.resource_db,
            models=args.models,
            corpora=args.corpora,
            generated=args.generated,
        )
    if args.command == "init-db":
        from src.reproduction.db.registry import initialize_database

        initialize_database(args.database)
        print(f"ready\t{args.database}")
        return 0
    if args.command == "snapshot":
        if not args.dataset.strip():
            raise ReproductionError("DATASET is required")
        from src.reproduction.db.snapshot import export_snapshot

        digest = export_snapshot(
            args.database,
            args.dataset,
            args.snapshots,
            allow_unknown_leak_check=args.allow_unknown_leak_check,
            corpora=args.corpora,
            generated=args.generated,
            derived=args.derived,
        )
        print(f"snapshot_hash\t{digest}")
        return 0
    if args.command == "rebuild-db":
        if not args.out.strip():
            raise ReproductionError("OUT is required")
        from src.reproduction.db.registry import rebuild_database

        reference = args.reference.strip() or args.database
        resource = args.resource_db.strip() or None
        rebuild_database(args.staging, args.out, reference=reference, resource_db=resource)
        print(f"rebuilt\t{args.out}")
        return 0
    if args.command == "verify":
        from src.reproduction.db.verify import verify_files

        ok = verify_files(
            args.database,
            staging=args.staging,
            dataset=args.dataset,
            build=args.build,
            check_hash=args.hash,
            corpora=args.corpora,
            generated=args.generated,
            derived=args.derived,
        )
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    if args.command == "prepare":
        from src.reproduction.prepare.run import PreparePaths, execute

        resource = args.resource_db.strip()
        execute(
            args.step,
            PreparePaths(
                database=Path(args.database),
                staging=Path(args.staging),
                snapshots=Path(args.snapshots),
                resource_db=Path(resource) if resource else None,
                corpora=Path(args.corpora),
                derived=Path(args.derived),
                repo_root=Path(args.repo_root),
                config_root=Path(args.config_root),
                workers=args.workers,
            ),
        )
        return 0
    raise ReproductionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
