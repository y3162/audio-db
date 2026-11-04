"""Create data/corpora symlinks for the standard catalog.

Existing links that already point at the same directory are left in place.
A link that points somewhere else, or a real directory or file, is an error.
Corpus files are not copied or moved.

Path priority for each corpus is an explicit override, then that corpus's
environment variable, then ``AUDIO_DB_DATASET_ROOT / source_name``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data.constants import CORPORA_ROOT
from src.data.corpora_catalog import CATALOG, CorpusSpec, corpus_env_var


def dataset_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="dataset root; overrides AUDIO_DB_DATASET_ROOT")


def resolve_dataset_root(parser: argparse.ArgumentParser, value: Path | None) -> Path:
    value = value or (Path(os.environ["AUDIO_DB_DATASET_ROOT"]) if os.environ.get("AUDIO_DB_DATASET_ROOT") else None)
    if value is None:
        parser.error("specify --dataset-root or AUDIO_DB_DATASET_ROOT")
    return value


def select_corpora(only: list[str] | None) -> tuple[CorpusSpec, ...]:
    if not only:
        return CATALOG
    wanted = {item.strip() for item in only if item.strip()}
    found = tuple(spec for spec in CATALOG if spec.source_name in wanted or spec.link_name in wanted)
    known = {spec.source_name for spec in found} | {spec.link_name for spec in found}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"unknown corpus: {', '.join(unknown)}")
    return found


def corpus_source(spec: CorpusSpec, dataset_root: Path | None, overrides: dict[str, Path] | None) -> Path | None:
    if overrides and overrides.get(spec.source_name):
        return Path(overrides[spec.source_name])
    if dataset_root is None:
        return None
    return dataset_root / spec.source_name


def env_overrides() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for spec in CATALOG:
        value = os.environ.get(corpus_env_var(spec), "").strip()
        if value:
            found[spec.source_name] = Path(value)
    return found


def link_corpora(
    dataset_root: Path | None,
    corpora_root: Path = CORPORA_ROOT,
    *,
    include_non_audio: bool = False,
    dry_run: bool = False,
    only: list[str] | None = None,
    overrides: dict[str, Path] | None = None,
    allow_partial: bool = False,
) -> list[str]:
    """Link the selected catalog entries.

    The default refuses a missing directory before creating any link. ``allow_partial``
    records missing sources and links the rest. Conflicts are never overwritten.
    """
    if dataset_root is not None:
        dataset_root = Path(dataset_root)
        if dataset_root.exists():
            dataset_root = dataset_root.resolve()
    lines: list[str] = []
    planned: list[tuple[CorpusSpec, Path | None, Path]] = []
    missing: list[str] = []
    for spec in select_corpora(only):
        if not spec.index and not include_non_audio:
            lines.append(f"skip {spec.source_name}: {spec.note}")
            continue
        source = corpus_source(spec, dataset_root, overrides)
        dest = corpora_root / spec.link_name
        if source is None or not source.is_dir():
            shown = source if source is not None else spec.source_name
            missing.append(f"missing {spec.link_name}: {shown}")
            continue
        if dest.is_symlink() and dest.resolve() != source.resolve():
            raise FileExistsError(f"{dest} points at {dest.resolve()}, expected {source}")
        if not dest.is_symlink() and dest.exists():
            raise FileExistsError(f"{dest} exists and is not a symlink")
        planned.append((spec, source, dest))
    if missing and not allow_partial:
        raise FileNotFoundError(missing[0] if len(missing) == 1 else "; ".join(missing))
    lines.extend(missing)
    if dry_run:
        for spec, source, dest in planned:
            if dest.is_symlink():
                lines.append(f"ok {spec.link_name} -> {source}")
            else:
                lines.append(f"would-link {spec.link_name} -> {source}")
        return lines
    if planned:
        corpora_root.mkdir(parents=True, exist_ok=True)
    for spec, source, dest in planned:
        if dest.is_symlink():
            current = dest.resolve()
            if current != source.resolve():
                raise FileExistsError(f"{dest} points at {current}, expected {source}")
            lines.append(f"ok {spec.link_name} -> {source}")
            continue
        if dest.exists():
            raise FileExistsError(f"{dest} exists and is not a symlink")
        dest.symlink_to(source)
        lines.append(f"linked {spec.link_name} -> {source}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Link standard corpora into data/corpora")
    dataset_root_argument(parser)
    parser.add_argument("--corpora-root", type=Path, default=CORPORA_ROOT)
    parser.add_argument("--include-non-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="source or link name; repeat or comma-separate")
    parser.add_argument("--corpus-path", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    names: list[str] = []
    for item in args.only:
        names.extend(part.strip() for part in item.split(",") if part.strip())
    overrides = env_overrides()
    for item in args.corpus_path:
        if "=" not in item:
            parser.error(f"--corpus-path must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        overrides[name] = Path(path)
    root = args.dataset_root
    if root is None and os.environ.get("AUDIO_DB_DATASET_ROOT"):
        root = Path(os.environ["AUDIO_DB_DATASET_ROOT"])
    if root is None and not overrides and not args.allow_partial:
        parser.error("specify --dataset-root or AUDIO_DB_DATASET_ROOT")
    for line in link_corpora(
        root,
        args.corpora_root,
        include_non_audio=args.include_non_audio,
        dry_run=args.dry_run,
        only=names or None,
        overrides=overrides or None,
        allow_partial=args.allow_partial,
    ):
        print(line)


if __name__ == "__main__":
    main()
