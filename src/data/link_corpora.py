"""Create data/corpora symlinks for the standard catalog.

Existing links that already point at the same directory are left in place.
A link that points somewhere else is an error. Corpus files are not copied.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.data.constants import CORPORA_ROOT
from src.data.corpora_catalog import CATALOG

def dataset_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="dataset root; overrides AUDIO_DB_DATASET_ROOT")


def resolve_dataset_root(parser: argparse.ArgumentParser, value: Path | None) -> Path:
    value = value or (Path(os.environ["AUDIO_DB_DATASET_ROOT"]) if os.environ.get("AUDIO_DB_DATASET_ROOT") else None)
    if value is None:
        parser.error("specify --dataset-root or AUDIO_DB_DATASET_ROOT")
    return value


def link_corpora(dataset_root: Path, corpora_root: Path = CORPORA_ROOT, *, include_non_audio: bool = False) -> list[str]:
    dataset_root = dataset_root.resolve()
    lines: list[str] = []
    # Preflight the complete catalog before creating any links.
    for spec in CATALOG:
        if not spec.index and not include_non_audio:
            continue
        source, dest = dataset_root / spec.source_name, corpora_root / spec.link_name
        if not source.is_dir():
            raise FileNotFoundError(f"missing dataset directory: {source}")
        if dest.is_symlink() and dest.resolve() != source.resolve():
            raise FileExistsError(f"{dest} points at {dest.resolve()}, expected {source}")
        if not dest.is_symlink() and dest.exists():
            raise FileExistsError(f"{dest} exists and is not a symlink")
    corpora_root.mkdir(parents=True, exist_ok=True)
    for spec in CATALOG:
        if not spec.index and not include_non_audio:
            lines.append(f"skip {spec.source_name}: {spec.note}")
            continue
        source = dataset_root / spec.source_name
        dest = corpora_root / spec.link_name
        if not source.is_dir():
            raise FileNotFoundError(f"missing dataset directory: {source}")
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
    args = parser.parse_args()
    for line in link_corpora(resolve_dataset_root(parser, args.dataset_root), args.corpora_root, include_non_audio=args.include_non_audio):
        print(line)


if __name__ == "__main__":
    main()
