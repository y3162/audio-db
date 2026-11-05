"""Parse pinned VoiceBank train and test lists.

MP-SENet keeps the utterance id before ``|`` and ignores the rest of the line.
MossFormerGAN scp lines are ``noisy_path clean_path``. The local corpus is
resolved from the utterance id, not from the path stored in the list.
"""

from __future__ import annotations

from pathlib import Path

from src.data.recipe import sha256_file
from src.reproduction.errors import PreparationBlocked, ReproductionError

UTTERANCE_RE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def parse_mp_senet(text: str, *, where: str) -> list[str]:
    return _parse(text, where=where, kind="mp_senet")


def parse_mossformergan_scp(text: str, *, where: str) -> list[str]:
    return _parse(text, where=where, kind="mossformergan_scp")


def speaker_of(utterance_id: str) -> str:
    if "_" not in utterance_id:
        raise ReproductionError(f"utterance id has no speaker prefix: {utterance_id}")
    speaker = utterance_id.split("_", 1)[0]
    if not speaker:
        raise ReproductionError(f"utterance id has an empty speaker prefix: {utterance_id}")
    return speaker


def holdout_split(train_ids: list[str], speakers: tuple[str, ...] | list[str]) -> tuple[list[str], list[str]]:
    """Keep official order. Holdout speakers become validation. No RNG."""
    held = tuple(speakers)
    if len(held) != len(set(held)) or not held:
        raise ReproductionError("holdout speakers must be a non-empty unique list")
    held_set = set(held)
    train: list[str] = []
    valid: list[str] = []
    for utterance_id in train_ids:
        if speaker_of(utterance_id) in held_set:
            valid.append(utterance_id)
        else:
            train.append(utterance_id)
    return train, valid


def load_manifest(path: Path, repo_root: Path) -> dict[str, dict]:
    """Load the pin manifest and require each file's bytes to match its SHA-256."""
    document = _yaml(path)
    if not isinstance(document, dict) or not document:
        raise ReproductionError(f"list manifest is empty: {path}")
    loaded: dict[str, dict] = {}
    for model, spec in document.items():
        if not isinstance(model, str) or not isinstance(spec, dict):
            raise ReproductionError(f"list manifest entry {model} is invalid")
        loaded[model] = _model_lists(model, spec, repo_root)
    return loaded


def _model_lists(model: str, spec: dict, repo_root: Path) -> dict:
    kind = spec.get("format")
    if kind not in ("mp_senet", "mossformergan_scp"):
        raise ReproductionError(f"{model} list format is unknown: {kind}")
    for key in ("source", "repo", "commit"):
        if not isinstance(spec.get(key), str) or not spec[key]:
            raise ReproductionError(f"{model} manifest {key} must be a string")
    sides = {}
    for side in ("train", "test"):
        item = spec.get(side)
        if not isinstance(item, dict):
            raise ReproductionError(f"{model} manifest is missing {side}")
        file_name = item.get("file")
        digest = item.get("sha256")
        upstream = item.get("path")
        if not isinstance(file_name, str) or not isinstance(digest, str) or not isinstance(upstream, str):
            raise ReproductionError(f"{model} {side} needs file, path, and sha256")
        path = repo_root / file_name
        if not path.is_file():
            raise PreparationBlocked(f"official list is missing: {file_name}")
        actual = sha256_file(path)
        if actual != digest:
            raise ReproductionError(
                f"official list {file_name} sha256 is {actual}, manifest has {digest}"
            )
        text = path.read_text(encoding="utf-8")
        parser = parse_mp_senet if kind == "mp_senet" else parse_mossformergan_scp
        ids = parser(text, where=file_name)
        sides[side] = {
            "path": upstream,
            "file": file_name,
            "sha256": digest,
            "ids": ids,
        }
    return {
        "source": spec["source"],
        "repo": spec["repo"],
        "commit": spec["commit"],
        "format": kind,
        "train": sides["train"],
        "test": sides["test"],
    }


def _parse(text: str, *, where: str, kind: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    lines = text.splitlines()
    if text.endswith("\n"):
        # splitlines drops the trailing empty line; a blank line in the middle remains
        pass
    for line_no, line in enumerate(lines, 1):
        if line.strip() == "":
            raise ReproductionError(f"{where}:{line_no} is blank")
        utterance_id = _utterance(line, kind=kind, where=f"{where}:{line_no}")
        if utterance_id in seen:
            raise ReproductionError(f"{where}:{line_no} repeats {utterance_id}")
        seen.add(utterance_id)
        ids.append(utterance_id)
    if not ids:
        raise ReproductionError(f"{where} has no utterances")
    return ids


def _utterance(line: str, *, kind: str, where: str) -> str:
    if kind == "mp_senet":
        utterance_id = line.split("|", 1)[0].strip()
    else:
        parts = line.split()
        if len(parts) != 2:
            raise ReproductionError(f"{where} must contain noisy and clean paths")
        noisy = Path(parts[0]).name
        clean = Path(parts[1]).name
        if noisy != clean:
            raise ReproductionError(f"{where} noisy and clean filenames differ: {noisy} vs {clean}")
        utterance_id = Path(noisy).stem
    if not utterance_id or any(char not in UTTERANCE_RE_CHARS for char in utterance_id):
        raise ReproductionError(f"{where} has an invalid utterance id: {utterance_id!r}")
    if "_" not in utterance_id:
        raise ReproductionError(f"{where} utterance id has no speaker prefix: {utterance_id}")
    return utterance_id


def _yaml(path: Path):
    import yaml

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PreparationBlocked(f"cannot read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ReproductionError(f"cannot parse {path}: {exc}") from exc
