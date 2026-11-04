"""Fetch pinned official sources and weights.

An existing checkout at the pinned commit is reused. A different commit or a
local edit is left in place and reported. Downloads land in a temporary file
and replace the destination only after the size and SHA-256 match.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tomllib
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "requirements" / "official-assets.toml"


class AssetError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST) -> dict:
    return tomllib.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise AssetError(f"{' '.join(command)} failed:\n{result.stdout}")
    return result.stdout


def _ignored(path: str) -> bool:
    parts = Path(path).parts
    generated = {"__pycache__", "build", "target", "dist"}
    return any(part in generated or part.endswith(".pyc") or part.endswith(".egg-info") for part in parts)


def dirty_entries(repo: Path) -> list[str]:
    text = _run(["git", "-C", str(repo), "status", "--porcelain", "-uall"])
    rows = []
    for line in text.splitlines():
        path = line[3:].split(" -> ")[-1]
        if _ignored(path):
            continue
        rows.append(line)
    return rows


def ensure_git(spec: dict, root: Path) -> str:
    dest = root / spec["dest"]
    commit = spec["commit"]
    if dest.exists() and not (dest / ".git").exists():
        raise AssetError(f"{dest} exists and is not the pinned git checkout")
    if (dest / ".git").exists():
        head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"]).strip()
        dirty = dirty_entries(dest)
        if head != commit or dirty:
            detail = f"HEAD {head}"
            if dirty:
                detail += "; local changes: " + ", ".join(dirty[:8])
            raise AssetError(
                f"{dest} is not the pinned commit {commit} ({detail}). "
                "It was not modified. Inspect it, or move it aside and rerun."
            )
        return f"unchanged {spec['id']} {commit}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    sparse = list(spec.get("sparse") or [])
    if sparse:
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["url"], str(dest)])
        _run(["git", "-C", str(dest), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(dest), "sparse-checkout", "set", *sparse])
        _run(["git", "-C", str(dest), "checkout", "--detach", commit])
    else:
        _run(["git", "clone", "--filter=blob:none", spec["url"], str(dest)])
        _run(["git", "-C", str(dest), "checkout", "--detach", commit])
    head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"]).strip()
    if head != commit:
        raise AssetError(f"{dest} checked out {head}, expected {commit}")
    return f"cloned {spec['id']} {commit}"


def place_file(dest: Path, source: Path, sha256: str, size: int | None) -> str:
    if not sha256:
        raise AssetError(f"{dest} has no pinned sha256; not installed")
    actual_size = source.stat().st_size
    actual = sha256_file(source)
    if actual != sha256 or (size is not None and actual_size != size):
        raise AssetError(f"checksum mismatch for {source}: sha256 {actual} size {actual_size}")
    if dest.is_file() or dest.is_symlink():
        if dest.is_symlink() or sha256_file(dest) != sha256 or dest.stat().st_size != actual_size:
            raise AssetError(f"{dest} already exists and does not match the pinned file. It was not replaced.")
        return f"unchanged {dest}"
    if dest.exists():
        raise AssetError(f"{dest} exists and is not a file. It was not replaced.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    if source.resolve() == temporary.resolve():
        os.replace(temporary, dest)
        return f"installed {dest}"
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(source, temporary)
    os.replace(temporary, dest)
    return f"installed {dest}"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "audio-db-setup"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_http(spec: dict, root: Path) -> str:
    dest = root / spec["dest"]
    sha256 = spec.get("sha256") or ""
    size = spec.get("size")
    if dest.is_file() and not dest.is_symlink() and sha256 and sha256_file(dest) == sha256:
        if size is None or dest.stat().st_size == size:
            return f"unchanged {dest}"
    temporary = dest.with_name(dest.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    download(spec["url"], temporary)
    try:
        return place_file(dest, temporary, sha256, size)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_git_file(spec: dict, root: Path, sources: dict[str, dict]) -> str:
    source_spec = sources[spec["source"]]
    origin = root / source_spec["dest"] / spec["path"]
    if not origin.is_file():
        raise AssetError(f"missing {origin} in the pinned {spec['source']} tree")
    return place_file(root / spec["dest"], origin, spec.get("sha256") or "", spec.get("size"))


def ensure_zip(spec: dict, root: Path) -> str:
    sha256 = spec.get("sha256") or ""
    if not sha256:
        raise AssetError(f"{spec['id']} has no pinned sha256; not installed")
    cache = root / "build" / "asset-cache" / f"{spec['id']}.zip"
    if not (cache.is_file() and sha256_file(cache) == sha256):
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".zip.partial")
        if temporary.exists():
            temporary.unlink()
        download(spec["url"], temporary)
        if sha256_file(temporary) != sha256 or temporary.stat().st_size != spec.get("size"):
            temporary.unlink()
            raise AssetError(f"checksum mismatch for {spec['url']}")
        os.replace(temporary, cache)
    dest_dir = root / spec["dest_dir"]
    lines = [f"verified-archive {spec['id']}"]
    with zipfile.ZipFile(cache) as archive:
        names = set(archive.namelist())
        for member in spec.get("member") or []:
            path = member["path"]
            if path not in names:
                raise AssetError(f"{spec['id']} archive has no {path}")
            target = dest_dir / path
            payload = archive.read(path)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != member["sha256"] or len(payload) != member["size"]:
                raise AssetError(f"member checksum mismatch: {path}")
            if target.is_file() and sha256_file(target) == digest:
                lines.append(f"unchanged {target}")
                continue
            if target.exists():
                raise AssetError(f"{target} exists and does not match the archive member. It was not replaced.")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            temporary.write_bytes(payload)
            os.replace(temporary, target)
            lines.append(f"installed {target}")
    return "\n".join(lines)


def selected(items: list[dict], models: set[str] | None, key: str = "id") -> list[dict]:
    if not models:
        return items
    chosen = []
    for item in items:
        identity = item.get("model") or item.get(key)
        if identity in models or item.get(key) in models:
            chosen.append(item)
    return chosen


def fetch(root: Path = ROOT, models: set[str] | None = None, *, sources: bool = True, weights: bool = True) -> list[str]:
    manifest = load_manifest()
    lines: list[str] = []
    source_rows = {row["id"]: row for row in manifest.get("source", [])}
    if sources:
        for spec in selected(manifest.get("source", []), models):
            lines.append(ensure_git(spec, root))
    if weights:
        for spec in selected(manifest.get("weight", []), models):
            kind = spec["kind"]
            if kind == "http":
                lines.append(ensure_http(spec, root))
            elif kind == "git-file":
                lines.append(ensure_git_file(spec, root, source_rows))
            elif kind == "zip":
                lines.append(ensure_zip(spec, root))
            else:
                raise AssetError(f"unknown weight kind {kind}")
    return lines


def check(root: Path = ROOT, models: set[str] | None = None) -> list[str]:
    """Read-only verification. Missing files are reported and not downloaded."""
    manifest = load_manifest()
    lines: list[str] = []
    failed = False
    source_rows = {row["id"]: row for row in manifest.get("source", [])}
    for spec in selected(manifest.get("source", []), models):
        dest = root / spec["dest"]
        if not (dest / ".git").exists():
            lines.append(f"MISSING source {spec['id']} {dest}")
            failed = True
            continue
        head = _run(["git", "-C", str(dest), "rev-parse", "HEAD"]).strip()
        dirty = dirty_entries(dest)
        if head != spec["commit"] or dirty:
            lines.append(f"CONFLICT source {spec['id']} HEAD {head}")
            failed = True
        else:
            lines.append(f"OK source {spec['id']} {head}")
    for spec in selected(manifest.get("weight", []), models):
        if spec["kind"] == "zip":
            for member in spec.get("member") or []:
                path = root / spec["dest_dir"] / member["path"]
                if not path.is_file() or sha256_file(path) != member["sha256"]:
                    lines.append(f"MISSING weight {spec['id']} {path}")
                    failed = True
                else:
                    lines.append(f"OK weight {path.name}")
            continue
        if spec["kind"] == "git-file":
            path = root / spec["dest"]
        else:
            path = root / spec["dest"]
        sha = spec.get("sha256") or ""
        if not path.is_file() or not sha or sha256_file(path) != sha:
            lines.append(f"MISSING weight {spec['id']} {path}")
            failed = True
        else:
            lines.append(f"OK weight {path.name}")
        if spec["kind"] == "git-file" and spec["source"] not in source_rows:
            lines.append(f"MISSING source-ref {spec['source']}")
            failed = True
    if failed:
        raise SystemExit("\n".join(lines))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fetch", "check"))
    parser.add_argument("--models", default="")
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args()
    models = {item.strip() for item in args.models.split(",") if item.strip()} or None
    try:
        if args.command == "check":
            lines = check(models=models)
        else:
            lines = fetch(models=models, sources=not args.weights_only, weights=not args.sources_only)
    except AssetError as error:
        print(f"FAIL {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("\n".join(lines))


if __name__ == "__main__":
    main()
