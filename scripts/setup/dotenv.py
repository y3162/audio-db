"""Load a trusted .env without overriding variables already in the environment.

The file is the same format as ``source .env``: comments, optional ``export``,
and quoted values. Command substitution is rejected. Empty values are ignored
so a template line does not set a blank path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError(f".env:{lineno}: expected NAME=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.isidentifier():
            raise ValueError(f".env:{lineno}: invalid name {key!r}")
        values[key] = _decode(value.strip(), lineno)
    return {key: value for key, value in values.items() if value != ""}


def _decode(value: str, lineno: int) -> str:
    if "$(" in value or "`" in value:
        raise ValueError(f".env:{lineno}: command substitution is not allowed")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        body = value[1:-1]
        if quote == '"':
            body = body.replace('\\"', '"').replace("\\\\", "\\")
        return body
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def bash_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def emit_unset(path: Path, environ: dict[str, str] | None = None) -> str:
    current = os.environ if environ is None else environ
    lines = []
    for key, value in parse_env(path.read_text()).items():
        if key not in current:
            lines.append(f"export {key}={bash_quote(value)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-unset", type=Path, required=True)
    args = parser.parse_args()
    if not args.emit_unset.is_file():
        return
    sys.stdout.write(emit_unset(args.emit_unset))
    if sys.stdout.isatty():
        return
    if emit_unset(args.emit_unset):
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
