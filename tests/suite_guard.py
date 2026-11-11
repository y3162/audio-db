"""Read-only view of the production data for test processes.

``data/`` holds the production databases, staging, snapshots, checkpoints, and
results. Corpora are symlinks from it to directories outside the repository.
A test may read all of them and may write none.

``enter`` moves the calling process into a new user and mount namespace and
remounts those directories read-only there. A write then fails with EROFS in
this process and in every process it starts. Nothing changes for processes
outside the namespace. The user and group ids stay the same.

The process that calls ``enter`` keeps the capabilities of the new namespace
until it executes a program. ``run_guarded`` and the test runner execute the
tests in a fresh program, which has no capability to undo the mounts.

The guard covers the paths ``production_paths`` returns. A second path to the
same files, such as another mount of the same export, is not covered.

    python -m tests.suite_guard COMMAND [ARG...]
"""

from __future__ import annotations

import ctypes
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNGUARDED_ENV = "AUDIO_DB_TEST_UNGUARDED"
# The runner reports the guard state to its test processes here. It is a
# report for the tests that check the guard, not a switch.
STATE_ENV = "AUDIO_DB_TEST_DATA_GUARD"

_CLONE_NEWNS = 0x00020000
_CLONE_NEWUSER = 0x10000000
_MS_RDONLY = 0x1
_MS_REMOUNT = 0x20
_MS_BIND = 0x1000
_MS_REC = 0x4000
_MS_RELATIME = 0x200000
# A mount inherited from the host keeps these flags locked. A remount that
# drops one of them is refused, so the remount repeats the ones that are set.
# The names are the statvfs flags; all but relatime share the mount value.
_LOCKED = (
    (os.ST_NOSUID, os.ST_NOSUID),
    (os.ST_NODEV, os.ST_NODEV),
    (os.ST_NOEXEC, os.ST_NOEXEC),
    (os.ST_NOATIME, os.ST_NOATIME),
    (os.ST_NODIRATIME, os.ST_NODIRATIME),
    (os.ST_RELATIME, _MS_RELATIME),
)


# mountinfo writes space, tab, newline, and backslash in a path as \ooo.
_ESCAPE = re.compile(r"\\([0-7]{3})")


class GuardError(RuntimeError):
    """The production data could not be made read-only."""


def unguarded() -> bool:
    """True when the caller chose to run without the guard."""
    return os.environ.get(UNGUARDED_ENV, "").strip() == "1"


def production_paths(root: Path = ROOT) -> list[Path]:
    """``data/`` and the places its symlinks lead to, without nested entries.

    Symlinks are followed for ``data`` itself, its entries, and their entries.
    That covers ``data/corpora/<name>`` and a results or snapshot directory
    kept on another disk.
    """
    data = root / "data"
    if not data.exists():
        return []
    found = {data.resolve()}
    for entry in _entries(data):
        if entry.is_symlink():
            if entry.exists():
                found.add(entry.resolve())
            continue
        if entry.is_dir():
            for inner in _entries(entry):
                if inner.is_symlink() and inner.exists():
                    found.add(inner.resolve())
    kept: list[Path] = []
    for path in sorted(found, key=lambda item: len(item.parts)):
        if not any(path.is_relative_to(parent) for parent in kept):
            kept.append(path)
    return sorted(kept)


def protected(paths: list[Path]) -> bool:
    """True when every path is on a read-only mount."""
    return all(os.statvfs(path).f_flag & os.ST_RDONLY for path in paths)


def enter(paths: list[Path]) -> None:
    """Make ``paths`` read-only for this process and its children.

    The process must have one thread. On failure the process may already be
    in the new namespace, and the caller must not go on to run tests.
    """
    if not paths:
        return
    if threading.active_count() != 1:
        raise GuardError("the guard must be entered before any thread starts")
    libc = ctypes.CDLL(None, use_errno=True)
    uid, gid = os.getuid(), os.getgid()
    if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS) != 0:
        code = ctypes.get_errno()
        raise GuardError(f"cannot create a user and mount namespace: {os.strerror(code)}")
    try:
        Path("/proc/self/setgroups").write_text("deny", encoding="ascii")
        Path("/proc/self/uid_map").write_text(f"{uid} {uid} 1", encoding="ascii")
        Path("/proc/self/gid_map").write_text(f"{gid} {gid} 1", encoding="ascii")
    except OSError as exc:
        raise GuardError(f"cannot map the user into the new namespace: {exc}") from exc
    for path in paths:
        target = os.fsencode(path)
        if libc.mount(target, target, None, _MS_BIND | _MS_REC, None) != 0:
            code = ctypes.get_errno()
            raise GuardError(f"cannot bind {path}: {os.strerror(code)}")
        for mount in _mounts_under(path):
            _remount_read_only(libc, mount)
    for path in paths:
        if not os.statvfs(path).f_flag & os.ST_RDONLY or os.access(path, os.W_OK):
            raise GuardError(f"{path} is still writable after the remount")


def run_guarded(command: list[str], root: Path = ROOT) -> None:
    """Replace this process with ``command`` behind the guard. Does not return."""
    if not unguarded():
        paths = production_paths(root)
        if not protected(paths):
            enter(paths)
    os.execvp(command[0], command)


def main(argv: list[str] | None = None) -> int:
    command = list(sys.argv[1:] if argv is None else argv)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        print("usage: python -m tests.suite_guard COMMAND [ARG...]", file=sys.stderr)
        return 2
    try:
        run_guarded(command)
    except GuardError as exc:
        print(f"{exc}. Set {UNGUARDED_ENV}=1 to run without the guard.", file=sys.stderr)
        return 2
    return 0


def _entries(directory: Path) -> list[Path]:
    try:
        return sorted(directory.iterdir())
    except OSError:
        return []


def _mounts_under(path: Path) -> list[Path]:
    """Mount points at or below ``path`` in this namespace, parents first."""
    found: set[Path] = set()
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8", errors="surrogateescape").splitlines():
        fields = line.split(" ")
        if len(fields) < 5:
            continue
        point = Path(_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), fields[4]))
        if point == path or point.is_relative_to(path):
            found.add(point)
    return sorted(found, key=lambda item: len(item.parts))


def _remount_read_only(libc: ctypes.CDLL, mount: Path) -> None:
    flags = _MS_REMOUNT | _MS_BIND | _MS_RDONLY
    current = os.statvfs(mount).f_flag
    for seen, wanted in _LOCKED:
        if current & seen:
            flags |= wanted
    if libc.mount(None, os.fsencode(mount), None, flags, None) != 0:
        code = ctypes.get_errno()
        raise GuardError(f"cannot remount {mount} read-only: {os.strerror(code)}")


if __name__ == "__main__":
    raise SystemExit(main())
