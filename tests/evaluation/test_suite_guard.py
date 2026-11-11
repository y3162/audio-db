"""The read-only guard that keeps tests from writing production data.

The guarded cases use a stand-in ``data/`` tree in a temporary directory. No
case writes to the real one.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import suite_guard

ROOT = Path(__file__).resolve().parents[2]

# Runs behind the guard. Each attempt reports the errno it met, or None.
_PROBE = """
import json, os, subprocess, sys
root, outside = sys.argv[1], sys.argv[2]

def attempt(action):
    try:
        action()
    except OSError as exc:
        return exc.errno
    return None

def write(path):
    return attempt(lambda: open(path, "w").close())

child = subprocess.run(["sh", "-c", 'echo x > "$1"', "sh", root + "/data/db/from-child"], stderr=subprocess.DEVNULL)
print(json.dumps({
    "uid": os.getuid(),
    "gid": os.getgid(),
    "new_file": write(root + "/data/db/new"),
    "existing_file": attempt(lambda: open(root + "/data/db/kept", "a").close()),
    "through_link": write(root + "/data/corpora/Speech/new"),
    "link_target": write(outside + "/corpus/new"),
    "read": open(root + "/data/db/kept").read(),
    "rename": attempt(lambda: os.rename(root + "/data/db/kept", root + "/data/db/moved")),
    "remove": attempt(lambda: os.unlink(root + "/data/db/kept")),
    "child_status": child.returncode,
    "elsewhere": write(outside + "/scratch/new"),
    "capabilities": [line for line in open("/proc/self/status") if line.startswith("CapEff")][0].split()[1],
}))
"""


def _tree(base: Path) -> tuple[Path, Path]:
    root = base / "repo"
    outside = base / "outside"
    (root / "data/db").mkdir(parents=True)
    (root / "data/corpora").mkdir()
    (outside / "corpus").mkdir(parents=True)
    (outside / "scratch").mkdir()
    (root / "data/db/kept").write_text("kept", encoding="utf-8")
    (root / "data/corpora/Speech").symlink_to(outside / "corpus")
    return root, outside


class ProductionPathTests(unittest.TestCase):
    def test_links_are_followed_and_nested_entries_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw).resolve()
            root, outside = _tree(base)
            (outside / "results").mkdir()
            (root / "data/results").symlink_to(outside / "results")
            (root / "data/corpora/Inside").symlink_to(root / "data/db")
            (root / "data/corpora/Missing").symlink_to(outside / "absent")
            (root / "data/corpora/Deeper").symlink_to(outside / "corpus")
            self.assertEqual(
                suite_guard.production_paths(root),
                sorted([root / "data", outside / "corpus", outside / "results"]),
            )

    def test_a_tree_without_data_has_nothing_to_guard(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(suite_guard.production_paths(Path(raw)), [])
            suite_guard.enter([])

    def test_the_repository_guards_data_and_every_linked_corpus(self) -> None:
        data = ROOT / "data"
        if not data.exists():
            self.skipTest("this checkout has no data directory")
        paths = suite_guard.production_paths(ROOT)
        self.assertIn(data.resolve(), paths)
        for link in sorted((data / "corpora").glob("*")):
            if link.is_symlink() and link.exists():
                target = link.resolve()
                self.assertTrue(any(target == path or target.is_relative_to(path) for path in paths), link)


class GuardedProcessTests(unittest.TestCase):
    def _guarded(self, root: Path, outside: Path, environment: dict[str, str] | None = None) -> dict[str, object]:
        launcher = (
            "import sys; from pathlib import Path; from tests import suite_guard\n"
            "try:\n"
            "    suite_guard.run_guarded([sys.executable, '-c', sys.argv[3], sys.argv[1], sys.argv[2]], Path(sys.argv[1]))\n"
            "except suite_guard.GuardError as exc:\n"
            "    print('GUARD_ERROR ' + str(exc)); raise SystemExit(3)\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", launcher, str(root), str(outside), _PROBE],
            cwd=ROOT,
            env={**os.environ, **(environment or {})},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if done.returncode == 3 and "GUARD_ERROR cannot create a user and mount namespace" in done.stdout:
            self.skipTest("NOT_RUN: this host does not allow user namespaces: " + done.stdout.strip())
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        return json.loads(done.stdout.strip().splitlines()[-1])

    def test_guarded_data_refuses_every_write_and_stays_readable(self) -> None:
        import errno

        with tempfile.TemporaryDirectory() as raw:
            root, outside = _tree(Path(raw).resolve())
            environment = {suite_guard.UNGUARDED_ENV: ""}
            report = self._guarded(root, outside, environment)
            for name in ("new_file", "existing_file", "through_link", "link_target", "remove", "rename"):
                self.assertEqual(report[name], errno.EROFS, name)
            self.assertNotEqual(report["child_status"], 0)
            self.assertEqual(report["read"], "kept")
            self.assertIsNone(report["elsewhere"])
            self.assertEqual((report["uid"], report["gid"]), (os.getuid(), os.getgid()))
            self.assertEqual(int(str(report["capabilities"]), 16), 0)
            # Nothing reached the tree, and the caller's view is still writable.
            self.assertEqual(sorted(path.name for path in (root / "data/db").iterdir()), ["kept"])
            self.assertEqual(list((outside / "corpus").iterdir()), [])
            self.assertTrue((outside / "scratch/new").is_file())
            (root / "data/db/after").write_text("x", encoding="utf-8")

    def test_the_opt_out_leaves_the_tree_writable(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root, outside = _tree(Path(raw).resolve())
            report = self._guarded(root, outside, {suite_guard.UNGUARDED_ENV: "1"})
            self.assertIsNone(report["new_file"])
            self.assertTrue((root / "data/db/new").is_file())

    def test_suite_processes_see_read_only_production_data(self) -> None:
        if os.environ.get(suite_guard.STATE_ENV) != "read-only":
            self.skipTest("the runner's guard is not active in this process")
        paths = suite_guard.production_paths(ROOT)
        self.assertTrue(paths)
        self.assertTrue(suite_guard.protected(paths))
        for path in paths:
            self.assertFalse(os.access(path, os.W_OK), path)


class RunnerGuardTests(unittest.TestCase):
    def test_the_runner_enters_the_guard_once(self) -> None:
        from scripts.tests import run_suite

        paths = [Path("/data"), Path("/corpus")]
        with (
            patch.dict(os.environ, {suite_guard.UNGUARDED_ENV: ""}),
            patch.object(suite_guard, "production_paths", return_value=paths),
            patch.object(suite_guard, "enter") as enter,
        ):
            with patch.object(suite_guard, "protected", return_value=False):
                self.assertEqual(run_suite.guard_production_data(), ("read-only", ["/data", "/corpus"]))
            enter.assert_called_once_with(paths)
            enter.reset_mock()
            with patch.object(suite_guard, "protected", return_value=True):
                self.assertEqual(run_suite.guard_production_data(), ("read-only", ["/data", "/corpus"]))
            enter.assert_not_called()

    def test_the_opt_out_is_explicit_and_reported(self) -> None:
        from scripts.tests import run_suite

        with patch.object(suite_guard, "enter") as enter:
            with patch.dict(os.environ, {suite_guard.UNGUARDED_ENV: "1"}):
                self.assertEqual(run_suite.guard_production_data(), ("off", []))
            enter.assert_not_called()
            for value in ("", "0", "true"):
                with (
                    patch.dict(os.environ, {suite_guard.UNGUARDED_ENV: value}),
                    patch.object(suite_guard, "production_paths", return_value=[]),
                ):
                    self.assertEqual(run_suite.guard_production_data(), ("none", []))

    def test_the_runner_starts_no_test_when_the_guard_fails(self) -> None:
        from scripts.tests import run_suite

        with (
            patch.object(run_suite, "guard_production_data", side_effect=suite_guard.GuardError("no namespaces")),
            patch.object(run_suite, "run_lane", side_effect=AssertionError("a lane started")) as lane,
            patch("sys.stderr") as stderr,
        ):
            self.assertEqual(run_suite.main(["--cpu-only"]), 2)
        lane.assert_not_called()
        message = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("no namespaces", message)
        self.assertIn(suite_guard.UNGUARDED_ENV, message)


if __name__ == "__main__":
    unittest.main()
