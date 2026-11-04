import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.setup.assets import AssetError, ensure_git, place_file


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, stdout=subprocess.DEVNULL)


class AssetTests(unittest.TestCase):
    def test_checksum_match_and_conflict(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bin"
            source.write_bytes(b"abc")
            digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            dest = root / "dest.bin"
            self.assertTrue(place_file(dest, source, digest, 3).startswith("installed"))
            self.assertTrue(place_file(dest, source, digest, 3).startswith("unchanged"))
            source.write_bytes(b"zzz")
            with self.assertRaises(AssetError):
                place_file(root / "other.bin", source, digest, 3)
            dest.write_bytes(b"nope")
            with self.assertRaises(AssetError):
                place_file(dest, source, digest, 3)
            self.assertEqual(dest.read_bytes(), b"nope")

    def test_git_commit_is_reused_and_conflicts_are_kept(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            origin = root / "origin"
            origin.mkdir()
            _git(origin, "init")
            _git(origin, "config", "user.email", "test@example.com")
            _git(origin, "config", "user.name", "test")
            _git(origin, "config", "commit.gpgsign", "false")
            (origin / "README").write_text("one\n")
            _git(origin, "add", "README")
            _git(origin, "commit", "-m", "one")
            commit = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
            spec = {"id": "demo", "url": str(origin), "commit": commit, "dest": "src/demo"}
            first = ensure_git(spec, root)
            self.assertTrue(first.startswith("cloned"))
            second = ensure_git(spec, root)
            self.assertTrue(second.startswith("unchanged"))
            (root / "src" / "demo" / "README").write_text("edited\n")
            with self.assertRaises(AssetError):
                ensure_git(spec, root)
            self.assertIn("edited", (root / "src" / "demo" / "README").read_text())
            _git(root / "src" / "demo", "checkout", "--", "README")
            (origin / "README").write_text("two\n")
            _git(origin, "add", "README")
            _git(origin, "commit", "-m", "two")
            moved = subprocess.check_output(["git", "-C", str(origin), "rev-parse", "HEAD"], text=True).strip()
            spec["commit"] = moved
            with self.assertRaises(AssetError):
                ensure_git(spec, root)
            head = subprocess.check_output(["git", "-C", str(root / "src" / "demo"), "rev-parse", "HEAD"], text=True).strip()
            self.assertEqual(head, commit)


if __name__ == "__main__":
    unittest.main()
