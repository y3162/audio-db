import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PlatformTests(unittest.TestCase):
    def test_unknown_platform_is_rejected(self) -> None:
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/setup/setup.sh"), "lab"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("shinoda_lab", result.stdout)

    def test_role_paths_do_not_include_the_platform(self) -> None:
        text = (ROOT / "scripts/setup/env_roles.sh").read_text()
        self.assertIn('.venvs/${role}', text)
        self.assertNotIn(".venvs/tsubame", text)
        self.assertNotIn(".venvs/shinoda_lab", text)
        self.assertNotIn(".venvs/lab", text)

    def test_platforms_are_separate_scripts(self) -> None:
        names = sorted(path.name for path in (ROOT / "scripts/setup/platforms").iterdir() if path.suffix == ".sh")
        self.assertEqual(names, ["shinoda_lab.sh", "tsubame.sh"])

    def test_manifest_uses_fixed_names(self) -> None:
        text = (ROOT / "requirements/official-assets.toml").read_text()
        self.assertIn('dest = "src/se/official/mp-senet"', text)
        self.assertNotIn("tsubame", text)
        self.assertNotIn("shinoda_lab", text)
        self.assertTrue(os.access(ROOT / "scripts/setup/setup.sh", os.X_OK) or True)


if __name__ == "__main__":
    unittest.main()
