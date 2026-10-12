"""Exercise history preconditions and Git diff contracts with local repositories."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reproduce


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, timeout=30,
    ).strip()


class ReproduceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="history-reproduce-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        git(self.repo, "init", "--quiet")
        git(self.repo, "config", "user.name", "History test")
        git(self.repo, "config", "user.email", "history@example.invalid")
        git(self.repo, "config", "core.filemode", "true")
        for name, content in (("mode", "unchanged\n"), ("type", "mode")):
            (self.repo / name).write_text(content, encoding="ascii")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "--quiet", "-m", "baseline")
        self.base = git(self.repo, "rev-parse", "HEAD")

    def test_shallow_rejected_before_creating_or_overwriting_outputs(self) -> None:
        git(self.repo, "commit", "--quiet", "--allow-empty", "-m", "second")
        clone = self.root / "shallow"
        git(self.root, "clone", "--quiet", "--depth=1", self.repo.as_uri(), str(clone))
        self.assertEqual(git(clone, "rev-list", "--count", "HEAD"), "1")
        self.assertEqual(git(clone, "rev-parse", "--is-shallow-repository"), "true")
        data = clone / "data"
        ledger = clone / "05-master-ledger.md"
        ledger.write_bytes(b"existing ledger\n")
        with patch.object(reproduce, "ROOT", clone), patch.object(reproduce, "HERE", clone), \
                patch.object(reproduce, "DATA", data):
            with self.assertRaisesRegex(SystemExit, "git fetch --unshallow"):
                reproduce.main()
            self.assertFalse(data.exists())
            data.mkdir()
            (data / "summary.json").write_bytes(b"existing inventory\n")
            with self.assertRaisesRegex(SystemExit, "git fetch --unshallow"):
                reproduce.main()
        # The rejection must preserve existing evidence as well as avoid new output.
        self.assertEqual(ledger.read_bytes(), b"existing ledger\n")
        self.assertEqual(list(data.iterdir()), [data / "summary.json"])
        self.assertEqual((data / "summary.json").read_bytes(), b"existing inventory\n")

    def test_mode_and_type_changes_have_numstat_entries(self) -> None:
        (self.repo / "mode").chmod(0o755)
        (self.repo / "type").unlink()
        (self.repo / "type").symlink_to("mode")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "--quiet", "-m", "mode and type")
        data = self.root / "data"
        data.mkdir()
        with patch.object(reproduce, "ROOT", self.repo), patch.object(reproduce, "FORK", "HEAD"), \
                patch.object(reproduce, "DATA", data):
            rows = {row[0]: row for row in reproduce.delta(self.base, "delta.tsv")}
        # Both metadata-only changes retain the blob and still get a zero-count entry.
        self.assertEqual(rows["mode"][1:6], ["M", "0", "0", "100644", "100755"])
        self.assertEqual(rows["type"][1:6], ["T", "0", "0", "100644", "120000"])
        for row in rows.values():
            self.assertEqual(row[6], row[7])


if __name__ == "__main__":
    unittest.main()
