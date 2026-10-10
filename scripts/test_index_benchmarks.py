#!/usr/bin/env python3
"""Checks for archive inventory coverage and byte-identity semantics."""

import csv
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from index_benchmarks import inventory


class InventoryTests(unittest.TestCase):
    def test_hidden_artifacts_and_duplicates_keep_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "case").mkdir()
            (root / "oneapi-ab").mkdir()
            (root / "README.md").write_text("guide, excluded")
            (root / "FILES.tsv").write_text("old inventory, excluded")
            (root / "AGENTS.md").write_text("archive guide, excluded")
            (root / "case/README.md").write_text("archived report, included")
            (root / "case/result.json").write_bytes(b"same")
            (root / "oneapi-ab/result.json").write_bytes(b"same")
            (root / ".hidden.log").write_bytes(b"")
            rows = {str(row["path"]): row for row in inventory(root)}
            self.assertEqual(set(rows), {"case/README.md", "case/result.json",
                                         "oneapi-ab/result.json", ".hidden.log"})
            self.assertEqual(rows["case/result.json"]["sha256"],
                             hashlib.sha256(b"same").hexdigest())
            self.assertEqual(rows["oneapi-ab/result.json"]["identical_to"], "case/result.json")
            self.assertEqual(rows["case/result.json"]["bytes"], 4)
            self.assertEqual(rows[".hidden.log"]["bytes"], 0)

    def test_links_are_recorded_without_reading_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outside").symlink_to("/does/not/exist")
            (root / "loop").symlink_to(root, target_is_directory=True)
            rows = list(inventory(root))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["kind"] == "symlink" and not row["sha256"] for row in rows))
            self.assertEqual(rows[1]["link_target"], "/does/not/exist")


REPO = Path(__file__).resolve().parent.parent
LFS_POINTER = "version https://git-lfs.github.com/spec/v1"


def git(*args: str, stdin: str = "") -> list[str]:
    done = subprocess.run(["git", "-C", str(REPO), *args], input=stdin, capture_output=True,
                          text=True, check=True)
    return done.stdout.splitlines()


class CommittedArchiveTests(unittest.TestCase):
    """The committed inventory and LFS pointers must be reproducible from a clone."""

    def setUp(self) -> None:
        try:
            self.tracked = set(git("ls-files", "docs/benchmarks"))
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("not a git checkout")
        if not self.tracked:
            self.skipTest("benchmark archive is not tracked")

    def test_every_inventory_path_is_tracked(self) -> None:
        with (REPO / "docs/benchmarks/FILES.tsv").open(newline="") as stream:
            paths = [row["path"] for row in csv.DictReader(stream, delimiter="\t")]
        missing = [p for p in paths if f"docs/benchmarks/{p}" not in self.tracked]
        self.assertEqual(missing[:5], [], f"{len(missing)} FILES.tsv paths are not in the Git tree")

    def test_every_lfs_pointer_has_the_lfs_attribute(self) -> None:
        pointers = [line.split(":", 2)[1] for line in git(
            "grep", "-l", "-e", f"^{LFS_POINTER}$", "HEAD", "--", ".", ":!*.md")]
        self.assertTrue(pointers)
        attrs = git("check-attr", "filter", "--stdin", stdin="\n".join(pointers) + "\n")
        bad = [line for line in attrs if not line.endswith(": lfs")]
        self.assertEqual(bad[:5], [], f"{len(bad)} LFS pointers lack filter=lfs in .gitattributes")


if __name__ == "__main__":
    unittest.main()
