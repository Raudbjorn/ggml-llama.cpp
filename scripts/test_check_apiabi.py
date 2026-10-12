"""Fail-closed ABI wrapper and explicit C++ type-coverage fixtures (no compiler)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET

SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("apiabi_types", SCRIPTS / "check-apiabi-types.py")
TYPES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TYPES)
CONTRACT = {"libsample": {"complete_types": ["sample"], "opaque_types": ["handle"], "symbols": ["_Zf"]}}
CORPUS = """<abi-corpus tracking-non-reachable-types='yes'>
<elf-function-symbols><elf-symbol name='_Zf' is-defined='yes'/></elf-function-symbols>
<abi-instr><type-decl name='int' id='int' size-in-bits='32'/>
<class-decl name='sample' id='record' size-in-bits='32'>
<data-member layout-offset-in-bits='0'><var-decl name='n' type-id='int'/></data-member></class-decl>
<class-decl name='handle' id='opaque' is-declaration-only='yes'/>
<function-decl elf-symbol-id='_Zf'><parameter type-id='record'/></function-decl>
</abi-instr></abi-corpus>"""


class TypeCoverageTests(unittest.TestCase):
    def test_complete_and_explicit_opaque_contract(self):
        self.assertEqual(TYPES.check(CONTRACT, "libsample", ET.fromstring(CORPUS)), [])

    def test_missing_or_incomplete_record_fails(self):
        for corpus in (CORPUS.replace("name='sample'", "name='other'"),
                       CORPUS.replace("name='sample'", "is-declaration-only='yes' name='sample'"),
                       CORPUS.replace("size-in-bits='32'", ""),
                       CORPUS.replace("layout-offset-in-bits='0'", "")):
            with self.subTest(corpus=corpus):
                self.assertTrue(TYPES.check(CONTRACT, "libsample", ET.fromstring(corpus)))

    def test_missing_typed_symbol_or_export_fails(self):
        for marker in ("elf-symbol-id='_Zf'", "is-defined='yes'"):
            self.assertTrue(TYPES.check(CONTRACT, "libsample", ET.fromstring(CORPUS.replace(marker, ""))))

    def test_dangling_type_reference_fails(self):
        self.assertTrue(TYPES.check(CONTRACT, "libsample", ET.fromstring(CORPUS.replace("type-id='record'", "type-id='absent'"))))

    def test_incomplete_opaque_is_not_a_complete_record(self):
        contract = {"libsample": {"complete_types": ["handle"], "opaque_types": [], "symbols": []}}
        self.assertTrue(TYPES.check(contract, "libsample", ET.fromstring(CORPUS)))

    def test_alias_requires_complete_underlying_record(self):
        contract = {"libsample": {"complete_types": ["alias"], "opaque_types": [], "symbols": []}}
        for target, valid in (("record", True), ("opaque", False), ("alias", False), ("absent", False)):
            corpus = CORPUS.replace("</abi-instr>",
                f"<typedef-decl name='alias' id='alias' type-id='qualified'/><qualified-type-def id='qualified' type-id='{target}'/></abi-instr>")
            with self.subTest(target=target):
                self.assertEqual(not TYPES.check(contract, "libsample", ET.fromstring(corpus)), valid)

    def test_alias_can_resolve_separate_definition(self):
        contract = {"libsample": {"complete_types": ["alias"], "opaque_types": [], "symbols": []}}
        corpus = CORPUS.replace("</abi-instr>", "<typedef-decl name='alias' id='alias' type-id='opaque'/></abi-instr>")
        corpus = corpus.replace("id='record'", "id='record' def-of-decl-id='opaque'")
        self.assertEqual(TYPES.check(contract, "libsample", ET.fromstring(corpus)), [])

    def test_empty_and_conflicting_manifests_fail(self):
        for contract in ({}, {"complete_types": [], "opaque_types": [], "symbols": []},
                         {"complete_types": ["sample"], "opaque_types": ["sample"], "symbols": []}):
            self.assertTrue(TYPES.check({"libsample": contract}, "libsample", ET.fromstring(CORPUS)))

    def test_symbol_only_or_missing_debug_corpus_fails(self):
        for corpus in ("<abi-corpus/>", CORPUS.replace("tracking-non-reachable-types='yes'", ""),
                       "<abi-corpus tracking-non-reachable-types='yes'><abi-types/></abi-corpus>"):
            self.assertTrue(TYPES.check(CONTRACT, "libsample", ET.fromstring(corpus)))

    def test_namespace_qualified_name(self):
        corpus = CORPUS.replace("<class-decl name='sample'", "<namespace-decl name='ns'><class-decl name='sample'", 1)
        corpus = corpus.replace("</data-member></class-decl>", "</data-member></class-decl></namespace-decl>", 1)
        contract = {"libsample": dict(CONTRACT["libsample"], complete_types=["ns::sample"])}
        self.assertEqual(TYPES.check(contract, "libsample", ET.fromstring(corpus)), [])


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ["PATH"])
        self.tool("abi-compliance-checker", '#!/bin/sh\nexit "${API_EXIT:-0}"\n')
        self.tool("abidw", '#!/bin/sh\necho "abidw: 3.0"\n')
        self.tool("abidiff", '#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_LOG"\nexit "${ABI_EXIT:-0}"\n')
        self.env["ARGS_LOG"] = str(self.root / "args")
        self.old = self.root / "old"
        self.new = self.root / "new"
        for build in (self.old, self.new):
            (build / "apiabi").mkdir(parents=True)
            (build / "apiabi/libsample.abi.xml").write_text(CORPUS)
            (build / "apiabi/api.dump").touch()
            (build / "apiabi/mode").write_text("cxx\n")
            (build / "apiabi/cxx-type-manifest.json").write_text(json.dumps(CONTRACT))

    def tool(self, name, text):
        path = self.bin / name
        path.write_text(text)
        path.chmod(0o755)

    def run_check(self, *flags, cxx=True):
        return subprocess.run(["sh", str(SCRIPTS / "check-apiabi-compat.sh"),
                               *(["--cxx"] if cxx else []), *flags, "--check", str(self.old), str(self.new)],
                              cwd=SCRIPTS.parent, env=self.env, capture_output=True, text=True)

    def test_unchanged_passes_and_cxx_bit4_requires_review(self):
        self.assertEqual(self.run_check().returncode, 0)
        args = (self.root / "args").read_text()
        self.assertIn("--no-default-suppression", args)
        self.assertIn("--non-reachable-types", args)
        self.assertNotIn("--suppressions", args)
        # Real libabigail 3.0 returns bit 4 for an ABI-breaking record offset.
        # A source check can pass that change, so bit 8 alone is insufficient.
        self.env["ABI_EXIT"] = "4"
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_every_error_or_incompatible_bit_fails(self):
        for code in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 127, 139):
            with self.subTest(code=code):
                self.env["ABI_EXIT"] = str(code)
                self.assertNotEqual(self.run_check().returncode, 0)

    def test_api_tool_failure_fails(self):
        self.env["API_EXIT"] = "2"
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_missing_dumps_or_generation_marker_fail(self):
        for path in (self.old / "apiabi/libsample.abi.xml", self.new / "apiabi/libsample.abi.xml",
                     self.old / "apiabi/mode"):
            content = path.read_text()
            path.unlink()
            self.assertNotEqual(self.run_check().returncode, 0)
            path.write_text(content)

    def test_manifest_mismatch_fails(self):
        (self.new / "apiabi/cxx-type-manifest.json").write_text("{}")
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_library_missing_from_both_corpora_fails(self):
        for build in (self.old, self.new):
            (build / "apiabi/cxx-type-manifest.json").write_text(json.dumps(dict(CONTRACT, libmissing=CONTRACT["libsample"])))
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing library dump: libmissing", result.stdout + result.stderr)

    def test_unlisted_or_empty_library_manifest_fails(self):
        for build in (self.old, self.new):
            (build / "apiabi/libextra.abi.xml").write_text(CORPUS)
        self.assertNotEqual(self.run_check().returncode, 0)
        for build in (self.old, self.new):
            (build / "apiabi/cxx-type-manifest.json").write_text("{}")
        self.assertNotEqual(self.run_check().returncode, 0)

    def test_stale_library_cannot_satisfy_partial_generation(self):
        manifest = self.old / "apiabi/cxx-type-manifest.json"
        manifest.write_text(json.dumps(dict(CONTRACT, libmissing=CONTRACT["libsample"])))
        (self.old / "apiabi/libmissing.abi.xml").write_text(CORPUS)
        result = subprocess.run(["sh", str(SCRIPTS / "check-apiabi-compat.sh"), "--cxx",
                                 "--type-manifest", str(manifest), "--generate", str(self.old), "libsample"],
                                cwd=SCRIPTS.parent, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing library dump: libmissing", result.stdout + result.stderr)
        self.assertFalse((self.old / "apiabi/mode").exists())

    def test_cxx_dump_cannot_use_c_mode(self):
        self.assertNotEqual(self.run_check(cxx=False).returncode, 0)

    def test_c_mode_also_fails_tool_errors(self):
        for build in (self.old, self.new):
            (build / "apiabi/mode").write_text("c\n")
        self.env["ABI_EXIT"] = "1"
        self.assertNotEqual(self.run_check(cxx=False).returncode, 0)
        self.env["ABI_EXIT"] = "4"
        self.assertEqual(self.run_check(cxx=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
