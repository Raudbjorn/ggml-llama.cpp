#!/usr/bin/env python3
"""Check an explicit C++ ABI contract against an unsuppressed abidw corpus.

Manifest: {"libfoo": {"complete_types": ["ns::record"],
                       "opaque_types": ["handle"], "symbols": ["_Z..."]}}
Names must be derived from the baseline headers, not merely from available DWARF.
This checks that manifest, not the completeness of its header inventory. Opaque
names explicitly permit an incomplete declaration; complete records do not.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check(manifest: dict, library: str, root: ET.Element) -> list[str]:
    errors: list[str] = []
    contract = manifest.get(library)
    keys = {"complete_types", "opaque_types", "symbols"}
    if not isinstance(contract, dict) or set(contract) != keys:
        return [f"{library}: expected complete_types, opaque_types, symbols in manifest"]
    for key in keys:
        values = contract[key]
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            return [f"{library}: invalid {key} list"]
        if len(set(values)) != len(values):
            return [f"{library}: duplicate {key} entries"]
    if not any(contract.values()):
        return [f"{library}: empty contract"]
    if set(contract["complete_types"]) & set(contract["opaque_types"]):
        return [f"{library}: conflicting complete/opaque types"]
    if root.tag != "abi-corpus" or root.get("tracking-non-reachable-types") != "yes":
        return [f"{library}: expected unsuppressed --load-all-types corpus"]
    if not root.findall("abi-instr"):
        return [f"{library}: missing debug translation units"]

    names: dict[str, list[ET.Element]] = {}
    ids: dict[str, ET.Element] = {}
    symbols: set[str] = set()
    exports: set[str] = set()

    def visit(node: ET.Element, scope: str = "") -> None:
        name = node.get("name", "")
        if "id" in node.attrib:
            ids[node.attrib["id"]] = node
            if name:
                names.setdefault(scope + name, []).append(node)
        symbol = node.get("elf-symbol-id")
        if symbol:
            symbols.add(symbol)
        if node.tag == "elf-symbol" and node.get("is-defined") == "yes":
            exports.add(name)
        if node.tag in {"namespace-decl", "class-decl", "union-decl"} and name:
            scope += name + "::"
        for child in node:
            visit(child, scope)

    visit(root)
    definitions = {n.get("def-of-decl-id"): n for n in root.iter() if n.get("def-of-decl-id")}

    def underlying(node: ET.Element) -> ET.Element | None:
        seen: set[int] = set()
        while node.tag in {"typedef-decl", "qualified-type-def"} or node.get("is-declaration-only") == "yes":
            if id(node) in seen:
                return None
            seen.add(id(node))
            if node.get("is-declaration-only") == "yes":
                node = definitions.get(node.get("id"))
            else:
                node = ids.get(node.get("type-id"))
            if node is None:
                return None
        return node

    for name in contract["opaque_types"]:
        if name not in names:
            errors.append(f"opaque type absent: {name}")
    for name in contract["complete_types"]:
        candidates = [resolved for n in names.get(name, []) if (resolved := underlying(n)) is not None]
        if not candidates:
            errors.append(f"complete type absent: {name}")
        for node in candidates:
            if node.tag in {"class-decl", "union-decl"}:
                if not node.get("size-in-bits"):
                    errors.append(f"record size absent: {name}")
                for member in node.findall("data-member"):
                    if member.get("static") != "yes" and "layout-offset-in-bits" not in member.attrib:
                        errors.append(f"member offset absent: {name}")
    for symbol in contract["symbols"]:
        if symbol not in exports:
            errors.append(f"required export absent: {symbol}")
        if symbol not in symbols:
            errors.append(f"required export lacks typed declaration: {symbol}")
    # An unresolved edge is incomplete type evidence, even if names are present.
    for node in root.iter():
        for key, value in node.attrib.items():
            if key in {"type-id", "return-type-id", "naming-typedef-id"} and value not in ids:
                errors.append(f"unresolved {key}: {value}")
    return sorted(set(errors))


def check_library_names(manifest: dict, names: list[str]) -> list[str]:
    if not manifest or any(not re.fullmatch(r"lib[A-Za-z0-9_-]+", name) for name in manifest):
        return ["invalid or empty library manifest"]
    if len(names) != len(set(names)):
        return ["duplicate requested library"]
    actual = set(names)
    expected = set(manifest)
    return ([f"missing library dump: {name}" for name in sorted(expected - actual)] +
            [f"unlisted library dump: {name}" for name in sorted(actual - expected)])


def check_libraries(manifest: dict, directory: Path) -> list[str]:
    return check_library_names(manifest, [p.name.removesuffix(".abi.xml")
                                         for p in directory.glob("lib*.abi.xml") if p.is_file()])


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: check-apiabi-types.py MANIFEST LIBRARY CORPUS | MANIFEST --libraries DIRECTORY", file=sys.stderr)
        return 1
    manifest_path, library, corpus_path = sys.argv[1:]
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        if not isinstance(manifest, dict):
            print("ABI coverage manifest must be an object", file=sys.stderr)
            return 1
        if library == "--requested-libraries":
            errors = check_library_names(manifest, corpus_path.split())
        elif library == "--libraries":
            errors = check_libraries(manifest, Path(corpus_path))
        else:
            errors = check(manifest, library, ET.parse(corpus_path).getroot())
    except (OSError, ValueError, ET.ParseError) as error:
        print(f"ABI coverage unavailable: {error}", file=sys.stderr)
        return 1
    for error in errors:
        print(f"ABI coverage incomplete: {error}", file=sys.stderr)
    if not errors:
        print(f"ABI manifest coverage passed: {library}; header inventory completeness requires separate review")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
