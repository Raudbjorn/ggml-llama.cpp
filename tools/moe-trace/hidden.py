#!/usr/bin/env python3
"""Reader for llama-moe-trace --hidden-out files (format v1, stdlib only).

The file starts with text lines "key=value" after the line
"llama-moe-trace hidden v1" and up to the line "end". Records follow:
pos, layer, tag, n as little-endian int32, then n little-endian float32.
Tags: 0 = layer output, 1 = residual after attention, 2 = router input.

    python3 hidden.py FILE           # header and record counts per tag
"""

from __future__ import annotations

import argparse
import array
import struct
import sys
from typing import BinaryIO, Iterator, NamedTuple

MAGIC = b"llama-moe-trace hidden v1\n"
TAGS = {0: "layer_out", 1: "residual_after_attention", 2: "router_input"}
_REC = struct.Struct("<4i")


class Record(NamedTuple):
    pos: int
    layer: int
    tag: int
    values: array.array  # float32


class HiddenFormatError(ValueError):
    pass


def read_header(f: BinaryIO) -> dict[str, str]:
    if f.readline() != MAGIC:
        raise HiddenFormatError("not a llama-moe-trace hidden v1 file")
    header: dict[str, str] = {}
    for _ in range(64):
        line = f.readline()
        if not line.endswith(b"\n"):
            raise HiddenFormatError("truncated header")
        text = line[:-1].decode("utf-8", "replace")
        if text == "end":
            if header.get("byte_order") != "little":
                raise HiddenFormatError(f"unsupported byte_order {header.get('byte_order')!r}")
            return header
        key, sep, value = text.partition("=")
        if not sep:
            raise HiddenFormatError(f"malformed header line {text!r}")
        header[key] = value
    raise HiddenFormatError("header has no end line")


def read_records(f: BinaryIO, n_embd: int) -> Iterator[Record]:
    while True:
        head = f.read(_REC.size)
        if not head:
            return
        if len(head) != _REC.size:
            raise HiddenFormatError("truncated record header")
        pos, layer, tag, n = _REC.unpack(head)
        if n <= 0 or n > n_embd or tag not in TAGS:
            raise HiddenFormatError(f"bad record pos={pos} layer={layer} tag={tag} n={n}")
        values = array.array("f")
        data = f.read(4 * n)
        if len(data) != 4 * n:
            raise HiddenFormatError("truncated record values")
        values.frombytes(data)
        if sys.byteorder != "little":
            values.byteswap()
        yield Record(pos, layer, tag, values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    args = ap.parse_args()
    try:
        with open(args.file, "rb") as f:
            header = read_header(f)
            counts: dict[int, int] = {}
            steps: set[int] = set()
            for rec in read_records(f, int(header["n_embd"])):
                counts[rec.tag] = counts.get(rec.tag, 0) + 1
                steps.add(rec.pos)
    except (OSError, HiddenFormatError, KeyError, ValueError) as e:
        print(f"error: {args.file}: {e}", file=sys.stderr)
        return 1
    for key in ("session", "arch", "n_layer", "n_embd", "n_decode", "model"):
        print(f"{key}: {header.get(key, '')}")
    print(f"decode steps: {len(steps)}")
    for tag, name in TAGS.items():
        print(f"records {name}: {counts.get(tag, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
