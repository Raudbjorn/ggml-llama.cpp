#!/usr/bin/env python3
"""Oracle for llama-moe-trace --spec-router and --hidden-out on an F32 fixture.

Recomputes on the host, from the dumped hidden states and the model's weights:
- every decode step's routed experts from the router input (tag 2), against -o;
- the pre-attention router of layer L >= 1 from layer L-1's output (tag 0), against
  --spec-out (layer 0's input is the token embedding, which is not dumped).
A set mismatch is accepted only when the k-th and (k+1)-th logits tie within 1e-4
relative. Needs numpy and the repository's gguf-py.

usage: moe_trace_oracle.py MODEL.gguf TRACE.csv SPEC.csv HIDDEN K
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gguf-py"))
sys.path.insert(0, os.path.join(HERE, "..", "tools", "moe-trace"))

import gguf  # noqa: E402
import hidden  # noqa: E402

ROUTER_NORM = {"qwen3moe": "ffn_norm", "qwen35moe": "post_attention_norm"}


def decode_rows(path: str) -> dict[tuple[int, int], list[int]]:
    rows: dict[tuple[int, int], list[int]] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith("pos,"):
                continue
            parts = [int(x) for x in line.split(",")]
            if parts[0] >= 0:
                rows[(parts[0], parts[1])] = parts[2:]
    return rows


def same_top(logits: np.ndarray, got: list[int]) -> bool:
    k = len(got)
    order = np.argsort(-logits, kind="stable")
    if set(order[:k].tolist()) == set(got):
        return True
    if k >= len(logits):
        return False
    a, b = logits[order[k - 1]], logits[order[k]]
    return abs(a - b) <= 1e-4 * max(abs(a), abs(b), 1e-6)


def main() -> int:
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    model, trace_csv, spec_csv, hidden_path, k = sys.argv[1:5] + [int(sys.argv[5])]

    reader = gguf.GGUFReader(model)
    fields = reader.fields
    arch = bytes(fields["general.architecture"].parts[-1]).decode()
    eps = float(fields[f"{arch}.attention.layer_norm_rms_epsilon"].parts[-1][0])
    tensors = {t.name: t for t in reader.tensors}

    def weight(name: str) -> np.ndarray:
        t = tensors[name]
        if t.tensor_type != gguf.GGMLQuantizationType.F32:
            raise SystemExit(f"{name}: the oracle needs F32 weights")
        return np.asarray(t.data, dtype=np.float64)

    with open(hidden_path, "rb") as f:
        header = hidden.read_header(f)
        recs = {(r.pos, r.layer, r.tag): np.asarray(r.values, dtype=np.float64)
                for r in hidden.read_records(f, int(header["n_embd"]))}

    actual = decode_rows(trace_csv)
    spec = decode_rows(spec_csv)
    n_route = n_spec = 0
    failures = []
    for (pos, layer, tag), x in sorted(recs.items()):
        gate = weight(f"blk.{layer}.ffn_gate_inp.weight")  # [n_expert, n_embd]
        if tag == 2 and (pos, layer) in actual:
            n_route += 1
            if not same_top(gate @ x, actual[(pos, layer)]):
                failures.append(f"router input of step {pos} layer {layer} does not give {actual[(pos, layer)]}")
        nxt = layer + 1
        if tag == 0 and (pos, nxt) in spec:
            w = weight(f"blk.{nxt}.{ROUTER_NORM[arch]}.weight")
            xn = x / np.sqrt(np.mean(x * x) + eps) * w
            n_spec += 1
            if not same_top(weight(f"blk.{nxt}.ffn_gate_inp.weight") @ xn, spec[(pos, nxt)]):
                failures.append(f"layer {layer} output does not give the spec ids {spec[(pos, nxt)]} of layer {nxt}")
            if len(spec[(pos, nxt)]) != k:
                failures.append(f"step {pos} layer {nxt}: {len(spec[(pos, nxt)])} spec ids, expected {k}")
    if n_route == 0 or n_spec == 0:
        failures.append(f"nothing to check ({n_route} router, {n_spec} spec records)")
    for msg in failures:
        print(f"FAIL: {msg}", file=sys.stderr)
    if failures:
        return 1
    print(f"ok: {arch}: {n_route} routed and {n_spec} spec decode rows match the host recomputation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
