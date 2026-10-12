#!/usr/bin/env python3
"""Replay llama-moe-trace CSV files against MoE expert residency policies.

A hit means the routed expert of a layer was resident on the GPU when the
router selected it. Hit rate bounds the routed decode traffic a policy can
move off the CPU; it does not include kernels, activation transfers,
synchronization or graph compilation.

Per-layer policies use S slots per MoE layer (S = budget x n_expert):
  static           top-S experts per layer from a separate --profile trace
                   (deployable, the static expert pack)
  static-oracle    top-S from the evaluated trace itself; optimistic and not
                   deployable, shown only as a reference
  static-prefill   top-S from the evaluated sessions' own prompt routing
  lru              per-layer least recently used
  lfu-decay        per-layer exponentially decayed frequency
  reelect-K        per-layer decayed counts, membership re-elected every K steps

The dynamic policy models the --moe-cache provider: one pool shared by all
layers with S x n_layers (layer, expert) units, LRU order, and resident
experts with more than --hot-uses uses evicted only when every unit is hot.

Sessions ('# session=' lines) are replayed in file order and never
interleaved; cache state carries over between sessions as in a long-running
server. Only decode rows (pos >= 0) are scored.

Usage: simulate.py TRACE [TRACE ...] [--profile PROFILE ...] [--budgets ...]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field

Step = list[int]


@dataclass
class Session:
    sid: str
    prefill: dict[int, list[tuple[int, Step]]] = field(default_factory=lambda: defaultdict(list))
    decode: dict[int, list[tuple[int, Step]]] = field(default_factory=lambda: defaultdict(list))


@dataclass
class Trace:
    sessions: list[Session]
    n_expert: int | None
    n_used: int


@dataclass
class Result:
    hit: float
    inserts_per_step: float


def parse_traces(paths: list[str]) -> tuple[Trace | None, str | None]:
    """Parse one or more trace files; returns (trace, None) or (None, error)."""
    sessions: list[Session] = []
    n_expert: int | None = None
    n_used = 0
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError as e:
            return None, f"{path}: {e.strerror}"
        current: Session | None = None
        for lineno, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, _, value = line[1:].strip().partition("=")
                key = key.strip()
                if key == "session":
                    current = Session(sid=f"{path}:{value.strip()}")
                    sessions.append(current)
                elif key == "kind":
                    # e.g. llama-moe-trace --spec-out: predicted ids, not routed ones
                    return None, f"{path}:{lineno}: a '{value.strip()}' trace holds no routing"
                elif key == "n_expert":
                    try:
                        n = int(value)
                    except ValueError:
                        return None, f"{path}:{lineno}: invalid n_expert '{value}'"
                    if n <= 0 or (n_expert is not None and n != n_expert):
                        return None, f"{path}:{lineno}: n_expert {n} conflicts with {n_expert}"
                    n_expert = n
                continue
            if line.startswith("pos,"):
                continue
            if current is None:
                # traces written before the session header: one session per file
                current = Session(sid=path)
                sessions.append(current)
            fields = line.split(",")
            if len(fields) < 3:
                return None, f"{path}:{lineno}: expected pos,layer,id0,... got '{line}'"
            try:
                pos, layer = int(fields[0]), int(fields[1])
                ids = [int(x) for x in fields[2:]]
            except ValueError:
                return None, f"{path}:{lineno}: non-integer field in '{line}'"
            if layer < 0:
                return None, f"{path}:{lineno}: negative layer {layer}"
            if n_used and len(ids) != n_used:
                return None, f"{path}:{lineno}: {len(ids)} ids, earlier rows have {n_used}"
            n_used = len(ids)
            (current.prefill if pos < 0 else current.decode)[layer].append((pos, ids))
    if not sessions:
        return None, "no trace rows found"
    for s in sessions:
        for rows in s.prefill.values():
            rows.sort(key=lambda r: -r[0])  # pos = -(p + 1): prompt order
        for rows in s.decode.values():
            rows.sort(key=lambda r: r[0])
    return Trace(sessions=sessions, n_expert=n_expert, n_used=n_used), None


def validate_ids(trace: Trace, n_expert: int) -> str | None:
    for s in trace.sessions:
        for part in (s.prefill, s.decode):
            for layer, rows in part.items():
                for pos, ids in rows:
                    for e in ids:
                        if e < 0 or e >= n_expert:
                            return f"session {s.sid}: layer {layer} pos {pos}: expert id {e} outside [0, {n_expert})"
    return None


def decode_steps(trace: Trace) -> dict[int, list[Step]]:
    """Per-layer decode steps, sessions concatenated in file order."""
    out: dict[int, list[Step]] = defaultdict(list)
    for s in trace.sessions:
        for layer, rows in s.decode.items():
            out[layer].extend(ids for _, ids in rows)
    return out


def prefill_steps(trace: Trace) -> dict[int, list[Step]]:
    out: dict[int, list[Step]] = defaultdict(list)
    for s in trace.sessions:
        for layer, rows in s.prefill.items():
            out[layer].extend(ids for _, ids in rows)
    return out


def top_by_freq(steps_by_layer: dict[int, list[Step]], slots: int) -> dict[int, set[int]]:
    resident: dict[int, set[int]] = {}
    for layer, steps in steps_by_layer.items():
        freq: dict[int, int] = defaultdict(int)
        for ids in steps:
            for e in ids:
                freq[e] += 1
        # ties break on the lower expert id, like the C++ profile loader
        ranked = sorted(freq, key=lambda e: (-freq[e], e))
        resident[layer] = set(ranked[:slots])
    return resident


def n_decode_steps(decode: dict[int, list[Step]]) -> int:
    return max((len(s) for s in decode.values()), default=0)


def sim_static(decode: dict[int, list[Step]], resident: dict[int, set[int]]) -> Result:
    hits = total = 0
    for layer, steps in decode.items():
        r = resident.get(layer, set())
        for ids in steps:
            for e in ids:
                hits += e in r
                total += 1
    return Result(hits / max(total, 1), 0.0)


def sim_lru(decode: dict[int, list[Step]], slots: int) -> Result:
    hits = total = inserts = 0
    for steps in decode.values():
        cache: OrderedDict[int, None] = OrderedDict()
        for ids in steps:
            for e in ids:
                total += 1
                if e in cache:
                    hits += 1
                    cache.move_to_end(e)
                    continue
                if len(cache) >= slots:
                    cache.popitem(last=False)
                cache[e] = None
                inserts += 1
    return Result(hits / max(total, 1), inserts / max(n_decode_steps(decode), 1))


def sim_lfu_decay(decode: dict[int, list[Step]], slots: int, alpha: float = 0.95) -> Result:
    hits = total = inserts = 0
    for steps in decode.values():
        score: dict[int, float] = defaultdict(float)
        cache: set[int] = set()
        for ids in steps:
            for k in score:
                score[k] *= alpha
            for e in ids:
                total += 1
                score[e] += 1.0
                if e in cache:
                    hits += 1
                elif len(cache) < slots:
                    cache.add(e)
                    inserts += 1
                else:
                    victim = min(cache, key=lambda k: (score[k], k))
                    if score[e] >= score[victim]:
                        cache.discard(victim)
                        cache.add(e)
                        inserts += 1
    return Result(hits / max(total, 1), inserts / max(n_decode_steps(decode), 1))


def sim_reelect(decode: dict[int, list[Step]], prefill: dict[int, list[Step]], slots: int, interval: int,
                alpha: float = 0.98) -> Result:
    hits = total = inserts = 0
    for layer, steps in decode.items():
        score: dict[int, float] = defaultdict(float)
        for ids in prefill.get(layer, []):
            for e in ids:
                score[e] += 1.0
        cache = set(sorted(score, key=lambda k: (-score[k], k))[:slots])
        for step, ids in enumerate(steps):
            for e in ids:
                hits += e in cache
                total += 1
                score[e] += 1.0
            if step % interval == interval - 1:
                for k in score:
                    score[k] *= alpha
                new = set(sorted(score, key=lambda k: (-score[k], k))[:slots])
                inserts += len(new - cache)
                cache = new
    return Result(hits / max(total, 1), inserts / max(n_decode_steps(decode), 1))


def sim_dynamic(decode: dict[int, list[Step]], units: int, hot_uses: int) -> Result:
    """Global LRU pool over (layer, expert) units with the provider's hot rule."""
    hits = total = inserts = 0
    uses: OrderedDict[tuple[int, int], int] = OrderedDict()  # LRU order, head = oldest
    layers = sorted(decode)
    n_steps = n_decode_steps(decode)
    for step in range(n_steps):
        for layer in layers:
            steps = decode[layer]
            if step >= len(steps):
                continue
            for e in steps[step]:
                key = (layer, e)
                total += 1
                if key in uses:
                    hits += 1
                    uses[key] += 1
                    uses.move_to_end(key)
                    continue
                if units <= 0:
                    continue
                if len(uses) >= units:
                    victim = next((k for k, u in uses.items() if u <= hot_uses), None)
                    if victim is None:
                        victim = next(iter(uses))
                    del uses[victim]
                uses[key] = 0
                inserts += 1
    return Result(hits / max(total, 1), inserts / max(n_steps, 1))


def skew_lines(decode: dict[int, list[Step]]) -> list[str]:
    freq: dict[tuple[int, int], int] = defaultdict(int)
    for layer, steps in decode.items():
        for ids in steps:
            for e in ids:
                freq[(layer, e)] += 1
    ranked = sorted(freq.values(), reverse=True)
    total = sum(ranked)
    if total == 0:
        return []
    out = []
    for frac in (0.1, 0.25, 0.5):
        k = max(1, int(len(ranked) * frac))
        out.append(f"top {frac:>4.0%} of observed (layer, expert) pairs carry {sum(ranked[:k]) / total:.1%} of decode routing")
    return out


def parse_budgets(text: str) -> tuple[list[float], str | None]:
    try:
        values = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        return [], f"invalid --budgets '{text}'"
    if not values or any(not (0.0 < v <= 1.0) or math.isnan(v) for v in values):
        return [], f"--budgets must be fractions in (0, 1], got '{text}'"
    return values, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="+", help="trace(s) to evaluate")
    ap.add_argument("--profile", nargs="+", default=[], help="separate trace(s) that select the static pack")
    ap.add_argument("--n-expert", type=int, default=None, help="experts per layer when traces lack '# n_expert='")
    ap.add_argument("--budgets", default="0.0625,0.125,0.25,0.5", help="slots per layer as a fraction of n_expert")
    ap.add_argument("--reelect", type=int, default=32, help="re-election interval in decode steps")
    ap.add_argument("--hot-uses", type=int, default=4, help="dynamic policy hot threshold (GGML_CUDA_MOE_CACHE_HOT_USES)")
    ap.add_argument("--slab-mb", type=float, default=None,
                    help="MiB per (layer, expert) gate+up+down; enables the assumed cost columns")
    ap.add_argument("--ram-gbs", type=float, default=None, help="assumed host read bandwidth, GB/s")
    ap.add_argument("--link-gbs", type=float, default=None, help="assumed host-to-device bandwidth, GB/s")
    args = ap.parse_args(argv)

    budgets, err = parse_budgets(args.budgets)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.reelect <= 0:
        print("error: --reelect must be positive", file=sys.stderr)
        return 2
    if args.hot_uses < 0:
        print("error: --hot-uses must be >= 0", file=sys.stderr)
        return 2
    cost_args = (args.slab_mb, args.ram_gbs, args.link_gbs)
    if any(v is not None for v in cost_args) and not all(v is not None and v > 0 for v in cost_args):
        print("error: --slab-mb, --ram-gbs and --link-gbs must all be given and positive", file=sys.stderr)
        return 2

    trace, err = parse_traces(args.trace)
    if trace is None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    n_expert = args.n_expert if args.n_expert is not None else trace.n_expert
    if n_expert is None or n_expert <= 0:
        print("error: the trace has no '# n_expert=' line; pass --n-expert", file=sys.stderr)
        return 2
    err = validate_ids(trace, n_expert)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    profile: Trace | None = None
    if args.profile:
        profile, err = parse_traces(args.profile)
        if profile is None:
            print(f"error: profile: {err}", file=sys.stderr)
            return 2
        if profile.n_expert is not None and profile.n_expert != n_expert:
            print(f"error: profile n_expert {profile.n_expert} differs from {n_expert}", file=sys.stderr)
            return 2
        err = validate_ids(profile, n_expert)
        if err:
            print(f"error: profile: {err}", file=sys.stderr)
            return 2

    decode = decode_steps(trace)
    prefill = prefill_steps(trace)
    if not decode:
        print("error: no decode rows (pos >= 0) in the evaluated trace", file=sys.stderr)
        return 2

    print(f"trace: {len(trace.sessions)} session(s), {len(decode)} MoE layers, {n_expert} experts, "
          f"{trace.n_used} active/token, {n_decode_steps(decode)} decode steps")
    for line in skew_lines(decode):
        print(line)
    print()

    with_cost = args.slab_mb is not None
    reads_per_step = sum(len(steps[0]) for steps in decode.values() if steps)

    hdr = f"{'budget':>7} {'slots':>5} | {'policy':>14} {'hit':>7} {'ins/tok':>8}"
    if with_cost:
        hdr += f" {'miss-ms*':>9} {'up-ms*':>7}"
    print(hdr)
    print("-" * len(hdr))
    for frac in budgets:
        slots = max(1, int(n_expert * frac))
        rows: list[tuple[str, Result | None]] = []
        if profile is not None:
            rows.append(("static", sim_static(decode, top_by_freq(decode_steps(profile), slots))))
        rows.append(("static-oracle", sim_static(decode, top_by_freq(decode, slots))))
        rows.append(("static-prefill", sim_static(decode, top_by_freq(prefill, slots)) if prefill else None))
        rows.append(("lru", sim_lru(decode, slots)))
        rows.append(("lfu-decay", sim_lfu_decay(decode, slots)))
        rows.append((f"reelect-{args.reelect}", sim_reelect(decode, prefill, slots, args.reelect)))
        rows.append(("dynamic", sim_dynamic(decode, slots * len(decode), args.hot_uses)))
        for name, res in rows:
            if res is None:
                print(f"{frac:>7.4f} {slots:>5} | {name:>14} {'n/a':>7}")
                continue
            line = f"{frac:>7.4f} {slots:>5} | {name:>14} {res.hit:>7.1%} {res.inserts_per_step:>8.2f}"
            if with_cost:
                miss_mb = reads_per_step * (1.0 - res.hit) * args.slab_mb
                up_mb = res.inserts_per_step * args.slab_mb
                line += f" {miss_mb / args.ram_gbs:>9.2f} {up_mb / args.link_gbs:>7.2f}"
            print(line)
        print()
    if with_cost:
        print("* assumed cost model from --slab-mb/--ram-gbs/--link-gbs; excludes kernels, "
              "activation transfers, synchronization and compilation")
    print("static-oracle selects residents from the scored trace itself and is optimistic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
