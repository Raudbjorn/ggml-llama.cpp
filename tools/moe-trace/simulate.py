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

The dynamic policy models the --moe-cache provider: LRU order, and resident
units with more than --hot-uses uses evicted only when every unit is hot. Each
plan fills at most --inserts-per-plan misses and never evicts a unit it hit or
filled; its other misses stay on the CPU. With the trace's '# expert_tensors='
lines, every expert tensor of a layer is its own plan on the pool of its weight
type and expert size, which holds min(S x member tensors, n_expert) slabs, so
gate, up and down of one type compete for one pool; hit and ins/tok are then
weighted by bytes. Without them, one pool of min(S x n_layers, n_expert)
(layer, expert) units with one plan per layer is an upper bound.

Sessions ('# session=' lines) are replayed in file order and never
interleaved; cache state carries over between sessions as in a long-running
server. Only decode rows (pos >= 0) are scored.

Usage: simulate.py TRACE [TRACE ...] [--profile PROFILE ...] [--budgets ...]
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Hashable, Iterable
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
    widths: dict[int, int]  # routed experts per row, per layer (expert_used_count may vary by layer)
    arch: str | None = None
    n_layer: int | None = None
    # per MoE layer, its expert tensors as (weight type, bytes per expert), from '# expert_tensors='
    tensors: dict[int, list[tuple[str, int]]] = field(default_factory=dict)


@dataclass
class Result:
    hit: float
    inserts_per_step: float


def parse_traces(paths: list[str]) -> tuple[Trace | None, str | None]:
    """Parse one or more trace files; returns (trace, None) or (None, error)."""
    sessions: list[Session] = []
    n_expert: int | None = None
    widths: dict[int, int] = {}
    arch: str | None = None
    n_layer: int | None = None
    tensors: dict[int, list[tuple[str, int]]] = {}
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
                elif key == "arch":
                    # traces of different models cannot be combined: their layer ids do not line up
                    a = value.strip()
                    if arch is not None and a != arch:
                        return None, f"{path}:{lineno}: arch {a} conflicts with {arch}"
                    arch = a
                elif key == "n_layer":
                    try:
                        n = int(value)
                    except ValueError:
                        return None, f"{path}:{lineno}: invalid n_layer '{value}'"
                    if n <= 0 or (n_layer is not None and n != n_layer):
                        return None, f"{path}:{lineno}: n_layer {n} conflicts with {n_layer}"
                    n_layer = n
                elif key == "expert_tensors":
                    parsed = parse_expert_tensors(value.strip())
                    if parsed is None:
                        return None, f"{path}:{lineno}: invalid expert_tensors '{value.strip()}'"
                    layer, layout = parsed
                    if tensors.setdefault(layer, layout) != layout:
                        return None, f"{path}:{lineno}: layer {layer}: expert tensors conflict with an earlier trace"
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
            if widths.setdefault(layer, len(ids)) != len(ids):
                return None, f"{path}:{lineno}: layer {layer}: {len(ids)} ids, earlier rows of the layer have {widths[layer]}"
            (current.prefill if pos < 0 else current.decode)[layer].append((pos, ids))
    if not sessions:
        return None, "no trace rows found"
    if n_layer is not None:
        # a row past the declared layers would add a layer, and dynamic pool units, to the replay
        for s in sessions:
            for layer in sorted(set(s.prefill) | set(s.decode)):
                if layer >= n_layer:
                    return None, f"session {s.sid}: layer {layer} is outside n_layer {n_layer}"
    for s in sessions:
        for rows in s.prefill.values():
            rows.sort(key=lambda r: -r[0])  # pos = -(p + 1): prompt order
        for rows in s.decode.values():
            rows.sort(key=lambda r: r[0])
    return Trace(sessions=sessions, n_expert=n_expert, widths=widths, arch=arch, n_layer=n_layer, tensors=tensors), None


def parse_expert_tensors(value: str) -> tuple[int, list[tuple[str, int]]] | None:
    """'LAYER:TYPE/BYTES,TYPE/BYTES,...' as llama-moe-trace writes it, or None if malformed."""
    layer_text, sep, rest = value.partition(":")
    if not sep or not rest:
        return None
    try:
        layer = int(layer_text)
    except ValueError:
        return None
    layout: list[tuple[str, int]] = []
    for item in rest.split(","):
        ty, sep, size = item.partition("/")
        try:
            nbytes = int(size)
        except ValueError:
            return None
        if not sep or not ty or nbytes <= 0:
            return None
        layout.append((ty, nbytes))
    return (layer, layout) if layer >= 0 else None


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


def decode_events(trace: Trace) -> list[list[tuple[int, Step]]]:
    """Decode steps in file order across layers: per session, per position, (layer, ids)
    for every layer routed there. A session whose layers hold different row counts stays
    aligned by position, unlike the per-layer lists of decode_steps."""
    out: list[list[tuple[int, Step]]] = []
    for s in trace.sessions:
        by_pos: dict[int, list[tuple[int, Step]]] = defaultdict(list)
        for layer in sorted(s.decode):
            for pos, ids in s.decode[layer]:
                by_pos[pos].append((layer, ids))
        out.extend(by_pos[p] for p in sorted(by_pos))
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


def sim_static(decode: dict[int, list[Step]], resident: dict[int, set[int]]) -> Result:
    hits = total = 0
    for layer, steps in decode.items():
        r = resident.get(layer, set())
        for ids in steps:
            for e in ids:
                hits += e in r
                total += 1
    return Result(hits / max(total, 1), 0.0)


def sim_lru(decode: dict[int, list[Step]], slots: int, n_steps: int) -> Result:
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
    return Result(hits / max(total, 1), inserts / max(n_steps, 1))


def sim_lfu_decay(decode: dict[int, list[Step]], slots: int, n_steps: int, alpha: float = 0.95) -> Result:
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
    return Result(hits / max(total, 1), inserts / max(n_steps, 1))


def sim_reelect(trace: Trace, slots: int, interval: int, n_steps: int, alpha: float = 0.98) -> Result:
    """Per-layer decayed counts re-elected every `interval` decode events. Sessions run in file
    order: each session's prompt routing joins the counts, and re-elects, when it is reached.
    The clock counts every decode event of the model, also those where a layer has no row."""
    hits = total = inserts = 0
    layers = sorted({layer for s in trace.sessions for layer in s.decode})
    score: dict[int, dict[int, float]] = {layer: defaultdict(float) for layer in layers}
    cache: dict[int, set[int]] = {layer: set() for layer in layers}

    def elect(layer: int) -> set[int]:
        sc = score[layer]
        return set(sorted(sc, key=lambda k: (-sc[k], k))[:slots])

    step = 0
    for i, s in enumerate(trace.sessions):
        for layer in layers:
            prompt = s.prefill.get(layer, [])
            for _, ids in prompt:
                for e in ids:
                    score[layer][e] += 1.0
            if i == 0 or prompt:
                new = elect(layer)
                if i > 0:
                    inserts += len(new - cache[layer])
                cache[layer] = new
        by_pos: dict[int, list[tuple[int, Step]]] = defaultdict(list)
        for layer in layers:
            for pos, ids in s.decode.get(layer, []):
                by_pos[pos].append((layer, ids))
        for pos in sorted(by_pos):
            for layer, ids in by_pos[pos]:
                for e in ids:
                    hits += e in cache[layer]
                    total += 1
                    score[layer][e] += 1.0
            if step % interval == interval - 1:
                for layer in layers:
                    for k in score[layer]:
                        score[layer][k] *= alpha
                    new = elect(layer)
                    inserts += len(new - cache[layer])
                    cache[layer] = new
            step += 1
    return Result(hits / max(total, 1), inserts / max(n_steps, 1))


PROVIDER_INSERTS_PER_PLAN = 8  # GGML_CUDA_MOE_CACHE_INSERTS default, ggml-moe-cache-common.h


def plan(uses: OrderedDict[Hashable, int], units: int, keys: Iterable[Hashable], hot_uses: int,
         inserts_per_plan: int) -> tuple[int, int]:
    """One provider plan on one pool (uses: LRU order, head = oldest): it fills at most
    inserts_per_plan of its misses, and it cannot evict a unit it hit or filled (the
    provider pins those); its other misses run on the CPU and are not inserted.
    Returns (hits, inserts)."""
    hits = inserts = fills = 0
    pinned: set[Hashable] = set()
    for key in keys:
        if key in uses:
            hits += 1
            uses[key] += 1
            uses.move_to_end(key)
            pinned.add(key)
            continue
        if units <= 0 or fills >= inserts_per_plan:
            continue
        fills += 1  # the provider spends the budget before it looks for a victim
        if len(uses) >= units:
            victim = next((k for k, u in uses.items() if u <= hot_uses and k not in pinned), None)
            if victim is None:
                victim = next((k for k in uses if k not in pinned), None)
            if victim is None:
                continue  # every unit is pinned by this plan
            del uses[victim]
        uses[key] = 0
        pinned.add(key)
        inserts += 1
    return hits, inserts


def sim_dynamic(events: list[list[tuple[int, Step]]], units: int, hot_uses: int,
                inserts_per_plan: int = PROVIDER_INSERTS_PER_PLAN) -> Result:
    """Global LRU pool over (layer, expert) units with the provider's rules, replayed in
    the order of decode_events, one plan per (step, layer). An upper bound when a layer's
    expert tensors share a pool: see sim_dynamic_tensors."""
    hits = total = inserts = 0
    uses: OrderedDict[Hashable, int] = OrderedDict()
    for step in events:
        for layer, ids in step:
            h, i = plan(uses, units, [(layer, e) for e in ids], hot_uses, inserts_per_plan)
            hits += h
            inserts += i
            total += len(ids)
    return Result(hits / max(total, 1), inserts / max(len(events), 1))


def dynamic_pools(tensors: dict[int, list[tuple[str, int]]], layers: Iterable[int], slots: int,
                  n_expert: int) -> dict[tuple[str, int], int]:
    """Slabs of each provider pool: one pool per weight type and expert size, S slabs per
    member tensor, capped at n_expert slabs (moe-cache.cpp)."""
    members = Counter(t for layer in layers for t in tensors[layer])
    return {t: min(slots * n, n_expert) for t, n in members.items()}


def sim_dynamic_tensors(events: list[list[tuple[int, Step]]], tensors: dict[int, list[tuple[str, int]]],
                        pools: dict[tuple[str, int], int], hot_uses: int,
                        inserts_per_plan: int = PROVIDER_INSERTS_PER_PLAN) -> Result:
    """The provider's keys: an entry per (expert tensor, expert), and a plan per MUL_MAT_ID,
    so each expert tensor of a layer is its own plan on the pool of its type and size, in
    the trace's tensor order. Hits are the fraction of routed expert bytes served from the
    pools; inserts are counted in (layer, expert) units of all the layer's tensors."""
    hit_bytes = total_bytes = 0
    inserted = 0.0
    state: dict[tuple[str, int], OrderedDict[Hashable, int]] = {t: OrderedDict() for t in pools}
    for step in events:
        for layer, ids in step:
            layer_bytes = sum(nbytes for _, nbytes in tensors[layer])
            for ti, t in enumerate(tensors[layer]):
                h, i = plan(state[t], pools[t], [(layer, ti, e) for e in ids], hot_uses, inserts_per_plan)
                hit_bytes += h * t[1]
                total_bytes += len(ids) * t[1]
                inserted += i * t[1] / layer_bytes
    return Result(hit_bytes / max(total_bytes, 1), inserted / max(len(events), 1))


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
    ap.add_argument("--inserts-per-plan", type=int, default=PROVIDER_INSERTS_PER_PLAN,
                    help="dynamic policy fills per layer per step, 1..1024 (GGML_CUDA_MOE_CACHE_INSERTS)")
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
    if not 1 <= args.inserts_per_plan <= 1024:
        print("error: --inserts-per-plan must be in 1..1024", file=sys.stderr)
        return 2
    cost_args = (args.slab_mb, args.ram_gbs, args.link_gbs)
    if any(v is not None for v in cost_args) and not all(v is not None and v > 0 for v in cost_args):
        print("error: --slab-mb, --ram-gbs and --link-gbs must all be given and positive", file=sys.stderr)
        return 2

    trace, err = parse_traces(args.trace)
    if trace is None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if args.n_expert is not None and trace.n_expert is not None and args.n_expert != trace.n_expert:
        print(f"error: --n-expert {args.n_expert} conflicts with the trace's n_expert {trace.n_expert}", file=sys.stderr)
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
        for what, ours, theirs in (("arch", trace.arch, profile.arch), ("n_layer", trace.n_layer, profile.n_layer)):
            if ours is not None and theirs is not None and ours != theirs:
                print(f"error: profile {what} {theirs} differs from the trace's {ours}", file=sys.stderr)
                return 2
        err = validate_ids(profile, n_expert)
        if err:
            print(f"error: profile: {err}", file=sys.stderr)
            return 2
        # packs are chosen from decode routing, like the model's profile loader
        if not decode_steps(profile):
            print("error: profile: no decode rows (pos >= 0) to choose a static pack from", file=sys.stderr)
            return 2

    decode = decode_steps(trace)
    events = decode_events(trace)
    prefill = prefill_steps(trace)
    if not decode:
        print("error: no decode rows (pos >= 0) in the evaluated trace", file=sys.stderr)
        return 2
    missing = sorted(set(decode) - set(trace.tensors))
    if trace.tensors and missing:
        print(f"error: layer {missing[0]} routes experts but has no '# expert_tensors=' line", file=sys.stderr)
        return 2

    lo, hi = min(trace.widths.values()), max(trace.widths.values())
    active = str(lo) if lo == hi else f"{lo}-{hi}"
    print(f"trace: {len(trace.sessions)} session(s), {len(decode)} MoE layers, {n_expert} experts, "
          f"{active} active/token, {len(events)} decode steps")
    for line in skew_lines(decode):
        print(line)
    print()

    with_cost = args.slab_mb is not None
    # routed expert reads per decode step, from the rows themselves: a layer may route on fewer steps
    reads_per_step = sum(len(ids) for step in events for _, ids in step) / len(events)

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
        # rates per decode event: a layer may route on fewer steps than the model decodes
        rows.append(("lru", sim_lru(decode, slots, len(events))))
        rows.append(("lfu-decay", sim_lfu_decay(decode, slots, len(events))))
        rows.append((f"reelect-{args.reelect}", sim_reelect(trace, slots, args.reelect, len(events))))
        # the provider shares one pool per expert size and weight type across all layers and caps
        # it at the creating tensor's n_expert slabs (moe-cache.cpp), whatever the budget
        if trace.tensors:
            pools = dynamic_pools(trace.tensors, decode, slots, n_expert)
            rows.append(("dynamic", sim_dynamic_tensors(events, trace.tensors, pools, args.hot_uses,
                                                        args.inserts_per_plan)))
        else:
            units = min(slots * len(decode), n_expert)
            rows.append(("dynamic", sim_dynamic(events, units, args.hot_uses, args.inserts_per_plan)))
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
    if trace.tensors:
        print("dynamic replays each expert tensor as its own plan on the pool of its weight type and size; "
              "hit and ins/tok are weighted by bytes")
    else:
        print("dynamic counts one pool unit per (layer, expert): the trace has no '# expert_tensors=' lines, so it is an "
              "upper bound where gate, up and down of one weight type share a pool")
    print("dynamic assumes the --moe-cache provider takes every MoE layer; it leaves on the CPU expert tensors below its "
          "minimum (logged as min-expert=, GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB) and unsupported weight types, which the "
          "replay does not apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
