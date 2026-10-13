"""Tests for tools/moe-trace/simulate.py (stdlib only; runs under unittest or pytest)."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from collections import OrderedDict
from collections.abc import Hashable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "moe-trace"))

import simulate  # noqa: E402


def write(dirpath: str, name: str, text: str) -> str:
    path = os.path.join(dirpath, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = simulate.main(argv)
    return code, out.getvalue(), err.getvalue()


HEADER = "# llama-moe-trace v1\n# session={sid}\n# n_expert=4\npos,layer,ids...\n"


class ParseTests(unittest.TestCase):
    def test_same_expert_id_in_two_layers_stays_distinct(self) -> None:
        # expert 0 is hot in layer 0 and never used in layer 1
        rows = "".join(f"{s},0,0,1\n{s},1,2,3\n" for s in range(8))
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "t.csv", HEADER.format(sid="a") + rows)])
        self.assertIsNone(err)
        assert trace is not None
        decode = simulate.decode_steps(trace)
        resident = simulate.top_by_freq(decode, 2)
        self.assertEqual(resident[0], {0, 1})
        self.assertEqual(resident[1], {2, 3})
        # four observed (layer, expert) pairs with equal traffic: the top 25 % carry 25 %
        lines = simulate.skew_lines(decode)
        self.assertIn("top  25% of observed (layer, expert) pairs carry 25.0%", lines[1])

    def test_expert_tensors_lines(self) -> None:
        line = "# expert_tensors=1:q4_K/100,q4_K/100,q6_K/150\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "t.csv", HEADER.format(sid="a") + line + "0,1,0,1\n")])
            self.assertIsNone(err)
            assert trace is not None
            self.assertEqual(trace.tensors, {1: [("q4_K", 100), ("q4_K", 100), ("q6_K", 150)]})
            self.assertEqual(trace.dims, {})
            for bad in ("1", "1:", "x:f32/1", "-1:f32/1", "1:f32", "1:f32/0", "1:/4", "1:f32/1,", "1:f32/1/2",
                        "1:f32/1/2/0", "1:f32/1/2/3,f32/1"):
                with self.subTest(bad=bad):
                    text = HEADER.format(sid="a") + f"# expert_tensors={bad}\n0,1,0,1\n"
                    trace, err = simulate.parse_traces([write(d, "bad.csv", text)])
                    self.assertIsNone(trace)
                    self.assertIn("invalid expert_tensors", err or "")
            # two traces of one model agree; another layout for the same layer is an error
            a = write(d, "a.csv", HEADER.format(sid="a") + line + "0,1,0,1\n")
            b = write(d, "b.csv", HEADER.format(sid="b") + "# expert_tensors=1:f32/100\n0,1,0,1\n")
            self.assertIsNone(simulate.parse_traces([a, a])[1])
            trace, err = simulate.parse_traces([a, b])
            self.assertIsNone(trace)
            self.assertIn("expert tensors conflict", err or "")

    def test_expert_tensors_lines_with_widths(self) -> None:
        line = "# expert_tensors=1:q4_K/100/256/64,q6_K/150/64/256\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "t.csv", HEADER.format(sid="a") + line + "0,1,0,1\n")])
            self.assertIsNone(err)
            assert trace is not None
            self.assertEqual(trace.tensors, {1: [("q4_K", 100), ("q6_K", 150)]})
            self.assertEqual(trace.dims, {1: [(256, 64), (64, 256)]})
            other = HEADER.format(sid="b") + "# expert_tensors=1:q4_K/100/256/32,q6_K/150/64/256\n0,1,0,1\n"
            trace, err = simulate.parse_traces([write(d, "t.csv", HEADER.format(sid="a") + line + "0,1,0,1\n"),
                                                write(d, "o.csv", other)])
            self.assertIsNone(trace)
            self.assertIn("expert tensors conflict", err or "")

    def test_spec_router_trace_is_refused(self) -> None:
        text = HEADER.format(sid="a").replace("# n_expert", "# kind=spec-router k=2\n# n_expert", 1) + "0,0,0,1\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "s.csv", text)])
        self.assertIsNone(trace)
        assert err is not None
        self.assertIn("holds no routing", err)

    def test_sessions_are_not_interleaved(self) -> None:
        a = HEADER.format(sid="a") + "0,0,0,1\n1,0,0,1\n"
        b = HEADER.format(sid="b") + "0,0,2,3\n1,0,2,3\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "a.csv", a), write(d, "b.csv", b)])
        self.assertIsNone(err)
        assert trace is not None
        self.assertEqual(len(trace.sessions), 2)
        # sorting by position across sessions would give [0,1],[2,3],[0,1],[2,3]
        self.assertEqual(simulate.decode_steps(trace)[0], [[0, 1], [0, 1], [2, 3], [2, 3]])

    def test_prefill_rows_keep_prompt_order(self) -> None:
        # pos = -(p + 1): -1 is prompt token 0
        text = HEADER.format(sid="a") + "-2,0,1,2\n-1,0,3,0\n-3,0,0,1\n0,0,0,1\n"
        with tempfile.TemporaryDirectory() as d:
            trace, _ = simulate.parse_traces([write(d, "t.csv", text)])
        assert trace is not None
        self.assertEqual(simulate.prefill_steps(trace)[0], [[3, 0], [1, 2], [0, 1]])

    def test_header_without_session_counts_as_one_session(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "t.csv", "0,0,1,2\n1,0,1,3\n")])
        self.assertIsNone(err)
        assert trace is not None
        self.assertEqual(len(trace.sessions), 1)
        self.assertIsNone(trace.n_expert)

    def test_malformed_rows_are_errors(self) -> None:
        cases = {
            "short": "0,0\n",
            "text": "0,x,1,2\n",
            "width": "0,0,1,2\n1,0,1\n",
            "layer": "0,-1,1,2\n",
            "n_expert": "# n_expert=abc\n0,0,1,2\n",
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in cases.items():
                with self.subTest(case=name):
                    trace, err = simulate.parse_traces([write(d, f"{name}.csv", text)])
                    self.assertIsNone(trace)
                    self.assertIsNotNone(err)

    def test_routing_width_may_differ_between_layers(self) -> None:
        # a per-layer expert_used_count: layer 0 routes 2 experts, layer 1 routes 3
        text = HEADER.format(sid="w") + "0,0,0,1\n0,1,0,1,2\n1,0,2,3\n1,1,1,2,3\n"
        with tempfile.TemporaryDirectory() as d:
            path = write(d, "w.csv", text)
            trace, err = simulate.parse_traces([path])
            self.assertIsNone(err)
            assert trace is not None
            self.assertEqual(trace.widths, {0: 2, 1: 3})
            code, out, err = run([path, "--budgets", "0.5"])
            self.assertEqual(code, 0, err)
            self.assertIn("2-3 active/token", out)
            # within one layer the width stays fixed
            trace, err = simulate.parse_traces([write(d, "x.csv", text + "2,1,0,1\n")])
            self.assertIsNone(trace)
            assert err is not None
            self.assertIn("layer 1", err)

    def test_prompt_header_sets_the_prompt_call_size(self) -> None:
        head = HEADER.format(sid="c") + "# n_prompt=6 n_predict=2 n_ctx=64 n_batch=8 n_ubatch=4\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "c.csv", head + "0,0,0,1\n"),
                                                write(d, "o.csv", HEADER.format(sid="o") + "0,0,0,1\n")])
            self.assertIsNone(err)
            assert trace is not None
            self.assertEqual([s.chunk for s in trace.sessions], [4, None])
            bad = head.replace("n_ubatch=4", "n_ubatch=x")
            trace, err = simulate.parse_traces([write(d, "b.csv", bad + "0,0,0,1\n")])
        self.assertIsNone(trace)
        assert err is not None
        self.assertIn("invalid prompt sizes", err)

    def test_missing_file_is_an_error(self) -> None:
        trace, err = simulate.parse_traces(["/nonexistent/trace.csv"])
        self.assertIsNone(trace)
        self.assertIn("/nonexistent/trace.csv", err or "")


class PolicyTests(unittest.TestCase):
    def test_held_out_profile_differs_from_oracle(self) -> None:
        evaluated = {0: [[2, 3]] * 4}
        profile = {0: [[0, 1]] * 4}
        self.assertEqual(simulate.sim_static(evaluated, simulate.top_by_freq(profile, 2)).hit, 0.0)
        self.assertEqual(simulate.sim_static(evaluated, simulate.top_by_freq(evaluated, 2)).hit, 1.0)

    def test_lru_evicts_least_recent(self) -> None:
        res = simulate.sim_lru({0: [[0], [1], [0], [2], [1]]}, 2, n_steps=5)
        # hits: third access of 0 only; 2 evicts 1, so the final 1 misses
        self.assertAlmostEqual(res.hit, 1 / 5)

    def test_dynamic_keeps_hot_units(self) -> None:
        # (0,0) used five times becomes hot (uses > 1); a stream of cold experts
        # must evict the other unit, not the hot one
        steps = [[0]] * 6 + [[1], [2], [3], [0]]
        res = simulate.sim_dynamic([[(0, ids)] for ids in steps], units=2, hot_uses=1)
        self.assertAlmostEqual(res.hit, 6 / 10)

    def test_dynamic_pool_is_shared_across_layers(self) -> None:
        # one unit total: alternating layers evict each other
        res = simulate.sim_dynamic([[(0, [0]), (1, [0])]] * 3, units=1, hot_uses=4)
        self.assertEqual(res.hit, 0.0)

    def test_dynamic_fills_at_most_inserts_per_plan(self) -> None:
        # 16 unseen experts twice in a row: the provider fills 8 per (step, layer) plan,
        # so the second step hits only the first 8 and fills the other 8
        steps = [[(0, list(range(16)))]] * 2
        res = simulate.sim_dynamic(steps, units=32, hot_uses=4, inserts_per_plan=8)
        self.assertAlmostEqual(res.hit, 8 / 32)
        self.assertAlmostEqual(res.inserts_per_step, 16 / 2)
        # the default is the provider's (GGML_CUDA_MOE_CACHE_INSERTS = 8)
        self.assertAlmostEqual(simulate.sim_dynamic(steps, units=32, hot_uses=4).hit, 8 / 32)
        res = simulate.sim_dynamic(steps, units=32, hot_uses=4, inserts_per_plan=16)
        self.assertAlmostEqual(res.hit, 16 / 32)

    def test_dynamic_plan_never_evicts_its_own_units(self) -> None:
        # one unit: the plan fills expert 0 and pins it, so expert 1 runs on the CPU
        # and the next step still hits 0
        res = simulate.sim_dynamic([[(0, [0, 1])], [(0, [0])]], units=1, hot_uses=4)
        self.assertAlmostEqual(res.hit, 1 / 3)
        self.assertAlmostEqual(res.inserts_per_step, 1 / 2)

    def test_lfu_decay_counts_events_without_the_layer(self) -> None:
        # one slot. Layer 0 routes expert 0 twice (score 1.95), then sits out 13 events that layer 1
        # routes. Decayed by those 13 events and the step itself, expert 0 scores 1.95 x 0.95^14 =
        # 0.95 < 1, so expert 1 replaces it at once and hits next step. A per-layer clock decays
        # once (1.85) and admits expert 1 a step later, without the hit.
        events = [[(0, [0])]] * 2 + [[(1, [5])]] * 13 + [[(0, [1])]] * 2
        res = simulate.sim_lfu_decay(events, 1)
        self.assertAlmostEqual(res.hit, 14 / 17)
        self.assertAlmostEqual(res.inserts_per_step, 3 / 17)

    def test_dynamic_tensors_of_one_type_share_the_pool(self) -> None:
        # experts 0 and 1 on every step, S = 2 of 4 experts. Gate, up and down of one type and
        # size share one pool of min(2 x 3, 4) = 4 slabs, too few for 2 experts x 3 tensors:
        # each plan evicts the slabs of the plans before it, so nothing ever hits.
        steps = [[(0, [0, 1])]] * 8
        shared = {0: [("f32", 100)] * 3}
        pools = simulate.dynamic_pools(shared, [0], 2, 4, min_slots=1)
        self.assertEqual(pools, {("f32", 100): 4})
        self.assertEqual(simulate.sim_dynamic_tensors(steps, shared, pools, hot_uses=4).hit, 0.0)
        # one unit per (layer, expert) holds both experts: the upper bound
        self.assertAlmostEqual(simulate.sim_dynamic(steps, units=2, hot_uses=4).hit, 14 / 16)

    def test_dynamic_tensors_of_two_types_use_two_pools(self) -> None:
        # gate and up share a pool of 4 slabs, down has its own of 2: both experts stay resident.
        # Step 0 fills 2 slabs of each tensor, 2 (layer, expert) units over 8 steps.
        steps = [[(0, [0, 1])]] * 8
        split = {0: [("q4_K", 100), ("q4_K", 100), ("q6_K", 150)]}
        # budget 2 x (100 + 100 + 150) = 700 bytes: q4_K takes min(7, 4) slabs, q6_K the 300 left
        pools = simulate.dynamic_pools(split, [0], 2, 4, min_slots=1)
        self.assertEqual(pools, {("q4_K", 100): 4, ("q6_K", 150): 2})
        res = simulate.sim_dynamic_tensors(steps, split, pools, hot_uses=4)
        self.assertAlmostEqual(res.hit, 7 / 8)
        self.assertAlmostEqual(res.inserts_per_step, 2 / 8)

    def test_dynamic_pools_share_one_byte_budget(self) -> None:
        # two layers with one tensor each, of equal size but different types, S = 64 of 128: the
        # first pool met takes the whole budget of 128 experts, the second is not created
        tensors = {0: [("q4_K", 100)], 1: [("q5_K", 100)]}
        pools = simulate.dynamic_pools(tensors, [0, 1], 64, 128)
        self.assertEqual(pools, {("q4_K", 100): 128, ("q5_K", 100): 0})
        steps = [[(0, [0, 1]), (1, [0, 1])]] * 4
        res = simulate.sim_dynamic_tensors(steps, tensors, pools, hot_uses=4)
        self.assertAlmostEqual(res.hit, 3 / 8)  # layer 0 hits from step 1, layer 1 never

    def test_vulkan_pools_each_take_the_whole_budget(self) -> None:
        # the Vulkan provider passes the configured budget to every new pool: both get 128
        tensors = {0: [("q4_K", 100)], 1: [("q5_K", 100)]}
        pools = simulate.dynamic_pools(tensors, [0, 1], 64, 128, shared=False)
        self.assertEqual(pools, {("q4_K", 100): 128, ("q5_K", 100): 128})
        steps = [[(0, [0, 1]), (1, [0, 1])]] * 4
        self.assertAlmostEqual(simulate.sim_dynamic_tensors(steps, tensors, pools, hot_uses=4).hit, 6 / 8)

    def test_dynamic_pools_below_the_provider_minimum_are_not_created(self) -> None:
        # S = 30 of 128 is a budget of 30 x (100 + 50) = 4500 bytes: 45 f32 slabs, below the
        # provider's 64, so that pool is not created and the f16 pool met later gets 90
        tensors = {0: [("f32", 100)], 1: [("f16", 50)]}
        self.assertEqual(simulate.dynamic_pools(tensors, [0, 1], 30, 128),
                         {("f32", 100): 0, ("f16", 50): 90})

    def test_sycl_pools_hold_back_the_scratch(self) -> None:
        # 64 slabs of 1000 bytes fit a budget of 64 x 1000 exactly; the SYCL provider first holds
        # back the scratch of the plan, 64 x (1 x 4 + 1 x 36) = 2560 bytes for 32 inputs and one
        # output, which leaves 61 slabs, below the 64-slot minimum. Vulkan reserves nothing.
        self.assertEqual(simulate.sycl_scratch_bytes(32, 1), 2560)
        tensors = {0: [("f32", 1000)]}
        self.assertEqual(simulate.dynamic_pools(tensors, [0], 64, 128), {("f32", 1000): 64})
        self.assertEqual(simulate.dynamic_pools(tensors, [0], 64, 128, scratch={0: [2560]}), {("f32", 1000): 0})
        self.assertEqual(simulate.dynamic_pools(tensors, [0], 64, 128, shared=False, scratch={0: [2560]}),
                         {("f32", 1000): 64})
        # the reserve is the largest scratch met so far: a pool created before a larger plan keeps its size
        two = {0: [("f32", 1000)], 1: [("f16", 1000)]}
        self.assertEqual(simulate.dynamic_pools(two, [0, 1], 64, 256, scratch={0: [100], 1: [64000]}),
                         {("f32", 1000): 127, ("f16", 1000): 0})

    def test_dynamic_tensor_plans_fill_separately(self) -> None:
        # each tensor's plan has its own fill budget: one fill per plan still inserts both
        # tensors of expert 0 in step 0, so step 1 hits it in both (2 of 4 tensor reads)
        two = {0: [("f32", 100), ("f32", 100)]}
        steps = [[(0, [0, 1])], [(0, [0, 2])]]
        res = simulate.sim_dynamic_tensors(steps, two, {("f32", 100): 4}, hot_uses=4, inserts_per_plan=1)
        self.assertAlmostEqual(res.hit, 2 / 8)

    def test_reelect_applies_each_prompt_when_its_session_starts(self) -> None:
        # a: prompt and decode on expert 0; b: a long prompt on expert 1, then expert 1.
        # Seeding with both prompts up front would give a expert 1 and miss its 3 steps.
        a = HEADER.format(sid="a") + "-1,0,0\n0,0,0\n1,0,0\n2,0,0\n"
        b = HEADER.format(sid="b") + "".join(f"-{p},0,1\n" for p in range(1, 6)) + "0,0,1\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "a.csv", a), write(d, "b.csv", b)])
        self.assertIsNone(err)
        assert trace is not None
        res = simulate.sim_reelect(trace, slots=1, interval=100, n_steps=len(simulate.decode_events(trace)))
        self.assertAlmostEqual(res.hit, 1.0)
        # b's prompt re-elects expert 1: one insertion over 4 decode steps
        self.assertAlmostEqual(res.inserts_per_step, 1 / 4)

    def test_reelect_clock_counts_every_decode_event(self) -> None:
        # layer 0 routes at pos 0 and 2, layer 1 at pos 1: with reelect-2 the election after
        # event 1 makes expert 1 resident in layer 0 before pos 2, a hit; a clock that counts
        # only layer 0's own rows elects after pos 2 and misses
        text = HEADER.format(sid="c") + "0,0,1\n1,1,3\n2,0,1\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "c.csv", text)])
        self.assertIsNone(err)
        assert trace is not None
        res = simulate.sim_reelect(trace, slots=1, interval=2, n_steps=len(simulate.decode_events(trace)))
        self.assertAlmostEqual(res.hit, 1 / 3)

    def test_prompt_plans_follow_the_prompt_calls(self) -> None:
        # 6 prompt tokens in calls of 4: layer 0 routes every row, layer 1 (output-pruned) only the
        # last row of each call. With max_batch 2 the 4-row plan of layer 0 bypasses the provider.
        rows = "".join(f"-{p + 1},0,{p % 4},{(p + 1) % 4}\n" for p in range(6)) + "-4,1,3,0\n-6,1,1,2\n0,0,0,1\n0,1,0,1\n"
        head = HEADER.format(sid="p") + "# n_prompt=6 n_predict=1 n_ctx=64 n_batch=4 n_ubatch=4\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "p.csv", head + rows)])
        self.assertIsNone(err)
        assert trace is not None
        s = trace.sessions[0]
        self.assertEqual(simulate.prompt_plans(s, trace.widths, {0, 1}, 8),
                         [[(0, [0, 1, 1, 2, 2, 3, 3, 0]), (1, [3, 0])], [(0, [0, 1, 1, 2]), (1, [1, 2])]])
        self.assertEqual(simulate.prompt_plans(s, trace.widths, {0, 1}, 2),
                         [[(1, [3, 0])], [(0, [0, 1, 1, 2]), (1, [1, 2])]])
        self.assertEqual(simulate.prompt_plans(s, trace.widths, {1}, 2), [[(1, [3, 0])], [(1, [1, 2])]])
        rows_max = simulate.PROVIDER_PLAN_ROWS_MAX
        simulate.PROVIDER_PLAN_ROWS_MAX = 3  # the 2-row plans hold 4 ids
        try:
            self.assertEqual(simulate.prompt_plans(s, trace.widths, {0, 1}, 8), [[(1, [3, 0])], [(1, [1, 2])]])
        finally:
            simulate.PROVIDER_PLAN_ROWS_MAX = rows_max
        events, scored = simulate.dynamic_events(trace, {0, 1}, 2)
        self.assertEqual(events, [[(1, [3, 0])], [(0, [0, 1, 1, 2]), (1, [1, 2])], [(0, [0, 1]), (1, [0, 1])]])
        self.assertEqual(scored, [False, False, True])

    def test_plan_rides_recurring_keys_on_their_fill(self) -> None:
        # three tokens of a prompt plan route expert 1: one fill serves all three. The SYCL
        # provider counts no use for the riders, the Vulkan provider (host-mapped slabs) does.
        for fill_uses, want in ((False, 0), (True, 2)):
            with self.subTest(fill_uses=fill_uses):
                uses: OrderedDict[Hashable, int] = OrderedDict()
                self.assertEqual(simulate.plan(uses, 4, [1, 1, 1], 4, 8, fill_uses), (2, 1))
                self.assertEqual(uses[1], want)
        # hits on an earlier resident count a use each, as before
        uses = OrderedDict([(1, 0)])
        self.assertEqual(simulate.plan(uses, 4, [1, 1], 4, 8), (2, 0))
        self.assertEqual(uses[1], 2)

    def test_dynamic_scores_only_marked_steps(self) -> None:
        # a warm step fills expert 1: the scored decode step hits it and inserts nothing
        events = [[(0, [1])], [(0, [1])]]
        res = simulate.sim_dynamic(events, units=2, hot_uses=4, scored=[False, True])
        self.assertEqual((res.hit, res.inserts_per_step), (1.0, 0.0))
        res = simulate.sim_dynamic_tensors(events, {0: [("f32", 100)]}, {("f32", 100): 2}, hot_uses=4,
                                           scored=[False, True])
        self.assertEqual((res.hit, res.inserts_per_step), (1.0, 0.0))

    def test_dynamic_replays_sessions_in_file_order(self) -> None:
        # session a stops after layer 0 at pos 1; indexing the per-layer lists would pair
        # a's layer-1 row at pos 1 with b's layer-0 row at pos 0
        a = HEADER.format(sid="a") + "0,0,0\n0,1,0\n1,1,1\n"
        b = HEADER.format(sid="b") + "0,0,2\n0,1,2\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "a.csv", a), write(d, "b.csv", b)])
        self.assertIsNone(err)
        assert trace is not None
        self.assertEqual(simulate.decode_events(trace),
                         [[(0, [0]), (1, [0])], [(1, [1])], [(0, [2]), (1, [2])]])
        # one unit: a(0,0) miss, a(1,0) miss, a(1,1) miss, b(0,2) miss, b(1,2) miss
        res = simulate.sim_dynamic(simulate.decode_events(trace), units=1, hot_uses=4)
        self.assertEqual(res.hit, 0.0)
        self.assertAlmostEqual(res.inserts_per_step, 5 / 3)


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        # the fixtures have 4 experts, far below the provider's 64-slot pool minimum
        self.pool_min = simulate.PROVIDER_POOL_SLOTS_MIN
        simulate.PROVIDER_POOL_SLOTS_MIN = 1
        self.tmp = tempfile.TemporaryDirectory()
        self.trace = write(self.tmp.name, "t.csv", HEADER.format(sid="a") + "".join(
            f"-{p},0,0,1\n" for p in range(1, 4)) + "".join(f"{s},0,{s % 4},{(s + 1) % 4}\n" for s in range(16)))

    def tearDown(self) -> None:
        simulate.PROVIDER_POOL_SLOTS_MIN = self.pool_min
        self.tmp.cleanup()

    def test_report_runs(self) -> None:
        code, out, err = run([self.trace, "--budgets", "0.5", "--profile", self.trace])
        self.assertEqual(code, 0, err)
        for policy in ("static ", "static-oracle", "static-prefill", "lru", "lfu-decay", "reelect-32", "dynamic"):
            self.assertIn(policy, out)
        self.assertNotIn("miss-ms", out)
        self.assertIn("dynamic assumes the --moe-cache provider takes every MoE layer", out)

    def test_cost_columns_need_all_three_values(self) -> None:
        code, _, err = run([self.trace, "--slab-mb", "2"])
        self.assertEqual(code, 2)
        self.assertIn("--link-gbs", err)
        code, out, _ = run([self.trace, "--slab-mb", "2", "--ram-gbs", "60", "--link-gbs", "25"])
        self.assertEqual(code, 0)
        self.assertIn("assumed cost model", out)

    def test_cost_reads_follow_the_routed_rows(self) -> None:
        # layer 1 routes only at pos 0: 6 expert reads over 2 decode steps is 3 per step, not
        # one row of each layer (4). One slot per layer misses every read under lru.
        uneven = write(self.tmp.name, "u.csv", HEADER.format(sid="u") + "0,0,0,1\n1,0,2,3\n0,1,0,1\n")
        code, out, err = run([uneven, "--budgets", "0.25", "--slab-mb", "1000", "--ram-gbs", "1", "--link-gbs", "1"])
        self.assertEqual(code, 0, err)
        lru = next(line for line in out.splitlines() if "| " in line and line.split("|")[1].split()[0] == "lru")
        cols = lru.split("|")[1].split()
        self.assertEqual(cols[1], "0.0%")
        self.assertEqual(cols[3], "3000.00")

    def test_insert_rates_follow_the_decode_events(self) -> None:
        # session a routes only layer 0, session b only layer 1: two decode events, one insert
        # each, so 1.00 per token, although the busiest layer holds one step. reelect inserts
        # once, when session b's prompt re-elects layer 1: 0.50.
        a = HEADER.format(sid="a") + "0,0,0\n"
        b = HEADER.format(sid="b") + "-1,1,2\n0,1,2\n"
        path = write(self.tmp.name, "ab.csv", a + b)
        code, out, err = run([path, "--budgets", "0.25"])
        self.assertEqual(code, 0, err)
        for policy, want in (("lru", "1.00"), ("lfu-decay", "1.00"), ("dynamic", "1.00"), ("reelect-32", "0.50")):
            with self.subTest(policy=policy):
                line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == policy)
                self.assertEqual(line.split("|")[1].split()[2], want)

    def test_dynamic_pool_holds_at_most_n_expert_units(self) -> None:
        # 4 layers route experts 0 and 1 on every step: 8 (layer, expert) units. budget 0.5 gives
        # 2 slots x 4 layers, but the provider caps its shared pool at n_expert = 4 slabs, so
        # the working set thrashes instead of hitting from step 2 on
        rows = "".join(f"{s},{layer},0,1\n" for s in range(3) for layer in range(4))
        path = write(self.tmp.name, "pool.csv", HEADER.format(sid="p") + rows)
        code, out, err = run([path, "--budgets", "0.5"])
        self.assertEqual(code, 0, err)
        line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
        self.assertEqual(line.split("|")[1].split()[1], "0.0%")

    def test_invalid_arguments(self) -> None:
        for argv in (["--budgets", "0"], ["--budgets", "1.5"], ["--budgets", "x"], ["--reelect", "0"], ["--hot-uses", "-1"],
                     ["--inserts-per-plan", "0"], ["--inserts-per-plan", "1025"], ["--max-batch", "0"],
                     ["--max-batch", "9"]):
            with self.subTest(argv=argv):
                code, _, _ = run([self.trace, *argv])
                self.assertEqual(code, 2)

    def test_n_expert_override_must_match_the_trace(self) -> None:
        code, _, err = run([self.trace, "--n-expert", "8"])
        self.assertEqual(code, 2)
        self.assertIn("conflicts with the trace's n_expert 4", err)
        code, _, err = run([self.trace, "--n-expert", "4", "--budgets", "0.5"])
        self.assertEqual(code, 0, err)

    def test_prefill_only_profile_is_rejected(self) -> None:
        prompt_only = write(self.tmp.name, "p.csv", HEADER.format(sid="p") + "-1,0,0,1\n-2,0,2,3\n")
        code, _, err = run([self.trace, "--profile", prompt_only])
        self.assertEqual(code, 2)
        self.assertIn("no decode rows", err)

    def test_out_of_range_expert_is_rejected(self) -> None:
        bad = write(self.tmp.name, "bad.csv", HEADER.format(sid="b") + "0,0,0,7\n")
        code, _, err = run([bad])
        self.assertEqual(code, 2)
        self.assertIn("outside [0, 4)", err)

    def test_missing_n_expert_needs_flag(self) -> None:
        bare = write(self.tmp.name, "bare.csv", "0,0,0,1\n1,0,1,2\n")
        code, _, err = run([bare])
        self.assertEqual(code, 2)
        self.assertIn("--n-expert", err)
        code, _, _ = run([bare, "--n-expert", "4"])
        self.assertEqual(code, 0)

    def test_traces_of_different_models_are_rejected(self) -> None:
        # the same n_expert, but another architecture or layer count: layer ids do not line up
        def model(sid: str, arch: str, n_layer: int) -> str:
            head = HEADER.format(sid=sid).replace("# n_expert", f"# arch={arch}\n# n_layer={n_layer}\n# n_expert", 1)
            return write(self.tmp.name, f"{sid}.csv", head + "0,0,0,1\n1,0,1,2\n")
        a = model("a", "qwen3moe", 2)
        for other, what in ((model("b", "qwen35moe", 2), "arch"), (model("c", "qwen3moe", 3), "n_layer")):
            for argv in ([a, other], [a, "--profile", other]):
                with self.subTest(argv=argv):
                    code, _, err = run(argv)
                    self.assertEqual(code, 2)
                    self.assertIn(what, err)
        code, _, err = run([a, model("d", "qwen3moe", 2), "--profile", a, "--budgets", "0.5"])
        self.assertEqual(code, 0, err)

    def test_traces_of_other_routers_are_rejected(self) -> None:
        # same arch, layers and experts, but another checkpoint (a fine-tune): its experts are numbered differently
        def model(sid: str, router: str) -> str:
            head = HEADER.format(sid=sid).replace("# n_expert", f"# arch=qwen3moe\n# router_fingerprint={router}\n# n_expert", 1)
            return write(self.tmp.name, f"{sid}.csv", head + "0,0,0,1\n1,0,1,2\n")
        a = model("a", "00000000000000a1")
        b = model("b", "00000000000000b2")
        for argv in ([a, b], [a, "--profile", b]):
            with self.subTest(argv=argv):
                code, _, err = run(argv)
                self.assertEqual(code, 2)
                self.assertIn("router fingerprint", err)
        for bad in ("0", "00000000000000a1x", "0000000000000000", "00000000000000g1"):
            with self.subTest(bad=bad):
                code, _, err = run([model("x", bad)])
                self.assertEqual(code, 2)
                self.assertIn("invalid router_fingerprint", err)
        code, _, err = run([a, model("c", "00000000000000A1"), "--profile", a, "--budgets", "0.5"])
        self.assertEqual(code, 0, err)

    def test_layers_past_n_layer_are_rejected(self) -> None:
        # a row of layer 2 in a 2-layer trace would add a layer, and pool units, to the replay
        head = HEADER.format(sid="n").replace("# n_expert", "# n_layer=2\n# n_expert", 1)
        bad = write(self.tmp.name, "n.csv", head + "0,0,0,1\n0,2,0,1\n")
        good = write(self.tmp.name, "g.csv", head + "0,0,0,1\n0,1,0,1\n")
        for argv in ([bad], [good, "--profile", bad]):
            with self.subTest(argv=argv):
                code, _, err = run(argv)
                self.assertEqual(code, 2)
                self.assertIn("layer 2", err)
        code, _, err = run([good, "--budgets", "0.5"])
        self.assertEqual(code, 0, err)

    def test_dynamic_follows_the_expert_tensors_lines(self) -> None:
        rows = "".join(f"{p},0,0,1\n" for p in range(8))

        def dynamic_hit(text: str) -> str:
            code, out, err = run([write(self.tmp.name, "e.csv", text), "--budgets", "0.5"])
            self.assertEqual(code, 0, err)
            line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
            return line.split("|")[1].split()[1]

        self.assertEqual(dynamic_hit(HEADER.format(sid="e") + rows), "87.5%")
        shared = HEADER.format(sid="e") + "# expert_tensors=0:f32/100,f32/100,f32/100\n" + rows
        self.assertEqual(dynamic_hit(shared), "0.0%")
        split = HEADER.format(sid="e") + "# expert_tensors=0:q4_K/100,q4_K/100,q6_K/150\n" + rows
        self.assertEqual(dynamic_hit(split), "87.5%")
        code, out, _ = run([write(self.tmp.name, "e.csv", shared)])
        self.assertIn("weighted by bytes", out)
        code, out, _ = run([self.trace])
        self.assertIn("upper bound", out)

    def test_provider_selects_the_pool_budget(self) -> None:
        # two layers of one tensor each, equal size, different types, n_expert 128 at budget 0.5
        text = HEADER.format(sid="v").replace("n_expert=4", "n_expert=128")
        text += "# expert_tensors=0:q4_K/100\n# expert_tensors=1:q5_K/100\n"
        text += "".join(f"{p},{layer},0,1\n" for p in range(4) for layer in (0, 1))
        path = write(self.tmp.name, "v.csv", text)
        simulate.PROVIDER_POOL_SLOTS_MIN = self.pool_min

        def dynamic(argv: list[str]) -> list[str]:
            code, out, err = run([path, "--budgets", "0.5", *argv])
            self.assertEqual(code, 0, err)
            line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
            return [line.split("|")[1].split()[1], out]

        hit, out = dynamic([])
        self.assertEqual(hit, "37.5%")
        self.assertIn("SYCL provider: the pools share one byte budget", out)
        hit, out = dynamic(["--provider", "vulkan"])
        self.assertEqual(hit, "75.0%")
        self.assertIn("Vulkan provider: every pool sizes itself", out)

    def test_sycl_scratch_can_disable_a_pool(self) -> None:
        # n_expert 128 at budget 0.5: 64 slabs of 1000 bytes, exactly the provider's minimum. With the
        # tensor's widths the SYCL replay holds back 2560 bytes of scratch and has no pool; Vulkan
        # reserves nothing; without the widths the report says the SYCL figure is an upper bound
        simulate.PROVIDER_POOL_SLOTS_MIN = self.pool_min
        head = HEADER.format(sid="s").replace("n_expert=4", "n_expert=128")
        rows = "".join(f"{p},0,0,1\n" for p in range(4))

        def dynamic(line: str, *argv: str) -> tuple[str, str]:
            code, out, err = run([write(self.tmp.name, "s.csv", head + line + rows), "--budgets", "0.5", *argv])
            self.assertEqual(code, 0, err)
            row = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
            return row.split("|")[1].split()[1], out

        widths = "# expert_tensors=0:f32/1000/32/1\n"
        self.assertEqual(dynamic(widths)[0], "0.0%")
        self.assertEqual(dynamic(widths, "--provider", "vulkan")[0], "75.0%")
        hit, out = dynamic("# expert_tensors=0:f32/1000\n")
        self.assertEqual(hit, "75.0%")
        self.assertIn("device scratch is not held back: an upper bound", out)
        self.assertNotIn("not held back", dynamic(widths)[1])

    def test_dynamic_needs_the_provider_minimum(self) -> None:
        simulate.PROVIDER_POOL_SLOTS_MIN = self.pool_min
        code, out, err = run([self.trace, "--budgets", "0.5"])
        self.assertEqual(code, 0, err)
        line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
        self.assertEqual(line.split("|")[1].split()[1:3], ["0.0%", "0.00"])

    def test_dynamic_is_warmed_by_the_prompt_plans(self) -> None:
        # a 3-token prompt in one call routes experts 0 and 1, then 2 decode steps route them again.
        # Two units: the prompt plan fills both, so decode hits every read and inserts nothing;
        # cold, step 0 misses both (50.0%, 1.00 per token)
        head = HEADER.format(sid="w") + "# n_prompt=3 n_predict=2 n_ctx=64 n_batch=8 n_ubatch=8\n"
        rows = "-1,0,0,1\n-2,0,1,0\n-3,0,0,1\n0,0,0,1\n1,0,1,0\n"

        def dynamic(text: str, *argv: str) -> tuple[list[str], str]:
            code, out, err = run([write(self.tmp.name, "w.csv", text), "--budgets", "0.5", *argv])
            self.assertEqual(code, 0, err)
            line = next(x for x in out.splitlines() if "| " in x and x.split("|")[1].split()[0] == "dynamic")
            return line.split("|")[1].split()[1:3], out

        cols, out = dynamic(head + rows)
        self.assertEqual(cols, ["100.0%", "0.00"])
        self.assertIn("prompt plans of at most 8 tokens (the trace's prompt calls of 8 tokens)", out)
        tensors = head + "# expert_tensors=0:f32/100\n"
        self.assertEqual(dynamic(tensors + rows)[0], ["100.0%", "0.00"])
        # a plan of 3 tokens bypasses a provider limited to 2
        self.assertEqual(dynamic(head + rows, "--max-batch", "2")[0], ["50.0%", "1.00"])
        # without the header the prompt calls are unknown
        cols, out = dynamic(HEADER.format(sid="w") + rows)
        self.assertEqual(cols, ["50.0%", "1.00"])
        self.assertIn("1 session(s) without a '# n_prompt=' line", out)

    def test_expert_tensors_must_cover_every_routed_layer(self) -> None:
        text = HEADER.format(sid="m") + "# expert_tensors=0:f32/100\n0,0,0,1\n0,1,0,1\n"
        code, _, err = run([write(self.tmp.name, "m.csv", text)])
        self.assertEqual(code, 2)
        self.assertIn("layer 1 routes experts but has no '# expert_tensors=' line", err)

    def test_prefill_only_trace_is_rejected(self) -> None:
        pre = write(self.tmp.name, "pre.csv", HEADER.format(sid="c") + "-1,0,0,1\n")
        code, _, err = run([pre])
        self.assertEqual(code, 2)
        self.assertIn("no decode rows", err)


if __name__ == "__main__":
    unittest.main()
