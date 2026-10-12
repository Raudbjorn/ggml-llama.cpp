"""Tests for tools/moe-trace/simulate.py (stdlib only; runs under unittest or pytest)."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
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
        self.tmp = tempfile.TemporaryDirectory()
        self.trace = write(self.tmp.name, "t.csv", HEADER.format(sid="a") + "".join(
            f"-{p},0,0,1\n" for p in range(1, 4)) + "".join(f"{s},0,{s % 4},{(s + 1) % 4}\n" for s in range(16)))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_report_runs(self) -> None:
        code, out, err = run([self.trace, "--budgets", "0.5", "--profile", self.trace])
        self.assertEqual(code, 0, err)
        for policy in ("static ", "static-oracle", "static-prefill", "lru", "lfu-decay", "reelect-32", "dynamic"):
            self.assertIn(policy, out)
        self.assertNotIn("miss-ms", out)

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

    def test_invalid_arguments(self) -> None:
        for argv in (["--budgets", "0"], ["--budgets", "1.5"], ["--budgets", "x"], ["--reelect", "0"], ["--hot-uses", "-1"],
                     ["--inserts-per-plan", "0"], ["--inserts-per-plan", "1025"]):
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

    def test_prefill_only_trace_is_rejected(self) -> None:
        pre = write(self.tmp.name, "pre.csv", HEADER.format(sid="c") + "-1,0,0,1\n")
        code, _, err = run([pre])
        self.assertEqual(code, 2)
        self.assertIn("no decode rows", err)


if __name__ == "__main__":
    unittest.main()
