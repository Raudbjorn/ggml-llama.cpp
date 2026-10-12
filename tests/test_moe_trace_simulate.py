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
        res = simulate.sim_lru({0: [[0], [1], [0], [2], [1]]}, 2)
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

    def test_reelect_applies_each_prompt_when_its_session_starts(self) -> None:
        # a: prompt and decode on expert 0; b: a long prompt on expert 1, then expert 1.
        # Seeding with both prompts up front would give a expert 1 and miss its 3 steps.
        a = HEADER.format(sid="a") + "-1,0,0\n0,0,0\n1,0,0\n2,0,0\n"
        b = HEADER.format(sid="b") + "".join(f"-{p},0,1\n" for p in range(1, 6)) + "0,0,1\n"
        with tempfile.TemporaryDirectory() as d:
            trace, err = simulate.parse_traces([write(d, "a.csv", a), write(d, "b.csv", b)])
        self.assertIsNone(err)
        assert trace is not None
        res = simulate.sim_reelect(trace, slots=1, interval=100)
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

    def test_invalid_arguments(self) -> None:
        for argv in (["--budgets", "0"], ["--budgets", "1.5"], ["--budgets", "x"], ["--reelect", "0"], ["--hot-uses", "-1"]):
            with self.subTest(argv=argv):
                code, _, _ = run([self.trace, *argv])
                self.assertEqual(code, 2)

    def test_n_expert_override_must_match_the_trace(self) -> None:
        code, _, err = run([self.trace, "--n-expert", "8"])
        self.assertEqual(code, 2)
        self.assertIn("conflicts with the trace's n_expert 4", err)
        code, _, err = run([self.trace, "--n-expert", "4", "--budgets", "0.5"])
        self.assertEqual(code, 0, err)

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

    def test_prefill_only_trace_is_rejected(self) -> None:
        pre = write(self.tmp.name, "pre.csv", HEADER.format(sid="c") + "-1,0,0,1\n")
        code, _, err = run([pre])
        self.assertEqual(code, 2)
        self.assertIn("no decode rows", err)


if __name__ == "__main__":
    unittest.main()
