"""P04 verifier negative fixtures; optional compiled recorder checks via SPEC_CYCLE_TEST_BIN."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_spec_cycle", Path(__file__).parent / "perf/verify-spec-cycle-log.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def phase(ident, members, start, end):
    return {"phase_id": ident, "participants": [{"proposal_id": p, "sequence_id": s} for p, s in members],
            "participant_count": len(members), "timing_scope": "single_sequence" if len(members) == 1 else "shared_batch",
            "start_us": start, "end_us": end, "wall_us": end - start, "synchronization_us": 0}


def row():
    return {"schema_version": 1, "launch_id": 100, "request_id": 7, "proposal_id": 1, "sequence_id": 0,
            "cycle_id": 1, "starting_position": 5, "selected": True, "outcome": "completed", "drafted_count": 1, "attempted_count": 3,
            "accepted_count": 1, "attempts": [{"token": 42, "top1_probability": .9, "emitted": True},
                {"token": 43, "top1_probability": .8, "emitted": True},
                {"token": 44, "top1_probability": .1, "emitted": False}],
            "top1_probabilities": [.9, .8, .1], "proposal_tokens": [42],
            "draft_phase": phase(1, [(1, 0)], 0, 10), "verify_phases": [phase(2, [(1, 0)], 15, 25)],
            "draft_wall_us": 10, "verify_sample_wall_us": 10, "cycle_wall_us": 25,
            "inter_phase_wait_us": 5, "additive_residual_us": 5, "timing_mode": "host_unsynchronized"}


class CycleVerifierTests(unittest.TestCase):
    def test_clamp_retains_stop_and_reconciles(self):
        record = row()
        counts = VERIFY.validate_rows([record], "spec_cycle_start: launch=100 request=7 proposal=1 sequence=0 cycle=1")
        self.assertEqual(counts["draft_phases"], 1)
        self.assertEqual(VERIFY.reconcile([record], {"timings": {"draft_n": 1, "draft_n_accepted": 1}}), 7)

    def test_unselected_clear(self):
        record = row()
        record.update(selected=False, drafted_count=0, accepted_count=0, proposal_tokens=[], verify_phases=[],
                      inter_phase_wait_us=None, verify_sample_wall_us=None, additive_residual_us=None)
        VERIFY.validate_rows([record])
        self.assertEqual(len(record["attempts"]), 3)

    def test_shared_mixed_membership(self):
        a = row()
        c = copy.deepcopy(a)
        c.update(proposal_id=3, sequence_id=2, request_id=8)
        a["draft_phase"] = phase(1, [(1, 0), (2, 1)], 0, 10)
        c["draft_phase"] = phase(3, [(3, 2)], 0, 10)
        a["verify_phases"] = c["verify_phases"] = [phase(4, [(1, 0), (3, 2)], 15, 25)]
        a["additive_residual_us"] = c["additive_residual_us"] = None
        counts = VERIFY.validate_rows([a, c])
        self.assertEqual((counts["draft_phases"], counts["verify_phases"], counts["sequence_cycles"]), (2, 1, 2))
        c = copy.deepcopy(c)
        c["verify_phases"][0].update(wall_us=11, end_us=26)
        with self.assertRaisesRegex(ValueError, "shared phase metadata differs"):
            VERIFY.validate_rows([a, c])

    def test_replay_has_multiple_invocations(self):
        record = row()
        record["verify_phases"].append(phase(3, [(1, 0)], 30, 40))
        record.update(cycle_wall_us=40, verify_sample_wall_us=None, additive_residual_us=None)
        self.assertEqual(VERIFY.validate_rows([record])["verify_phases"], 2)

    def test_cancelled_phase_evidence(self):
        cancelled = row()
        cancelled.update(outcome="cancelled", selected=False, drafted_count=0, accepted_count=0,
                         proposal_tokens=[], additive_residual_us=None)
        counts = VERIFY.validate_rows([cancelled])
        self.assertEqual(counts["verify_phases"], 1)
        cancelled["verify_phases"].append(phase(3, [(1, 0)], 30, 40))
        cancelled.update(cycle_wall_us=50, verify_sample_wall_us=None)
        self.assertEqual(VERIFY.validate_rows([cancelled])["verify_phases"], 2)
        with self.assertRaisesRegex(ValueError, "cancelled proposals"):
            VERIFY.reconcile([cancelled], {"timings": {"draft_n": 0, "draft_n_accepted": 0}})

    def test_shared_cancelled_and_completed(self):
        cancelled = row()
        success = copy.deepcopy(cancelled)
        success.update(sequence_id=1, proposal_id=2)
        cancelled.update(outcome="cancelled", selected=False, drafted_count=0, accepted_count=0, proposal_tokens=[])
        for record in (cancelled, success):
            record["draft_phase"] = phase(1, [(1, 0), (2, 1)], 0, 10)
            record["verify_phases"] = [phase(2, [(1, 0), (2, 1)], 15, 25)]
            record["additive_residual_us"] = None
        counts = VERIFY.validate_rows([cancelled, success])
        self.assertEqual((counts["draft_phases"], counts["verify_phases"], counts["sequence_cycles"]), (1, 1, 2))

    def test_reject_corrupt_rows(self):
        mutations = [lambda r: r.update(attempted_count=2), lambda r: r.update(accepted_count=2),
                     lambda r: r.update(proposal_tokens=[99]), lambda r: r.update(additive_residual_us=None),
                     lambda r: r["draft_phase"].update(synchronization_us=1),
                     lambda r: r["attempts"][0].update(emitted=False),
                     lambda r: r["verify_phases"][0]["participants"][0].update(proposal_id=77)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = row()
                mutate(record)
                with self.assertRaises(ValueError):
                    VERIFY.validate_rows([record])
        with self.assertRaisesRegex(ValueError, "duplicate proposal"):
            VERIFY.validate_rows([row(), row()])
        with self.assertRaisesRegex(ValueError, "started/completed"):
            VERIFY.validate_rows([row()], "")
        with self.assertRaisesRegex(ValueError, "accepted total"):
            VERIFY.reconcile([row()], {"timings": {"draft_n": 1, "draft_n_accepted": 0}})

    @unittest.skipUnless(os.environ.get("SPEC_CYCLE_TEST_BIN"), "compiled recorder binary not supplied")
    def test_compiled_recorder_and_failures(self):
        binary = os.environ["SPEC_CYCLE_TEST_BIN"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cycles.jsonl"
            result = subprocess.run([binary, "--profile-fixture", str(path)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            counts = VERIFY.validate_rows(VERIFY.read_rows(path), result.stderr)
            self.assertEqual([counts[k] for k in ("draft_phases", "verify_phases", "sequence_cycles")], [64, 64, 64])
            self.assertEqual(len(VERIFY.PROFILE.findall(result.stderr)), 1)
            VERIFY.validate_profile(VERIFY.read_rows(path), result.stderr)
            cancelled_path = Path(directory) / "cancelled.jsonl"
            result = subprocess.run([binary, "--cancellation-fixture", str(cancelled_path)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            cancelled_rows = VERIFY.read_rows(cancelled_path)
            counts = VERIFY.validate_rows(cancelled_rows, result.stderr)
            self.assertEqual([counts[k] for k in ("draft_phases", "verify_phases", "sequence_cycles")], [64, 85, 85])
            self.assertEqual(sum(r["outcome"] == "cancelled" for r in cancelled_rows), 64)
            self.assertEqual(sum(r["selected"] for r in cancelled_rows), 21)
            for record in cancelled_rows:
                self.assertEqual(len(record["verify_phases"]), 2 if record["request_id"] % 3 == 1 else 1)
            VERIFY.validate_profile(cancelled_rows, result.stderr)
            for args, diagnostic in [(["--bad-identity"], "identity mismatch"),
                    (["--write", str(Path(directory) / "missing/row")], "cannot open"),
                    (["--write", "/dev/full"], "append or flush failed")]:
                failure = subprocess.run([binary, *args], capture_output=True, text=True, timeout=30)
                self.assertNotEqual(failure.returncode, 0)
                self.assertIn(diagnostic, failure.stderr)


if __name__ == "__main__":
    unittest.main()
