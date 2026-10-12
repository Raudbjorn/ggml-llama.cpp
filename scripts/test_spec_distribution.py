"""DistributionTests are CPU-light mocks; RealProcessTests require a separate host lease."""
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.test_spec_cycle_log import row as p04_row, phase

SPEC = importlib.util.spec_from_file_location("verify_spec_distribution", Path(__file__).parent / "perf/verify-spec-distribution.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def settings():
    return SimpleNamespace(seed=123, target_temperature=.8, draft_temperature=.8, draft_top_p=1.0)


def row(enabled=True):
    record = p04_row()
    record.update(drafted_count=2, proposal_tokens=[42, 43], sampling_reason="distribution" if enabled else "disabled",
                  sampling_mode="distribution" if enabled else "exact_match", distribution_attempts=int(enabled),
                  exact_match_attempts=int(not enabled), q_records=None, draft_sampling=None)
    for attempt in record["attempts"]:
        attempt.update(top1_token=attempt["token"], sampled_probability=attempt["top1_probability"])
    if enabled:
        # Sampled token 42 is deliberately not the top-1 token 99.
        record["attempts"][0].update(top1_token=99, top1_probability=.75, sampled_probability=.25)
        record["top1_probabilities"][0] = .75
        record["draft_sampling"] = {"temperature": .8, "top_p": 1.0, "top_k": 10, "backend_sampling": False}
        record["q_records"] = [
            {"schema_version": 1, "proposal_id": 1, "draft_index": 0, "draft_token": 42,
             "support": [{"token": 42, "probability": .25}, {"token": 99, "probability": .75}], "probability_sum": 1.0},
            {"schema_version": 1, "proposal_id": 1, "draft_index": 1, "draft_token": 43,
             "support": [{"token": 43, "probability": .8}, {"token": 99, "probability": .2}], "probability_sum": 1.0}]
    return record


def response(payload):
    return {"tokens_predicted": 3, "stop": True, "truncated": False,
            "generation_settings": {**payload, "speculative.types": "none,draft-mtp"},
            "timings": {"draft_n": 2, "draft_n_accepted": 1}}


def campaign(enabled=True):
    rows, responses = [], []
    for n in range(6):
        record = row(enabled)
        record.update(request_id=7 + n * 3, proposal_id=n + 1, cycle_id=n + 1)
        record["draft_phase"] = phase(2 * n + 1, [(n + 1, 0)], 0, 10)
        record["verify_phases"] = [phase(2 * n + 2, [(n + 1, 0)], 15, 25)]
        for q in record["q_records"] or []:
            q["proposal_id"] = n + 1
        payload = VERIFY.request_payload(f"prompt {n // 2}", 256, 123, .8)
        rows.append(record)
        responses.append({"client_request": n + 1, "prompt_index": n // 2, "repeat": n % 2,
                          "request_id": record["request_id"], "payload": payload, "response": response(payload)})
    return rows, responses


class DistributionTests(unittest.TestCase):
    def validate(self, record, enabled=True):
        return VERIFY.validate_distribution([record], enabled, .8, 1.0)

    def test_baseline_null_and_sampled_not_top1(self):
        self.assertEqual(self.validate(row())["accepted_per_drafted"], .5)
        self.assertEqual(self.validate(row(False), False)["distribution_attempts"], 0)

    def test_probability_tolerance_and_zero_sampled_mass(self):
        record = row()
        record["q_records"][-1]["support"][-1]["probability"] += .000005
        self.validate(record)
        record["q_records"][-1]["support"][-1]["probability"] += .00001
        with self.assertRaisesRegex(ValueError, "normalized"):
            self.validate(record)
        record = row()
        record["q_records"][-1]["support"] = [{"token": 43, "probability": 0}, {"token": 99, "probability": 1}]
        with self.assertRaisesRegex(ValueError, "positive q mass"):
            self.validate(record)

    def test_repair_and_real_replay_do_not_replace_first_decision(self):
        for exact in (0, 1, 3):
            with self.subTest(exact=exact):
                record = row()
                record["verify_phases"].append(phase(3, [(1, 0)], 30, 40))
                record.update(exact_match_attempts=exact, cycle_wall_us=40,
                              verify_sample_wall_us=None, additive_residual_us=None)
                self.assertEqual(self.validate(record)["distribution_attempts"], 1)

    def test_q_corruption_including_final_record(self):
        changes = [lambda r: r.pop("q_records"), lambda r: r.update(q_records=None),
                   lambda r: r.update(q_records=[]), lambda r: r["q_records"].pop(),
                   lambda r: r["q_records"].append(copy.deepcopy(r["q_records"][0])),
                   lambda r: r["q_records"][-1].update(draft_index=0),
                   lambda r: r["q_records"][-1].update(draft_index=True),
                   lambda r: r["q_records"][-1].update(proposal_id=2),
                   lambda r: r["q_records"][-1].update(schema_version=2),
                   lambda r: r["q_records"][-1].update(draft_token=42),
                   lambda r: r["q_records"][-1].update(probability_sum=.9),
                   lambda r: r["q_records"][-1].update(probability_sum=float("nan")),
                   lambda r: r["q_records"][-1].update(support=[]),
                   lambda r: r["q_records"][-1].update(support=[{"token": n, "probability": 1 / 11} for n in range(11)]),
                   lambda r: r["q_records"][-1]["support"][1].update(token=43),
                   lambda r: r["q_records"][-1]["support"][0].update(probability=float("inf")),
                   lambda r: r["q_records"][-1]["support"][0].update(probability=-.2),
                   lambda r: r["q_records"][-1]["support"][0].update(probability=.1),
                   lambda r: r["q_records"][-1].update(support=[{"token": 99, "probability": 1}]),
                   lambda r: r["attempts"][0].update(sampled_probability=.75),
                   lambda r: r["attempts"][0].update(top1_token=42)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                record = row()
                change(record)
                with self.assertRaises((ValueError, KeyError)):
                    self.validate(record)

    def test_no_false_positive_or_ineligible_fallback(self):
        changes = [lambda r: r.update(distribution_attempts=0, exact_match_attempts=1,
                                      sampling_mode="exact_match", sampling_reason="unsupported_target_sampler"),
                   lambda r: r.update(distribution_attempts=2),
                   lambda r: r.update(distribution_attempts=True),
                   lambda r: r.update(exact_match_attempts=-1),
                   lambda r: r.update(sampling_mode="exact_match"),
                   lambda r: r.update(sampling_reason="chained_draft"),
                   lambda r: r["draft_sampling"].update(backend_sampling=True),
                   lambda r: r["draft_sampling"].update(top_k=11),
                   lambda r: r["draft_sampling"].update(temperature=0)]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                record = row()
                change(record)
                with self.assertRaises(ValueError):
                    self.validate(record)
        with self.assertRaises(ValueError):
            self.validate(row(False))
        with self.assertRaises(ValueError):
            VERIFY.validate_distribution([], True, .8, 1)

    def test_baseline_requires_actual_exact_attempt(self):
        record = row(False)
        record["exact_match_attempts"] = 0
        with self.assertRaises(ValueError):
            self.validate(record, False)
        with self.assertRaises(ValueError):
            self.validate(row(), False)

    def test_duplicate_proposal_and_schema_ids(self):
        rows, _ = campaign()
        rows[1]["proposal_id"] = 1
        for p in [rows[1]["draft_phase"], *rows[1]["verify_phases"]]:
            p["participants"][0]["proposal_id"] = 1
        with self.assertRaisesRegex(ValueError, "duplicate"):
            VERIFY.validate_distribution(rows, True, .8, 1)

    def test_fallback_cannot_supply_positive_distribution_evidence(self):
        rows, responses = campaign()
        rows[-1].update(distribution_attempts=0, exact_match_attempts=1,
                        sampling_mode="exact_match", sampling_reason="consumed_or_replayed")
        self.assertEqual(VERIFY.validate_campaign(rows, responses, True, settings())["distribution_attempts"], 5)
        for record in rows:
            record.update(distribution_attempts=0, exact_match_attempts=1,
                          sampling_mode="exact_match", sampling_reason="unsupported_target_sampler")
        with self.assertRaisesRegex(ValueError, "no actual distribution"):
            VERIFY.validate_campaign(rows, responses, True, settings())

    def test_unselected_attempt_diagnostics_are_retained(self):
        rows, responses = campaign()
        extra = row()
        extra.update(proposal_id=100, cycle_id=100, selected=False, drafted_count=0, accepted_count=0,
                     proposal_tokens=[], q_records=None, sampling_mode="exact_match", sampling_reason="pending",
                     distribution_attempts=0, exact_match_attempts=0, verify_phases=[],
                     additive_residual_us=None, inter_phase_wait_us=None, verify_sample_wall_us=None)
        extra["draft_phase"] = phase(100, [(100, 0)], 0, 10)
        rows.append(extra)
        self.assertEqual(VERIFY.validate_campaign(rows, responses, True, settings())["drafted"], 12)
        self.assertEqual(len(extra["attempts"]), 3)

    def test_six_request_reconciliation(self):
        for enabled in (False, True):
            rows, responses = campaign(enabled)
            summary = VERIFY.validate_campaign(rows, responses, enabled, settings())
            self.assertEqual((summary["requests"], summary["drafted"], summary["accepted"]), (6, 12, 6))
            for count in (0, 1, 5):
                with self.subTest(enabled=enabled, count=count), self.assertRaises(ValueError):
                    VERIFY.validate_campaign(rows[:count], responses[:count], enabled, settings())

    def test_invalid_request_and_response_evidence(self):
        mutations = [lambda r: r[-1]["response"].update(tokens_predicted=0),
                     lambda r: r[-1]["response"].update(stop=False),
                     lambda r: r[-1]["response"]["timings"].update(draft_n=99),
                     lambda r: r[-1].update(request_id=r[0]["request_id"]),
                     lambda r: r[-1].update(prompt_index=0),
                     lambda r: r[-1]["payload"].update(seed=999),
                     lambda r: r[-1]["response"]["generation_settings"].update(temperature=0),
                     lambda r: r[-1]["response"]["generation_settings"].update(samplers=["penalties", "temperature"]),
                     lambda r: r[-1]["response"]["generation_settings"].update(backend_sampling=True),
                     lambda r: r[-1]["response"]["generation_settings"].update(grammar="root ::= 'x'"),
                     lambda r: r[-1]["response"]["generation_settings"].update(**{"speculative.types": "ngram-mod"})]
        for change in mutations:
            with self.subTest(change=mutations.index(change)):
                rows, responses = campaign()
                change(responses)
                with self.assertRaises(ValueError):
                    VERIFY.validate_campaign(rows, responses, True, settings())

    def test_environment_isolation(self):
        parent = {"LLAMA_SPEC_SAMPLE_TEMP": "wrong", "LLAMA_SPEC_DRAFT_TOPP": ".3", "LLAMA_SPEC_CHAIN": "1",
                  "LLAMA_SPEC_PROFILE": "1", "LLAMA_ARG_SEED": "999", "LLAMA_DRAFT_TOP_K": "1",
                  "PATH": "/bin", "OCL_ICD_VENDORS": "/approved/intel", "UR_L0_USE_COPY_ENGINE": "0"}
        original = parent.copy()
        for enabled in (False, True):
            env = VERIFY.child_environment(parent, Path("/build/bin/llama-server"), Path("/tmp/p06.jsonl"), enabled, settings())
            self.assertEqual(env["LLAMA_DRAFT_TOP_K"], "10")
            self.assertEqual(env["LLAMA_SPEC_CHAIN"], "0")
            self.assertNotIn("LLAMA_SPEC_PROFILE", env)
            self.assertNotIn("LLAMA_ARG_SEED", env)
            self.assertEqual(env["OCL_ICD_VENDORS"], parent["OCL_ICD_VENDORS"])
            self.assertEqual(env["UR_L0_USE_COPY_ENGINE"], "0")
            if enabled:
                self.assertEqual((env["LLAMA_SPEC_SAMPLE_TEMP"], env["LLAMA_SPEC_DRAFT_TOPP"]), ("0.8", "1.0"))
            else:
                self.assertNotIn("LLAMA_SPEC_SAMPLE_TEMP", env)
                self.assertNotIn("LLAMA_SPEC_DRAFT_TOPP", env)
        self.assertEqual(parent, original)

    def test_owned_tenancy_uses_p02_attribution(self):
        bench = SimpleNamespace(RENDER_NODE="/dev/dri/renderD128", curve_children=Mock(return_value={124}))
        for markers in ("", "m", "e", "cerm", " m e\n"):
            with self.subTest(markers=markers):
                result = subprocess.CompletedProcess(["fuser", bench.RENDER_NODE], 0,
                                                     " 123 124\n", bench.RENDER_NODE + ": " + markers)
                probes = []
                with patch.object(VERIFY.subprocess, "run", return_value=result) as run:
                    self.assertEqual(VERIFY.owned_tenancy(bench, SimpleNamespace(pid=123), probes), [123, 124])
                run.assert_called_once_with(result.args, check=False, capture_output=True, text=True, timeout=30)
                self.assertEqual(probes, [{"command": result.args, "returncode": 0, "stdout": result.stdout,
                                          "stderr": result.stderr, "holders": [123, 124], "owned_pids": [123, 124],
                                          "attribution_ok": True}])

    def test_owned_tenancy_rejects_foreign_empty_failed_and_malformed_probes(self):
        bench = SimpleNamespace(RENDER_NODE="/dev/dri/renderD128", curve_children=Mock(return_value={124}))
        label = bench.RENDER_NODE + ":"
        for rc, stdout, stderr in ((0, "123 999", label + "m"), (0, "", label), (1, "", ""),
                                   (2, "123", label), (-15, "123", label), (0, "0", label),
                                   (0, "123m", label), (0, "123", ""), (0, "123", label + " unexpected"),
                                   (0, "123", label + "\nPermission denied"),
                                   (0, "123", "/dev/dri/renderD129:m")):
            with self.subTest(rc=rc, stdout=stdout, stderr=stderr):
                probes = []
                result = subprocess.CompletedProcess(["fuser", bench.RENDER_NODE], rc, stdout, stderr)
                with patch.object(VERIFY.subprocess, "run", return_value=result), self.assertRaises(ValueError):
                    VERIFY.owned_tenancy(bench, SimpleNamespace(pid=123), probes)
                self.assertEqual((probes[0]["returncode"], probes[0]["stdout"], probes[0]["stderr"]),
                                 (rc, stdout, stderr))
                self.assertFalse(probes[0]["attribution_ok"])

    def test_owned_tenancy_records_probe_exceptions_and_timeout_streams(self):
        bench = SimpleNamespace(RENDER_NODE="/dev/dri/renderD128")
        for error in (OSError("fuser missing"), subprocess.TimeoutExpired(["fuser", bench.RENDER_NODE], 30,
                                                                         output=b"123", stderr=b"partial")):
            with self.subTest(error=type(error).__name__):
                probes = []
                with patch.object(VERIFY.subprocess, "run", side_effect=error), self.assertRaises(type(error)):
                    VERIFY.owned_tenancy(bench, SimpleNamespace(pid=123), probes)
                saved = json.loads(json.dumps(probes))[0]
                self.assertEqual(saved["command"], ["fuser", bench.RENDER_NODE])
                self.assertIsNone(saved["returncode"])
                self.assertFalse(saved["attribution_ok"])
                self.assertEqual(saved["exception"], {"type": type(error).__name__, "message": str(error)})
                self.assertEqual(saved["stdout"], "123" if isinstance(error, subprocess.TimeoutExpired) else "")
                self.assertEqual(saved["stderr"], "partial" if isinstance(error, subprocess.TimeoutExpired) else "")

    def test_explicit_spawn_registers_p02_ownership_without_real_process(self):
        with tempfile.TemporaryDirectory() as directory:
            bench = SimpleNamespace(_owned_servers={}, _PR_SET_CHILD_SUBREAPER=36,
                                    process_table=lambda: {10: (os.getpid(), 1, "S")})
            proc = SimpleNamespace(pid=123)
            env = {"PATH": "/mock"}
            with patch.object(VERIFY.ctypes, "CDLL", return_value=SimpleNamespace(prctl=Mock(return_value=0))), \
                 patch.object(VERIFY.signal, "pthread_sigmask", return_value=set()) as mask, \
                 patch.object(VERIFY.subprocess, "Popen", return_value=proc) as popen:
                self.assertIs(VERIFY.start_server(bench, ["/mock/server"], env, Path(directory) / "server.log"), proc)
            self.assertIs(bench._owned_servers[123], proc)
            self.assertIsNone(proc.owned_pgid)
            self.assertEqual(proc.existing_children, {10})
            self.assertIs(popen.call_args.kwargs["env"], env)
            self.assertNotIn("start_new_session", popen.call_args.kwargs)
            self.assertEqual(mask.call_count, 2)

    def test_cli_rejects_changed_campaign(self):
        base = [arg for key in ("server", "model", "draft-model", "prompts", "baseline-log", "enabled-log", "q-artifact", "report")
                for arg in ("--" + key, "/tmp/" + key)]
        self.assertEqual(VERIFY.parse_args(base).seed, 123)
        for extra in (["--repeats", "1"], ["--seed", "42"], ["--draft-top-p", ".5"],
                      ["--spec-args=--spec-type draft-mtp --spec-draft-backend-sampling"], ["--ctx", "4096"]):
            with self.assertRaises(ValueError):
                VERIFY.parse_args(base + extra)

    def test_preflight_fails_closed_without_real_processes(self):
        bench = SimpleNamespace(gpu_holders=Mock(return_value=[]))
        idle = SimpleNamespace(returncode=1, stdout="", stderr="")
        inactive = SimpleNamespace(returncode=3, stdout="inactive\ninactive\n", stderr="")
        with patch.object(VERIFY.subprocess, "run", side_effect=[idle, idle, inactive]), \
             patch.object(VERIFY.os, "getloadavg", return_value=(0, 0, 0)):
            VERIFY.preflight(bench)
        for failure in (SimpleNamespace(returncode=0, stdout="foreign process", stderr=""),
                        SimpleNamespace(returncode=2, stdout="", stderr="unavailable")):
            with patch.object(VERIFY.subprocess, "run", return_value=failure), self.assertRaises(ValueError):
                VERIFY.preflight(bench)
        active = SimpleNamespace(returncode=0, stdout="active\n", stderr="")
        with patch.object(VERIFY.subprocess, "run", side_effect=[idle, idle, active]), self.assertRaises(ValueError):
            VERIFY.preflight(bench)
        with patch.object(VERIFY.subprocess, "run", side_effect=[idle, idle, inactive]), \
             patch.object(VERIFY.os, "getloadavg", return_value=(1000, 1000, 1000)), \
             patch.object(VERIFY.time, "monotonic", side_effect=[0, 0, 181]), self.assertRaises(ValueError):
            VERIFY.preflight(bench)

    def test_quiet_settle_observes_load_and_rechecks_all_idle_gates(self):
        for failure in (None, "timeout", "holder", "process", "service", "cancel", "signal", "invalid_load"):
            with self.subTest(failure=failure):
                clock = [0.0]
                observations = []
                def load():
                    return (float("nan"), 0, 0) if failure == "invalid_load" else ((12.0, 0, 0) if failure == "timeout" else ((12.0 if not clock[0] else 6.0), 0, 0))
                def sleep(seconds):
                    self.assertGreater(seconds, 0)
                    self.assertLessEqual(seconds, VERIFY.LOAD_POLL_SECONDS)
                    clock[0] += seconds
                    if failure == "signal":
                        raise KeyboardInterrupt()
                def probe(command, **kwargs):
                    self.assertGreater(kwargs["timeout"], 0)
                    self.assertLessEqual(kwargs["timeout"], 10)
                    if command[0] == "pgrep":
                        return subprocess.CompletedProcess(command, 0 if clock[0] and failure == "process" else 1,
                                                           "123 foreign" if clock[0] and failure == "process" else "", "")
                    active = bool(clock[0] and failure == "service")
                    return subprocess.CompletedProcess(command, 0 if active else 3, "active\n" if active else "inactive\nfailed\n", "")
                bench = SimpleNamespace(gpu_holders=Mock(side_effect=lambda: ["999"] if clock[0] and failure == "holder" else []))
                with patch.object(VERIFY.os, "getloadavg", side_effect=load), \
                     patch.object(VERIFY.os, "cpu_count", return_value=24), \
                     patch.object(VERIFY.time, "monotonic", side_effect=lambda: clock[0]), \
                     patch.object(VERIFY.time, "sleep", side_effect=sleep), \
                     patch.object(VERIFY.subprocess, "run", side_effect=probe) as run:
                    if failure:
                        with self.assertRaises(KeyboardInterrupt if failure == "signal" else ValueError):
                            VERIFY.preflight(bench, observations, lambda: bool(clock[0] and failure == "cancel"))
                    else:
                        result = VERIFY.preflight(bench, observations)
                        self.assertIs(result, observations[-1])
                        self.assertEqual([r["load1"] for r in observations], [12, 6])
                        self.assertEqual(run.call_count, 6)
                        self.assertEqual(bench.gpu_holders.call_count, 2)
                self.assertTrue(observations)
                self.assertTrue(all(r["load_limit"] == 6 for r in observations))
                self.assertEqual("error" in observations[-1], failure is not None)
                if failure == "timeout":
                    self.assertEqual(clock[0], VERIFY.LOAD_SETTLE_SECONDS)
                    self.assertTrue(all(r["load1"] == 12 for r in observations))
                elif failure != "invalid_load":
                    self.assertEqual(clock[0], VERIFY.LOAD_POLL_SECONDS)

    def test_quiet_settle_rechecks_load_after_resource_probes(self):
        clock = [0.0]
        observations = []
        idle = SimpleNamespace(returncode=1, stdout="", stderr="")
        inactive = SimpleNamespace(returncode=3, stdout="inactive\n", stderr="")
        bench = SimpleNamespace(gpu_holders=Mock(return_value=[]))
        with patch.object(VERIFY.os, "getloadavg", side_effect=[(0, 0, 0), (12, 0, 0), (0, 0, 0), (0, 0, 0)]), \
             patch.object(VERIFY.os, "cpu_count", return_value=24), \
             patch.object(VERIFY.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(VERIFY.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)) as sleep, \
             patch.object(VERIFY.subprocess, "run", side_effect=[idle, idle, inactive] * 2):
            VERIFY.preflight(bench, observations)
        self.assertEqual([r["entry_load1"] for r in observations], [0, 0])
        self.assertEqual([r["load1"] for r in observations], [12, 0])
        self.assertEqual(bench.gpu_holders.call_count, 2)
        sleep.assert_called_once_with(VERIFY.LOAD_POLL_SECONDS)

    def test_quiet_settle_cannot_pass_after_slow_tenancy_probe(self):
        clock = [0.0]
        observations = []
        def holders():
            clock[0] = VERIFY.LOAD_SETTLE_SECONDS + 1
            return []
        bench = SimpleNamespace(gpu_holders=holders)
        with patch.object(VERIFY.os, "getloadavg", return_value=(0, 0, 0)), \
             patch.object(VERIFY.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(VERIFY.subprocess, "run") as run, \
             patch.object(VERIFY.time, "sleep") as sleep, \
             self.assertRaisesRegex(ValueError, "deadline expired"):
            VERIFY.preflight(bench, observations)
        run.assert_not_called()
        sleep.assert_not_called()
        self.assertEqual(observations[0]["load1"], 0)
        self.assertIn("deadline expired", observations[0]["error"])

    def test_main_publishes_only_complete_matched_arms(self):
        for failure in (None, "mismatch", "incomplete", "reused_launch", "cancelled", "shutdown"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = {name: root / name for name in ("prompts", "baseline-log", "enabled-log", "q-artifact", "report")}
                paths["prompts"].write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": str(n)}]}) for n in range(3)))
                argv = [arg for key, value in paths.items() for arg in ("--" + key, str(value))]
                argv += [arg for key in ("server", "model", "draft-model") for arg in ("--" + key, __file__)]
                bench = SimpleNamespace(render_driver=lambda node: "xe", arm_library_path=lambda server: "/mock",
                                        resolved_libraries=lambda *args: {}, build_hashes=lambda *args: {"mock": "digest"})
                def arm(bench, args, prompts, enabled, command):
                    for flag, value in {"--n-gpu-layers": "auto", "--device": "SYCL0", "--fit": "on",
                                        "--fit-target": "1024", "-ctk": "q8_0", "-ctv": "q8_0", "-c": "16384",
                                        "--spec-draft-n-min": "1", "--spec-draft-n-max": "7"}.items():
                        self.assertEqual(command[command.index(flag) + 1], value)
                    self.assertIn("--mmap", command)
                    self.assertNotIn("-ngl", command)
                    rows, responses = campaign(enabled)
                    if enabled:
                        for record in rows:
                            record["launch_id"] = 101 if failure != "reused_launch" else 100
                        if failure == "mismatch":
                            responses[-1]["payload"]["prompt"] = "wrong prompt"
                        if failure == "incomplete":
                            responses.pop()
                    summary = VERIFY.validate_campaign(rows, responses, enabled, args)
                    log = args.enabled_log if enabled else args.baseline_log
                    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
                    return {"responses": responses, "summary": summary,
                            "ok": failure != "shutdown", "cancelled": failure == "cancelled"}
                with patch.object(VERIFY, "load_helper", return_value=bench), patch.object(VERIFY, "run_arm", side_effect=arm), \
                     patch.object(VERIFY.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="intel-compute-runtime mock\n")):
                    result = VERIFY.main(argv)
                self.assertEqual(result, int(failure is not None))
                report = json.loads(paths["report"].read_text())
                self.assertEqual(report["ok"], failure is None)
                self.assertEqual(paths["q-artifact"].exists(), failure is None)
                if failure is None:
                    self.assertEqual(len(VERIFY.cycle.read_rows(paths["q-artifact"])), 6)
                    self.assertIn("not established", report["distribution_parity"])

    def test_mocked_arm_orchestration_and_failure_cleanup(self):
        for enabled, fail in ((False, False), (True, False), (True, "output"), (True, "fault"), (True, "startup"),
                              (True, "crash"), (True, "early_zero"), (True, "abort"), (True, "escalation"),
                              (True, "tenancy"), (True, "probe_timeout"), (True, "load")):
            with self.subTest(enabled=enabled, fail=fail), tempfile.TemporaryDirectory() as directory:
                args = settings()
                args.server = Path("/mock/llama-server")
                args.baseline_log = Path(directory) / "off.jsonl"
                args.enabled_log = Path(directory) / "on.jsonl"
                log = args.enabled_log if enabled else args.baseline_log
                rows, responses = campaign(enabled)
                proc = SimpleNamespace(pid=123, returncode=None)
                proc.poll = lambda: proc.returncode
                proc.wait = lambda: proc.returncode
                def send(sig):
                    if fail != "escalation" or sig == signal.SIGTERM:
                        proc.returncode = -signal.SIGABRT if fail == "abort" else 0
                proc.send_signal = Mock(side_effect=send)
                def start(bench, command, env, server_log):
                    bench._owned_servers[proc.pid] = proc
                    server_log.write_text("common_speculative_impl_draft_mtp\noffloaded 9/49 layers\n")
                    self.assertEqual("LLAMA_SPEC_SAMPLE_TEMP" in env, enabled)
                    return proc
                called = []
                def post(path, payload):
                    n = len(called)
                    called.append(payload)
                    log.write_text("".join(json.dumps(r) + "\n" for r in rows[:n + 1]))
                    with log.with_suffix(".server.log").open("a") as stream:
                        r = rows[n]
                        stream.write(f"spec_cycle_start: launch=100 request={r['request_id']} proposal={r['proposal_id']} sequence=0 cycle={r['cycle_id']}\n")
                    result = responses[n]["response"]
                    if fail == "output":
                        result["tokens_predicted"] = 0
                    if n == 5 and fail in {"crash", "early_zero"}:
                        proc.returncode = 42 if fail == "crash" else 0
                    return result
                bench = SimpleNamespace(RENDER_NODE="/dev/dri/renderD128",
                                        dmesg_lines=Mock(return_value=["old kernel line"]), wait_health=Mock(return_value=fail != "startup"),
                                        _owned_servers={}, curve_children=lambda proc: set(), new_gpu_faults=Mock(return_value=["xe fault"] if fail == "fault" else []), post_json=post,
                                        apply_chat_template=lambda messages: messages[0]["content"])
                prompts = [{"messages": [{"content": f"prompt {n}"}]} for n in range(3)]
                probe_result = subprocess.CompletedProcess(["fuser", bench.RENDER_NODE], 0,
                                                           "999" if fail == "tenancy" else "123", bench.RENDER_NODE + ":m")
                probe_error = subprocess.TimeoutExpired(probe_result.args, 30, output=b"123", stderr=b"partial")
                def preflight(_bench, observations, cancelled):
                    observations.append({"load1": 12 if fail == "load" and called else 0, "load_limit": 6})
                    if fail == "load" and called:
                        observations[-1]["error"] = "host quiet-settle deadline expired"
                        raise ValueError(observations[-1]["error"])
                    return observations[-1]
                with patch.object(VERIFY, "preflight", side_effect=preflight), patch.object(VERIFY, "start_server", side_effect=start), \
                     patch.object(VERIFY.subprocess, "run", return_value=probe_result,
                                  side_effect=probe_error if fail == "probe_timeout" else None), \
                     patch.object(VERIFY.time, "monotonic", side_effect=range(0, 10000, 40)), \
                     patch.object(VERIFY.time, "sleep"):
                    if fail:
                        with self.assertRaises(ValueError):
                            VERIFY.run_arm(bench, args, prompts, enabled, [str(args.server)])
                    else:
                        result = VERIFY.run_arm(bench, args, prompts, enabled, [str(args.server)])
                        self.assertEqual(result["summary"]["requests"], 6)
                        self.assertEqual(len(called), 6)
                self.assertFalse(bench._owned_servers)
                saved = json.loads(log.with_suffix(".arm.json").read_text())
                self.assertEqual(saved["ok"], not bool(fail))
                self.assertEqual(saved["shutdown"][0]["escalated"], fail == "escalation")
                self.assertEqual(saved["shutdown"][0]["returncode"], proc.returncode)
                if not fail:
                    self.assertEqual(saved["startup_placement_lines"], ["offloaded 9/49 layers"])
                    self.assertEqual(len(saved["tenancy_probes"]), 13)
                    self.assertTrue(all(p["attribution_ok"] for p in saved["tenancy_probes"]))
                if fail == "load":
                    self.assertEqual(saved["postflight_observations"], [{"load1": 12, "load_limit": 6,
                                      "error": "host quiet-settle deadline expired"}])
                    self.assertEqual(len(called), 6)
                if fail in {"tenancy", "probe_timeout"}:
                    self.assertEqual(len(saved["tenancy_probes"]), 1)
                    self.assertFalse(saved["tenancy_probes"][0]["attribution_ok"])
                    self.assertEqual(saved["tenancy_probes"][0]["stdout"], "999" if fail == "tenancy" else "123")
                bench.new_gpu_faults.assert_called_once()
                self.assertEqual(log.with_suffix(".responses.json").exists(), fail not in {"startup", "tenancy", "probe_timeout"})

    def test_mock_cancellation_handoff_and_repeated_cleanup_signals(self):
        for stage in ("handoff", "health", "cleanup"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                args = settings()
                args.server = Path("/mock/server")
                args.baseline_log = Path(directory) / "off.jsonl"
                args.enabled_log = Path(directory) / "on.jsonl"
                proc = SimpleNamespace(pid=123)
                installed = {}
                original = {sig: object() for sig in (signal.SIGINT, signal.SIGTERM)}
                def install(sig, handler):
                    previous = installed.get(sig, original[sig])
                    installed[sig] = handler
                    return previous
                def start(bench, *_):
                    bench._owned_servers[123] = proc
                    if stage == "handoff":
                        installed[signal.SIGINT](signal.SIGINT, None)
                    return proc
                def health(*_):
                    if stage == "health":
                        installed[signal.SIGTERM](signal.SIGTERM, None)
                    return False
                def stop(bench, child, status):
                    self.assertIs(child, proc)
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGTERM):
                        installed[sig](sig, None)
                    bench._owned_servers.pop(child.pid)
                    status.update(clean=True, returncode=0)
                bench = SimpleNamespace(_owned_servers={}, wait_health=health,
                                        dmesg_lines=lambda: [], new_gpu_faults=lambda *args: [])
                with patch.object(VERIFY.signal, "signal", side_effect=install), \
                     patch.object(VERIFY, "start_server", side_effect=start), \
                     patch.object(VERIFY, "stop_server", side_effect=stop) as stopped, \
                     patch.object(VERIFY, "preflight", return_value={}):
                    with self.assertRaises(ValueError):
                        VERIFY.run_arm(bench, args, [], True, [str(args.server)])
                stopped.assert_called_once()
                self.assertFalse(bench._owned_servers)
                self.assertEqual(installed, original)
                saved = json.loads(args.enabled_log.with_suffix(".arm.json").read_text())
                self.assertTrue(saved["cancelled"])
                self.assertFalse(saved["ok"])
                self.assertEqual("pid" in saved, stage != "handoff")

    def test_shutdown_escalation_for_owned_descendant_invalidates_zero_exit(self):
        proc = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda: 0)
        proc.poll = Mock(side_effect=[None, None, 0, 0, 0])
        proc.send_signal = Mock()
        bench = SimpleNamespace(_owned_servers={123: proc},
                                curve_children=Mock(side_effect=[{124}, {124}, {124}, set()]))
        status = {}
        with patch.object(VERIFY.os, "kill") as kill, \
             patch.object(VERIFY.time, "monotonic", side_effect=range(0, 1000, 40)):
            VERIFY.stop_server(bench, proc, status)
        self.assertTrue(status["escalated"])
        self.assertFalse(status["clean"])
        self.assertEqual(status["returncode"], 0)
        self.assertFalse(bench._owned_servers)
        kill.assert_called_once_with(124, signal.SIGINT)

    def test_forced_kill_and_cleanup_timeout_never_clean(self):
        for exits in (True, False):
            with self.subTest(exits=exits):
                proc = SimpleNamespace(pid=123, returncode=None)
                proc.poll = lambda: proc.returncode
                proc.wait = lambda: proc.returncode
                def send(sig):
                    if exits and sig == signal.SIGKILL:
                        proc.returncode = -signal.SIGKILL
                proc.send_signal = Mock(side_effect=send)
                bench = SimpleNamespace(_owned_servers={123: proc}, curve_children=lambda _: set())
                status = {}
                with patch.object(VERIFY.time, "monotonic", side_effect=range(0, 1000, 40)):
                    if exits:
                        VERIFY.stop_server(bench, proc, status)
                    else:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            VERIFY.stop_server(bench, proc, status)
                self.assertFalse(status["clean"])
                self.assertTrue(status["escalated"])
                self.assertEqual(status["signals"], ["SIGINT", "SIGTERM", "SIGKILL"])
                self.assertEqual(bool(bench._owned_servers), not exits)



class RealProcessTests(unittest.TestCase):
    """Post-timing only: real Linux parent/child signals, with HTTP/GPU probes mocked."""

    def run_cancellation(self, stage):
        self.assertEqual(sys.platform, "linux", "process qualification requires Linux")
        bench = VERIFY.load_helper("p06_process_bench", "bench_spec.py")
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready.json"
            helper = root / "helper.py"
            helper.write_text("""import json, os, signal, sys, time
from pathlib import Path
parent = os.getppid()
def stop(*args):
    if sys.argv[2] == 'cleanup' and os.getppid() == parent:
        os.kill(parent, signal.SIGTERM)
        time.sleep(.02)
        os.kill(parent, signal.SIGINT)
        time.sleep(.02)
        os.kill(parent, signal.SIGTERM)
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
child = os.fork()
if child:
    print('common_speculative_impl_draft_mtp', flush=True)
    Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child]))
while True: time.sleep(.01)
""")
            args = settings()
            args.server = Path(sys.executable)
            args.baseline_log, args.enabled_log = root / "off.jsonl", root / "on.jsonl"
            owned = []
            real_popen = subprocess.Popen

            def await_ready():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        pids = json.loads(ready.read_text())
                    except (FileNotFoundError, json.JSONDecodeError):
                        time.sleep(.01)
                        continue
                    return pids
                self.fail("helper did not become ready")

            def spawn(*args, **kwargs):
                existing = {pid for pid, row in bench.process_table().items() if row[0] == os.getpid()}
                child = real_popen(*args, **kwargs)
                owned.append(child)
                child.existing_children = existing
                await_ready()
                self.assertEqual(os.getpgid(child.pid), os.getpgrp())
                if stage == "handoff":
                    os.kill(os.getpid(), signal.SIGINT)
                return child

            def health(*_):
                await_ready()
                if stage == "health":
                    os.kill(os.getpid(), signal.SIGTERM)
                # Enter normal cleanup with no cancellation yet for the repeated-signal case.
                return False

            try:
                with patch.object(VERIFY.subprocess, "Popen", side_effect=spawn), \
                     patch.object(VERIFY, "preflight", return_value={}), \
                     patch.multiple(bench, wait_health=health, dmesg_lines=lambda: [], new_gpu_faults=lambda *args: []):
                    with self.assertRaises(ValueError):
                        VERIFY.run_arm(bench, args, [], True, [sys.executable, str(helper), str(ready), stage])
                self.assertEqual({sig: signal.getsignal(sig) for sig in old_handlers}, old_handlers)
                self.assertFalse(bench._owned_servers)
                self.assertTrue(all(child.poll() == 0 for child in owned))
                self.assertTrue(all(pid not in bench.process_table() for pid in await_ready()))
                saved = json.loads(args.enabled_log.with_suffix(".arm.json").read_text())
                self.assertTrue(saved["cancelled"])
                self.assertFalse(saved["ok"])
                self.assertEqual("pid" in saved, stage != "handoff")
                self.assertEqual(len(saved["shutdown"]), 1)
            finally:
                # Failure cleanup is limited to the actual Popen child and its owned descendants.
                for child in owned:
                    for pid in bench.curve_children(child) if hasattr(child, "existing_children") else set():
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
                    if hasattr(child, "existing_children"):
                        bench.curve_children(child)

    def test_parent_only_sigterm(self):
        self.run_cancellation("health")

    def test_pending_signal_at_ownership_handoff(self):
        self.run_cancellation("handoff")

    def test_repeated_signals_during_cleanup(self):
        self.run_cancellation("cleanup")


if __name__ == "__main__":
    unittest.main()
