#!/usr/bin/env python3
"""Offline tests for the explicit stochastic benchmark and paired-block analysis."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("speed", Path(__file__).parent / "perf" / "bench_spec_speed.py")
SPEED = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPEED)
BENCH = SPEED.bench
SAMPLING_QUALIFICATION = {"suppressed_token_ids": [], "unreported_settings": {}}


def response(payload, speculative=False):
    return {"tokens": list(range(512)), "tokens_predicted": 512, "stop": True, "truncated": False,
            "stop_type": "limit", "generation_settings": {**payload, "speculative.types": "draft-mtp" if speculative else "none"},
            "timings": {"prompt_n": len(payload["prompt"]), "predicted_n": 512,
                        "predicted_per_second": 30.0, "prompt_per_second": 100.0,
                        "draft_n": 600 if speculative else 0, "draft_n_accepted": 100 if speculative else 0}}


def records():
    result = []
    for mi, model in enumerate(SPEED.MODELS):
        for block in range(10):
            for arm in ("ABC" if model == "qwen4exp" else "AB"):
                for ci, cell in enumerate(SPEED.CELLS):
                    ratio = 1 if arm == "A" else 1.15 + .005 * math.sin(block + ci + mi)
                    if arm == "C":
                        ratio *= 1.08 + .005 * math.cos(block + ci)
                    result.append({"model": model, "block": block, "arm": arm, "cell": cell,
                                   "elapsed_s": (10 + .1 * block) / ratio, "fixed_work_valid": True})
    return result


class StochasticTests(unittest.TestCase):
    def test_real_response_masks_and_unserialized_reasoning_require_qualification(self):
        payload = BENCH.stochastic_payload([1] * 512, 3)
        resp = response(payload)
        settings = resp["generation_settings"]
        del settings["reasoning_budget_tokens"], settings["reasoning_control"]
        # Actual Qwen3 responses duplicate these EOG masks from CLI and request settings.
        ids = [128247, 151643, 151645, 151662, 151663, 151664]
        settings["logit_bias"] = [{"token": token, "bias": None} for token in ids] * 2
        qualified = {"suppressed_token_ids": ids,
                     "unreported_settings": {"reasoning_budget_tokens": -1, "reasoning_control": False}}
        self.assertTrue(BENCH.analyze_stochastic_response(resp, payload, False, qualified)["fixed_work_valid"])
        for masks in ([], settings["logit_bias"] + [{"token": 123, "bias": None}],
                      settings["logit_bias"] + [{"token": ids[0], "bias": 0}],
                      [{"token": True, "bias": None}], [{"token": ids[0]}]):
            broken = copy.deepcopy(resp)
            broken["generation_settings"]["logit_bias"] = masks
            with self.subTest(masks=masks), self.assertRaisesRegex(ValueError, "logit_bias"):
                BENCH.analyze_stochastic_response(broken, payload, False, qualified)
        for missing in ("reasoning_budget_tokens", "reasoning_control"):
            broken = copy.deepcopy(qualified)
            del broken["unreported_settings"][missing]
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, missing):
                BENCH.analyze_stochastic_response(resp, payload, False, broken)
        for key, value in (("reasoning_budget_tokens", 0), ("reasoning_control", True)):
            broken = copy.deepcopy(resp)
            broken["generation_settings"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                BENCH.analyze_stochastic_response(broken, payload, False, qualified)
        for broken in ({**qualified, "suppressed_token_ids": ids + ids},
                       {**qualified, "suppressed_token_ids": [True]},
                       {**qualified, "unreported_settings": {"grammar": ""}},
                       {**qualified, "unreported_settings": {"reasoning_control": 0}}):
            with self.assertRaisesRegex(ValueError, "sampling qualification"):
                BENCH.analyze_stochastic_response(resp, payload, False, broken)

    def test_non_argmax_sample_is_valid_only_in_explicit_stochastic_path(self):
        payload = BENCH.stochastic_payload([1] * 512, 3)
        resp = response(payload, True)
        result = BENCH.analyze_stochastic_response(resp, payload, True, SAMPLING_QUALIFICATION)
        self.assertTrue(result["fixed_work_valid"])
        self.assertFalse(result["distribution_equality_proven"])
        resp["completion_probabilities"] = [{"id": token, "logprob": -.5,
                                             "top_logprobs": [{"id": token + 1}]} for token in resp["tokens"]]
        self.assertFalse(BENCH.analyze_native_response(resp)["verifier_invariant_ok"])

    def test_bad_settings_counts_or_route_fail_closed(self):
        payload = BENCH.stochastic_payload([1] * 512, 3)
        changes = [("generation_settings", "temperature", 0), ("generation_settings", "ignore_eos", False),
                   ("generation_settings", "samplers", ["temperature", "top_p"]),
                   ("generation_settings", "reasoning_budget_tokens", 0),
                   ("generation_settings", "reasoning_budget_tokens", None),
                   ("generation_settings", "reasoning_control", True),
                   ("generation_settings", "reasoning_control", None),
                   ("generation_settings", "speculative.types", "none"), ("timings", "draft_n", 0),
                   ("timings", "predicted_per_second", math.nan), ("timings", "prompt_n", 511)]
        for parent, key, value in changes:
            resp = response(payload, True)
            resp[parent][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                BENCH.analyze_stochastic_response(resp, payload, True, SAMPLING_QUALIFICATION)
        for key, value in (("tokens", [True] * 512), ("stop_type", "eos"), ("truncated", True),
                           ("tokens_predicted", True), ("tokens", [1] * 511)):
            resp = response(payload)
            resp[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                BENCH.analyze_stochastic_response(resp, payload, False, SAMPLING_QUALIFICATION)

    def test_payload_and_fixture_are_exact(self):
        for count in (0, 511, 513, 4095):
            with self.assertRaises(ValueError):
                BENCH.stochastic_payload([1] * count, 3)
        for seed in (-1, True, 2**32 - 1):
            with self.assertRaises(ValueError):
                BENCH.stochastic_payload([1] * 512, seed)
        fixture = SPEED.fixed_fixture([8, 9], [2, 3, 4], 512)
        self.assertEqual(len(fixture), 512)
        self.assertEqual(fixture[-2:], [8, 9])
        self.assertEqual(fixture[:6], [2, 3, 4, 2, 3, 4])

    def test_schedule_balances_every_pair_and_fixes_seeds(self):
        orders = SPEED.schedule("qwen4exp")
        for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
            self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), 5)
        seeds = {SPEED.request_seed(r, b, c, warmup) for r in ("initial", "heldout")
                 for b in range(10) for c in range(6) for warmup in (True, False)}
        self.assertEqual(len(seeds), 240)

    def test_bootstrap_keeps_complete_matrix_and_refuses_degenerate_data(self):
        valid = records()
        for bad in (valid[:-1], valid + [valid[0]], [{**row, "elapsed_s": 1} for row in valid]):
            with self.assertRaises(ValueError):
                SPEED.analyze(bad, draws=100)
        bad = copy.deepcopy(valid)
        bad[0]["elapsed_s"] = math.nan
        with self.assertRaises(ValueError):
            SPEED.analyze(bad, draws=100)

    def test_simultaneous_bounds_and_structural_reuse(self):
        result = SPEED.analyze(records(), draws=100)
        self.assertEqual(len(result["bounds"]), 57)
        self.assertEqual(result["round_family_alpha"], .025)
        self.assertTrue(result["claims"]["default"]["passed"])
        self.assertTrue(result["claims"]["enabled"]["passed"])
        self.assertNotIn("attribution/llama31/code_edit-512", result["bounds"])
        for bound in result["bounds"].values():
            self.assertLess(bound["lower"], bound["ratio"])
            self.assertGreater(bound["upper"], bound["ratio"])

    def test_single_cell_regression_prevents_aggregate_claim(self):
        data = records()
        for row in data:
            if row["model"] == "llama31" and row["cell"] == "code_edit-512" and row["arm"] == "B":
                row["elapsed_s"] *= 1.3
        result = SPEED.analyze(data, draws=100)
        self.assertFalse(result["claims"]["enabled"]["all_cells_exclude_3pct_latency_regression"])
        self.assertFalse(result["claims"]["enabled"]["passed"])

    def test_logging_environment_is_removed(self):
        with mock.patch.dict(SPEED.os.environ, {"LLAMA_SPEC_LOG": "/bad", "LLAMA_TRACE": "1",
                                              "LLAMA_SPEC_SAMPLE_TEMP": "9", "GGML_SYCL_FA_XMX": "1",
                                              "LD_PRELOAD": "/bad.so", "LD_AUDIT": "/audit.so"}):
            env = SPEED.child_environment(Path("/tmp/build/bin/llama-server"), False)
        self.assertNotIn("LLAMA_SPEC_LOG", env)
        self.assertNotIn("LLAMA_TRACE", env)
        self.assertNotIn("LLAMA_SPEC_SAMPLE_TEMP", env)
        self.assertNotIn("GGML_SYCL_FA_XMX", env)
        self.assertNotIn("LD_PRELOAD", env)
        self.assertNotIn("LD_AUDIT", env)

    def test_changed_cpu_runtime_control_rejects_launch_before_preflight(self):
        server = Path("/tmp/build/bin/llama-server")
        controls = {"OMP_NUM_THREADS": "12", "KMP_BLOCKTIME": "0", "GOMP_CPU_AFFINITY": "0-11",
                    "MKL_NUM_THREADS": "12", "OPENBLAS_NUM_THREADS": "12", "BLIS_NUM_THREADS": "12",
                    "VECLIB_MAXIMUM_THREADS": "12", "TBB_NUM_THREADS": "12", "DNNL_MAX_CPU_ISA": "AVX2",
                    "ONEDNN_MAX_CPU_ISA": "AVX2"}
        with mock.patch.dict(SPEED.os.environ, controls):
            env = SPEED.qualified_environment(SPEED.child_environment(server, False))
            self.assertEqual({k: env[k] for k in controls}, controls)
            config = {"arms": {"A": {"server": {"path": str(server)}, "placement_args": []}},
                      "model_files": [{"path": "/tmp/model.gguf"}], "context": 8192, "threads": 12,
                      "effective_environment": {"A": env}}
            for key in controls:
                with self.subTest(key=key), mock.patch.dict(SPEED.os.environ, {key: "changed"}), \
                        mock.patch.object(SPEED.distribution, "preflight") as preflight:
                    with self.assertRaisesRegex(ValueError, "environment differs"):
                        SPEED.run_launch({"round": "initial"}, "llama31", config, [], 0, "A", Path("/unused"))
                    preflight.assert_not_called()

    def test_analysis_binds_every_payload_to_hash_checked_fixture_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures, summaries = {}, []
            def store(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return {"path": str(path), "sha256": SPEED.digest(path)}
            for model in SPEED.MODELS:
                cells = [{"id": cell, "tokens": [1] * int(cell.rsplit("-", 1)[1])} for cell in SPEED.CELLS]
                fixtures[model] = {"fixtures": store(model + "-fixture.json", {"model": model, "round": "initial", "cells": cells}),
                                   "sampling_qualification": {**SAMPLING_QUALIFICATION, "report": store(model + "-sampling.json", {"scope": "unit fixture"})}}
            manifest = store("manifest.json", {"round": "initial", "models": fixtures})
            for model in SPEED.MODELS:
                launches = []
                for block, order in enumerate(SPEED.schedule(model)):
                    for arm in order:
                        rows = []
                        for ci, cell in enumerate(SPEED.CELLS):
                            for warmup in (True, False):
                                payload = BENCH.stochastic_payload([1] * int(cell.rsplit("-", 1)[1]),
                                    SPEED.request_seed("initial", block, ci, warmup))
                                resp = response(payload, model == "qwen4exp")
                                del resp["generation_settings"]["prompt"]
                                rows.append({"model": model, "block": block, "arm": arm, "cell": cell,
                                             "warmup": warmup, "payload": payload, "response": resp,
                                             "fixed_work_valid": True, "elapsed_s": 1.0})
                        launches.append({"model": model, "block": block, "arm": arm, "ok": True,
                                         "faults": [], "responses": rows})
                summaries.append({"model": model, "round": "initial", "manifest": manifest,
                                  "manifest_sha256": manifest["sha256"], "ok": True, "launches": launches})
            paths = [store(model + "-summary.json", summary)["path"] for model, summary in zip(SPEED.MODELS, summaries)]
            output = root / "analysis.json"
            argv = ["bench_spec_speed.py", "analyze", "--inputs", *paths, "--output", str(output)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(SPEED, "analyze", return_value={"fixture": "passed"}) as analyze:
                self.assertEqual(SPEED.main(), 0)
                self.assertEqual(len(analyze.call_args.args[0]), 540)
                output.unlink()
                for index in (0, 1):
                    payload = summaries[0]["launches"][0]["responses"][index]["payload"]
                    payload["prompt"][0] = 2
                    store("llama31-summary.json", summaries[0])
                    analyze.reset_mock()
                    with self.subTest(warmup=index == 0), self.assertRaisesRegex(ValueError, "prompt differs from frozen fixture"):
                        SPEED.main()
                    analyze.assert_not_called()
                    payload["prompt"][0] = 1
                store("llama31-summary.json", summaries[0])
                Path(fixtures["llama31"]["fixtures"]["path"]).write_text("changed fixture")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    SPEED.main()

    def test_mapped_project_and_runtime_images_require_exact_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / name for name in ("libggml-sycl.so", "libze_intel_gpu.so", "libc.so.6", "model.gguf")]
            for path in files:
                path.write_bytes(b"GGUF" if path.suffix == ".gguf" else b"\x7fELFfixture")
            rows = [{"path": str(p), "sha256": SPEED.digest(p)} for p in files[:-1]]
            lines = []
            for path in files:
                st = path.stat()
                lines.append(f"1000-2000 r--p 00000000 {SPEED.os.major(st.st_dev):02x}:{SPEED.os.minor(st.st_dev):02x} {st.st_ino} {path}")
            proc = mock.Mock(pid=321)
            with mock.patch.object(Path, "read_text", return_value="\n".join(lines)):
                result = SPEED.capture_maps(proc, rows)
                self.assertEqual(set(result["image_hashes"]), {str(p) for p in files[:-1]})
                for missing in range(3):
                    with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, "unqualified image"):
                        SPEED.capture_maps(proc, rows[:missing] + rows[missing + 1:])
                changed = copy.deepcopy(rows)
                changed[1]["sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    SPEED.capture_maps(proc, changed)
                with self.assertRaisesRegex(ValueError, "conflicting image"):
                    SPEED.capture_maps(proc, rows + [changed[1]])
            for bad in (lines[0] + " (deleted)", lines[0].replace(str(files[0].stat().st_ino), "0")):
                with mock.patch.object(Path, "read_text", return_value=bad), self.assertRaises(ValueError):
                    SPEED.capture_maps(proc, rows)

    def test_manifest_requires_matching_qualification_and_complete_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def store(name, value):
                path = root / name
                path.write_text(json.dumps(value))
                return {"path": str(path), "sha256": SPEED.digest(path)}
            artifact = store("artifact", "fixture bytes")
            prompt = store("prompts.json", [])
            prereg = store("protocol.json", {"analysis_seed": SPEED.ANALYSIS_SEED, "bootstrap_draws": SPEED.DRAWS,
                            "round_family_alpha": SPEED.ROUND_ALPHA, "context": 8192, "threads": 12,
                            "prompts": {"initial-prompts.json": prompt}})
            configs = {}
            for model in SPEED.MODELS:
                fixtures = store(model + ".json", {"schema_version": 1, "model": model, "round": "initial",
                                 "prompt_source": prompt, "tokenizer_parity": True,
                                 "tokenizer_flags": {"add_special": False, "parse_special": True},
                                 "cells": [{"id": c, "tokens": [1] * int(c.rsplit("-", 1)[1])} for c in SPEED.CELLS]})
                build = {"server": artifact, "artifacts": [artifact], "source_commit": "candidate", "placement_args": []}
                configs[model] = {"context": 8192, "threads": 12, "model_files": [artifact], "fixtures": fixtures,
                                  "sampling_qualification": {**SAMPLING_QUALIFICATION, "report": artifact},
                                  "draft_model": artifact["path"], "arms": {"A": {**build, "source_commit": "2f623c6607c078d1069dac23e9d8af9cb146e771"},
                                  "B": build, "C": build}}
            platform = {"kernel": SPEED.os.uname().release, "driver": "xe", "runtime_files": [artifact]}
            qualification = {"status": "qualified", "distribution_qualification": "established", "scope": "fixture",
                             "platform": platform, "models": configs, **{field: artifact for field in
                             ("abi_report", "correctness_report", "distribution_report", "batch_sequential_report", "route_report")}}
            manifest = {"schema_version": 1, "round": "initial", "preregistration": prereg,
                        "qualification": store("qualification.json", qualification), "qualification_scope": "fixture",
                        "platform": platform, "models": configs}
            with mock.patch.object(SPEED, "PREREGISTRATION_SHA256", prereg["sha256"]), \
                    mock.patch.object(BENCH, "render_driver", return_value="xe"):
                for model in SPEED.MODELS:
                    self.assertEqual(len(SPEED.validate_manifest(manifest, model)[1]), 6)
                for field, value in (("status", "incomplete"), ("distribution_qualification", "incomplete"),
                                     ("scope", "unqualified"), ("route_report", {"path": str(root / "missing"), "sha256": "0"})):
                    broken = copy.deepcopy(manifest)
                    broken["qualification"] = store("broken-qualification.json", {**qualification, field: value})
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        SPEED.validate_manifest(broken, "llama31")
                broken = copy.deepcopy(manifest)
                broken["models"]["llama31"]["context"] = 4096
                with self.assertRaisesRegex(ValueError, "context/thread"):
                    SPEED.validate_manifest(broken, "llama31")
                for field, value in (("runtime_files", []), ("driver", "i915")):
                    broken = copy.deepcopy(manifest)
                    broken["platform"][field] = value
                    q = copy.deepcopy(qualification)
                    q["platform"] = broken["platform"]
                    broken["qualification"] = store("broken-qualification.json", q)
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        SPEED.validate_manifest(broken, "llama31")
                Path(artifact["path"]).write_text("changed after qualification")
                with self.assertRaisesRegex(ValueError, "identity changed"):
                    SPEED.validate_manifest(manifest, "llama31")
            with mock.patch.object(SPEED, "PREREGISTRATION_SHA256", "0" * 64), self.assertRaisesRegex(ValueError, "frozen protocol"):
                SPEED.validate_manifest(manifest, "llama31")

    def test_failure_retains_evidence_and_stops_owned_server(self):
        payload = BENCH.stochastic_payload([1] * 512, 1)
        prompts = [{"id": cell, "tokens": [1] * int(cell.rsplit("-", 1)[1])} for cell in SPEED.CELLS]
        server = Path("/tmp/build/bin/llama-server")
        env = SPEED.child_environment(server, False)
        relevant = SPEED.qualified_environment(env)
        config = {"arms": {"A": {"server": {"path": str(server)}, "placement_args": [], "artifacts": []}},
                  "model_files": [{"path": "/tmp/model.gguf"}], "context": 8192, "threads": 4,
                  "effective_environment": {"A": relevant}}
        proc = mock.Mock()
        proc.poll.return_value = None
        def start(*_):
            BENCH._owned_servers[123] = proc
            return proc
        def stop(_bench, _proc, status):
            status["clean"] = True
            BENCH._owned_servers.clear()
        with tempfile.TemporaryDirectory() as root, \
                mock.patch.multiple(SPEED.distribution, preflight=mock.Mock(), owned_tenancy=mock.Mock(),
                                    start_server=mock.Mock(side_effect=start), stop_server=mock.Mock(side_effect=stop)), \
                mock.patch.multiple(BENCH, render_driver=mock.Mock(return_value="xe"),
                                    dmesg_lines=mock.Mock(return_value=["old"]),
                                    wait_health=mock.Mock(return_value=True),
                                    post_json=mock.Mock(side_effect=RuntimeError("request failure"))), \
                mock.patch.object(SPEED, "capture_maps", return_value={}):
            with self.assertRaisesRegex(ValueError, "request failure"):
                SPEED.run_launch({"round": "initial", "platform": {"runtime_files": []}},
                                 "llama31", config, prompts, 0, "A", Path(root))
            self.assertTrue((Path(root) / "00-A.json").exists())
            command = json.loads((Path(root) / "00-A.json").read_text())["command"]
            self.assertEqual(command[command.index("--reasoning") + 1], "off")
            self.assertEqual(command[command.index("--reasoning-budget") + 1], "-1")
            self.assertFalse(BENCH._owned_servers)


if __name__ == "__main__":
    unittest.main()
