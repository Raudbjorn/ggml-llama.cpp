#!/usr/bin/env python3
"""P06 six-request OFF/ON evidence, not a distribution-parity or speedup test.

Runtime requires the coordinator's binary/model identity, GPU lease, idle builds,
README regression floor, and outer `flock /tmp/a770.lock timeout ...` protocol.
No services or downloads. The registered test is scripts.test_spec_distribution.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time


def load_helper(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cycle = load_helper("p06_cycle", "verify-spec-cycle-log.py")
require = cycle.require
TOLERANCE = 1e-5
LOAD_SETTLE_SECONDS = 180
LOAD_POLL_SECONDS = 2
IDENTITY = ("launch_id", "request_id", "proposal_id", "sequence_id", "cycle_id")
REGRESSION_FLOOR = [
    "timeout 240 ./build-sycl/bin/test-qwen4exp-mtp",
    "timeout 240 ./build-sycl/bin/test-qwen4exp-mtp --q8-kv",
    "timeout 240 python3 -m unittest scripts.test_bench_spec",
]


def integer(value, label: str, maximum: int = 2**64 - 1) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"invalid {label}")
    return value


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_distribution(rows: list[dict], enabled: bool, temperature: float, top_p: float) -> dict:
    require(bool(rows), "empty instrumentation")
    cycle.validate_rows(rows)
    proposals = set()
    selected = distribution = exact = 0
    for row in rows:
        require(type(row["schema_version"]) is int and row["schema_version"] == 1, "unknown root schema")
        for field in IDENTITY:
            integer(row[field], field)
        for field in ("drafted_count", "accepted_count", "attempted_count"):
            integer(row[field], field)
        proposal = integer(row["proposal_id"], "proposal ID")
        key = (row["launch_id"], proposal)
        require(proposal > 0 and key not in proposals, "duplicate or zero proposal identity")
        proposals.add(key)
        nd = integer(row["distribution_attempts"], "distribution attempts", 1)
        ne = integer(row["exact_match_attempts"], "exact-match attempts")
        require(type(row["selected"]) is bool, "invalid selection")
        require(isinstance(row["sampling_reason"], str) and row["sampling_reason"], "missing sampling reason")
        require(row["sampling_mode"] in {"distribution", "exact_match"}, "unknown sampling mode")
        require((row["sampling_mode"] == "distribution") == (nd == 1), "first decision/counter mismatch")
        if nd:
            require(row["sampling_reason"] == "distribution", "distribution reason mismatch")
        else:
            require(row["sampling_reason"] != "distribution", "exact reason mismatch")
        for attempt in row["attempts"]:
            integer(attempt["token"], "sampled token", 2**31 - 1)
            integer(attempt["top1_token"], "top1 token", 2**31 - 1)
            require(finite(attempt["sampled_probability"]) and 0 <= attempt["sampled_probability"] <= 1,
                    "invalid sampled probability")
        if not enabled:
            require(row["q_records"] is None and row["draft_sampling"] is None,
                    "baseline must have null q and draft sampling")
            require(nd == 0 and row["sampling_mode"] == "exact_match", "baseline used distribution")
        else:
            settings = row["draft_sampling"]
            require(isinstance(settings, dict), "missing enabled draft sampling")
            require(finite(settings["temperature"]) and abs(settings["temperature"] - temperature) <= TOLERANCE
                    and finite(settings["top_p"]) and abs(settings["top_p"] - top_p) <= TOLERANCE
                    and type(settings["top_k"]) is int and settings["top_k"] == 10,
                    "draft sampling identity mismatch")
            require(settings["backend_sampling"] is False, "backend draft sampling is ineligible")
            records = row["q_records"]
            if row["selected"]:
                require(isinstance(records, list) and len(records) == row["drafted_count"] > 0,
                        "missing or misaligned selected q records")
            else:
                require(records is None, "unselected proposal retained q records")
            indices = set()
            for q in records or []:
                require(type(q["schema_version"]) is int and q["schema_version"] == 1, "unknown q schema")
                require(integer(q["proposal_id"], "q proposal ID") == proposal, "cross-proposal q record")
                index = integer(q["draft_index"], "draft index")
                require(index not in indices and index < row["drafted_count"], "duplicate or invalid draft index")
                indices.add(index)
                token = integer(q["draft_token"], "q draft token", 2**31 - 1)
                require(token == row["proposal_tokens"][index], "q draft token mismatch")
                support = q["support"]
                require(isinstance(support, list) and 0 < len(support) <= 10, "invalid sparse support size")
                masses = {}
                for entry in support:
                    tid = integer(entry["token"], "support token", 2**31 - 1)
                    probability = entry["probability"]
                    require(tid not in masses, "duplicate support token")
                    require(finite(probability) and probability >= 0, "invalid q probability")
                    masses[tid] = probability
                total = math.fsum(masses.values())
                require(abs(total - 1) <= TOLERANCE, "q is not normalized")
                require(finite(q["probability_sum"]) and abs(q["probability_sum"] - total) <= TOLERANCE,
                        "q sum field mismatch")
                require(masses.get(token, 0) > 0, "sampled token has no positive q mass")
                attempt = row["attempts"][index]
                require(abs(masses[token] - attempt["sampled_probability"]) <= TOLERANCE,
                        "sampled diagnostic differs from q")
                require(attempt["top1_token"] in masses and
                        abs(masses[attempt["top1_token"]] - attempt["top1_probability"]) <= TOLERANCE and
                        abs(max(masses.values()) - attempt["top1_probability"]) <= TOLERANCE,
                        "top1 diagnostic differs from q")
        if row["selected"]:
            require(row["outcome"] == "completed" and bool(row["verify_phases"]), "selected cycle did not complete")
            require(nd + ne >= 1, "selected cycle has no actual acceptance")
            selected += 1
        else:
            require(nd == ne == 0, "unselected cycle counted acceptance")
        # A repair phase is not an acceptance invocation; never compare their counts.
        distribution += nd
        exact += ne
    require(selected > 0, "no selected proposals")
    if enabled:
        require(distribution > 0, "no actual distribution verification; q recording alone is insufficient")
    drafted = sum(row["drafted_count"] for row in rows if row["selected"])
    accepted = sum(row["accepted_count"] for row in rows if row["selected"])
    require(drafted > 0, "zero drafted tokens")
    return {"selected_cycles": selected, "distribution_attempts": distribution, "exact_match_attempts": exact,
            "drafted": drafted, "accepted": accepted, "accepted_per_drafted": accepted / drafted}


def request_payload(prompt: str, n_predict: int, seed: int, temperature: float) -> dict:
    return {"prompt": prompt, "n_predict": n_predict, "seed": seed, "temperature": temperature,
            "samplers": ["temperature"], "top_k": 0, "top_p": 1.0, "min_p": 0.0,
            "dynatemp_range": 0.0, "repeat_penalty": 1.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0, "dry_multiplier": 0.0, "mirostat": 0,
            "grammar": "", "logit_bias": [], "backend_sampling": False,
            "reasoning_budget_tokens": -1, "reasoning_control": False,
            "cache_prompt": False, "stream": False}


def validate_response(response: dict, payload: dict) -> None:
    require(integer(response["tokens_predicted"], "output count") > 0, "zero-output response")
    require(response["stop"] is True and response["truncated"] is False, "incomplete or truncated response")
    settings = response["generation_settings"]
    for key in ("seed", "temperature", "samplers", "top_k", "top_p", "min_p", "dynatemp_range",
                "repeat_penalty", "presence_penalty", "frequency_penalty", "dry_multiplier", "mirostat",
                "grammar", "logit_bias", "backend_sampling", "n_predict", "stream"):
        expected, actual = payload[key], settings[key]
        if type(expected) is float:
            require(finite(actual) and abs(actual - expected) <= TOLERANCE, f"effective {key} mismatch")
        else:
            require(type(actual) is type(expected) and actual == expected, f"effective {key} mismatch")
    # NONE is the source's serialized placeholder, not an active implementation.
    require(settings["speculative.types"] in {"draft-mtp", "none,draft-mtp"}, "effective MTP types mismatch")


def validate_campaign(rows: list[dict], responses: list[dict], enabled: bool, args) -> dict:
    require(len(responses) == 6, "campaign requires all six requests")
    require(len({r["launch_id"] for r in rows}) == 1, "expected one fresh launch per arm")
    request_ids = set()
    for number, item in enumerate(responses):
        require(item["client_request"] == number + 1 and item["prompt_index"] == number // 2
                and item["repeat"] == number % 2, "request matrix mismatch")
        request_id = item["request_id"]
        require(request_id not in request_ids, "duplicate request ID")
        request_ids.add(request_id)
        expected = request_payload(item["payload"]["prompt"], item["payload"]["n_predict"],
                                   args.seed, args.target_temperature)
        require(item["payload"] == expected, "request sampling identity mismatch")
        validate_response(item["response"], expected)
        cycle.reconcile([r for r in rows if r["request_id"] == request_id], item["response"])
    require({r["request_id"] for r in rows} == request_ids, "orphan request rows")
    summary = validate_distribution(rows, enabled, args.draft_temperature, args.draft_top_p)
    return {**summary, "requests": 6, "reconciled_request_ids": sorted(request_ids)}


def child_environment(parent: dict, server: Path, log: Path, enabled: bool, args) -> dict:
    env = {key: value for key, value in parent.items() if not key.startswith("LLAMA_")}
    env.update(LLAMA_SPEC_LOG=str(log.resolve()), LLAMA_TRACE="1", LLAMA_SPEC_CHAIN="0", LLAMA_DRAFT_TOP_K="10")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(p for p in [str(server.parent), *parent.get("LD_LIBRARY_PATH", "").split(os.pathsep)] if p)
    if enabled:
        env.update(LLAMA_SPEC_SAMPLE_TEMP=str(args.draft_temperature), LLAMA_SPEC_DRAFT_TOPP=str(args.draft_top_p))
    return env


def preflight(bench, observations: list[dict] | None = None, cancelled=lambda: False) -> dict:
    observations = observations if observations is not None else []
    started = time.monotonic()
    deadline = started + LOAD_SETTLE_SECONDS
    threshold = max(2, (os.cpu_count() or 1) / 4)
    while True:
        observation = {"elapsed_seconds": time.monotonic() - started, "load_limit": threshold,
                       "settle_limit_seconds": LOAD_SETTLE_SECONDS, "process_checks": []}
        observations.append(observation)
        try:
            load = os.getloadavg()[0]
            observation.update(entry_load1=load, load1=load)
            require(finite(load) and load >= 0, "host load unavailable")
            require(not cancelled(), "host quiet-settle cancelled")
            require(time.monotonic() < deadline, "host quiet-settle deadline expired")
            holders = bench.gpu_holders()
            observation["holders"] = holders
            require(not holders, f"GPU tenancy unavailable: {holders}")
            for pattern in ("llama-(server|bench|cli|completion|perplexity)", "ninja|icpx|icx|cc1|cc1plus|cargo|rustc"):
                remaining = deadline - time.monotonic()
                require(remaining > 0, "host quiet-settle deadline expired")
                result = subprocess.run(["pgrep", "-a", pattern], capture_output=True, text=True,
                                        timeout=min(10, remaining))
                observation["process_checks"].append({"pattern": pattern, "returncode": result.returncode,
                                                       "stdout": result.stdout, "stderr": result.stderr})
                require(result.returncode == 1 and not result.stdout and not result.stderr,
                        f"foreign process or unavailable process gate: {result.stdout} {result.stderr}")
            remaining = deadline - time.monotonic()
            require(remaining > 0, "host quiet-settle deadline expired")
            services = subprocess.run(["systemctl", "is-active", "llama-sycl.cpp.service", "llama-gpu@*",
                                       "llama-vulkan.cpp.service"], capture_output=True, text=True,
                                      timeout=min(10, remaining))
            states = services.stdout.splitlines()
            observation.update(service_states=states, service_returncode=services.returncode,
                               service_stderr=services.stderr)
            require(services.returncode in {3, 4} and not services.stderr and bool(states)
                    and all(s in {"inactive", "failed", "unknown"} for s in states), "service gate unavailable or active")
            require(not cancelled(), "host quiet-settle cancelled")
            load = os.getloadavg()[0]
            observation["load1"] = load
            require(finite(load) and load >= 0, "host load unavailable")
            require(time.monotonic() < deadline, "host quiet-settle deadline expired")
            if load <= threshold:
                return observation
            # Load averages retain workload history; require the same idle gates while they settle.
            time.sleep(min(LOAD_POLL_SECONDS, max(0, deadline - time.monotonic())))
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            observation["error"] = str(error) or "cancelled"
            raise


def start_server(bench, command: list[str], env: dict, log: Path):
    """P02's shared-timeout-group ownership, with an explicit child environment."""
    require(not bench._owned_servers, "previous server still owned")
    if ctypes.CDLL(None, use_errno=True).prctl(bench._PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot own server descendants")
    children = {pid for pid, row in bench.process_table().items() if row[0] == os.getpid()}
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        with log.open("x") as output:
            proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env,
                                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, blocked))
        proc.owned_pgid = None
        proc.existing_children = children
        bench._owned_servers[proc.pid] = proc
        return proc
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def stop_server(bench, proc, status: dict) -> None:
    """P02's bounded shared-group drain, retaining shutdown evidence per signal."""
    status.update(pid=proc.pid, returncode_before=proc.poll(), signals=[], escalated=False, clean=False)
    for sig, grace in ((signal.SIGINT, 20), (signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        sent = set()
        leader_alive = proc.poll() is None
        children = bench.curve_children(proc)
        if leader_alive or children:
            status["signals"].append(sig.name)
            status["escalated"] |= sig != signal.SIGINT
        if leader_alive:
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while True:
            children = bench.curve_children(proc)
            for pid in children - sent:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                sent.add(pid)
            if proc.poll() is not None and not children:
                status["returncode"] = proc.wait()
                bench._owned_servers.pop(proc.pid, None)
                status["clean"] = (status["returncode_before"] is None and
                                   status["signals"] == ["SIGINT"] and status["returncode"] == 0)
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(.05)
    raise subprocess.TimeoutExpired("server process tree cleanup", 35)


def owned_tenancy(bench, proc, probes: list[dict]) -> list[int]:
    probe = {"command": ["fuser", bench.RENDER_NODE], "returncode": None,
             "stdout": "", "stderr": "", "attribution_ok": False}
    probes.append(probe)
    try:
        result = subprocess.run(probe["command"], check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        probe.update(exception={"type": type(error).__name__, "message": str(error)},
                     stdout=os.fsdecode(getattr(error, "stdout", None) or ""),
                     stderr=os.fsdecode(getattr(error, "stderr", None) or ""))
        raise
    probe.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    # fuser(1): stdout is PIDs; stderr is the device label and default c/e/r/m access markers.
    require(result.returncode == 0 and
            bool(re.fullmatch(r"[1-9][0-9]*(?:\s+[1-9][0-9]*)*", result.stdout.strip())) and
            bool(re.fullmatch(r"\s*" + re.escape(bench.RENDER_NODE) + r":\s*[cerm\s]*", result.stderr)),
            "GPU holder attribution unavailable")
    holders = set(map(int, result.stdout.split()))
    probe["holders"] = sorted(holders)
    probe["owned_pids"] = sorted({proc.pid} | bench.curve_children(proc))
    require(holders <= set(probe["owned_pids"]), "foreign GPU holder during requests")
    probe["attribution_ok"] = True
    return sorted(holders)


def run_arm(bench, args, prompts: list[dict], enabled: bool, command: list[str]) -> dict:
    log = args.enabled_log if enabled else args.baseline_log
    server_log = log.with_suffix(".server.log")
    env = child_environment(os.environ, args.server, log, enabled, args)
    result = {"enabled": enabled, "command": command, "ok": False, "cancelled": False,
              "server_log": str(server_log.resolve()),
              "environment": {k: v for k, v in env.items() if k.startswith(("LLAMA_", "GGML_", "SYCL_", "UR_", "ZE_", "ZES_"))
                              or k in {"LD_LIBRARY_PATH", "ONEAPI_DEVICE_SELECTOR", "OCL_ICD_VENDORS"}},
              "unset_environment": sorted(k for k in os.environ if k.startswith("LLAMA_") and k not in env),
              "responses": [], "shutdown": [], "tenancy_probes": [],
              "preflight_observations": [], "postflight_observations": []}
    proc, before = None, None
    handlers = {}
    cleaning = False

    def cancel(signum, _frame):
        result["cancelled"] = True
        result["ok"] = False
        result["error"] = f"cancelled by signal {signum}"
        if not cleaning:
            raise SystemExit(result["error"])

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, cancel)
        result["preflight"] = preflight(bench, result["preflight_observations"], lambda: result["cancelled"])
        before = bench.dmesg_lines()
        log.with_suffix(".dmesg-before.txt").write_text("\n".join(before) if before is not None else "UNAVAILABLE")
        require(before is not None, "kernel log unavailable")
        started = time.monotonic()
        proc = start_server(bench, command, env, server_log)
        result["pid"] = proc.pid
        require(bench.wait_health(300) and proc.poll() is None, "server startup failed")
        result["startup_seconds"] = time.monotonic() - started
        startup = server_log.read_text()
        result["startup_build_lines"] = [line for line in startup.splitlines() if "build:" in line or "build =" in line]
        result["startup_placement_lines"] = [line for line in startup.splitlines()
                                             if re.search(r"offload|buffer size|fit:|n_gpu_layers|device", line, re.I)]
        require("common_speculative_impl_draft_mtp" in startup, "active MTP implementation missing")
        result["startup_holders"] = owned_tenancy(bench, proc, result["tenancy_probes"])
        rows = []
        for index, prompt in enumerate(prompts):
            rendered = bench.apply_chat_template(prompt["messages"])
            for repeat in range(2):
                owned_tenancy(bench, proc, result["tenancy_probes"])
                payload = request_payload(rendered, prompt.get("n_predict", 256), args.seed, args.target_temperature)
                response = bench.post_json("/completion", payload)
                item = {"client_request": len(result["responses"]) + 1, "prompt_index": index, "repeat": repeat,
                        "payload": payload, "response": response}
                result["responses"].append(item)
                log.with_suffix(".responses.json").write_text(json.dumps(result["responses"], indent=2))
                validate_response(response, payload)
                owned_tenancy(bench, proc, result["tenancy_probes"])
                deadline = time.monotonic() + 10
                while True:
                    try:
                        latest = cycle.read_rows(log)
                    except json.JSONDecodeError:
                        require(time.monotonic() < deadline, "incomplete cycle JSON at deadline")
                        time.sleep(.02)
                        continue
                    # P04 flushes phase completion just after the response is published.
                    if latest and len(latest) > len(rows) and cycle.START.findall(server_log.read_text()) and \
                            len(latest) == len(cycle.START.findall(server_log.read_text())):
                        cycle.validate_rows(latest, server_log.read_text())
                        item["request_id"] = cycle.reconcile(latest[len(rows):], response)
                        rows = latest
                        break
                    require(time.monotonic() < deadline, "cycle completion deadline expired")
                    time.sleep(.02)
                log.with_suffix(".responses.json").write_text(json.dumps(result["responses"], indent=2))
        result["summary"] = validate_campaign(rows, result["responses"], enabled, args)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        result["error"] = str(error) or "cancelled"
        result["cancelled"] |= isinstance(error, (SystemExit, KeyboardInterrupt))
    finally:
        # One handler covers the handoff and repeated signals without interrupting cleanup.
        cleaning = True
        try:
            owned = list(bench._owned_servers.values())
            if proc is not None and proc not in owned:
                owned.append(proc)
            for child in owned:
                status = {"clean": False}
                result["shutdown"].append(status)
                try:
                    stop_server(bench, child, status)
                except (OSError, subprocess.SubprocessError) as error:
                    status["error"] = str(error)
            require(bool(result["shutdown"]) and all(s["clean"] for s in result["shutdown"])
                    and not bench._owned_servers, "server did not terminate cleanly after SIGINT")
            require(not result.get("error") and not result["cancelled"], result.get("error", "cancelled"))
            final_rows = cycle.read_rows(log)
            cycle.validate_rows(final_rows, server_log.read_text())
            require(final_rows == rows, "cycle log changed after reconciliation")
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            result["error"] = result.get("error") or str(error)
            result["cancelled"] |= isinstance(error, (SystemExit, KeyboardInterrupt))
        finally:
            try:
                after = bench.dmesg_lines()
                log.with_suffix(".dmesg-after.txt").write_text("\n".join(after) if after is not None else "UNAVAILABLE")
                result["new_gpu_faults"] = bench.new_gpu_faults(before, after)
                require(result["new_gpu_faults"] == [], "new GPU faults or unreadable kernel delta")
                result["postflight"] = preflight(bench, result["postflight_observations"],
                                                 lambda: result["cancelled"] or bool(result.get("error")))
            except Exception as error:
                result["error"] = result.get("error") or str(error)
            finally:
                try:
                    result["ok"] = not result.get("error") and not result["cancelled"]
                    log.with_suffix(".arm.json").write_text(json.dumps(result, indent=2))
                finally:
                    for sig, handler in handlers.items():
                        signal.signal(sig, handler)
    require(result["ok"] and not result["cancelled"], result.get("error", "cancelled"))
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "draft-model", "prompts", "baseline-log", "enabled-log", "q-artifact", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--target-temperature", type=float, default=.8)
    parser.add_argument("--draft-temperature", type=float, default=.8)
    parser.add_argument("--draft-top-p", type=float, default=1.0)
    parser.add_argument("--spec-args", default="--spec-type draft-mtp --no-spec-draft-backend-sampling")
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args(argv)
    require(args.repeats == 2 and args.seed == 123 and args.target_temperature == args.draft_temperature == .8
            and args.draft_top_p == 1.0, "campaign identity must be repeats=2 seed=123 target/draft=.8 top-p=1")
    require(shlex.split(args.spec_args) == ["--spec-type", "draft-mtp", "--no-spec-draft-backend-sampling"],
            "campaign requires non-chained host draft-mtp arguments")
    require(args.ctx == 16384 and 1024 <= args.port <= 65535, "invalid context or port")
    args.server = args.server.resolve()
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    outputs = [args.report, args.q_artifact]
    for log in (args.baseline_log, args.enabled_log):
        outputs += [log, *(log.with_suffix(suffix) for suffix in
                          (".server.log", ".responses.json", ".dmesg-before.txt", ".dmesg-after.txt", ".arm.json"))]
    require(len({p.resolve() for p in outputs}) == len(outputs), "artifact paths collide")
    require(all(not p.exists() and not p.is_symlink() and p.parent.is_dir() for p in outputs),
            "use fresh artifact paths in existing directories")
    prompts = cycle.read_rows(args.prompts)
    require(len(prompts) == 3 and all(isinstance(p["messages"], list) and p["messages"] and
            type(p.get("n_predict", 256)) is int and 0 < p.get("n_predict", 256) <= 4096 for p in prompts),
            "three bounded prompts required")
    bench = load_helper("p06_bench", "bench_spec.py")
    bench.BASE = f"http://127.0.0.1:{args.port}"
    bench.RENDER_NODE = "/dev/dri/renderD128"
    bench.REQ_TIMEOUT = 300
    command = [str(args.server), "-m", str(args.model.resolve()), "-md", str(args.draft_model.resolve()),
               "--host", "127.0.0.1", "--port", str(args.port), "-c", str(args.ctx), "-np", "1",
               "--n-gpu-layers", "auto", "--device", "SYCL0", "--fit", "on", "--fit-target", "1024",
               "--mmap", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "on",
               "--spec-draft-n-min", "1", "--spec-draft-n-max", "7", "--reasoning", "off", "--reasoning-budget", "-1",
               *shlex.split(args.spec_args)]
    report = {"schema_version": 1, "ok": False, "arms": [], "command": command,
              "result_scope": "campaign artifacts and operational gates only; not P06 acceptance",
              "kernel": os.uname().release, "driver": bench.render_driver(bench.RENDER_NODE),
              "regression_floor": {"commands_before_and_after": REGRESSION_FLOOR, "status": "external gate, not run here"},
              "distribution_parity": "not established here; requires the test-sampling Pearson/Holm corpus",
              "performance": "no speedup or confidence interval claimed; one process per arm"}
    try:
        require(report["driver"] in {"xe", "i915"}, "unknown render driver")
        packages = subprocess.run(["pacman", "-Q"], capture_output=True, text=True, timeout=10)
        require(packages.returncode == 0, "host stack package provenance unavailable")
        report["stack_packages"] = [line for line in packages.stdout.splitlines()
                                    if any(name in line.split()[0] for name in ("compute-runtime", "level-zero", "intel-igc", "intel-oneapi"))]
        require(any("compute-runtime" in p for p in report["stack_packages"]), "compute-runtime identity unavailable")
        report["models"] = {}
        for name, path in (("target", args.model), ("draft", args.draft_model)):
            stat = path.stat()
            require(path.is_file() and stat.st_size > 0, "missing local model")
            report["models"][name] = {"path": str(path.resolve()), "size": stat.st_size,
                                       "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino,
                                       "identity_limit": "metadata only; coordinator supplies accepted model identity"}
        libraries = bench.resolved_libraries(args.server, bench.arm_library_path(args.server))
        require(libraries is not None, "library provenance unavailable")
        report["libraries"] = libraries
        report["build_hashes"] = bench.build_hashes(args.server, libraries)
        require(all(report["build_hashes"].values()), "incomplete build hashes")
        for enabled in (False, True):
            report["arms"].append(run_arm(bench, args, prompts, enabled, command))
        off, on = report["arms"]
        require(all(arm["ok"] and not arm["cancelled"] for arm in report["arms"]), "failed or cancelled arm")
        require(cycle.read_rows(args.baseline_log)[0]["launch_id"] != cycle.read_rows(args.enabled_log)[0]["launch_id"],
                "arms reused a launch identity")
        require([r["payload"] for r in off["responses"]] == [r["payload"] for r in on["responses"]],
                "arms used different requests")
        require(bench.build_hashes(args.server, libraries) == report["build_hashes"], "build changed during campaign")
        for identity in report["models"].values():
            stat = Path(identity["path"]).stat()
            require((stat.st_size, stat.st_mtime_ns, stat.st_ino) ==
                    (identity["size"], identity["mtime_ns"], identity["inode"]), "model metadata changed during campaign")
        with args.q_artifact.open("x") as output:
            for row in cycle.read_rows(args.enabled_log):
                output.write(json.dumps({k: row[k] for k in (*IDENTITY, "schema_version", "q_records", "attempts",
                                         "draft_sampling", "sampling_mode", "sampling_reason", "distribution_attempts",
                                         "exact_match_attempts", "drafted_count", "accepted_count")}) + "\n")
        report["ok"] = True
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        args.report.write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
