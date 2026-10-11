#!/usr/bin/env python3
"""Frozen stochastic speed experiment using bench_spec's HTTP and ownership gates.

No services or downloads. Run only under the coordinator's GPU lease and
`flock -w 60 /tmp/a770.lock timeout --kill-after=45s ...`. Preregistration,
qualification, and token fixtures must be frozen before timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import time


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("speed_bench", "bench_spec.py")
distribution = load("speed_distribution", "verify-spec-distribution.py")
require = distribution.require
MODELS = ("llama31", "mistral7", "qwen3", "qwen4exp")
CATEGORIES = ("code_edit", "multi_turn", "free_prose")
CELLS = tuple(f"{category}-{length}" for category in CATEGORIES for length in (512, 4096))
CONTRASTS = (("default", "A", "B"), ("enabled", "A", "C"), ("attribution", "B", "C"))
ORDERS = ("ABC", "CBA", "BCA", "ACB", "CAB", "BAC", "ABC", "CBA", "BCA", "ACB")
ANALYSIS_SEED = 68020261011
DRAWS = 20000
ROUND_ALPHA = 0.025
PREREGISTRATION_SHA256 = "3d533e803eb871dcb435538e0822e7ec82e07db2288c0926bf602509919426b3"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(row: dict) -> Path:
    path = Path(row["path"])
    require(path.is_absolute() and path.is_file(), f"missing absolute evidence path: {path}")
    require(digest(path) == row["sha256"], f"identity changed: {path}")
    return path


def write_json(path: Path, value) -> None:
    # Exclusive campaign directories prevent overwriting a prior attempt.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def schedule(model: str) -> list[str]:
    require(model in MODELS, "unknown model")
    return list(ORDERS if model == "qwen4exp" else ("AB", "BA") * 5)


def fixed_fixture(rendered: list[int], filler: list[int], length: int) -> list[int]:
    """Keep the whole rendered chat suffix; pad its prefix with fixed neutral context."""
    require(length in (512, 4096) and 0 < len(rendered) <= length and bool(filler), "invalid token fixture sources")
    require(all(type(t) is int and t >= 0 for t in rendered + filler), "invalid fixture token")
    needed = length - len(rendered)
    return (filler * ((needed + len(filler) - 1) // len(filler)))[:needed] + rendered


def request_seed(round_name: str, block: int, cell: int, warmup: bool) -> int:
    require(round_name in {"initial", "heldout"} and 0 <= block < 10 and 0 <= cell < 6,
            "invalid request identity")
    return 680000 + (10000 if round_name == "heldout" else 0) + block * 100 + cell * 2 + int(warmup)


def analyze(records: list[dict], draws: int = DRAWS) -> dict:
    """Stratified paired-cluster bootstrap max-|t| simultaneous log-ratio bounds.

    Each model has ten independent launch blocks; all cells and arms within a
    block remain together. Bootstrap coverage is approximate, not finite-sample proof.
    """
    expected = {(m, b, a, c) for m in MODELS for b in range(10)
                for a in ("ABC" if m == "qwen4exp" else "AB") for c in CELLS}
    values = {}
    for row in records:
        key = (row["model"], row["block"], row["arm"], row["cell"])
        require(key in expected and key not in values, "unexpected or duplicate measured cell")
        value = row["elapsed_s"]
        require(row.get("fixed_work_valid") is True and type(value) in (int, float)
                and math.isfinite(value) and value > 0, "invalid measured request")
        values[key] = value
    require(values.keys() == expected, "incomplete paired experiment; no cell dropping")
    # Each model contributes 18 correlated endpoint coordinates per paired block.
    vectors = {}
    for model in MODELS:
        vectors[model] = []
        for block in range(10):
            vector = []
            for _name, reference, candidate in CONTRASTS:
                for cell in CELLS:
                    actual = "B" if model != "qwen4exp" and candidate == "C" else candidate
                    vector.append(math.log(values[model, block, reference, cell] /
                                           values[model, block, actual, cell]))
            vectors[model].append(vector)
    structural = {(model, index) for model in MODELS if model != "qwen4exp" for index in range(12, 18)}

    def estimates(sample: dict) -> tuple[list[float], list[float]]:
        means, variances = {}, {}
        for model in MODELS:
            for coordinate in range(18):
                series = [row[coordinate] for row in sample[model]]
                means[model, coordinate] = statistics.fmean(series)
                variances[model, coordinate] = statistics.variance(series) / 10
        # Per-cell contrasts first, excluding algebraic C=B identities.
        theta = [means[m, i] for m in MODELS for i in range(18) if (m, i) not in structural]
        se = [math.sqrt(variances[m, i]) for m in MODELS for i in range(18) if (m, i) not in structural]
        for contrast in range(3):
            model_series = {m: [statistics.fmean(row[contrast * 6:(contrast + 1) * 6])
                                for row in sample[m]] for m in MODELS}
            theta.append(statistics.fmean(statistics.fmean(s) for s in model_series.values()))
            se.append(math.sqrt(sum(statistics.variance(s) / 10 for s in model_series.values()) / 16))
        return theta, se

    theta, se = estimates(vectors)
    require(all(v > 1e-14 for v in se), "degenerate observed variance; inference incomplete")
    rng = random.Random(ANALYSIS_SEED)
    maxima = []
    for _ in range(draws):
        sample = {m: [vectors[m][rng.randrange(10)] for _ in range(10)] for m in MODELS}
        boot, boot_se = estimates(sample)
        # A degenerate bootstrap draw widens the bound; it is never silently discarded.
        maxima.append(max(abs(x - original) / error if error > 1e-14 else math.inf
                          for x, original, error in zip(boot, theta, boot_se)))
    maxima.sort()
    rank = min(draws - 1, math.ceil((1 - ROUND_ALPHA) * (draws + 1)) - 1)
    critical = maxima[rank]
    require(math.isfinite(critical), "bootstrap tail degenerate; inference incomplete")
    names = [f"{CONTRASTS[i // 6][0]}/{m}/{CELLS[i % 6]}"
             for m in MODELS for i in range(18) if (m, i) not in structural]
    names += [f"{name}/aggregate" for name, _, _ in CONTRASTS]
    bounds = {name: {"ratio": math.exp(value), "lower": math.exp(value - critical * error),
                     "upper": math.exp(value + critical * error)}
              for name, value, error in zip(names, theta, se)}
    claims = {}
    for name, _, _ in CONTRASTS:
        cells = [value for key, value in bounds.items() if key.startswith(name + "/") and not key.endswith("aggregate")]
        guard = all(value["lower"] >= 1 / 1.03 for value in cells)
        claims[name] = {"aggregate_above_2pct": bounds[name + "/aggregate"]["lower"] > 1.02,
                        "all_cells_exclude_3pct_latency_regression": guard,
                        "passed": bounds[name + "/aggregate"]["lower"] > 1.02 and guard}
    return {"method": "stratified paired-block studentized bootstrap max-absolute-t",
            "coverage": "approximate; independent launch blocks and regular sampling distribution assumed",
            "round_family_alpha": ROUND_ALPHA, "two_round_family_alpha": 2 * ROUND_ALPHA,
            "draws": draws, "analysis_seed": ANALYSIS_SEED, "critical_value": critical,
            "bounds": bounds, "claims": claims,
            "structural_reuse": "C equals B by observation reuse on non-Qwen4Exp models; ratio exactly 1"}


def validate_manifest(manifest: dict, model: str) -> tuple[dict, list[dict]]:
    require(manifest["schema_version"] == 1 and manifest["round"] in {"initial", "heldout"}, "invalid manifest")
    require(set(manifest["models"]) == set(MODELS), "frozen suite must contain all four models")
    prereg = file_identity(manifest["preregistration"])
    require(manifest["preregistration"]["sha256"] == PREREGISTRATION_SHA256, "preregistration differs from frozen protocol")
    protocol = json.loads(prereg.read_text())
    require(protocol["analysis_seed"] == ANALYSIS_SEED and protocol["bootstrap_draws"] == DRAWS
            and protocol["round_family_alpha"] == ROUND_ALPHA, "analysis protocol mismatch")
    qualification = json.loads(file_identity(manifest["qualification"]).read_text())
    require(qualification["status"] == "qualified" and qualification["distribution_qualification"] == "established",
            "sampling/distribution qualification is incomplete")
    require(qualification["scope"] == manifest["qualification_scope"], "qualification scope mismatch")
    for field in ("abi_report", "correctness_report", "distribution_report", "batch_sequential_report", "route_report"):
        file_identity(qualification[field])
    config = manifest["models"][model]
    require(qualification["platform"] == manifest["platform"], "platform qualification mismatch")
    require(os.uname().release == manifest["platform"]["kernel"], "kernel changed")
    require(bench.render_driver(bench.RENDER_NODE) == manifest["platform"]["driver"], "driver changed")
    require(manifest["platform"]["runtime_files"], "empty runtime inventory")
    for item in manifest["platform"]["runtime_files"]:
        file_identity(item)
    require(config["context"] == protocol["context"] and config["threads"] == protocol["threads"], "context/thread setting differs from protocol")
    require(qualification["models"][model] == config, "qualification does not cover model/build/placement/fixtures")
    require(config["arms"]["A"]["placement_args"] == config["arms"]["B"]["placement_args"],
            "placement arguments differ across arms")
    require(config["model_files"], "missing model files")
    if model == "qwen4exp":
        require(config["draft_model"] in {r["path"] for r in config["model_files"]}, "head is not hash-bound")
    # Hash each model shard/head before the first measured launch.
    for item in config["model_files"]:
        file_identity(item)
    for arm in ("ABC" if model == "qwen4exp" else "AB"):
        build = config["arms"][arm]
        file_identity(build["server"])
        require(build["artifacts"], "empty build artifact inventory")
        for item in build["artifacts"]:
            file_identity(item)
    require(config["arms"]["A"]["source_commit"] == "2f623c6607c078d1069dac23e9d8af9cb146e771", "wrong pristine baseline")
    if model == "qwen4exp":
        require(config["arms"]["B"] == config["arms"]["C"], "B and C must use identical build and placement")
    fixtures = json.loads(file_identity(config["fixtures"]).read_text())
    require(fixtures["schema_version"] == 1 and fixtures["model"] == model and
            fixtures["round"] == manifest["round"], "fixture model/round mismatch")
    require(fixtures["prompt_source"] == protocol["prompts"][manifest["round"] + "-prompts.json"],
            "fixture prompt source differs from preregistration")
    file_identity(fixtures["prompt_source"])
    require(fixtures["tokenizer_parity"] is True and fixtures["tokenizer_flags"] ==
            {"add_special": False, "parse_special": True}, "tokenizer parity/flags unqualified")
    prompts = fixtures["cells"]
    require([p["id"] for p in prompts] == list(CELLS), "fixture cells differ from frozen suite")
    for prompt in prompts:
        length = int(prompt["id"].rsplit("-", 1)[1])
        require(len(prompt["tokens"]) == length, "fixture length mismatch")
        bench.stochastic_payload(prompt["tokens"], 1)
    return config, prompts


def child_environment(server: Path, enabled: bool) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_"))
           and k not in {"LD_PRELOAD", "LD_AUDIT"}}
    env.update(LLAMA_SPEC_CHAIN="0", LLAMA_DRAFT_TOP_K="10", ONEAPI_DEVICE_SELECTOR="level_zero:0",
               SYCL_CACHE_PERSISTENT="1", GGML_SYCL_DISABLE_GRAPHS="1", GGML_SYCL_Q8_KV_QUANTS_FIRST="0",
               LD_LIBRARY_PATH=bench.library_path(str(server.parent), os.environ.get("LD_LIBRARY_PATH", "")))
    if enabled:
        env.update(LLAMA_SPEC_SAMPLE_TEMP="0.8", LLAMA_SPEC_DRAFT_TOPP="1")
    return env


def qualified_environment(env: dict) -> dict:
    return {k: v for k, v in env.items() if k.startswith((
        "LLAMA_", "GGML_", "SYCL_", "UR_", "ZE_", "ZES_", "OMP_", "KMP_", "GOMP_",
        "MKL_", "OPENBLAS_", "BLIS_", "VECLIB_", "TBB_", "DNNL_", "ONEDNN_"))
        or k in {"ONEAPI_DEVICE_SELECTOR", "LD_LIBRARY_PATH"}}


def capture_maps(proc, expected: list[dict]) -> dict:
    paths, images = set(), set()
    for line in Path(f"/proc/{proc.pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            require(path.exists(), f"mapped file missing or deleted: {path}")
            if path.is_file():
                st = path.stat()
                device = tuple(int(part, 16) for part in fields[3].split(":"))
                require(device == (os.major(st.st_dev), os.minor(st.st_dev)) and int(fields[4]) == st.st_ino,
                        f"mapped file replaced: {path}")
                paths.add(path.resolve())
                with path.open("rb") as stream:
                    if "x" in fields[1] or stream.read(4) == b"\x7fELF":
                        images.add(path.resolve())
    expected_paths = {}
    for row in expected:
        path = Path(row["path"]).resolve()
        require(path not in expected_paths or expected_paths[path] == row["sha256"], "conflicting image identities")
        expected_paths[path] = row["sha256"]
    project = {p for p in paths if p.name.startswith(("libllama", "libggml", "libcommon", "libmtmd"))}
    require(project and any(p.name.startswith("libggml-sycl") for p in project), "SYCL runtime image missing")
    images |= project
    require(images <= expected_paths.keys(), f"unqualified image mapped: {sorted(map(str, images - expected_paths.keys()))}")
    hashes = {str(p): digest(p) for p in images}
    require(all(hashes[str(p)] == expected_paths[p] for p in images), "mapped artifact changed")
    return {"pid": proc.pid, "image_hashes": hashes,
            "project_hashes": {str(p): hashes[str(p)] for p in project}, "all_file_paths": sorted(map(str, paths))}


def run_launch(manifest: dict, model: str, config: dict, prompts: list[dict], block: int,
               arm: str, directory: Path) -> dict:
    build = config["arms"][arm]
    server = Path(build["server"]["path"])
    speculative = model == "qwen4exp"
    command = [str(server), "-m", config["model_files"][0]["path"], "--host", "127.0.0.1",
               "--port", str(bench.PORT), "--ctx-size", str(config["context"]), "--parallel", "1",
               "--threads", str(config["threads"]), "--flash-attn", "on", "-ctk", "q8_0", "-ctv", "q8_0",
               "--reasoning", "off", "--reasoning-budget", "-1",
               *build["placement_args"], "--spec-type", "draft-mtp" if speculative else "none"]
    if speculative:
        command += ["--spec-draft-model", config["draft_model"], "--spec-draft-n-min", "1",
                    "--spec-draft-n-max", "7", "--spec-draft-p-min", "0"]
    env = child_environment(server, arm == "C")
    relevant_env = qualified_environment(env)
    require(relevant_env == config["effective_environment"][arm], "environment differs from route qualification")
    log = directory / f"{block:02d}-{arm}.log"
    output = log.with_suffix(".json")
    state = {"schema_version": 1, "model": model, "block": block, "arm": arm, "command": command,
             "environment": relevant_env,
             "ok": False, "responses": [], "tenancy": [], "shutdown": [], "preflight": [], "postflight": []}
    proc, before = None, None
    handlers = {}
    cleaning = False

    def cancel(signum, _frame):
        state["error"] = f"cancelled by signal {signum}"
        if not cleaning:
            raise SystemExit(state["error"])

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, cancel)
        distribution.preflight(bench, state["preflight"])
        require(bench.render_driver(bench.RENDER_NODE) in {"xe", "i915"}, "unknown render driver")
        before = bench.dmesg_lines()
        require(before is not None, "unreadable dmesg")
        log.with_suffix(".dmesg-before.txt").write_text("\n".join(before))
        proc = distribution.start_server(bench, command, env, log)
        require(bench.wait_health(600) and proc.poll() is None, "server startup failed")
        images = [build["server"], *build["artifacts"], *manifest["platform"]["runtime_files"]]
        state["maps"] = capture_maps(proc, images)
        for cell, prompt in enumerate(prompts):
            for warmup in (True, False):
                distribution.owned_tenancy(bench, proc, state["tenancy"])
                payload = bench.stochastic_payload(prompt["tokens"], request_seed(manifest["round"], block, cell, warmup))
                started = time.perf_counter()
                response = bench.post_json("/completion", payload)
                elapsed = time.perf_counter() - started
                item = {"model": model, "block": block, "arm": arm, "cell": prompt["id"], "warmup": warmup,
                        "elapsed_s": elapsed, "payload": payload, "response": response}
                state["responses"].append(item)
                write_json(output, state)
                item.update(bench.analyze_stochastic_response(response, payload, speculative))
                distribution.owned_tenancy(bench, proc, state["tenancy"])
                write_json(output, state)
        state["maps_after"] = capture_maps(proc, images)
    except (Exception, SystemExit, KeyboardInterrupt) as error:
        state["error"] = str(error) or "cancelled"
    finally:
        cleaning = True
        try:
            for child in list(bench._owned_servers.values()):
                shutdown = {}
                state["shutdown"].append(shutdown)
                distribution.stop_server(bench, child, shutdown)
            require(bool(state["shutdown"]) and all(s["clean"] for s in state["shutdown"]), "unclean shutdown")
        except Exception as error:
            state["error"] = state.get("error") or str(error)
        try:
            after = bench.dmesg_lines()
            log.with_suffix(".dmesg-after.txt").write_text("\n".join(after) if after is not None else "UNAVAILABLE")
            state["faults"] = bench.new_gpu_faults(before, after)
            require(state["faults"] == [], "new faults or unavailable fault comparison")
            distribution.preflight(bench, state["postflight"])
            state["ok"] = not state.get("error") and not bench._owned_servers
        except Exception as error:
            state["error"] = state.get("error") or str(error)
        finally:
            write_json(output, state)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    require(state["ok"], state.get("error", "launch failed"))
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--model", choices=MODELS, required=True)
    run.add_argument("--output", type=Path, required=True)
    analysis = sub.add_parser("analyze")
    analysis.add_argument("--inputs", type=Path, nargs=4, required=True)
    analysis.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "analyze":
        records = []
        manifests = set()
        models = set()
        for path in args.inputs:
            summary = json.loads(path.read_text())
            require(summary["ok"] is True and summary["model"] not in models, "invalid/duplicate model campaign")
            frozen = json.loads(file_identity(summary["manifest"]).read_text())
            require(summary["manifest_sha256"] == summary["manifest"]["sha256"], "manifest hash mismatch")
            require(summary["round"] == frozen["round"], "round mismatch")
            fixture = json.loads(file_identity(frozen["models"][summary["model"]]["fixtures"]).read_text())
            require(fixture["model"] == summary["model"] and fixture["round"] == summary["round"], "fixture model/round mismatch")
            require([p["id"] for p in fixture["cells"]] == list(CELLS), "fixture cells differ from frozen suite")
            tokens_by_cell = {p["id"]: p["tokens"] for p in fixture["cells"]}
            require(len(summary["launches"]) == sum(map(len, schedule(summary["model"]))), "incomplete launch schedule")
            expected_launches = [(block, arm) for block, order in enumerate(schedule(summary["model"])) for arm in order]
            require([(l["block"], l["arm"]) for l in summary["launches"]] == expected_launches, "launch order mismatch")
            models.add(summary["model"])
            manifests.add(summary["manifest_sha256"])
            for launch in summary["launches"]:
                require(launch["ok"] is True and launch["faults"] == [], "invalid launch")
                require(launch["model"] == summary["model"] and len(launch["responses"]) == 12, "request matrix mismatch")
                require([(r["cell"], r["warmup"]) for r in launch["responses"]] ==
                        [(cell, warmup) for cell in CELLS for warmup in (True, False)], "warmup/request order mismatch")
                for row in launch["responses"]:
                    require(row["model"] == launch["model"] and row["block"] == launch["block"] and
                            row["arm"] == launch["arm"], "response identity mismatch")
                    require(row["payload"]["seed"] == request_seed(summary["round"], row["block"], CELLS.index(row["cell"]), row["warmup"]), "request seed mismatch")
                    require(row["payload"]["prompt"] == tokens_by_cell[row["cell"]], "request prompt differs from frozen fixture")
                    bench.analyze_stochastic_response(row["response"], row["payload"], row["model"] == "qwen4exp")
                    if not row["warmup"]:
                        records.append(row)
        require(len(manifests) == 1 and models == set(MODELS), "mixed manifests/incomplete suite")
        require(not args.output.exists(), "refusing to overwrite analysis")
        write_json(args.output, analyze(records))
        return 0
    manifest = json.loads(args.manifest.read_text())
    config, prompts = validate_manifest(manifest, args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": 1, "ok": False, "model": args.model, "round": manifest["round"],
               "manifest_sha256": digest(args.manifest), "manifest": {"path": str(args.manifest.resolve()), "sha256": digest(args.manifest)}, "launches": []}
    try:
        for block, order in enumerate(schedule(args.model)):
            for arm in order:
                summary["launches"].append(run_launch(manifest, args.model, config, prompts, block, arm, args.output))
                write_json(args.output / "summary.json", summary)
        validate_manifest(manifest, args.model)
        summary["ok"] = True
    finally:
        write_json(args.output / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
