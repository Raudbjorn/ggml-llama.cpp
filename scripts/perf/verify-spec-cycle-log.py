#!/usr/bin/env python3
"""Reconcile P04 JSONL against six server responses, then reach the profile floor.

Run under flock /tmp/a770.lock and timeout, with sole GPU tenancy. No service
changes or downloads. Existing bench_spec helpers provide HTTP and fault gates.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

START = re.compile(r"spec_cycle_start: launch=(\d+) request=(-?\d+) proposal=(\d+) sequence=(\d+) cycle=(\d+)")
PROFILE = re.compile(r"spec_profile: draft_phases=(\d+) draft_mean_us=([\d.]+) draft_sync_mean_us=([\d.]+) "
                     r"verify_phases=(\d+) verify_mean_us=([\d.]+) verify_sync_mean_us=([\d.]+) "
                     r"sequence_cycles=(\d+) cycle_mean_us=([\d.]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_rows(rows: list[dict], starts: str | None = None) -> dict:
    """Validate identities, attempts, immutable counts and deduplicated phase metadata."""
    identities: set[tuple] = set()
    cycles: set[tuple] = set()
    phases: dict[str, dict[tuple, dict]] = {"draft": {}, "verify": {}}
    for row in rows:
        require(row["schema_version"] == 1, "unknown cycle schema")
        identity = tuple(row[key] for key in ("launch_id", "request_id", "proposal_id", "sequence_id", "cycle_id"))
        require(identity not in identities, "duplicate proposal completion")
        identities.add(identity)
        cycle = (row["launch_id"], row["sequence_id"], row["cycle_id"])
        require(cycle not in cycles, "duplicate sequence cycle")
        cycles.add(cycle)
        attempts = row["attempts"]
        require(row["attempted_count"] == len(attempts) == len(row["top1_probabilities"]), "attempt count mismatch")
        require([a["top1_probability"] for a in attempts] == row["top1_probabilities"], "attempt probabilities lost")
        require(all(math.isfinite(p) and 0 <= p <= 1 for p in row["top1_probabilities"]), "invalid probability")
        stops = [i for i, item in enumerate(attempts) if not item["emitted"]]
        require(not stops or stops == [len(attempts) - 1], "stop candidate must remain last")
        emitted = [a["token"] for a in attempts if a["emitted"]]
        require(row["proposal_tokens"] == emitted[:row["drafted_count"]], "selected proposal differs from attempted prefix")
        require(row["drafted_count"] == len(row["proposal_tokens"]), "draft count mismatch")
        require(0 <= row["accepted_count"] <= row["drafted_count"], "invalid accepted count")
        require(row["outcome"] in {"completed", "cancelled"}, "unknown cycle outcome")
        if row["outcome"] == "cancelled":
            require(not row["selected"], "cancelled proposal remains selected")
        if not row["selected"]:
            require(row["drafted_count"] == row["accepted_count"] == 0, "unselected proposal counted")
            require(row["outcome"] == "cancelled" or not row["verify_phases"], "unselected proposal was verified")
        owner = {"proposal_id": row["proposal_id"], "sequence_id": row["sequence_id"]}
        for kind, entries in (("draft", [row["draft_phase"]]), ("verify", row["verify_phases"])):
            seen = set()
            for phase in entries:
                key = (row["launch_id"], phase["phase_id"])
                require(key not in seen, "duplicate phase in a cycle")
                seen.add(key)
                members = phase["participants"]
                require(owner in members, "phase does not contain its proposal")
                require(len(members) == phase["participant_count"] > 0, "participant count mismatch")
                require(len({(p["proposal_id"], p["sequence_id"]) for p in members}) == len(members), "duplicate participant")
                require(phase["timing_scope"] == ("single_sequence" if len(members) == 1 else "shared_batch"), "timing scope mismatch")
                require(phase["wall_us"] == phase["end_us"] - phase["start_us"] >= phase["synchronization_us"] >= 0, "invalid phase time")
                if row["timing_mode"] == "host_unsynchronized":
                    require(phase["synchronization_us"] == 0, "log-only synchronization")
                else:
                    require(row["timing_mode"] == "synchronized", "unknown timing mode")
                if key in phases[kind]:
                    require(phases[kind][key] == phase, "shared phase metadata differs")
                phases[kind][key] = phase
        draft = row["draft_phase"]
        verify = row["verify_phases"]
        require(row["cycle_wall_us"] >= 0 and row["draft_wall_us"] == draft["wall_us"], "invalid cycle time")
        if verify:
            require(row["inter_phase_wait_us"] == verify[0]["start_us"] - draft["end_us"] >= 0, "invalid scheduling wait")
        else:
            require(row["inter_phase_wait_us"] is None, "fabricated verify wait")
        single = row["outcome"] == "completed" and len(verify) == 1 and draft["participant_count"] == verify[0]["participant_count"] == 1
        if single:
            residual = row["cycle_wall_us"] - draft["wall_us"] - verify[0]["wall_us"]
            require(row["additive_residual_us"] == residual, "incorrect cycle residual")
            require(abs(residual) <= row["inter_phase_wait_us"] + draft["synchronization_us"] + verify[0]["synchronization_us"], "unaccounted phase residual")
        else:
            require(row["additive_residual_us"] is None, "shared/replayed phases treated as exclusive")
        require(row["verify_sample_wall_us"] == (verify[0]["wall_us"] if len(verify) == 1 else None), "fabricated verify duration")
    require(not (phases["draft"].keys() & phases["verify"].keys()), "draft and verify identity collision")
    if starts is not None:
        started = [tuple(map(int, match)) for match in START.findall(starts)]
        require(len(started) == len(set(started)), "duplicate proposal start")
        require(set(started) == identities, "started/completed proposal identity mismatch")
    return {"draft_phases": len(phases["draft"]), "verify_phases": len(phases["verify"]),
            "sequence_cycles": len(cycles), "phases": phases}


def validate_profile(rows: list[dict], text: str, minimum: int = 64) -> None:
    counts = validate_rows(rows)
    summaries = PROFILE.findall(text)
    require(bool(summaries), "missing independently counted profile summary")
    for summary in summaries:
        d, dw, ds, v, vw, vs, c, cw = summary
        for kind, count, wall, sync in (("draft", int(d), float(dw), float(ds)),
                                       ("verify", int(v), float(vw), float(vs))):
            phases = sorted(counts["phases"][kind].values(), key=lambda phase: phase["phase_id"])
            require(64 <= count <= len(phases), "profile phase denominator differs from unique invocations")
            require(abs(sum(p["wall_us"] for p in phases[:count]) / count - wall) <= .001,
                    "profile phase mean used an incorrect denominator")
            require(abs(sum(p["synchronization_us"] for p in phases[:count]) / count - sync) <= .001,
                    "profile synchronization mean used an incorrect denominator")
        count = int(c)
        require(64 <= count <= len(rows), "profile cycle denominator differs from completed cycles")
        require(abs(sum(r["cycle_wall_us"] for r in rows[:count]) / count - float(cw)) <= .001,
                "profile cycle mean used an incorrect denominator")
    require(min(int(summaries[-1][i]) for i in (0, 3, 6)) >= minimum,
            "profile summary denominator below floor")


def reconcile(rows: list[dict], response: dict) -> int:
    require(bool(rows), "request produced no MTP cycles")
    require(all(row["outcome"] == "completed" for row in rows), "successful request contains cancelled proposals")
    requests = {row["request_id"] for row in rows}
    require(len(requests) == 1, "request rows joined across tasks")
    selected = [row for row in rows if row["selected"]]
    timings = response["timings"]
    require(sum(row["drafted_count"] for row in selected) == timings["draft_n"], "drafted total differs from server")
    require(sum(row["accepted_count"] for row in selected) == timings["draft_n_accepted"], "accepted total differs from server")
    return requests.pop()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ctx", type=int, default=16384)
    parser.add_argument("--spec-args", default="--spec-type draft-mtp --spec-draft-n-min 1 --spec-draft-n-max 7 --no-spec-draft-backend-sampling")
    parser.add_argument("--server-args", default="")
    parser.add_argument("--assert-impl", default="common_speculative_impl_draft_mtp")
    parser.add_argument("--min-profile-cycles", type=int, default=64)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()
    require(args.repeats == 2, "the reconciliation contract is three prompts x two repetitions")
    require(args.min_profile_cycles >= 64, "profile floor is at least 64 per scope")
    import bench_spec as bench
    bench.BASE = f"http://127.0.0.1:{args.port}"
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()][:3]
    require(len(prompts) == 3, "three prompts are required")
    require(not bench.gpu_holders(), "GPU is in use; coordinator must establish sole tenancy")
    driver = bench.render_driver(bench.RENDER_NODE)
    require(driver in {"xe", "i915"}, "unknown render driver")
    before = bench.dmesg_lines()
    require(before is not None, "kernel log unavailable")
    args.log.with_suffix(".dmesg-before.txt").write_text("\n".join(before))
    command = [str(args.server.resolve()), "-m", args.model, "-md", args.draft_model,
               "--host", "127.0.0.1", "--port", str(args.port), "-c", str(args.ctx), "-np", "1",
               "-ngl", "99", "-fa", "on", *shlex.split(args.spec_args), *shlex.split(args.server_args)]
    env = dict(os.environ)
    env.pop("LLAMA_SPEC_LOG", None)
    env.pop("LLAMA_SPEC_PROFILE", None)
    env["LD_LIBRARY_PATH"] = bench.arm_library_path(args.server.resolve())
    env["LLAMA_TRACE"] = "1"
    command_metadata = {"command": command, "driver": driver, "model": args.model,
                        "draft_model": args.draft_model, "server": str(args.server.resolve()),
                        "build_hashes": bench.build_hashes(args.server.resolve()), "repeats": args.repeats}
    args.log.with_suffix(".provenance.json").write_text(json.dumps(command_metadata, indent=2))
    try:
        for measured in (False, True):
            warmup_log_state = args.log.stat() if args.log.exists() else None
            if measured:
                if args.log.exists():
                    args.log.rename(args.log.with_name(args.log.name + f".previous-{time.time_ns()}"))
                env["LLAMA_SPEC_LOG"] = str(args.log.resolve())
                if args.profile:
                    env["LLAMA_SPEC_PROFILE"] = "1"
            server_log = args.log.with_suffix(".server.log" if measured else ".warmup.log")
            with server_log.open("w") as output:
                proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
                try:
                    require(bench.wait_health(300), "server did not become healthy")
                    if measured:
                        require(args.assert_impl in server_log.read_text(), "MTP implementation assertion failed before requests")
                    request_ids = []
                    rows = []
                    responses = []
                    for number in range(256 if measured else 1):
                        prompt = prompts[(number // args.repeats) % len(prompts)]
                        response = bench.post_json("/v1/chat/completions", {"messages": prompt["messages"],
                            "max_tokens": prompt.get("n_predict", 256), "temperature": 0, "seed": 42, "cache_prompt": False})
                        if not measured:
                            break
                        # The server finishes the shared phase just after publishing its response.
                        deadline = time.monotonic() + 10
                        while True:
                            try:
                                latest = read_rows(args.log)
                                validate_rows(latest, server_log.read_text())
                                request_id = reconcile(latest[len(rows):], response)
                                break
                            except (ValueError, json.JSONDecodeError):
                                if time.monotonic() >= deadline:
                                    raise
                                time.sleep(.02)
                        require(request_id not in request_ids, "server reused a request identity")
                        request_ids.append(request_id)
                        rows = latest
                        responses.append({"client_request": number + 1, "request_id": request_id, "response": response,
                                          "reconciliation": number < 6})
                        args.log.with_suffix(".responses.json").write_text(json.dumps(responses))
                        counts = validate_rows(rows, server_log.read_text())
                        if number >= 5 and (not args.profile or min(counts[k] for k in ("draft_phases", "verify_phases", "sequence_cycles")) >= args.min_profile_cycles):
                            break
                    else:
                        raise ValueError("profile floor was not reached in 256 requests")
                    if measured:
                        require(len(request_ids) >= 6, "six reconciliation requests did not complete")
                        if args.profile:
                            validate_profile(rows, server_log.read_text(), args.min_profile_cycles)
                        print(json.dumps({k: v for k, v in counts.items() if k != "phases"}, sort_keys=True))
                finally:
                    bench.stop_server(proc)
            if not measured:
                require("spec_cycle_start:" not in server_log.read_text(), "disabled warmup recorded cycles")
                current = args.log.stat() if args.log.exists() else None
                require(current == warmup_log_state, "disabled warmup created or changed the cycle log")
    finally:
        after = bench.dmesg_lines()
        args.log.with_suffix(".dmesg-after.txt").write_text("\n".join(after) if after is not None else "UNAVAILABLE")
        require(bench.new_gpu_faults(before, after) == [], "new GPU faults or unreadable kernel log")
        require(not bench.gpu_holders(), "GPU remained held after verifier cleanup")


if __name__ == "__main__":
    main()
