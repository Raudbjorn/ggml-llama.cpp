#!/usr/bin/env python3
"""Synthetic P06 executable wiring checks; never trained-model campaign evidence."""
import argparse
import contextlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import tempfile
import time


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def observations(path):
    return [json.loads(item) for item in re.findall(r'P06_CONSUMER (\{[^\n]*\})', path.read_text(errors="replace"))]


def group_alive(group):
    # Same Linux liveness criterion as bench_spec: zombies cannot hold resources
    # or respond to signals. Restrict membership to our new session and group.
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and int(fields[3]) == group and fields[0] != "Z":
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def drain(process):
    # A dead leader is not proof that its child group is empty.
    for sig, grace in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        if group_alive(process.pid):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        end = time.monotonic() + grace
        while True:
            if process.poll() is not None and not group_alive(process.pid):
                process.wait()
                return
            if time.monotonic() >= end:
                break
            time.sleep(.05)
    raise subprocess.TimeoutExpired("owned consumer process group cleanup", 10)


@contextlib.contextmanager
def child(command, env, log):
    require(Path("/proc/self/stat").is_file(), "consumer process-group checks require Linux /proc")
    protected = True
    pending = 0
    process = None
    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def interrupted(signum, _frame):
        nonlocal protected, pending
        pending = pending or signum
        if not protected:
            # The first interruption enters cleanup. Further signals cannot
            # interrupt the ownership handoff or the bounded group drain.
            protected = True
            raise SystemExit(128 + signum)

    try:
        for sig in signals:
            signal.signal(sig, interrupted)
        with log.open("w") as output:
            try:
                # Defer exceptions until Popen has returned ownership. Unlike an
                # inherited blocked signal mask, handlers reset at child exec.
                process = subprocess.Popen(command, env=env, stdout=output, stderr=output, start_new_session=True)
                protected = False
                if pending:
                    raise SystemExit(128 + pending)
                yield process
                protected = True
            except BaseException:
                protected = True
                raise
            finally:
                protected = True
                if process is not None:
                    drain(process)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending and sys.exc_info()[0] is None:
            raise SystemExit(128 + pending)


def request(port, method, path, data=None, timeout=10):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method, path, body=None if data is None else json.dumps(data),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        require(response.status == 200, f"HTTP {response.status}: {body}")
        return body
    finally:
        connection.close()


def wait_for(process, predicate, seconds=30):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        require(process.poll() is None, f"child exited {process.returncode}")
        if predicate():
            return
        time.sleep(.05)
    raise AssertionError("fixture deadline expired")


def validate(folder, case, ignore, consumer):
    events = observations(folder / "process.log")
    decisions = [e for e in events if e["event"] == "decision"]
    require(len(decisions) == 1, "must reach exactly one injected consumer decision")
    decision = decisions[0]
    proposal = decision["proposal"]
    require(decision["original_draft"] == 3, "must exercise three-token proposal")
    cycles = [json.loads(line) for line in (folder / "cycles.jsonl").read_text().splitlines()]
    cycle = [r for r in cycles if r["proposal_id"] == proposal]
    require(len(cycle) == 1, "missing actual P04 completion/cancellation")
    cycle = cycle[0]
    require(cycle["distribution_attempts"] == 1 and cycle["exact_match_attempts"] == 0,
            "repair must not add another acceptance attempt")
    phases = cycle["verify_phases"]
    require(all(p["end_us"] >= p["start_us"] > 0 for p in phases), "unfinished actual verify phase")
    start = events.index(decision)
    tail = events[start + 1:]
    if case == "cancel":
        require(consumer == "server", "cancel applies to server")
        require(any(e["event"] == "cancel_dispatched" and e["pending"] for e in tail),
                "real cancel task must observe pending repair")
        require(any(e["event"] == "reset" and e["pending_before"] and e["awaiting_before"] for e in tail),
                "reset must clear a pending correction")
        require(any(e["event"] == "reset_complete" and not e["pending"] for e in tail), "repair state survived reset")
        require(not any(e["event"] == "repaired" for e in tail), "cancelled correction must not be repaired")
        require(cycle["outcome"] == "cancelled" and len(phases) == 1, "cancelled proposal lifecycle")
        require(any(r["outcome"] == "completed" and r["request_id"] != cycle["request_id"] for r in cycles),
                "slot reuse must complete a new proposal")
        return {"case": case, "synthetic": True, "proposal": proposal, "outcome": "cancelled_then_reused"}
    # Both stop and ignore-EOG assertions concern the original request only.
    # A subsequent slot-reuse request must not supply its missing decode/visits.
    end = next((i for i, e in enumerate(tail) if e["event"] == "reset"), len(tail))
    require(consumer != "server" or end < len(tail), "missing original request reset boundary")
    tail = tail[:end]
    repairs = [e for e in tail if e["event"] == "repaired"]
    expected_repair = case.startswith("repair") or (case == "eog_correction" and ignore)
    require(len(repairs) == int(expected_repair), "wrong repair branch")
    require(len(phases) == (2 if expected_repair else 1), "wrong actual verify phase count")
    if expected_repair:
        repaired = repairs[0]
        require(repaired["selected"] == decision["selected"] and repaired["sampler_preserved"],
                "selected decision/RNG changed during repair")
        require(repaired["attempts"] == decision["attempts"] and repaired["original_draft"] == 3,
                "new acceptance or changed proposal denominator")
        span = tail[:tail.index(repaired)]
        require(not any(e["event"] == "fresh_draft" for e in span), "new proposal during pending repair")
        decodes = [e for e in span if e["event"] == "decode"]
        expected = [decision["seed"], *decision["selected"][:-1]]
        require(len(decodes) == 1 and decodes[0]["repairing"] and decodes[0]["tokens"] == expected,
                "repair must materialize only seed plus accepted prefix")
        require(cycle["drafted_count"] == 3 and cycle["accepted_count"] == len(decision["selected"]) - 1,
                "correction counted as an accepted draft")
        if case == "repair2":
            require(len(decision["selected"]) == len(decision["draft"]), "same-length case not reached")
    selected = decision["selected"]
    eog = case.startswith("eog_")
    effective = selected[:selected.index(31) + 1] if eog and not ignore else selected
    visits = [e["token"] for e in tail if e["event"] == "visit"]
    require(visits[:len(effective)] == effective, "consumer changed selected token order")
    if eog and not ignore:
        stopped = tail
        require([e["token"] for e in stopped if e["event"] == "visit"] == effective,
                "consumer visited a post-EOG token/bonus")
        require(not any(e["event"] == "decode" for e in stopped), "decode after stopping EOG")
        require(any(e["event"] == "policy" and e["stops"] for e in stopped), "EOG policy not reached")
    if eog and ignore:
        last_visit = [i for i, e in enumerate(tail) if e["event"] == "visit"][len(selected) - 1]
        require(any(e["event"] == "decode" for e in tail[last_visit + 1:]), "ignore-EOG did not continue decoding")
    if case in {"eog_early", "eog_final"}:
        require(len(selected) == 4, "acceptor must return G+1 even at EOG")
    return {"case": case, "ignore_eos": ignore, "synthetic": True, "proposal": proposal, "verified": True}


def run_case(args, models, consumer, case, ignore):
    folder = args.output / f"{consumer}-{case}-ignore{int(ignore)}"
    folder.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_") and k != "LD_LIBRARY_PATH"}
    env.update(LLAMA_TEST_SPEC_CONSUMER_CASE=case, LLAMA_SPEC_SAMPLE_TEMP="0.8", LLAMA_SPEC_DRAFT_TOPP="1",
               LLAMA_SPEC_CHAIN="0", LLAMA_DRAFT_TOP_K="10", LLAMA_SPEC_LOG=str(folder / "cycles.jsonl"), LLAMA_TRACE="1")
    command = [str(args.server if consumer == "server" else args.simple),
               "-m", str(models / "target.gguf"), "-md", str(models / "head.gguf"),
               "--spec-type", "draft-mtp", "--no-spec-draft-backend-sampling", "--spec-draft-n-max", "3",
               "--spec-draft-n-min", "0", "--spec-draft-p-min", "0", "-ngl", "0", "-ngld", "0",
               "-c", "128", "-b", "32", "-ub", "32", "-t", "2", "-tb", "2", "-fa", "off",
               "--fit", "off", "--temp", "0.8", "--seed", "1201", "--logit-bias", "31-inf"]
    if consumer == "simple":
        command += ["-p", "A small synthetic prompt", "-n", "12"]
        if ignore:
            command += ["--ignore-eos"]
        save(folder / "launch.json", {"synthetic": True, "command": command, "env": {k: v for k, v in env.items() if k.startswith("LLAMA_")}})
        with child(command, env, folder / "process.log") as process:
            require(process.wait(timeout=45) == 0, "simple fixture failed")
    else:
        with socket.socket() as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
        # The speculative example parser does not expose the server warmup option.
        command += ["--no-warmup", "--host", "127.0.0.1", "--port", str(port), "-np", "1", "--slots"]
        save(folder / "launch.json", {"synthetic": True, "command": command, "env": {k: v for k, v in env.items() if k.startswith("LLAMA_")}})
        with child(command, env, folder / "process.log") as process:
            def healthy():
                try:
                    return request(port, "GET", "/health", timeout=1).get("status") == "ok"
                except (OSError, http.client.HTTPException, AssertionError):
                    return False
            wait_for(process, healthy)
            payload = {"prompt": [4, 5, 6], "temperature": .8, "seed": 1201, "n_predict": 12,
                       "ignore_eos": ignore, "return_tokens": True, "cache_prompt": False, "logit_bias": [[31, False]]}
            if case == "cancel":
                # The compile-only hook posts a real cancellation task. Its HTTP
                # request has no final completion by design; bounded client timeout.
                cancelled = {}
                def pending_request():
                    try:
                        cancelled["response"] = request(port, "POST", "/completion", payload, timeout=5)
                    except (OSError, http.client.HTTPException, AssertionError) as error:
                        cancelled["client_result"] = str(error)
                thread = threading.Thread(target=pending_request)
                thread.start()
                try:
                    wait_for(process, lambda: any(e["event"] == "reset" and e["pending_before"]
                                                  for e in observations(folder / "process.log")), seconds=10)
                finally:
                    thread.join(timeout=7)
                    require(not thread.is_alive(), "cancelled client did not terminate")
                save(folder / "cancel-client.json", cancelled)
            else:
                response = request(port, "POST", "/completion", payload)
                save(folder / "response.json", response)
                require(response["timings"]["draft_n"] > 0, "no actual server drafting")
                if case.startswith("eog_") and not ignore:
                    require(response["stop_type"] == "eos" and response["tokens"][-1] == 31, "server EOG result")
                if case.startswith("eog_") and ignore:
                    require(response["tokens_predicted"] == payload["n_predict"] and response["stop_type"] == "limit",
                            "original ignore-EOG request did not reach its generation limit")
            reuse = request(port, "POST", "/completion", {**payload, "n_predict": 8, "ignore_eos": True})
            save(folder / "reuse.json", reuse)
            require(reuse["tokens_predicted"] == 8 and reuse["stop_type"] == "limit", "slot reuse failed")
    result = validate(folder, case, ignore, consumer)
    save(folder / "result.json", result)
    return {"consumer": consumer, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("generator", "server", "simple", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    # Preserve earlier evidence instead of overwriting a failed fixture run.
    args.output.mkdir(parents=True, exist_ok=True)
    args.output = Path(tempfile.mkdtemp(prefix="run-", dir=args.output))
    print(f"Synthetic consumer evidence: {args.output}", flush=True)
    models = args.output / "models"
    models.mkdir()
    clean = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH" and not k.startswith("LLAMA_")}
    with child([str(args.generator), "--write-consumer-fixture", str(models)], clean, args.output / "generator.log") as process:
        require(process.wait(timeout=30) == 0, "fixture export failed")
    cases = [(f"repair{i}", True) for i in range(3)]
    cases += [(case, ignore) for case in ("eog_early", "eog_final", "eog_correction") for ignore in (False, True)]
    results = []
    for consumer in ("simple", "server"):
        for case, ignore in cases:
            results.append(run_case(args, models, consumer, case, ignore))
    results.append(run_case(args, models, "server", "cancel", True))
    save(args.output / "report.json", {"synthetic": True, "trained_model_evidence": False, "cases": results})
    print(f"PASS: {len(results)} synthetic executable consumer cases; real-model P06 campaign remains separate")


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    main()
