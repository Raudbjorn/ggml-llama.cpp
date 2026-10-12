import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_batch = 256
    server.decision_seqs = 8


SCHEMA = {
    "category": {"type": "enum", "choices": ["billing", "technical", "tech support", "other"],
                 "description": "What type of request is this?"},
    "urgent": {"type": "boolean", "description": "Does it need urgent handling?"},
    "priority": {"type": "integer", "minimum": 1, "maximum": 12, "description": "Priority."},
    "score": {"type": "number", "minimum": 0, "maximum": 1, "step": 0.25, "description": "Confidence.",
              "aggregate": "median"},
}

CONTEXTS = [
    "I was charged twice and need this fixed today.",
    "The app crashes when I upload a photo.",
    "Do you ship to Iceland?",
]


def decide(contexts, **extra):
    body = {"instructions": "Answer about this support request.", "schema": SCHEMA, "contexts": contexts}
    body.update(extra)
    return server.make_request("POST", "/v1/decision", data=body)


def check_result(item):
    decision = item["decision"]
    assert decision["category"] in SCHEMA["category"]["choices"]
    assert decision["urgent"] in (True, False)
    assert 1 <= decision["priority"] <= 12
    assert decision["score"] in (0.0, 0.25, 0.5, 0.75, 1.0)
    for name, field in item["fields"].items():
        assert 0.0 < field["probability"] <= 1.0 + 1e-6, name
        assert field["value"] == decision[name]
    # numeric tree fields carry the distribution's 10th and 90th percentiles
    if item["fields"]["score"]["tree"]:
        assert len(item["fields"]["score"]["interval_p10_p90"]) == 2


def test_decision_disabled():
    global server
    server.decision_seqs = None
    server.start()
    res = decide(CONTEXTS[:1])
    assert res.status_code == 501
    assert "--decision-seqs" in res.body["error"]["message"]


def test_decision_basic_and_cached_prefix():
    global server
    server.start()
    res = decide(CONTEXTS)
    assert res.status_code == 200, res.body
    assert res.body["object"] == "decision"
    assert len(res.body["results"]) == len(CONTEXTS)
    for item in res.body["results"]:
        check_result(item)
    usage = res.body["usage"]
    assert usage["cached_tokens"] == 0
    assert usage["prompt_tokens"] > usage["context_tokens"] > 0

    # the same instructions and schema: the static prefix comes from the cache
    again = decide(CONTEXTS)
    assert again.status_code == 200
    assert again.body["usage"]["cached_tokens"] > 0
    for a, b in zip(res.body["results"], again.body["results"]):
        assert a["decision"] == b["decision"]
        for name in SCHEMA:
            assert a["fields"][name]["probability"] == pytest.approx(b["fields"][name]["probability"], abs=1e-4)

    # cache_prompt false rebuilds the prefix and gives the same answers
    fresh = decide(CONTEXTS, cache_prompt=False)
    assert fresh.status_code == 200
    assert fresh.body["usage"]["cached_tokens"] == 0
    assert [r["decision"] for r in fresh.body["results"]] == [r["decision"] for r in res.body["results"]]


def test_decision_batch_matches_single_contexts():
    global server
    server.start()
    batch = decide(CONTEXTS, mode="tree")
    assert batch.status_code == 200
    for i, context in enumerate(CONTEXTS):
        one = decide([context], mode="tree")
        assert one.status_code == 200
        for name in SCHEMA:
            assert one.body["results"][0]["fields"][name]["probability"] == pytest.approx(
                batch.body["results"][i]["fields"][name]["probability"], abs=2e-3)


def test_decision_modes():
    global server
    server.start()
    greedy = decide(CONTEXTS[:1], mode="greedy")
    assert greedy.status_code == 200
    assert all(not f["tree"] for f in greedy.body["results"][0]["fields"].values() if f["scored_nodes"] > 0)
    auto = decide(CONTEXTS[:1], mode="auto", tree_max=4)
    assert auto.status_code == 200
    fields = auto.body["results"][0]["fields"]
    assert fields["category"]["tree"] and not fields["priority"]["tree"]
    check_result(auto.body["results"][0])


@pytest.mark.parametrize("body,needle", [
    ({"schema": SCHEMA, "contexts": []}, "contexts"),
    ({"schema": SCHEMA, "contexts": [""]}, "context"),
    ({"schema": SCHEMA, "contexts": ["c"] * 257}, "contexts"),
    ({"schema": SCHEMA, "contexts": ["c"], "tree_max": 256}, "tree_max"),
    ({"schema": SCHEMA, "contexts": ["c"], "mode": "fast"}, "mode"),
    ({"schema": {"x": {"type": "integer", "minimum": 0, "maximum": 300, "description": "d"}}, "contexts": ["c"]}, "1-255"),
    ({"schema": {"type": "object", "properties": {"x": {"type": "number", "minimum": 0, "maximum": 1,
                                                        "multipleOf": 0.5, "exclusiveMinimum": 0}}},
      "contexts": ["c"]}, "exclusiveMinimum"),
    ({"contexts": ["c"]}, "schema"),
])
def test_decision_invalid_request(body, needle):
    global server
    server.start()
    res = server.make_request("POST", "/v1/decision", data=body)
    assert res.status_code == 400
    assert needle in res.body["error"]["message"]


def test_decision_context_too_long():
    global server
    server.n_ctx = 512
    server.start()
    res = decide(["word " * 2000])
    assert res.status_code == 400
    assert "positions" in res.body["error"]["message"]
    # the server and the engine keep working
    res = decide(CONTEXTS[:1])
    assert res.status_code == 200


def test_decision_interleaved_with_completions():
    global server
    server.temperature = 0.0
    server.start()
    prompt = "Once upon a time"

    def complete():
        res = server.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 32, "temperature": 0.0})
        assert res.status_code == 200
        return res.body["content"]

    reference = complete()
    results = parallel_function_calls([
        (complete, ()),
        (decide, (CONTEXTS,)),
        (complete, ()),
        (decide, (CONTEXTS[:1],)),
    ])
    # decisions run on their own sequences: the slots' output does not change
    assert results[0] == reference
    assert results[2] == reference
    assert results[1].status_code == 200 and results[3].status_code == 200
    assert results[3].body["results"][0]["decision"] == results[1].body["results"][0]["decision"]


def test_decision_after_sleep():
    global server
    server.sleep_idle_seconds = 1
    server.start()
    before = decide(CONTEXTS[:1])
    assert before.status_code == 200
    start = time.time()
    while not server.make_request("GET", "/props").body["is_sleeping"]:
        assert time.time() - start < 10, "server did not go to sleep"
        time.sleep(0.1)
    # sleeping freed the context and the engine; the request wakes the server and rebuilds both
    after = decide(CONTEXTS[:1])
    assert after.status_code == 200
    assert after.body["usage"]["cached_tokens"] == 0
    assert after.body["results"][0]["decision"] == before.body["results"][0]["decision"]


def test_decision_kv_exhaustion():
    global server
    server.n_ctx = 512
    server.start()
    probe = decide(["x"])
    assert probe.status_code == 200
    usage = probe.body["usage"]
    n_shared = usage["prompt_tokens"] - usage["context_tokens"]
    n_tok_x = len(server.make_request("POST", "/tokenize", data={"content": "x"}).body["tokens"])
    n_extra = usage["context_tokens"] - n_tok_x  # template tokens that stay in the trunk

    # a context that fits with its longest branch (under two dozen tokens here), but not with
    # a group of branches at once
    target = 512 - n_shared - 24 - n_extra
    words = []
    while len(server.make_request("POST", "/tokenize", data={"content": " ".join(words)}).body["tokens"]) < target:
        words.append("word")
    res = decide([" ".join(words[:-1])])
    assert res.status_code == 503, res.body
    assert "KV cache" in res.body["error"]["message"]
    # the failed request left nothing behind
    res = decide(CONTEXTS[:1])
    assert res.status_code == 200
