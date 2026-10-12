# llama-schema-decision

Answers a finite JSON schema about one or more contexts by scoring the allowed values of every
field instead of generating the JSON token by token. Each field's allowed values (enum strings,
booleans, bounded integers, numbers on a grid) form a token trie after the field's own suffix.
Every scored trie node runs as its own sequence forked from the context, so the fields are
scored in batched `llama_decode` calls and cannot see each other's answers. The JSON object is
assembled by code, so it always matches the schema.

The engine (`schema-decision.{h,cpp}`) is shared with `llama-server`'s `POST /v1/decision`. It was
ported from thecodacus/llama.cpp `decision-upstream` (a8d96e948, `tools/parallel-decision`); see
[differences from the source](#differences-from-the-source).

## What the numbers mean

- `tree` fields score every divergence node of the trie and return a distribution over all
  allowed values. At each node the softmax is taken over the allowed next tokens only; a value's
  score is the product along its path, normalised over the values. These are constrained-decoding
  probabilities: not calibrated probabilities that the answer is true, and not full-vocabulary
  sequence likelihoods.
- `greedy` fields walk the trie, taking the most likely allowed token at each divergence. They
  need one scoring round per divergence on the chosen path and can miss the value with the best
  complete-path score. `probability` is then the product of the step probabilities.
- `auto` (default) scores fields with up to `tree_max` values as `tree`, larger ones `greedy`.
- Fields are independent: no field conditions on another field's answer. Cross-field
  constraints and JSON Schema keywords beyond the subset below are not supported and are
  rejected rather than ignored.

## CLI

```sh
llama-schema-decision -m model.gguf -c 8192 --decision-seqs 16 --decision-request request.json
llama-schema-decision -m model.gguf --decision-request - < request.json
```

The request is the `/v1/decision` body; the response object is printed to stdout. The CLI's
context belongs to the engine: `--decision-seqs N` (default 16, minimum 3) sequences, a unified KV
cache, and the model's chat template (`--chat-template`, `--jinja` apply).

```json
{
  "instructions": "Answer each question about this support request.",
  "schema": {
    "category": {"type": "enum", "choices": ["billing", "technical", "other"], "description": "Request type."},
    "urgent":   {"type": "boolean", "description": "Does it need urgent handling?"},
    "priority": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Priority, 5 is highest."}
  },
  "contexts": ["I was charged twice and need this fixed today."]
}
```

Qwen3-0.6B Q8_0 on two CPU cores answers it as follows (abridged; a 0.6B model, so the
category is wrong):

```json
{
  "object": "decision",
  "mode": "auto",
  "results": [{
    "decision": {"category": "technical", "urgent": true, "priority": 5},
    "fields": {
      "category": {"value": "technical", "probability": 0.6526, "scored_nodes": 1, "tree": true},
      "urgent":   {"value": true, "probability": 0.9999, "scored_nodes": 1, "tree": true},
      "priority": {"interval_p10_p90": [5, 5], "value": 5, "probability": 0.9846, "scored_nodes": 1, "tree": true}
    },
    "usage": {"context_tokens": 18, "scored_rows": 20}
  }],
  "usage": {"prompt_tokens": 120, "cached_tokens": 0, "context_tokens": 18, "scored_rows": 20},
  "timings": {"prefill_ms": 435.0, "scoring_ms": 105.0, "total_ms": 546.8, "rounds": 1, "decodes": 3, "per_decision_ms": 546.8}
}
```

Numeric fields also return `interval_p10_p90`, and `mean` with the `mean` aggregate.

### Request

| key | default | meaning |
| --- | --- | --- |
| `schema` | required | compact fields or a JSON Schema object (below) |
| `contexts` | required | 1-256 non-empty strings, 1 MiB each, 16 MiB together; results keep their order |
| `instructions` | `""` | appended to the generated field catalogue; cached with it (64 KiB) |
| `mode` | `auto` | `auto`, `tree` or `greedy` |
| `tree_max` | 128 | integer 0-255: `auto` uses `tree` for fields with at most this many values |
| `cache_prompt` | `true` | reuse the cached instructions and catalogue when they match |

### Schema

Compact form: `{"name": {"type": ..., "description": ..., ...}}`, every field needs a
description. JSON Schema form: `{"type": "object", "properties": {...}}` (`required`,
`additionalProperties`, `title`, `description`, `$schema` and `$comment` are accepted and have no
effect; every field is always answered).

| type | keys | values |
| --- | --- | --- |
| `enum` (JSON Schema: `string` with `enum`) | `choices` or `enum` | 1-255 distinct strings of up to 256 bytes |
| `boolean` | | `true`, `false` |
| `integer` | `minimum`, `maximum` | every integer in the bounds, 1-255 values, full 64-bit range |
| `number`, compact | `minimum`, `maximum`, `step` | `minimum + k*step`, which must reach `maximum` exactly |
| `number`, JSON Schema | `minimum`, `maximum`, `multipleOf` | the multiples of `multipleOf` within the bounds |

Numbers must be finite, within 1e15 in magnitude, and need at most 9 decimals; every value of a
field prints with the same number of decimals. `aggregate` (or `x-aggregate`) picks the answer of
a numeric `tree` field: `mode` (default, the most likely value), `median`, or `mean` (the grid
value nearest to the expected value). Any other keyword (`exclusiveMinimum`, `const`, `pattern`,
`anyOf`, ...) is an error. At most 32 fields, 256-byte names, 4 KiB descriptions.

## Tokenization

The prompt is the chat template applied to a system message (field catalogue and instructions)
and a user message holding the context, then the generation prompt and `{\n`. Everything before
the user message is the cached prefix. Each value is scored as the tokens of
`  "name": value\n` after the prompt.

Tokenizing the prompt and a value apart gives tokens the joint text never tokenizes to. Probed on
20 vocabularies (17 in `models/`, SmolLM2, Qwen3, Qwen3.5), a prompt ending in `{\n` gave wrong
value tokens on 9 of them (SmolLM2, Phi-3, Llama 2 SPM, Starcoder, Falcon among them) and a prompt
ending in `{` on 7 (Llama 3, Qwen2 and Qwen3 among them): whitespace runs and `{\n` merge across
the boundary, and SPM adds a space prefix to every separately tokenized string.

The engine therefore cuts each context after its last special token near the end, where the
tokenizer partitions the text before any merge, and tokenizes the values
together with the text after the cut. Without such a token it moves the cut back from the end
until the values tokenize after a few tokens of left context without changing them. The test
checks that every scored path equals the tokenization of the joint text on the 17 vocabularies in
`models/` and on real models, with and without a chat template. The cached prefix is the longest
token prefix shared by the joint tokenizations of all contexts.

Special tokens in the context text are parsed like in llama-server's chat completions.

## Sequences and memory

The engine uses `N` sequences: one holds the cached prefix, one per context in flight holds its
trunk (prefix and context), and the rest score branches. Contexts are processed in groups that
fit the budget; more sequences mean fewer decodes, not different results. The KV cache must be
unified so that branches share the trunk's cells: a plain attention model needs about the prompt
length plus the scored rows of one group in cells. A sliding-window model allocates its window per
sequence, and a recurrent or hybrid model (Qwen3.5, Nemotron-H) one recurrent state per sequence.

Recurrent layers take a ubatch only when every sequence in it holds the same number of tokens.
On those models every group of branches is right-padded to its longest branch and bounded by
`n_ubatch`, so a group is one ubatch. The logits are read at each branch's last real token, so the
padding does not change them.

A request that runs out of KV space, is cancelled, or fails in any other way returns an error and
leaves none of the engine's sequences behind; the next request rebuilds the cached prefix.

## Validation

`tests/test-schema-decision.cpp`:

- `--vocab models` (ctest `test-schema-decision-vocab`): schema and request validation (64-bit
  integer bounds, non-finite and misaligned grids, unsupported keywords, `tree_max`, context
  limits, aggregates), and the tokenization identity above on every vocabulary in `models/`.
- `--model` (ctest `test-schema-decision`, on dense and hybrid test-llama-archs fixtures): token
  level requests against an oracle that decodes every trie prefix from an empty memory on one
  sequence. Tree, greedy and auto modes, cached and uncached prefix, a single context against the
  batch, 3 sequences against 16, a one-value field, cancellation, KV exhaustion and invalid plans.
  Results match the oracle within 4e-8.
- `tools/server/tests/unit/test_schema_decision.py`: the endpoint on a running server: disabled
  without `--decision-seqs`, cached prefix, batch against single contexts, modes, invalid requests
  (400), a context too long (400), KV exhaustion (503) with the server still answering after it,
  completions interleaved with decisions giving the same output as alone, and a decision after
  the server slept and woke.
- `--e2e MODEL...`: a text request through the chat template against the oracle on the joint
  tokenization. The engine is held to the backend's own spread, measured as the same oracle
  decoded in ubatches of 512 and 64 (about 0.03 in probability on the CPU, because the kernels are
  not batch-invariant). Run on SmolLM2-135M, Qwen3-0.6B and Qwen3.5-0.8B (hybrid).

## Differences from the source

- Errors are returned as values; nothing throws out of the engine, so a bad request cannot take a
  server down. Every limit above is enforced before anything is decoded.
- Integer bounds use overflow-free width arithmetic; `multipleOf` grids are aligned to multiples
  of the step (the source started them at `minimum`); non-finite numbers and grids needing more
  than 9 decimals are rejected; `-0` is never printed.
- Value tokenization follows the joint text (above); the source tokenized the value apart from
  the prompt, which gives non-canonical tokens on most of the vocabularies tested.
- Non-finite logits are an error instead of a silent NaN probability.
- Padded groups are bounded by `n_ubatch`, not `n_batch`, so a padded group is one ubatch.
- The stdin worker protocol of `llama-parallel-decision` (for llama-mojo) and its legacy
  `split_boundary` mode are not ported; the CLI reads the server's request body instead.
