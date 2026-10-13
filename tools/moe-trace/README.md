# llama-moe-trace

Records which experts every MoE layer routes to, for a prompt and a sampled
continuation, and replays those traces against residency policies. Use it to
decide whether a model's routing is skewed enough for a static expert pack
([`--moe-pack-profile`](../../docs/backend/MOE-PACK.md)) or the dynamic
`--moe-cache` to pay, and to build the profile a pack is selected from.

```bash
# record: the prompt is decoded one ubatch at a time, then -n tokens are sampled
# with the usual sampling flags (--temp, --top-k, --top-p, --min-p, --seed)
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f prompt.txt -n 512 \
    --temp 0.7 --seed 1 -o work.csv

# replay: score one workload, select the static pack from another
tools/moe-trace/simulate.py eval.csv --profile work.csv --budgets 0.125,0.25
```

## Trace format (v1)

Lines starting with `#` carry provenance: tool version, a per-run `session`
id, model path/description/size, architecture, `n_layer`, `n_expert`,
`n_expert_used`, prompt length and the context's `n_ctx`, `n_batch` and `n_ubatch`
(as the context resolved them, not as requested), and the sampling seed and settings. Without
`--seed` the tool draws one seed and gives it to every sampler (the distribution, XTC and mirostat
samplers would each draw their own), so the recorded seed replays the run.
Every sampling control that can change the continuation, and with it the routing, is
recorded: `# sampling_chain=` lists the sampler chain in order (`?` marks a sampler that
is inactive with its parameters), `# sampling_params=` the parameters
`common_params_sampling::print()` shows, as `name=value` pairs with every float exact
(the shortest text that reads back to the same value), and `# sampling_inputs=` the
others, with the grammar, the logit biases, the DRY sequence breakers and the lazy-grammar
triggers as `COUNT:FNV-1a-64` identities (`none` when empty).
One `# expert_tensors=LAYER:TYPE/BYTES/N_IN/N_OUT,...` line per MoE layer lists its
routed expert tensors in the order the graph multiplies them (up, gate, down; or
gate_up, down), with the weight type, bytes per expert and the expert's input and
output widths; the simulator groups them into the provider's pools and sizes the SYCL
provider's scratch from the widths (older traces write `TYPE/BYTES` only). A layer with a second router under the same
tensor name (GroveMoE's adjugate experts) keeps only the main router's rows and
tensors. Then a header row and one row per (token, layer):

```
pos,layer,id0,...,idK-1
```

Prompt token `p` is written as `pos = -(p + 1)`; decode step `s` as `pos = s`.
Readers select decode routing with `pos >= 0`. Concatenating trace files keeps
their sessions apart through the `# session=` lines.

Many models compute the last layer's FFN for output rows only. For those layers
the trace has one prompt row per ubatch (its last token), not one per prompt
token; layer 0 always has every prompt token.

## Limits

- Tracing copies every top-k tensor to the host and synchronizes. A traced run
  is a routing sample, not a throughput or overlap measurement.
- A backend that fuses the router away from `ffn_moe_topk-<il>` produces no rows;
  the tool then exits with an error instead of writing an empty trace.
- `-n` must be positive and the prompt plus `-n` must fit the context.

## Expert prediction diagnostics (experimental)

Two opt-in outputs for measuring whether expert routing can be predicted early
enough to prefetch. Nothing consumes what they add, but `--spec-router` adds graph
nodes and both add eval-callback reads, which can change backend fusion or graph
partitioning and so the last bits of later results. The test checks that the
routed experts of every row stay the same on the fixtures.

```bash
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f prompt.txt -n 256 --seed 1 -o work.csv \
    --spec-router 16 --spec-out spec.csv --hidden-out work.hid --hidden-n 64
```

`--spec-router K` evaluates every MoE layer's router on the layer input, before
attention, through the router's own RMS norm, and traces its top K as
`ffn_moe_spec_topk-<il>`. It is a heuristic built from the existing router weights,
not a trained predictor. Its lead time is the layer's attention block, because the
prediction depends only on the layer input. The ids go to `--spec-out` in the trace
format above, with a `# kind=spec-router k=K` line and the session id of the main
trace; `--moe-pack-profile` and `simulate.py` refuse such a file. The log reports
the recall: the share of the routed decode experts that were
among the K predicted ones, overall and for the worst and best layer. K is
1..4096, at most the model's expert count. Supported: qwen3moe and qwen35moe. On
any other architecture the tool exits with an error.

`--hidden-out FILE` writes, for the first `--hidden-n` decode steps (1..65536,
default 64), three float32 vectors per layer and step:

| tag | name | tensor |
| --- | --- | --- |
| 0 | `layer_out` | `l_out-<il>`, the next layer's input |
| 1 | `residual_after_attention` | `ffn_inp-<il>` or `attn_residual-<il>` |
| 2 | `router_input` | the input of the router matmul `ffn_moe_logits-<il>` |

A LoRA adapter on a router (`ffn_gate_inp`) wraps that matmul, so its input cannot
be recorded; such a run stops with an error after the prompt instead of writing an
incomplete file.

The file starts with text lines: `llama-moe-trace hidden v1`, then `key=value`
lines (session, `byte_order=little`, model path, description and size, arch,
`n_layer`, `n_embd`, `max_decode` (the requested `--hidden-n`; generation can end
earlier), the tag and record layouts), then `end`. Records follow: `pos`, `layer`,
`tag`, `n` as int32, then `n` float32 values. The file is at most
`max_decode x n_layer x 3 x (16 + 4 n_embd)` bytes; the tool logs that bound. `-o`,
`--spec-out` and `--hidden-out` must name different files, and none may name an
input (any shard of the model, or the `-f`/`-bf` prompt file). An output replaces
only an empty file or one llama-moe-trace wrote; any other existing file is
refused before the model loads, which protects adapters, control vectors and
grammar or schema files too.
`hidden.py` reads it (stdlib only) and prints the header and record counts.

Validation: `tests/moe_trace_oracle.py` (run by `test-moe-trace` when numpy is
available) recomputes on the host, from the dumped vectors and the fixture's F32
weights, the routed experts of every decode row (from tag 2) and the spec ids of
layer 1 (from layer 0's output). Both must match on the CPU, Vulkan and OpenVINO.
With K equal to the expert count the recall must be exactly 1.

### Why there is no dynamic ring

The source fork appends a refilled ring of slots to the static pack, driven by this
prediction (`fable5/expert-prediction`). It was not ported:

- The ring and its maps are mutable state of the shared model. Contexts that share a
  model would overwrite experts that another context's graph is still reading.
- Its staging fallback can point async copies at a reused host buffer, and it takes
  a second `ggml_backend_dev_init` as a separate upload stream. The target's
  equivalent contract is `ggml_backend_init_private_stream`, which only SYCL on
  Level Zero v1 provides; it cannot be exercised on the backends available to the
  test host.
- Runtime admission already exists as `--moe-cache`. A predictor would be an
  admission or prefetch policy of that provider, with its ownership and fallback
  rules, and not a second model-global cache.

The recall numbers above are what decide whether that is worth building. A useful
predictor needs high recall at a K whose transfer cost fits in the attention time of
one layer on the target link.

## Simulator

`simulate.py` scores decode rows only. Per-layer policies use `S = budget x
n_expert` slots: `static` (top-S from a separate `--profile` trace),
`static-oracle` (top-S from the scored trace itself; optimistic, not
deployable), `static-prefill`, `lru`, `lfu-decay` and `reelect-K`. The
`dynamic` policy models the `--moe-cache` provider. The provider keeps one pool
per weight type and expert size, caps it at the creating tensor's `n_expert`
slabs whatever the budget, keys its entries by (expert tensor, expert) and plans
each `MUL_MAT_ID` on its own. With the trace's `# expert_tensors=` lines the
replay does the same: each expert tensor of a layer is its own plan, in the
listed order, on the pool of its type and size. Gate, up and down of one type
and size therefore compete for one pool, and hit and `ins/tok` are weighted by
bytes, the inserts in (layer, expert) units. The byte budget holds `S` experts of
every expert tensor. `--provider sycl` (the default) follows the SYCL provider's
per-device budget: the pools take it in the order the graph meets them,
`min((left - scratch) / expert size, n_expert)` slabs each, so a later pool can get
less than its share or nothing. `scratch` is the device scratch the provider holds
back, sized by the largest plan met so far (f32 outputs and q8_1 activations for 64
ids, `sycl_moe_scratch_bytes`); without the widths in the trace it is left out and
the report calls the figure an upper bound. `--provider vulkan` follows the Vulkan provider, which sizes
every pool from the whole budget, so its pools together can exceed it. A trace without
those lines (older traces) gets one pool of `min(S x n_layers, n_expert)`
(layer, expert) units and one plan per layer: an upper bound wherever tensors of
a layer share a pool. In both, a pool below the providers' 64-slot minimum is
not created (models with fewer than 64 experts get no dynamic hits), the
order is LRU, units with more than `--hot-uses` uses are evicted only when every
unit is hot, and like the provider each plan fills at most `--inserts-per-plan`
misses (default 8, as `GGML_CUDA_MOE_CACHE_INSERTS`) and never evicts a unit it
hit or filled; the other misses run on the CPU and are not inserted. The provider also caps fills
at `GGML_CUDA_MOE_CACHE_QUEUE_MB / expert size`, which the simulator does not
model (with the 512 MiB default it binds only above 64 MiB per expert tensor).
The replay assumes the provider takes every MoE layer. It does not: expert tensors
below its minimum (`GGML_CUDA_MOE_CACHE_MIN_EXPERT_KB`; the provider logs the value
in use as `min-expert=`) and unsupported weight types stay on the CPU. The
`# expert_tensors=` sizes can be compared with that `min-expert=` value; the replay
does not apply it, nor the per-backend list of supported types.

The provider plans prompt calls too: a `MUL_MAT_ID` of at most
`GGML_CUDA_MOE_CACHE_MAX_BATCH` tokens (default 8, `--max-batch`) and 64 ids fills
and evicts like a decode step; larger ones bypass it. Before each session's decode
rows the replay runs the plans of its prompt calls that fit, from the session's
`# n_prompt=` line (calls of `min(n_batch, n_ubatch)` tokens, as llama-moe-trace
decodes the prompt): a short prompt, a short last call, and a last layer that routes
only the output row. They warm the pools and are not scored, and their fills do not
count in `ins/tok`. An expert that several tokens of one plan route is filled once:
the SYCL provider counts no use for the later tokens, the Vulkan provider with
host-mapped slabs counts one each, and `--provider` follows that (without
host-mapped slabs the Vulkan provider fills it again into a second slot, which the
replay does not model). A session without the line starts its decode from the
previous session's state, and the report says so.

Expert counts come from the trace's `# n_expert=` line, or from `--n-expert`
for a trace without one (a different value is an error), never from the
largest observed id. All traces and the `--profile` traces must agree on
`# arch=` and `# n_layer=` where they carry them: layer ids of different models
do not line up. Sessions are replayed in file order with cache state
carried over; `reelect-K` adds each session's prompt routing to its counts when
that session starts. The cost columns appear only when `--slab-mb`,
`--ram-gbs` and `--link-gbs` are all given; they are an assumed model that
excludes kernels, activation transfers, synchronization and compilation.

Tests: `tests/test-moe-trace.sh` (ctest `test-moe-trace`, with
`tests/moe_trace_oracle.py`) and `tests/test_moe_trace_simulate.py` (ctest
`test-moe-trace-simulate`).
