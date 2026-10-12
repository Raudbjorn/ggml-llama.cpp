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
`n_expert_used`, prompt and batch sizes, and the sampling seed and settings.
Then a header row and one row per (token, layer):

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

## Simulator

`simulate.py` scores decode rows only. Per-layer policies use `S = budget x
n_expert` slots: `static` (top-S from a separate `--profile` trace),
`static-oracle` (top-S from the scored trace itself; optimistic, not
deployable), `static-prefill`, `lru`, `lfu-decay` and `reelect-K`. The
`dynamic` policy models the `--moe-cache` provider: one pool of
`S x n_layers` units shared by all layers, LRU order, and units with more than
`--hot-uses` uses evicted only when every unit is hot.

Expert counts come from the trace's `# n_expert=` line or `--n-expert`, never
from the largest observed id. Sessions are replayed in file order with cache
state carried over. The cost columns appear only when `--slab-mb`,
`--ram-gbs` and `--link-gbs` are all given; they are an assumed model that
excludes kernels, activation transfers, synchronization and compilation.

Tests: `tests/test-moe-trace.sh` (ctest `test-moe-trace`) and
`tests/test_moe_trace_simulate.py` (ctest `test-moe-trace-simulate`).
