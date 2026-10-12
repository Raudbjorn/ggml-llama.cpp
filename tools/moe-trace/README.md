# llama-moe-trace

Records which experts every MoE layer routes to, for a prompt and a sampled
continuation. Use it to measure how skewed a model's routing is before
deciding on expert residency, and to build routing profiles.

```bash
# record: the prompt is decoded one ubatch at a time, then -n tokens are sampled
# with the usual sampling flags (--temp, --top-k, --top-p, --min-p, --seed)
llama-moe-trace -m model.gguf -ngl 99 --n-cpu-moe 99 -f prompt.txt -n 512 \
    --temp 0.7 --seed 1 -o work.csv
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

Test: `tests/test-moe-trace.sh` (ctest `test-moe-trace`).
