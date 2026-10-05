# P05 - Top-k prefilter in the CPU sampler

**Kind:** port
**Depends on:** none

## Purpose

When a sampler chain effectively begins with `top_k(k)`, nothing ahead of it can
change which tokens are the `k` largest logits. Kmic-68 therefore select them
straight from the logits rather than materialising and partially sorting the whole
vocabulary, and report that on a 248k vocab this is most of the host time in a
speculative cycle.

The guard passes in our fork without adaptation: the drafter sets
`sparams.top_k = LLAMA_DRAFT_TOP_K` (10) with `samplers = { TOP_K }` and no logit
bias.

**Scope.** This covers the non-chained MTP, eagle3 and dflash paths. Our
`--spec-chain` path samples greedily in-graph and emits `[id, prob]` pairs
(`common/speculative.cpp:2210-2216`), so it never reaches `set_logits` and is
unaffected by this plan.

## Source

- Kmic-68 `common/sampling.cpp` +335/-3: `top_k_from_logits`, `prefilter_k`,
  `can_prefilter`, the SSE2 block scan, and the guard in `common_sampler_init`.

## In this fork

- `common/sampling.cpp:130` `set_logits`, `:594` `common_sampler_sample`, `:607`
  the call site.
- `src/llama-ext.h:124` `LLAMA_DRAFT_TOP_K` = 10.
- `common/sampling.cpp:509-519` `common_sampler_clone` and `:521-539`
  `common_sampler_copy`, which must carry the new field.

## Requirements

- **R05.1** (ubiquitous) The <sampler> shall, where the chain's first non-no-op entry is TOP_K with `top_k <= 128` and no logit bias is configured, populate `cur_p` with the `k` largest logits descending without materialising the full vocabulary.
- **R05.2** (unwanted) IF a tie for the k-th position exists among the selected logits, THEN the <prefilter> shall fall back to the existing full-vocabulary path, because `llama_token_data_array_partial_sort_inplace` compares `a.logit > b.logit` only and `std::partial_sort` is not stable, so the existing order among equal logits is arbitrary and an ascending-id rule would not reproduce it.
- **R05.3** (unwanted) IF a logit bias is configured, or `mirostat` is non-zero, or `top_k > 128`, or `top_k <= 0`, THEN the <sampler> shall use the existing full-vocabulary path.
- **R05.4** (unwanted) IF a forcing reasoning budget is active, THEN the <sampler> shall use the existing full-vocabulary path.
- **R05.5** (unwanted) IF `LLAMA_SAMPLER_PREFILTER=0`, THEN the <sampler> shall use the existing full-vocabulary path.
- **R05.6** (event-driven) WHEN a grammar is applied ahead of sampling, the <sampler> shall use the existing full-vocabulary path.
- **R05.7** (ubiquitous) The <sampler> shall propagate `prefilter_k` through clone and copy.
- **R05.8** (event-driven) WHEN the prefilter path and the full path are given identical logits, the <test> shall assert identical sampled tokens across a seeded sweep of the vocabulary.

## Acceptance

Identical output tokens with the prefilter on and off over the oracle corpus.
Host time per speculative cycle before and after, A770, three reps, named driver.
