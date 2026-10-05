# P10 - Cumulative-probability draft width

**Kind:** decoupling
**Depends on:** P04 for the dataset that sets the default

## Purpose

Narrow the draft once the running product of the drafted tokens' top-1
probabilities falls below a threshold, so a deep context does not pay to verify
rows that cannot be accepted. Kmic-68 stop when the running product drops under a
threshold that ramps with depth, and report that the benefit only appears deep in
context, where each extra verify row costs a full attention pass over the KV
cache.

**Correction to the source, found while reading our tree.** Kmic-68's loop has a
per-token confidence stop already; ours does not. Our MTP drafter commits to a
width up front, `n_cap[seq_id] = adaptive ? adaptive_ctrl[seq_id].n_cur : params.n_max`
(`common/speculative.cpp:2285`), and never inspects a probability mid-draft. So
this is not a port but a new stop rule for our loop, and it has two distinct
sites:

- **non-chained MTP, eagle3, dflash**: the loop samples on the host
  (`common/speculative.cpp:2348`, candidates at `:2351`, push at `:2375`, stop at
  `:2404`). A running product breaks out of this loop cleanly.
- **`--spec-chain`**: all `n_chain` tokens come from one in-graph argmax decode
  (`common/speculative.cpp:2210-2216`). There is no host loop and therefore no
  per-token stop. Chain mode is explicitly out of scope for this plan.

## Source

- Kmic-68 `common/speculative.cpp`: `p_cum`, `p_cum_min`, and the stop condition
  `pc_next < p_cum_min(pos0)` alongside the existing `p_min` test.

## Requirements

- **R10.1** (optional feature) WHERE `LLAMA_SPEC_P_CUM` is set, the <drafter> shall multiply each successive drafted token's top-1 probability into a running product initialised to one, and shall stop drafting when that product falls below the threshold.
- **R10.2** (ubiquitous) The <stop rule> shall apply in addition to, and never in place of, any existing per-token stop rule.
- **R10.3** (optional feature) WHERE the threshold is not set, the <drafter> shall apply no cumulative-probability stop, regardless of padded verify state.
- **R10.4** (event-driven) WHEN the threshold is depth-dependent, the <default> shall be derived from a paired A770 measurement over our production depth range, not copied from Kmic-68's P100 values.
- **R10.5** (event-driven) WHEN `LLAMA_SPEC_P_CUM` is set to a negative value, the <drafter> shall use the measured default ramp.
- **R10.6** (unwanted) IF drafting stops by the cumulative rule at a depth of one, THEN the <drafter> shall still emit that first token.
- **R10.7** (event-driven) WHEN the running product falls below the threshold, the <drafter> shall discard that token and end the round, matching the existing `p_min` behaviour.
- **R10.8** (unwanted) IF the drafter is the chained MTP path, THEN the <cumulative rule> shall not apply, and the reason shall be documented at the call site.
- **R10.9** (event-driven) WHEN the adaptive controller selects the draft width, the <cumulative rule> shall clamp the selected width rather than replace the controller's choice.

## Acceptance

Paired depth sweep on the A770 reporting accepted-tokens-per-drafted-token and tg,
against P01's re-baselined numbers, with the P04 log as the dataset. Any default
threshold chosen here is recorded with the measurement that produced it.
