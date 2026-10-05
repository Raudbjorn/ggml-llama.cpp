# P07 - Block verification

**Kind:** port
**Depends on:** P06

## Purpose

Sun et al., arXiv:2403.10444, Algorithm 2. Under per-token rejection sampling a
rejection at position *i* ends the block. Block verification does not: it tracks
`keep[i]`, the probability that the prefix through position *i* survives, and a
later position can still accept the longer prefix. Kmic-68 cites their Theorem 2,
never fewer accepted tokens in expectation.

The instrumentation matters as much as the rule. `LLAMA_SPEC_BLOCK_VERIFY=2`
reports the expected accepted tokens per drafted token under both rules,
`sum_i keep[i]` against `sum_i prod_{j<=i} min(1, p_j/q_j)`, which measures the
gain with no timing run at all.

## Source

- Kmic-68 `common/sampling.cpp`: `common_spec_block_verify`, and the block branch
  inside `common_sampler_sample_and_accept_n_dist`.

## In this fork

- `common/sampling.cpp:678`, the per-token rule that P06 keeps for stateful chains
  and that this plan supersedes for stateless ones.

## Requirements

- **R07.1** (event-driven) WHEN `LLAMA_SPEC_BLOCK_VERIFY` is not zero and the chain is stateless, the <acceptor> shall evaluate every position in the block rather than stopping at the first rejection.
- **R07.2** (event-driven) WHEN position *i* is evaluated, the <acceptor> shall accumulate the unclipped cumulative ratio `cum[i] = cum[i-1] * p(x_i)/q(x_i)` and define `keep[i] = min(cum[i], 1)`.
- **R07.2a** (unwanted) IF the recurrence is implemented as `min(keep[i-1] * p/q, 1)`, the clipped-product form, THEN the <implementation> shall be rejected, because it is not the prefix acceptance probability. Consecutive ratios of 2 then 0.5 give `keep = 0.5` under it, where the cumulative prefix ratio is 1 and the prefix is acceptable with certainty. Kmic-68 ships exactly the rejected form, so it must not be copied.
- **R07.2b** (event-driven) WHEN the recurrence is chosen, the <plan> shall record which definition of prefix acceptance it follows and validate it against Sun et al. Algorithm 2, because two reviewers proposed different recurrences and R07.4's expected-gain figure is only meaningful under the correct one.
- **R07.3** (event-driven) WHEN a position is rejected, the <acceptor> shall continue evaluating later positions, accepting the longest prefix whose `keep` value covers the drawn variate.
- **R07.4** (event-driven) WHEN `LLAMA_SPEC_BLOCK_VERIFY=2`, the <acceptor> shall every 256 blocks log the expected accepted tokens per drafted token under both rules, `sum_i keep[i]` against `sum_i prod_{j<=i} min(1, p_j/q_j)`. The second term is the same recurrence as R07.2, so the two figures are comparable only because `keep` is non-increasing.
- **R07.5** (event-driven) WHEN a position is rejected, the <acceptor> shall draw the correction from the residual scaled by the surviving prefix's `keep` value.
- **R07.6** (unwanted) IF a grammar, penalty, DRY, mirostat or reasoning budget is active, THEN the <acceptor> shall use the per-token rule from P06.
- **R07.7** (unwanted) IF `LLAMA_SPEC_BLOCK_VERIFY=0`, THEN the <acceptor> shall use the per-token rule from P06.
- **R07.8** (ubiquitous) The <block rule> shall produce the same output distribution as the per-token rule.

## Acceptance

The R07.4 output shows a non-negative gain over a six-request run. Output
distribution parity against P06 over the seeded corpus, as P06 R06.6 requires.
