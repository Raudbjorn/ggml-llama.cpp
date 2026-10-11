# P06 distribution theorem and implementation obligations

The ideal rejection kernel preserves every normalized finite-vocabulary target
distribution under the assumptions below. This is a mathematical statement, not a
universal claim about finite-precision C++, a fixed PRNG seed, model batching, or
failed inference. The implementation audit at
`5acb895208d584c81779262fb5ec478e00d06035` found a concrete minimum-length selection
bug and an exact proposal-law mismatch. That historical audit is retained below.
The appended repair audit and CPU evidence identify subsequent source changes;
they establish only the stated bounded results, not universal runtime equality.

The starting algorithm is Algorithm 1 and Appendix A.1 of
[Leviathan, Kalman and Matias, Fast Inference from Transformers via Speculative
Decoding](https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf).
The proofs, error decomposition and source audit below apply it to this fork's
additional lifecycle and eligibility rules. They do not infer correctness from
the paper's performance results.

## Ideal one-token and sequence laws

Let V be a finite vocabulary, and p and q be nonnegative functions on V with
unit sums. Draw X from q. Independently draw U uniformly on [0, 1). Accept X
if `U * q(X) < p(X)`. Otherwise draw a new Y independently from
`r(y) / rho`, where `r(y) = max(p(y) - q(y), 0)` and `rho = sum_y r(y)`.

For q(x)>0 the accepted joint mass at x is `min(p(x), q(x))`. At q(x)=0
no proposal is drawn; its accepted joint mass is zero. Thus the total rejection
probability is

```
1 - sum_x min(p(x), q(x)) = sum_x max(p(x) - q(x), 0) = rho.
```

If rho>0, the output mass at y is

```
min(p(y), q(y)) + rho * r(y)/rho = p(y).
```

If rho=0 then p=q and every possible proposal is accepted, so the residual
branch is unreachable. This covers singleton, identical and disjoint supports;
division by q is never needed outside its support. No lower bound on positive
probabilities is needed in exact arithmetic.

For a sequence, condition on the committed history h, including any legitimate
adaptive controller state. Each q_h must be the conditional law actually used
to draw the next proposal. The verifier must use p_h for that same history.
Fresh accept/correction/bonus randomness must satisfy the preceding conditional
independence assumptions. The one-token identity then gives p_h at each step.
Applying the chain rule inductively gives
`Pr[Y_1...Y_n=y_1...y_n] = product_i p(y_i | y_<i)` for every finite n.

A proposed block can be evaluated in advance because its target row at i is
used only after all earlier proposals have been accepted. At that point its
prefix equals the committed prefix. After the first rejection, unused suffix
rows and proposals are discarded. If all G proposals are accepted, the bonus
draw uses p at the resulting history. Repeating these variable-length blocks
has the same finite-prefix law as ordinary autoregressive sampling.

This argument permits history-dependent q and draft width, with an important
restriction: the decision to attempt the next proposal must be made before
observing that proposal or an unused future proposal. A confidence rule based
on the next row's top-1 probability is predictable from the existing prefix.
Retroactively discarding earlier proposals because a later confidence check
failed is not covered. Nor may a future proposal select between different
verification kernels for an earlier token.

EOG stopping and a fixed generation limit are deterministic functions of the
output prefix. Applying the same stopping rule to equal sequence laws preserves
equality. Ignored EOG is an ordinary token; suppression by a logit bias is part
of p. Generating and then truncating an unused suffix is harmless to the
observable stopped sequence only if its sampler/cache state is never reused
as though that suffix had been committed.

## Implementation audit and concrete gaps

Source anchors in this section refer to the audited revision, not later shifted
line numbers. [sampling.cpp](../../../common/sampling.cpp),
[speculative.cpp](../../../common/speculative.cpp),
[llama-sampler.cpp](../../../src/llama-sampler.cpp) and
[speculative-cycle.cpp](../../../common/speculative-cycle.cpp) are the relevant
implementations.

| Obligation | Source and conclusion at the audited revision |
| --- | --- |
| Proposal identity and single use | `common_speculative_cycle_recorder::consume`, cycle lines 121-129, requires finalized/selected state, exact proposal ID/token vector and no replay; it consumes the authorization even on mismatch. Begin/cancel/invalidate end reuse. This establishes ownership, not the draw law. |
| Actual proposal token | MTP lines 2423-2466 keeps sampled token distinct from top-1 confidence and attaches q in the same order. Confidence observes the row maximum, not sampled-token probability. |
| Minimum-length selection | MTP lines 2521-2525 clears a valid short sampled prefix after later confidence failure. This conditions the initial proposal law; counterexample below. |
| Actual q | `llama_sampler_dist_apply`, sampler lines 1180-1210, draws from pre-normalization `expf` weights using a double cumulative sum, then stores normalized probabilities as float. MTP records those floats; P06 renormalizes them. Their exact laws need not match. |
| Target law | `common_sampler_sample_and_accept_n_dist`, sampling lines 877-906, obtains actual context rows and runs the common sampler chain. The extracted floats have the same normalization-rounding distinction as q. Batched target rows must separately match the intended sequential-history law. |
| Sampler-state transaction | Sampling lines 816-872 prepares a clone, discards it on failure, accepts the chosen tokens on the clone and swaps once. Clone/copy preserve the chain and rejection RNG. This prevents partial live-state mutation; it does not prevent selection bias. |
| Future-row eligibility | Sampling lines 818-823 validates even unreachable suffix target rows, and lines 806-814 checks all q/backend entries. Proposal-dependent failure can change earlier output law. Explicitly exclude invalid numerical rows from the ideal theorem; do not present fallback as a proof of equivalence. |
| Replay/repair | Both consumers use `common_speculative_repair::finish` for a distribution decision already made. FULL repair retains the selected prefix/correction and sampler state, invalidates q, restores model state and materializes the prefix without drawing again. The legacy exact-match replay instead restores the saved sampler before repeating. |
| EOG | Both consumers compute `common_sampler_eog_prefix` and truncate output. P06 has already accepted its returned suffix on the sampler; correctness of a stopped request relies on ending/resetting the request rather than continuing that hidden suffix state. |
| RNG | Common sampler lines 127-134 and 443 seed rejection `mt19937` with chain seed XOR `0x9e3779b9`. This is deterministic domain separation, not a proof of independent random variables. Fixed seeds yield deterministic streams. |

Consumer anchors: [server-context.cpp](../../../tools/server/server-context.cpp)
lines 4712-4810 and
[speculative-simple.cpp](../../../examples/speculative-simple/speculative-simple.cpp)
lines 291-365. The correction token is not decoded twice: repair decodes the
accepted draft prefix, then retains the correction as the next input. This is a
source argument; the executable consumer fixtures still need to run on any
repaired implementation.

### Built-in transform audit

Eligibility in `common_sampler_dist_supported` (sampling lines 733-758) is a
name allowlist plus configuration guards. Inspection of actual built-in apply,
accept and factory functions gives the following narrower result; arbitrary
custom samplers that spoof these names are not qualified.

| Allowed family | Actual behavior and constraint |
| --- | --- |
| top-k | Candidate sorting/truncation, no history or RNG (`llama_sampler_top_k_apply`, 1449). |
| top-p | Row softmax, sorting and cumulative threshold; reusable sort buffer is scratch, not sequence state (1549). |
| min-p | Current-row maximum/logit threshold, no sequence state (1749). |
| typical | Current-row entropy and sorting, no sequence state (1910). Zero probabilities can make `p*log(p)` NaN; the ideal qualification requires well-defined transformed rows and ordering. |
| top-n-sigma | Current-row mean/variance and mask, no sequence state (3237). Extreme finite logits can overflow its float arithmetic. |
| temp/temp-ext | Current-row scaling, or current-row entropy-derived temperature, no prior-token state (2021/2133). Positive finite settings do not by themselves exclude overflow or underflow. |
| logit-bias | Fixed bias vector; `to_search` is reset scratch on every apply (3902). Biases can remove support and must leave a valid row. |
| `?` variants | Factories return `llama_sampler_init_empty`; its apply is a no-op (450). `?penalties`, `?dry` and `?xtc` are such disabled instances, not active history/random transforms. |
| final dist | Stateless probabilities but stateful categorical RNG (1150). Scratch target draws advance this chain RNG even though P06 discards their selected token. Under ideal independent uniforms this changes no law; same-seed stream identity is not implied. |

Active penalties, DRY, XTC, adaptive-p, infill, mirostat, grammar, reasoning-budget
and backend-selected sampling do not qualify. The guards must remain. Testing
only a name or a null accept function is insufficient for new transforms.

### Valid-probability counterexample: minimum draft length

Use two tokens A/B, target p=(1/2,1/2), initial q=(3/4,1/4), `n_min=2`,
`n_max=2`, and confidence minimum 0.6. After proposing A the next q is
(9/10,1/10); after B it is (1/2,1/2). These distributions can be produced
from finite log probabilities. The first proposal passes confidence. The A
branch retains two tokens; the B branch fails the second confidence check and
the generic minimum-length rule clears its entire first-token prefix.

On A (probability 3/4), rejection sampling emits A with conditional probability
2/3 and otherwise B. On B (probability 1/4), target-only fallback emits each
token with probability 1/2. The first output law is therefore (5/8,3/8), not
(1/2,1/2). The exact oracle reproduces this without model or PRNG assumptions.

Selected repair: decide before any proposal draws that non-adaptive `n_min>1`
is ineligible for P06, retain existing minimum-length behavior, and record the
reason. Add a producer-level finite-logit regression; a test that only calls
the acceptance helper cannot exercise the clearing rule.

### Numerical failure is a separate selection boundary

For p=(3/4,1/4), q=(1/2,1/2), suppose an unused suffix row is invalid only
after proposal B. Whole-block validation selects distribution verification on A
and exact fallback on B. The first-token law becomes (7/8,1/8).

A concrete source-level trigger uses the actual temperature transform: current
logits `(0.25*log(3/4), 0.25*log(1/4))` at temperature 0.25 are valid; a future
raw row `(1e38,0)` passes the finite-logit guard but overflows its first scaled
logit to infinity. Softmax then computes `inf-inf`. This is a numerical-failure
domain, not evidence that ordinary trained-model logits reach it. The precise
C++ route must be tested with production transforms before calling it observed.

Preserving whole-block invalid-row fallback cannot also support an unconditional
distribution theorem for arbitrary proposal-dependent invalidity. Either qualify
only executions whose rows remain valid, or revise the failure contract. A
sequential validator could avoid letting unread future rows change an earlier
decision, but would need a defined error/fallback contract for reached invalid
rows and would change the original atomic whole-block fallback requirement.
No such semantic revision is implemented or silently assumed here.

## Finite arithmetic and conditional error bounds

Define canonical P and Q as exact real normalizations of stored nonnegative
float rows. Let Q_actual be the actual conditional proposal law. For a fixed
history, let `A(x)=min(1,P(x)/Q(x))`; let `A_actual(x)` be the conditional
accept probability. Let R be the exact residual law and bound every actual
correction law conditional on rejection by TV distance e_R from R. Write

```
e_Q = TV(Q_actual, Q)
e_A = sum_x Q(x) * abs(A_actual(x) - A(x))
rho = sum_x max(P(x)-Q(x), 0)
```

Couple proposal draws, acceptance decisions, then residual draws. Markov-kernel
contraction and a union bound give

```
TV(output, P) <= min(1, e_Q + e_A + min(1,rho+e_A)*e_R).
```

The bound allows correction randomness conditional on the rejected proposal;
e_R must bound each such conditional law. Independence cannot be replaced by a
small marginal histogram discrepancy. For a desired reference target T, add
`e_P=TV(P,T)`. For n tokens, if each conditional history has bounds e_i, maximal
coupling gives sequence TV at most `min(1,sum_i e_i)`. Stopping/truncation cannot
increase TV. These are non-vacuous mathematical bounds only when their inputs
are independently bounded; this audit does not assign unknown terms zero.

### Normalization and cumulative sums

For normal binary32 probabilities rounded once, the stored-normalization error
is at most `u32=2^-24` times each exact probability. Allowing subnormals adds
at most `eta32/2=2^-150` per element under round-to-nearest with gradual
underflow. For m entries let `e=u32+m*eta32/2`. Renormalizing the stored values
changes the original exact probability vector by TV at most `e/(1-e)` when
e<1. This follows from the L1 error at most e and the normalization denominator
being at least 1-e. It excludes flush-to-zero, nonstandard rounding, and invalid
arithmetic; those need separate qualifications.

This bound addresses the float normalization store, not `expf`'s accuracy
against mathematical exponentiation. If P is defined from the actual transformed
weights, transcendental rounding is already part of P. If T is an ideal
mathematical softmax, its library error must be bounded separately.

For nonnegative double summation, assuming ordinary rounded operations and no
overflow, `gamma_k = k*u64/(1-k*u64)`, `u64=2^-53`, bounds relative cumulative
sum error for k additions. Division, multiplication and interval-boundary
rounding contribute additional terms. A compiler's reassociation/fast-math
flags and the selected standard library must be inspected before specializing
this bound to actual binaries. The acceptance validator's mass tolerance of
1e-5 is an input check; it is not an output-law error bound.

The original sampler draws before its float normalization store, while P06 uses
the stored row. The oracle demonstrates unequal laws with exact binary32
weights `(1,0.1f,0.01f)`. Recommended bounded repair: in enabled host P06 only,
make the actual proposal draw from the canonical stored q using a private
per-sequence RNG. Treat the existing chain's categorical draw as discarded
scratch. Use a distinct documented seed domain from target acceptance; keep
disabled/chained/backend behavior unchanged. This removes the float-store
proposal mismatch under ideal uniforms, not finite-PRNG error.

### Residual cancellation, zero mass and random grids

The positive-part map is 1-Lipschitz. If exact residual r of mass rho and computed
nonnegative residual r_hat differ by L1 error E, then, for rho>E,
`TV(normalize(r_hat), normalize(r)) <= min(1,E/(rho-E))`. When rho<=E use the
bound 1. This can be large conditionally near p=q, but it is multiplied by the
small rejection probability in the output bound. Account for probability
normalization, subtraction and summation errors in E. Treating all rejected
zero-mass cases as impossible would assume exact arithmetic.

The implementation falls back to a target draw if computed residual mass is
zero. In exact arithmetic p=q makes rejection impossible. In finite arithmetic
the fallback can differ from the ideal residual; its contribution is bounded
by the probability of entering that branch, not automatically zero.

For a uniform discrete b-bit grid, a threshold probability differs from its
continuous counterpart by at most 2^-b; inverse-CDF intervals contribute at
most a constant number of grid/boundary errors per bin. The oracle's four-point
grid deliberately violates exact equality while using independent draws.
The actual `uniform_real_distribution<double>` mapping and rounding are
standard-library-specific. No b-bit-grid assumption is asserted for the current
binary without checking that implementation.

The shared sampler uses `cumulative >= target`; a leading zero-weight item is
selected when U=0. P06's local `dist_draw` skips nonpositive weights and uses a
strict comparison. Sorted top-k draft rows normally have a positive first item;
unsorted target rows can have zero weight first. A dedicated boundary test must
inject U=0 and preserve the disabled-path compatibility decision. The Python
test demonstrates the comparison difference, not a reachable mt19937 seed.

There are only finitely many PRNG states and a fixed seed produces one stream.
Neither seed XOR nor passing the 77-case corpus proves independent uniforms at
every history. Without an explicit stochastic seed/stream model, the only
general numerical sequence-distance bound is the trivial 1. This is why an
unconditional exact C++ distribution claim is not established.

### Model computation is another error term

Batch and sequential kernels can give different logits for the same textual
history. Define and measure `e_model(h)=TV(P_batch(h),P_sequential(h))` using
identical model, cache, backend, driver and sampling transforms. Measured prompts
only bound observed histories. Top-k/top-p boundary changes are discontinuous,
so small logit differences alone do not supply a universal transformed-law
bound. A target row that depends on unused future proposals, rather than only
its causal history, also violates the sequence theorem's premises.

## Initial executable evidence and remaining gates

This section records the initial proof-lane handoff before the repair appendix.

[Exact oracle](../../../scripts/spec_distribution_oracle.py) and
[unittest coverage](../../../scripts/test_spec_distribution_oracle.py) use only
the Python standard library and `fractions.Fraction`. They are discovered by
the repository's existing `scripts/test_*.py` convention. Run:

```bash
python3 -m unittest scripts.test_spec_distribution_oracle -v
```

Observed in the proof worktree: 10 tests passed; 5,778 one-token rational pairs
were checked across vocabulary sizes 2-4 and denominators 1-5. The sequence
checks compare a lazy block enumerator with an independently written sequential
enumerator at horizon four, widths 0-4, history-dependent rows, predictable
confidence stopping, and EOG/no-EOG policies. Other cases cover singleton and
disjoint support, tiny and nearly equal probabilities, normalization mismatch,
finite random grids, zero boundaries, and both selection counterexamples.

The oracle is finite executable support for the written proof, not a formal
refinement of C++ or an exhaustive test of every history. Its block/stream
entry points deliberately bound width to four and horizon to six; production
code has no dependency on the oracle.

The existing 77-case corpus in
[test-sampling-dist.cpp](../../../tests/test-sampling-dist.cpp), `collect` and
`compare` at lines 459-592, prints seeds, pooling maps and p-values but omits
complete observed bin counts. Before its next qualifying run, preserve every
unpooled ON/OFF count, zero-probability bin, expected probability, case identity,
seeds and exact N. Retain the existing deterministic pooling, three hypotheses
per case, 231-test Holm family, alpha and stopping criteria. This permits
independent recomputation; it does not convert statistical acceptance into a
proof. Note that existing pooling also examines pooled observed expectations;
it is the frozen existing method, not an exact finite-sample hypothesis test.

Required follow-up gates: implement and review the pre-draw minimum-length
guard and canonical q draw, run producer and consumer regressions, log the
complete statistical corpus, and measure batched/sequential target differences.
The acceptance proof must name its valid-row/built-in-sampler/randomness
assumptions even after these gates pass.

At that initial handoff, no C++ build, native sampler execution, model run, GPU
operation, service change or commit had been performed by this proof lane. No universal runtime distribution
equivalence, cross-backend equivalence or throughput improvement is claimed.


## Repair audit and CPU evidence, 2026-10-11

The ideal theorem above remains unchanged. Subsequent implementation work closes
specific source defects and adds bounded executable evidence. This appendix does
not turn the historical 5/8 or 7/8 counterexamples into observed native frequency
measurements. At this CPU-only handoff no trained-model GPU numerical comparison or
exact-Qwen4Exp runtime route qualification had been added. Later dense-model
measurements are recorded below.

### Source identity and ownership

The local evidence root is
`/home/svnbjrn/wt/kmic68-proof-evidence` (E below). The repairs start from
`5acb895208d584c81779262fb5ec478e00d06035`; the initial ABI worktree handoff had the
checker-only commit `e0a3e72e85a38f3bb763111434dfe05a550b458d` plus uncommitted
core changes. Final integration references appear in the final update below.
Two preserved patches identify the original sequential handoffs:

| Evidence relative to E | SHA256 | Scope |
| --- | --- | --- |
| `abi-layout/layout.patch` | `54bfd41d921a8de2c2d386ad3e36d57cc1edd312182fb9fa4cdf8ca2a82558cb` | Restore base parameter layout; move request/proposal/distribution state behind opaque ownership; update consumers. |
| `core-repairs/distribution.patch` | `52805cf6f469cf340111bd8fa1bd09a744edc87c04af8a9476cfeb02dd14b9f9` | Applies after the layout patch; pre-draw eligibility, canonical q draw, native regressions and complete statistical counts. |

`abi-layout/layout-source.json` and `core-repairs/source.json` retain their
source-only handoff status. They are not retroactively labeled compiled passes.
The later MTP callback and cycle-log fixture repairs are identified separately.
Historical handoff source hashes, read from the ABI worktree for this appendix:

| Source | SHA256 |
| --- | --- |
| `common/speculative.cpp` | `5c256b04ac47e7f87938dea09c72a0c14ae0f72d02595c152fe3f031986a3e21` |
| `common/speculative.h` | `ff53cf9fee5567124a64f80c0b7f4f2a96f646d6329e066eafb40aa00e84f077` |
| `common/sampling.cpp` | `e815aa94f97ee4bc46bc67bd8a4f85b5aa7967b1b4c6050a9aa5b3945ac8741a` |
| `common/sampling.h` | `46e24f88570e783db99a4f56f75ef19cbd6281511e8c0ccf3796eda6167dccaf` |
| `common/speculative-cycle.cpp` | `370fe8685256edc70a906226e4d6d15e52c4506578b75eced8780cf5e4644318` |
| `src/llama-sampler.cpp` | `b40cfaa2337575f9c68c6a291435b20f2d07bc4705484867946680b50db05660` |
| `tools/server/server-context.cpp` | `5ea465aa3897040a59151dab20d0363949497abd241372e4a6b0bbc0d1a214b6` |
| `examples/speculative-simple/speculative-simple.cpp` | `f2511463f51def8f4e492c6950281e2f9e5503760bd05d920b77230039bf9c73` |
| `tests/test-qwen4exp-mtp.cpp` | `129cb4dc5271df56c639b8528736a58d88b881ac237757926fe97a675a2c51a3` |
| `tests/test-qwen4exp-mtp-driver.h` | `42feaa4cbd2b39bc2f24320a868cfca8392cb26f3bb9ba52a5cc339eeb51ec6c` |
| `tests/test-sampling-dist.cpp` | `dd08bfc6378ef5c21bc254802385d8e383da1bd0a5512a42dd9d57515b379427` |
| `tests/test-speculative-cycle.cpp` | `dc1081918ad83406b63de6899f7f0d9807f8e378b8714689ab1d1b63f819ad09` |

These hashes describe the sampling audit surface, not whole-tree equivalence to
any binary. Build reports and source snapshots describe the binaries actually
executed; later unrelated ABI export repairs cannot inherit those binary identities.

### Repaired implementation argument

| Obligation | Repaired source argument | Qualification boundary |
| --- | --- | --- |
| Stable proposal ownership after layout restoration | Sequence-owned state behind `common_speculative` holds request identity, proposal identity and distributions. Consumers use `common_speculative_set_request_id` and `common_speculative_get_proposal_id`; the legacy `result_q` remains an output copy. Recorder finalization and single-use consumption retain their existing semantics. | This is a source/lifecycle argument; full C++ ABI compatibility is a separate report and gate. |
| Predictable minimum-length eligibility | MTP construction disables host distribution proposals for nonadaptive `n_min>1`, before sampler initialization or any proposal draw. The recorded reason is `minimum_length_selection`; chained-mode reason retains precedence. Adaptive mode remains eligible under its separate controller behavior. | Legacy minimum-length clearing remains. The native fixture tests the guard and confidence behavior, not an empirical 5/8 biased law. |
| Recorded q generates the actual proposal | The built-in draft sampler chain produces scratch candidates. After top-1 confidence passes, the producer validates and copies q, then calls `common_sampler_dist_draw` with that same float row. The helper sums its mass in double, skips zero weights, and uses strict positive-interval selection. The acceptor normalizes the same recorded row. | Under exact arithmetic and ideal uniforms this removes the previous float-store proposal-law mismatch. Finite summation, endpoints and PRNG errors remain in the stated bound. |
| Private proposal RNG lifecycle | `seed_seq(actual chain seed, 0x50303651, sequence ID)` initializes the private sequence generator. A changed seed or request begin reinitializes it; successive proposals continue its stream. The target rejection RNG has its distinct existing XOR domain. Repair/replay preserves the completed decision. | Distinct deterministic seed domains are not a proof of conditional independence. |
| Sampling and confidence stay distinct | The producer retains top-1 probability for confidence while the emitted token comes from q. Probability rows and finalized tokens stay aligned in recorder ownership. | Native producer checks include reconstructed private-stream draws; they do not quantify every possible model history. |
| Atomic fallback and commit | The acceptor still validates whole-block rows on a sampler clone and swaps live state only after a successful decision. Invalid rows select the retained exact fallback. | Transactional correctness does not remove proposal-dependent fallback selection bias. Valid future rows remain an explicit theorem assumption. |

Disabled, grammar/stateful, chained and backend-selected configurations retain
their existing guards and fallback contracts. The allowlist qualification still
refers to the inspected built-in implementations, not arbitrary custom samplers
using matching names.

### Observed CPU evidence and its limits

The verifier's
[CPU report](/home/svnbjrn/wt/kmic68-proof-evidence/candidate-cpu-build/report.md)
records a passing production CPU build and focused tests. Its SHA256 at this
append is `00d3c38fdebdc0a3cd0b806d4c9caed7004749105731f29cebf035d75540e9f8`.
The library build used GCC 16, optimized `-O3 -DNDEBUG`, DWARF 4 and
`-fno-eliminate-unused-debug-types`, shared libraries, native tuning disabled,
and downloads/UI disabled. A test-only `-UNDEBUG` overlay keeps fixture assertions
active; it does not remove release semantics from production libraries.

| Evidence relative to E | Observed result | What it establishes |
| --- | --- | --- |
| `candidate-cpu-build/mtp-retry-default.log`, `mtp-retry-q8.log` | Both print `PASS P06 finite-logit producer confidence/minimum selection fixture` and production MTP sampling/lifecycle success. | Actual synthetic model graphs and the production producer exercise the pre-draw minimum guard, confidence stopping, canonical proposal draws, adaptive/chained fallbacks, ownership and replay. |
| `candidate-cpu-build/mtp-retry-release.log` | Prints `PASS P06 release finite-logit overflow and whole-block invalid-suffix fallback`. | The actual temperature transform can invalidate the finite extreme suffix row; fallback matches the untouched transaction. It does not measure a native 7/8 output law. |
| `candidate-cpu-build/sampling.log`, `observed-counts.json` | 77 cases and 231 Holm hypotheses pass at family alpha 0.01. | The unchanged fixed corpus provides bounded statistical evidence and complete count data. It cannot establish exact equality. |
| `consumer-cpu/consumers.log` | `PASS: 19 synthetic executable consumer cases; real-model P06 campaign remains separate`. | Both executable consumers exercise synthetic EOG, ignored EOG, repair and cancellation paths. Instrumented consumer binaries are separate from ABI/performance binaries. |
| `candidate-cpu-build/cycle-retry-python.log` | All eight compiled-recorder Python tests pass without a skip. | Recorded identities, cancellations and profile reconciliation survive the fixture's explicit asynchronous-log flush. |
| `distribution/report.json`, `distribution/oracle-tests.log` | Initial exact-rational oracle exits zero with ten tests and 5,778 one-token pairs. | Independent finite support for the exact written theorem and counterexamples; no native or GPU inference is implied. |

Each statistical arm emits 100,000 tokens grouped into 25,000 four-token
observations. Every ON/OFF unpooled count vector sums to 25,000, not 100,000;
the larger number counts emitted tokens. `observed-counts.json` contains 77
unique case identities, both complete count vectors, and expected probabilities,
including zero-probability bins. Its SHA256 is
`5dd9796fcc9f496086e3d8f86c9bb28feb430972bf45d41a58b95093136762ea`.
The raw `sampling.log` retains the fixed arm seeds, pooling maps, p-values and
Holm thresholds. No seed search, case omission or statistical threshold change
is part of this repair.

The initial failures remain evidence. `mtp-default.log` failed because the
confidence callback asserted that a catch-up graph had one requested output.
The fixture now arms injection only before drafting; the one-row, two-attempt
and canonical-q assertions remain. This finite-logit fixture supplies (.75,.25)
then (.5,.5) independently of the sampled first token. It is a production-route
regression for the repair, not a native realization of the history-dependent
5/8 counterexample. That biased-law calculation remains exact-oracle evidence.

Likewise, the invalid-suffix native fixture checks overflow, fallback reason and
unchanged transactional state. It does not implement the proposal-dependent
branching needed to measure the 7/8 counterexample. The test is release-only:
the existing shared categorical sampler's `assert(found)` can abort on invalid
softmax values in assertion-enabled libraries. That numerical-domain limitation
was not repaired or hidden by changing assertions.

`python-compiled.log` retains the missing-start diagnostic failure. Cycle JSON
rows flush synchronously, while start/profile messages use the asynchronous
common logger; the fixture now flushes that logger after recorder destruction.
The later passing run closes that bounded fixture gate without relaxing identity
validation. No production recorder lifecycle change was made for this retry.

### Numerical probe readiness, not trained-model qualification

The native [probe](../../../tests/test-speculative-logits.cpp) passed its CPU
comparison/context self-test, recorded in
[numerical-probe-build/self-test.log](/home/svnbjrn/wt/kmic68-proof-evidence/numerical-probe-build/self-test.log).
The tested source SHA256 is
`69984186e8398767c184f5e0c2244f845ef4eab03bfabe70d6ab1115daa615c9`.
That self-test initializes no model or backend. No full-vocabulary trained-model
row measurements are established by it.

The probe requires explicit qualified target-context metadata. Qwen4Exp uses
seven recurrent rollback states, output caps of eight, and unmasked target
hidden-state export. Dense models retain output caps of one and compare
sequential rows with checkpoint-restored prefixes of lengths one through eight,
requesting only the final output. That axis is labeled separately from an
eight-output speculative verification block. Full host snapshots support the
repair comparison; snapshots above 2 GiB fail explicitly rather than silently
switching to partial state. Actual recurrent-state count and context dimensions
are checked; output caps, GDN replay and hidden masking are applied explicitly
where no public getter exists, and initialization logs remain required.

Only target weights and one target context are live at a time. This avoids
duplicating large model weights but leaves more free memory than a server with
an MTP head. A real run must therefore use the server's resolved target placement
and explicit cache budget/mode, not infer equivalence from `fit off` alone.
Raw full-vocabulary float rows, exact histories and maps snapshots still need
external artifact hashes and binary/model/project-library/system-library/driver
provenance. A completed probe report contains measurements, not an invented
universal tolerance or a substitute for exact-Qwen4Exp route validation.

### Remaining distribution qualification

1. Finite PRNG conditional independence remains unproved. No fixed-seed histogram
   or separate seed domain supplies that assumption for all histories.
2. Whole-block invalid-suffix selection remains outside the ideal theorem. The
   retained fallback is transactional but can select an earlier kernel based on
   a future proposal. Synthetic fallback success does not remove this exclusion.
3. P06's canonical target P normalizes stored float probabilities. Ordinary
   target categorical sampling draws from pre-store `expf` weights. Repairing q
   does not make those target laws exactly equal: the float-store difference,
   double arithmetic, residual cancellation and random-grid terms remain.
4. Current exact-Qwen4Exp route evidence and its trained numerical differences
   remain incomplete. Later dense-model observations below cover only their
   recorded histories/protocols. Prior campaign evidence cannot be promoted to
   the repaired implementation.
5. Final source/binary identities, independent review and post-integration CPU
   gates are now recorded below; they do not supply the missing GPU qualification.

The established result is the ideal theorem plus the identified source repairs
and bounded CPU evidence above. The implementation argument is partial, and
unknown numerical/randomness terms must not be assigned zero. No universal
runtime distribution equivalence or workload speedup is claimed.

## Final integration qualification update

The ABI layout repair is integrated as `eef488983`, the proposal-law/minimum
selection repair as `1a89077e5`, and the callback/log fixture corrections as
`a7830e9a8`. These are independently revertible changes. The combined source
review found no additional cross-change defect; runtime proof limitations above
remain explicit.

A later independent review found that the initial probe omitted the frozen
workload's EOG and model-suppression masks. That initial self-test remains a
historical result for its recorded source. The corrected probe reuses production
logit-bias before temperature and categorical normalization, records exact EOG
and model-suppression IDs, and matches the missing-EOS guard. It retains raw
logits independently and rejects fully masked rows. Its three added self-tests
cover support masking/renormalization, empty support and amplification of TV by
conditioning. Source review cleared SHA256
`72ff3741f394305d3c1577ff3994c7aab7019a6403a9aa22d08aa57bb1149c0d`.
The transform is fixed to this workload; common CLI sampling flags do not
configure the probability transform. Final compiled self-test evidence is
separately indexed and must not be inferred from that source review.

| Obligation | Disposition |
| --- | --- |
| DIST-01 arbitrary eligible p/q one-token theorem | Established under normalized finite support, correct history, ideal independent randomness and exact arithmetic. |
| DIST-02 finite sequences, bonus, predictable stopping and EOG | Established under the same assumptions and a common observable stopping rule. |
| DIST-03 actual proposal law and pre-draw eligibility | Identified recorded-q and minimum-selection defects repaired; source review and bounded native tests support those changes. |
| DIST-04 proposal ownership, transaction, replay and consumers | Reviewed source and synthetic CPU evidence; trained-model and every production rollback route not established. |
| DIST-05 exact rational oracle and 77-case family | Exact bounded reference checks and the unchanged statistical criterion pass; they are not universal runtime proof. |
| DIST-06 arithmetic and randomness error | Conservative conditional bound derived; unmeasured PRNG/numerical terms are not assigned zero. |
| DIST-07 same-history trained rows and current exact-model routes | Dense observations and cache-disabled exact-Qwen4Exp route checks are recorded below; frozen-profile numerics and cross-build routes remain incomplete. |
| DIST-08 universal finite-precision implementation equivalence | Incomplete; future-invalid-suffix selection, finite randomness and target-law rounding distinctions remain. |

The universal ideal theorem and the qualified implementation claim are separate
results. No synthetic pass erases missing trained-model numerical evidence.

## Final compiled evidence

Commit `56b69b831` integrates the exact oracle, corrected numerical probe and
CTest registration. Fresh [integration CPU verification](/home/svnbjrn/wt/kmic68-proof-evidence/integration-cpu/report.md)
passes MTP default/q8/release fixtures, all 77 statistical cases and 231 Holm
hypotheses, native cycle/adaptive/client checks, registered oracle/probe tests,
and 77 Python tests with no skips. Complete observed vectors are in
[integration observed counts](/home/svnbjrn/wt/kmic68-proof-evidence/integration-cpu/observed-counts.json).
The unchanged family criterion remains alpha 0.01; these are bounded observations.

A separate fresh [consumer build](/home/svnbjrn/wt/kmic68-proof-evidence/integration-consumer-cpu/report.md)
passes all 19 executable cases. These binaries contain synthetic hooks and do
not serve as ABI or performance artifacts. The corrected probe also passes its
model-free self-test with the oneAPI build; tracing records no GPU-device opens.
Final client compiler/debug-container guards are test-only changes, with their
own fresh client runs. Exact [source correspondence](/home/svnbjrn/wt/kmic68-proof-evidence/final-source-correspondence.json)
separates these refreshes from the earlier binary identities.

A later free-GPU window permitted dense-model numerical observations below.
The pristine offline server independently cannot link, so there is no pristine
cross-build comparison. The theorem is established under its written assumptions;
the unconditional finite-precision implementation claim remains incomplete.

## Trained dense-model observations

The [candidate numerical report](/home/svnbjrn/wt/kmic68-proof-evidence/gpu-numerical-qualification/report.md)
preserves 216 full-vocabulary comparisons on the A770/xe: 72 per model across
six initial category/length cells. All observed rows form valid distributions.
The models use the actually observed all-GPU placement, q8_0 KV, FA, context 8192,
batch/ubatch 512, one sequence and the matching masks/temperature transform.
Every valid GPU window completed with no new fault lines and released its owned
processes. Earlier failed orchestration/collector attempts remain separate.

| Candidate model | Sequential versus restored-prefix last-output comparisons | Maximum observed TV | Fresh versus full-snapshot rollback/replay comparisons | Maximum observed TV |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 1.7B | 48 | 0.07220091059675748 | 24 | 0 |
| Llama-3.1 8B | 48 | 0.07141787954159234 | 24 | 0 |
| Mistral 7B | 48 | 0.01745232314254596 | 24 | 0 |

All 72 rollback/replay pairs also have bit-identical raw logits. This is bounded
full-host-snapshot evidence; production partial/on-device restoration routes
are not thereby qualified. Dense contexts reserve one output: the first axis
restores a shared prefix, processes prefixes of lengths one through eight, and
requests only their final logits. It includes snapshot restoration and separate
context creation, so it cannot isolate a batching kernel as the cause.

Independent review checked the largest Qwen3 discrepancy against raw rows and
source histories: both predict after the same 514 explicit input IDs. It
recomputed maximum logit difference 0.9753928184509277 and masked double-softmax
TV 0.07220099242230724, corroborating the production-float result above while
using different arithmetic. These observations disprove exact numerical identity
for the tested protocols; they do not establish P06 sequence bias, a regression
against pristine binaries, a universal error bound, or any performance gain.
The continuation IDs are fixed fixture tokens, not sampled model output.

The eighteen HTTP context/count qualification responses omitted returned token
IDs and cannot pass the full stochastic benchmark validator. A separate canonical
Mistral request subsequently returned 512 explicit IDs and passed the repaired
validator with independently obtained EOG masks and source-qualified reasoning
settings. Neither those diagnostic response times nor these native probes are
speedup measurements. Exact-Qwen4Exp/MTP qualification remains separately gated.

## Exact-model terminal route evidence

The exact Qwen4Exp target/head reproduced all 113 tensor override matches and
43 allocation/context/cache observations with fitting disabled. The frozen
persistent-cache-enabled profile then crashed with SIGSEGV on its first request,
in the host SYCL image-sort path during BF16 conversion. This profile did not
qualify. The original failure, loaded-library identities and runtime diagnosis
are preserved in the [exact-model evidence](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/fitoff-disabled/report.md).

A separately named `SYCL_CACHE_PERSISTENT=0` diagnostic kept identical requests,
project library bytes and placement. Both responses report 512 prompt tokens,
512 output tokens and limit stopping. The [diagnostic ledger](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/cache-off-diagnostic-summary.json)
records the current repaired paths:

| Arm | Selected cycles | Distribution attempts | Exact-match attempts | Drafted | Accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| P06 disabled | 85 | 0 | 85 | 591 | 426 |
| P06 enabled | 66 | 66 | 0 | 458 | 445 |

The enabled arm records 458 normalized proposal distributions. Proposal identity,
cycle reconciliation, sampled-token versus top-1 and support checks pass. Every
sampled proposal in these particular diagnostics equals its top-1 token; the
runs add no trained non-top-1 example. Existing synthetic cases cover that
distinction. The
disabled wrapper's mask-serialization failure remains recorded; separate offline
adjudication uses independently logged target EOG IDs and source-qualified
reasoning settings. Responses omit explicit output-token arrays, so this is
scoped route evidence, not a full stochastic speed-harness pass. Both owned
servers shut down cleanly, released the GPU and produced no new matched faults.

This establishes bounded current-binary route execution in the cache-disabled
diagnostic profile. It does not amend the frozen cache-enabled profile, prove
stale cache contents caused the crash, or establish trained-output distribution
equality. The six-cell Qwen4Exp numerical phase remains unexecuted behind the
failed frozen runtime gate. Dense-model numerical observations above are unchanged.
