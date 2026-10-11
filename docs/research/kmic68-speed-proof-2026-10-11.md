# Kmic68 frozen workload speed claim

Result: **incomplete; no speedup has been measured or established**. The pristine
comparison server cannot link with the required downloads-disabled build. This
is a build prerequisite failure, not a failed measured speedup hypothesis.
The single permitted optimization round has therefore not been triggered.

## Fixed comparison and decision rule

The approved comparison is against pristine
`2f623c6607c078d1069dac23e9d8af9cb146e771`. Arm A uses that revision's existing
MTP behavior; arm B uses the repaired candidate with P06 disabled; arm C enables
P06 on exact Qwen4Exp and reuses B for the other models. B/A tests default
behavior, C/A tests the configured workload, and C/B attributes the P06 effect.
An enabled-only gain cannot establish a default gain.

The [preregistration](/home/svnbjrn/wt/kmic68-proof-evidence/speed/preregistered-protocol.json)
remains SHA256
`3d533e803eb871dcb435538e0822e7ec82e07db2288c0926bf602509919426b3`.
It freezes the initial and held-out prompts, seeds, sampling settings and analysis:

- Local Llama-3.1-8B, Mistral-7B, Qwen3-1.7B and exact-Qwen4Exp target/head.
- Code editing, multi-turn code and prose, each at exactly 512 and 4096 input
  tokens, followed by 512 generated tokens with EOG ignored.
- One request at a time, q8_0 KV, flash attention, temperature 0.8, top-p 1,
  no penalties/grammar, fixed context and placement; Qwen4Exp MTP width seven.
- Ten independent paired server launches per model, balanced arm order and an
  untimed warmup per cell. No build overlaps timing.
- Complete-request latency is primary. Equal-weight geometric ratios aggregate
  models/categories/lengths; decode throughput, prefill, TTFT and memory remain
  separate diagnostics.
- Twenty thousand paired-launch bootstrap resamples preserve within-launch
  correlations. Simultaneous bounds cover aggregate claims and cell regression
  guards; family-wise alpha is split across initial and held-out rounds.
- A claim requires a lower speed-ratio bound above 1.02 and every cell to exclude
  a latency regression above 3 percent. Preserve all valid measurements.

## Implemented preparation

Commit `b512876f5` extends the existing benchmark infrastructure with frozen
stochastic comparison and qualification. Deterministic identity checks remain.
The stochastic mode requires matching effective sampling, route evidence,
fixed token counts and distribution qualification rather than identical tokens.
It checks frozen prompt IDs, runtime environment, model/binary identities and
all mapped regular ELF dependencies against the qualified inventory. Snapshot
checks cannot observe a dependency loaded and unloaded entirely between them.

The harness and its statistical analysis were independently reviewed and their
mocked/unit checks passed. This is harness evidence, not server or hardware
performance. The runtime manifest is deliberately incomplete until real model,
placement, tokenizer and route qualification exists. See the retained
[integration brief](/home/svnbjrn/wt/kmic68-proof-evidence/speed/qualification-integration-brief.md)
and [review closure](/home/svnbjrn/wt/kmic68-proof-evidence/review-closure.md).

Real candidate responses exposed two additional validation mismatches: effective
EOG biases are serialized as null masks (including equivalent duplicates), and
reasoning budget/control are not serialized at all. The repaired analyzer checks
the exact model-qualified suppression set. Missing reasoning fields require
explicit source/route-qualified values in a hash-bound sampling report; they do
not default to disabled. Present contradictions and unexpected/finite biases
still fail. The frozen request, distributions and statistical thresholds remain
unchanged. Earlier numerical-qualification responses without returned token IDs
are not accepted as full stochastic samples.

All six local model files, including both target shards and the separate MTP
head, have fresh full-file hashes and stable pre/post file identities in
[model-full-hashes.json](/home/svnbjrn/wt/kmic68-proof-evidence/model-full-hashes.json).
Exact `qwen4exp` architecture metadata qualifies the target/head identity;
Ornith `qwen35` models are not substituted. Hashing does not establish that every
required context/placement/workload cell can execute.

## Blocking prerequisite and retained failures

The pristine CPU server's final link fails with undefined references to
`common_download_resolve_path`, called by `tools/server/server-models.cpp`.
The base declares this function in `common/download.h` and implements it only in
the download-enabled translation unit; its disabled-download stub omits it.
The [baseline report](/home/svnbjrn/wt/kmic68-proof-evidence/baseline-build/report.md)
and [raw link log](/home/svnbjrn/wt/kmic68-proof-evidence/baseline-build/cpu-build.log)
preserve the actual failure. Pristine libraries and test binaries are separate
successful artifacts; they do not supply a working arm-A server.

Candidate commit `ab4c1b5d4` supplies the missing empty-cache resolver and its
server links successfully. The pristine source remains unchanged. Applying the
repair to arm A, enabling downloads or comparing against a historical server
would change the approved comparison and has not been done.

Unrelated QSV media encoders occupied the A770 at earlier readiness checks.
After they exited, candidate-only dense numerical qualification ran in clean,
locked windows; no media process was terminated and no service was changed.
The [numerical report](/home/svnbjrn/wt/kmic68-proof-evidence/gpu-numerical-qualification/report.md)
contains three models' initial candidate fixtures, context/count observations
and 216 full-vocabulary comparisons. Tokenizer parity remains false because
arm A is unavailable. One canonical Mistral request passes the repaired response
validator with 512 returned IDs. These are neither paired performance samples
nor completion of the four-model qualification. Earlier GPU evidence retains
its own binary identities.

## Requirement disposition

| Obligation | Status and evidence |
| --- | --- |
| SPEED-01 frozen protocol and held-out prompts | Established as preparation; original registration hash unchanged. |
| SPEED-02 stochastic harness and failure guards | Established within reviewed unit/mock scope; one canonical real Mistral response also passes; complete paired execution unrun. |
| SPEED-03 local model identity | Full hashes retained; exact target/head metadata checked. Feasibility incomplete. |
| SPEED-04 pristine comparison server | Incomplete: disabled-download resolver link failure. |
| SPEED-05 tokenizer parity, 48 token arrays, placement and route qualification | Incomplete: candidate initial dense fixtures exist, but no arm-A parity, held-out arrays or exact-Qwen4Exp qualification. No cell removed. |
| SPEED-06 ABI/distribution prerequisites | Incomplete; see the separate claim reports. |
| SPEED-07 ten paired launch blocks and raw samples | Unrun; zero qualifying paired performance samples in this follow-up. |
| SPEED-08 aggregate gain and regression guards | Incomplete; no data, confidence intervals or positive speed claim. |
| SPEED-09 one measured-overhead optimization round | Not triggered; missing baseline is not an initial measured failure. |
| SPEED-10 service restoration and evidence preservation | No services changed by this follow-up; failed builds and all preparation retained. |

The frozen suite has not been reduced, thresholds have not been relaxed, and
mock data has not been represented as performance. Resolving the pristine
baseline contract and runtime qualification is required before measurement.

## Exact Qwen4Exp runtime gate

The [current exact-model run](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/fitoff-disabled/report.md)
reproduces all 113 target override names and 43 allocation/context/cache
observations with fitting disabled. Target GPU-layer setting 49 and head 50,
37 CPU override patterns, q8_0 target KV and f16 head KV are explicit. The large
PLE table is mapped lazily; mapped extents are not resident memory measurements.

The first 512-input/512-output request with P06 disabled crashes the server with
SIGSEGV. The captured host stack reaches SYCL persistent-cache image sorting
during BF16 conversion. There is no successful response or qualified route
from that run. The card was released and no new matched GPU fault line appeared;
neither observation converts the crash into a pass.

The frozen runtime explicitly requires `SYCL_CACHE_PERSISTENT=1`. Separately
recorded diagnostics use the installed runtime's supported explicit 0 control
to bypass that path. They do not qualify or amend the frozen speed profile,
even if requests complete. The [runtime diagnosis](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/cache-diagnosis/report.md)
retains exact loaded-runtime identities and the limits of that inference. No
shared cache, service, installed package, or source kernel was changed.

Consequently the prepared six-cell exact-model numerical phase remains
[unexecuted behind its runtime gate](/home/svnbjrn/wt/kmic68-proof-evidence/gpu-numerical-qualification/qwen4exp-final/report.md).
This is a second independent speed-suite blocker beside the missing pristine
server; it is not a measured speed regression or a reason to drop this model.

Both cache-disabled diagnostics subsequently report 512/512 limit completions:
85 exact-match cycles with P06 disabled and 66 distribution attempts with it
enabled. Their same-request/library identity and qualified response checks are
recorded in the [diagnostic summary](/home/svnbjrn/wt/kmic68-proof-evidence/qwen4exp-current-qualification/cache-off-diagnostic-summary.json).
The disabled wrapper's original mask mismatch is retained with separate offline
adjudication. These are route diagnostics with no explicit returned-token arrays;
neither their timings nor acceptance ratios establish the frozen speed claim.
