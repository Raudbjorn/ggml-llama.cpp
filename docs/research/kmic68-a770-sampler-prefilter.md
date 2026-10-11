# Kmic68 P05 sampler prefilter: rejected by the seeded oracle gate

Evidence collected on 2026-10-11. **P05 is rejected by its real-model token
identity gate.** All 36 requests completed, but all 18 paired OFF/ON requests
returned different native token IDs. The candidate is excluded from the
integration branch, including its unset-means-enabled behavior and telemetry.
The failure does not by itself identify the cause or establish distribution bias.

The [P05 plan](../plans/kmic68-05-topk-prefilter.md) requires both seeded token
identity and a lower median host sampling cost. Unit and backend passes cannot
override its real-model gate. There is no accepted P05 change to combine with
P06/P07; their eventual acceptance must use the existing sampler implementation.

## Frozen candidate and preservation

Candidate worktree: `/home/svnbjrn/wt/kmic68-speculative`, branch
`codex/kmic68-speculative`, base `f6a09bfaa2a7d8d369f9c8bd8eab78ba79ee4b6c`.
The 17 modified/new source files, tracked patch, per-file SHA256 hashes
and base revision are preserved under
`/home/svnbjrn/wt/kmic68-evidence/p05-rejected-source/`.
The source archive SHA256 is
`8346a97e56cc28a5d381ea23d0089fc0d9080a13072b0f319d82b82ed4d21f42`.
Archiving left candidate source and Git status unchanged.

The AOT server was `build-aot-p05/bin/llama-server`, SHA256
`3a17685eb6b15c996d312438ac7b5a51a89cfeee1f21c0ff3e929f0c2a816b12`.
Its build uses `acm-g10`, F16 and Release `-O3 -DNDEBUG -UNDEBUG` globally.
It therefore does not establish performance with production assertions disabled.
Both arms used this same frozen executable and libraries on the A770/xe host.

## Completed campaign and gate

The raw evidence root is
`/home/svnbjrn/wt/kmic68-evidence/p05-acceptance-cooldown/`.
It preserves the exact commands, process environments/maps, model-shard hashes,
resolved placement, runtime/driver identities, responses, metric deltas, boundary
checks, source/build manifests, and before/after fault logs.

The trained target is the two-shard Qwen3.8-Flash-Next-GSQ-RCO IQ1_M under
`/mnt/ssd1/gguf/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ1_M/IQ1_M/`; the canonical
Q8_0 MTP head is under the corresponding `/mnt/ssd2/gguf/.../mtp/` directory.
Their validated architecture is `qwen4exp`, as recorded in the
[P04 evidence](kmic68-speculative-cycle-evidence.md). No qwen35 substitute was used.

Each arm completed the same six seeded 256-token requests, repeated in three
paired launches, with host MTP sampling, draft top-k 10, widths 1 through 7,
context 16384, q8_0 KV and the same automatic fit settings. All 18 request
payloads and effective sampling settings matched their paired arm. Every ON
request recorded positive optimized calls; every OFF request recorded zero.
The ON total was 8,274 optimized calls. All metric generations and monotonic
counter checks passed.

`campaign.json` records `accepted=false` with error
`paired native token IDs differ`. `paired-audit.json` records all 18 failures;
for example, pair 1/request 1 first differs at zero-based token index 89.
Both responses in every pair contain 256 tokens. No request was discarded,
retried, or replaced in this completed campaign.

| Repetition | OFF host us/cycle | ON host us/cycle | Seeded token identity |
| --- | ---: | ---: | --- |
| 1 | 1539.91 | 2072.57 | All six pairs differ |
| 2 | 1592.32 | 2070.40 | All six pairs differ |
| 3 | 1569.76 | 2046.05 | All six pairs differ |

These are descriptive ratios of each launch's summed sampler microseconds to
summed cycles, preserved in `raw-launch-totals.json`. The raw medians are
1569.76 OFF and 2070.40 ON. Different token trajectories prevent interpreting
this comparison as a controlled performance effect; no throughput, speedup or
causal regression claim follows. The token identity failure alone rejects P05.

## Operational and earlier correctness evidence

Every launch passed its tenancy, process/build, service, readable-kernel-log and
fault checks. The reviewed boundary protocol waited at most 300 seconds, polling
every five seconds until all 1/5/15-minute loads were below six. All final
boundaries passed within that bound. The campaign and outer guard retained their
3600/3700-second bounds. Cooldown cannot prove absence of foreign CPU contention
during requests. No concurrent build, new GPU fault or source/build change was
observed. Services remained coordinator-owned.

The immediately preceding unchanged-binary after-floor supplied the before-floor.
After this campaign, both default and q8 MTP regression commands and the 33-test
Python benchmark suite passed. Release checks found no owned llama process or
render-node holder and an available GPU lock. All 625 prior evidence files
checked by the campaign remained unchanged. Earlier failed/partial campaigns
remain preserved and are not pooled with this result.

The separate `p05-aot-verification/report.md` records 12 passing command groups,
including guarded prefilter modes, sampling, queue metrics, adaptive, both MTP
modes and a CPU/SYCL oracle. Its requirement mapping covers R05.1-R05.11 static
and dynamic guards, all k=1..128 tie cases, positive/negative call counters,
clone/copy, backend-selected tokens, grammar-first and constrained fallback.
R05.12 telemetry also ran in the real campaign. The source kept `grammar_first`
protection, passing false only for verified private grammar-free draft samplers.
Those results remain scoped correctness evidence; they do not satisfy the failed
seeded real-model acceptance condition.

## Not claimed

No P05 integration, production default change, controlled performance result,
root-cause attribution, general distribution bias, or P05/P06/P07 combined
coverage. Candidate source and all raw results remain available for a separately
scoped investigation. This outcome introduces no runtime dependency or service.
