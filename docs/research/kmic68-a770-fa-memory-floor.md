# Kmic68 A770 FA memory-path proxy: readiness rejected

Report schema version: **1**. Evidence inspected on 2026-10-11.

**P03 is not accepted. `p11_eligible=false`.** Generated retained-path equivalence
is unproven and no proxy timing was collected. Target reachability is
**undetermined by proxy timing**. This is a readiness report, not a measured
memory floor or a performance conclusion.

The [corrected P03 plan](../plans/kmic68-03-memory-only-floor.md) applies to
same-type turbo3/turbo4 VEC decode, D128, one query, Hq32/Hkv8, one sequence,
depths 4096 and 16384. The proxy build option remains OFF by default. Its output
is non-correctness data. P11 remains blocked on P03 measurement readiness; no
P11 kernel changes were made. P12 is not gated by this experiment.

## Integration outcome

P03 measurement readiness is **rejected by R03.8**, because the generated
retained-path fingerprint has not been established. The incomplete decoder and
runtime launcher remain implementation gaps; this is not a finding that the
compiler cannot support the experiment or that the kernel paths differ.
Only this outcome report is integrated. The candidate proxy code, build option,
harness and fixtures remain outside the integration branch in
`/home/svnbjrn/wt/kmic68-sycl-attention`, branch
`codex/kmic68-sycl-attention`. Candidate source is preserved in
`/home/svnbjrn/wt/kmic68-evidence/p03-rejected-source/candidate-files.tar.gz`
with its base, per-file hashes, archive hash and tracked patch beside it.
All commands below refer to that candidate worktree, not the integration checkout.

P11 is **blocked before implementation** by its documented P03 prerequisite:
there is no valid paired proxy report with `p11_eligible=true`. R11.1-R11.7 and
the complete correctness/performance matrix were not exercised. No P11 tile-fold
kernel change is integrated. This outcome records the required P11 decision
before any later P12 VEC integration, without making P03 a P12 dependency.

## Available evidence

The source revision is `1a13eef9a38a8f0b232da13086ccc8b01cc0f425` plus the
uncommitted P03 implementation. Local raw evidence is retained under
`/home/svnbjrn/wt/kmic68-evidence/`; these paths describe this campaign and are
not portable repository fixtures.

- `p03-benchmark-preflight.json` hashes the actual full/floor executables,
  expected SYCL libraries, caches, and prepared probe. It records ELF sections,
  RUNPATHs, lease flags, and 24 **unexecuted** launch descriptions. The
  `loaded_library` and `memory_floor_observed` fields are null: a disk hash is
  not proof of the library used by a running process.
- Full build: `build-sycl`. Proxy snapshot:
  `build-sycl/p03-floor-snapshot`. Both caches have an empty
  `GGML_SYCL_DEVICE_ARCH`, i.e. these are JIT artifacts. The full library's ELF
  contains `__CLANG_OFFLOAD_BUNDLE__sycl-spir64`; this is not an interpreted
  generated EU instruction stream.
- `build-sycl/p03-probe-prepared/probe.json` has `prepared_only=true` and empty
  artifact lists for all four full/floor tq3/tq4 translation units. No generated
  descriptor, address, control, barrier, or output-store proof can be recovered
  from those command records.
- `ocloc`, `iga64`, `llvm-objdump`, and `readelf` are present. Package inspection
  reports IGC `1:2.41.10-1`, compute-runtime package
  `22.43.24558.r13063.gbe85a8d685-1`, and Level Zero loader
  `1.34.0.r2.gae3db48-1`. Tool presence does not prove that their output supports
  the required symbolic interpretation. No offline compilation was attempted.
- The measurement lease read during this work has `offline_compile=false` and
  `gpu_timing=false`. The coordinator owns service management and resource
  grants. No build, GPU launch, service change, commit, or download occurred
  during this benchmarker task.

The previous worker's `p03-verification/report.md` records normal OFF JIT
correctness passes: default and Turbo FA oracles have zero gate failures and
zero XPASS; the Turbo run has six expected turbo2 XFAILs. The filtered
FLASH_ATTN_EXT suite passed 282/282 cases. Its driver was xe and its saved
before/after fault checks found no new faults. Those existing logs were read,
not rerun here. They establish neither proxy validity nor AOT correctness nor
performance. The 282 cases are a filtered suite, not all FLASH_ATTN_EXT cases.

## Subsequent offline device-code probe

After the measurement AOT build released the compiler lease, the coordinator
ran the actual four-TU collector successfully:

```bash
source /opt/intel/oneapi/setvars.sh
timeout --kill-after=15s 1200 scripts/perf/probe-fa-memory-floor.sh \
  build-sycl build-sycl/p03-floor-snapshot \
  /home/svnbjrn/wt/kmic68-evidence/p03-device-probe
```

`p03-device-probe/probe.json` now contains real LLVM/SPIR-V, IGC dumps, and
acm-g10 Zebin artifacts for full/floor turbo3/turbo4. This supersedes the
prepared-only inventory above; the earlier benchmarker inspection remains a
record of its earlier inputs. No GPU timing was run for this probe.

An additional extraction examined the final turbo3 Zebin sections for the
exact D128/ncols1/no-softcap specialization, with workgroup sizes 128 and 256.
`ocloc disasm` returned success but did not write the long-named FA kernel
files. The coordinator therefore used `llvm-objcopy --dump-section` with the
exact ELF section names and short destination names, then `iga64 -d -p=12p71`.
The extraction manifest is `p03-device-probe/final-section-extraction.json`.
IGA warned about trailing padding in two sections; extracted raw sections are
retained.

The extraction command omitted a separate output ELF. `llvm-objcopy` therefore
rewrote both turbo3 parent containers, each 51,336 bytes smaller than the
collector version. Their original container bytes are no longer retained at
those paths. `p03-device-probe/container-transformation.json` records the exact
commands, original collector hashes/sizes, and current hashes/sizes; the
original `probe.json` records are unchanged. Extraction entries now identify
their current parent hash explicitly. Independent read-only ELF inspection
confirmed that all four extracted payloads match their exact named sections
in the current parents. This does not prove container identity to the original
compiler output. The four LLVM/SPIR-V inputs and both turbo4 parent containers
still match their collector hashes. This additional provenance gap also bars
using these artifacts as a qualified measurement manifest.

| Turbo3 workgroup | Arm | dc0 messages | ugm messages | slm messages | gateway messages |
| --- | --- | ---: | ---: | ---: | ---: |
| 128 | Full | 229 | 390 | 92 | 4 |
| 128 | Proxy | 153 | 376 | 92 | 4 |
| 256 | Full | 229 | 390 | 156 | 4 |
| 256 | Proxy | 154 | 376 | 156 | 4 |

These are static mnemonic inventories from final device sections, recorded in
`p03-device-probe/final-raw-message-inventory.json`. They are not dynamic byte
counts or a normalized retained-path manifest. In particular, they do not yet
separate K/V accesses from scratch spills and other loads. They demonstrate
that source similarity cannot establish identical generated memory traffic;
the differing messages still require semantic classification. Equal SLM and
gateway counts do not prove matching address, predicate, or control paths.

The offline outputs are not bound to the already-built JIT executable images.
Complete included-header provenance, a semantic decoder, exact image binding,
and the paired AOT timing campaign remain missing. R03.8 therefore continues
to reject measurement readiness, and `p11_eligible` remains false. The result
does not establish that installed tooling is incapable of completing the proof.

## Manifest and parser boundary

The candidate `scripts/perf/bench-sycl-fa-floor.py` deliberately
has **no runtime launcher** and **no accepted actual-artifact interpreter**.
`--check-manifest` always returns 42 with `unverifiable-retained-path`, including
when a document claims verification or has matching hashes/counts. The normal
preflight audits available files and writes a schema-v1 blocked JSON report.
It cannot be unlocked by granting a lease alone.

`--check-fixture` is a separate, explicitly synthetic comparison contract.
Its schema-v1 `kind` is `comparison-fixture-only`. Each arm carries ordered
records, exact category counts, a recomputed normalized SHA-256 fingerprint,
and a floating arithmetic count. The records contain operation, memory space,
surface, cache policy, width, vector length, predicate, address expression,
operands, and incidental instruction address. Only typed register and label
names and top-level instruction addresses are normalized; aliases, offsets,
strides, predicates, loop operands, instruction order, and store destinations
remain significant. Unknown descriptors and malformed/version-mismatched JSON
are rejected. Duplicate JSON keys and non-finite JSON constants are refused.

The fixtures are hand-authored comparison tests, **not compiler output**. They
prove the comparator's rejection behavior, not generated equivalence. They do
not claim the real kernel has the fixture's message count or memory layout.
Actual compiler-artifact interpretation remains unimplemented. In particular,
this work does not claim installed tooling is incapable of the task; obtaining
and qualifying its output requires a compiler lease and further implementation.

| Fixture check | Expected result |
| --- | --- |
| Intact pair in fixture mode | Exit 0; measurement readiness false |
| Same fixture in actual-manifest mode | Exit 42; `unverifiable-retained-path` |
| Missing required category | Exit 42; `missing-required-category` |
| Same counts, changed width/surface/stride/predicate/loop | Exit 42; `retained-path-mismatch` |
| Changed cache/vector/order/register alias | Exit 42; `retained-path-mismatch` |
| Unknown descriptor | Exit 42; `unverifiable-retained-path` |
| Register/label/incidental-PC renaming only | Fixture comparison passes |

Run the host checks from `/home/svnbjrn/wt/kmic68-sycl-attention`:

```bash
python3 -m pytest scripts/test_bench_sycl_fa_floor.py scripts/perf/test_probe_fa_memory_floor.py -q
python3 scripts/perf/bench-sycl-fa-floor.py --check-fixture scripts/perf/fixtures/fa-floor-valid.json
python3 scripts/perf/bench-sycl-fa-floor.py --check-fixture scripts/perf/fixtures/fa-floor-missing-category.json
```

Observed host result: **18 passed, 16 subtests passed**. ASCII and local-link
checks passed; all 768 pre-existing source/probe files checked against the
verification inventory and all eight floor snapshot binaries retained their
hashes. Test and preservation evidence are `p03-benchmarker-tests.log` and
`p03-benchmarker-preservation.json` in the local evidence directory.

The last command must return 42 and print exactly `missing-required-category`.
The existing `probe-fa-memory-floor.sh` remains an artifact collector; it has
not been modified to implement the plan's illustrative `--check-manifest`
wrapper syntax. Use the Python preflight entry point for that refusal check.

The actual artifact preflight, requiring neither compiler nor GPU access, is:

```bash
python3 scripts/perf/bench-sycl-fa-floor.py \
  --full build-sycl/bin/test-sycl-turbo-correctness \
  --floor build-sycl/p03-floor-snapshot/bin/test-sycl-turbo-correctness \
  --probe build-sycl/p03-probe-prepared/probe.json \
  --output /home/svnbjrn/wt/kmic68-evidence/p03-benchmark-preflight.json
```

It returned 42 with `unverifiable-retained-path`. Its output is JSON. The
preflight records planned library-path overrides but does not load either arm.
The floor snapshot's executable RUNPATH points to the normal `build-sycl/bin`:
a future launcher must prepend the selected arm's bin directory and independently
verify the actually loaded library hash and backend `memory_floor` boolean.
An `ldd` prediction or executable basename alone is insufficient.

## Measurement-readiness rule and outcome

| Type | Depth | Full samples | Proxy samples | Proxy gap | proxy_valid |
| --- | ---: | --- | --- | --- | --- |
| turbo3 | 4096 | Not measured | Not measured | Not computed | false |
| turbo3 | 16384 | Not measured | Not measured | Not computed | false |
| turbo4 | 4096 | Not measured | Not measured | Not computed | false |
| turbo4 | 16384 | Not measured | Not measured | Not computed | false |

There are no medians, confidence intervals, retained or discarded timing
samples, or same-campaign driver/kernel measurements to report. The readiness
rule requires exactly three finite positive samples per arm in all four cells,
a passing actual-artifact manifest, VEC route, matching build/driver identities,
verified runtime identity, and clean fault gates. Only then is the descriptive
median gap `100 * (full_time_us - proxy_time_us) / full_time_us` computed.
Unit fixtures confirm that small, zero, and negative gaps remain eligible; the
gap never acts as a readiness threshold. These unit values are synthetic, not
measurements. P11's separate oracle and 5% performance requirements are unchanged.

## Precise continuation requirements

1. Obtain the coordinator's offline compiler lease and matched acm-g10 AOT
   full/floor artifacts. Preserve exact build commands, tool identities, all
   included source/header hashes from the build-time dependency closure, and
   device-image extraction provenance. Existing changed-source hashes and TU
   hashes do not establish that complete closure.
2. Implement and qualify an actual generated-artifact decoder. Bind each exact
   D128/ncols1/turbo3 or turbo4 symbol and launch geometry to the device image
   embedded in the executable/backend being measured. Offline recompilation
   that is merely similar to the timed image is insufficient.
3. Prove retained memory descriptors (operation, surface/space/cache/width/
   vector/predicate), addressed bytes as functions of lane/query/KV indices,
   address/index dependencies, loop bounds and control edges, barriers and
   output stores including order. Unknown instructions/descriptors or unproved
   symbolic relationships reject with 42. Require fewer floating arithmetic
   instructions in the proxy as the negative control. Count equality alone is
   never evidence of equivalence.
4. Only after that gate passes, implement and test the runtime portion of the
   orchestrator. Validate actual loaded library identity and mode per process,
   one JSON row with `correctness_valid=false`, the exact shape/route, positive
   finite time, three warmups/100 iterations, and graph-compute-and-synchronize
   timing including split-K combine. Bind raw output, environment and commands
   to both build identities. Keep the P03 three-repetition protocol; this is
   not a general performance promotion campaign.
5. With a separate GPU timing lease, require sole tenancy, inactive competing
   services, low host load/no builds, and readable `sudo -n dmesg` before and
   after each process. Hold `/tmp/a770.lock` across the gates and bounded run;
   use GNU timeout, capture `fuser`/`pgrep`, and match both xe/i915 and fault
   terms in either order. A clean log cannot excuse a timeout. The coordinator
   alone restores services. Recheck source/binary/library hashes after runs.
6. Complete the four paired cells and emit schema-v1 eligibility from validity,
   without a gap threshold. Send any retained-path mismatch to the SYCL owner;
   do not weaken the gate to enable P11.

## Not claimed

Host tests and file/hash inspection do not prove real generated-path equality,
a source/include-to-device-image binding, actual runtime library selection,
proxy performance, a lower bound, a throughput ceiling, or target reachability.
The initial benchmarker lease prevented compilation; the coordinator's later
offline probe above obtained real artifacts. The missing decoder and runtime
launcher are additional implementation work, not just permissions.
No dependencies were added. Candidate state includes the Python preflight/tests, synthetic
fixtures, and local preflight/progress evidence. Integration adds only this
report and its index entry. Backend/test source
and the preserved floor snapshots were left intact.
