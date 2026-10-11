# Kmic68 C++ ABI restoration and qualification

Result: the identified draft-parameter layout and GCC vector-destructor export
breaks are repaired. **Exhaustive backward C++ ABI compatibility remains
incomplete.** Matching header layouts, retained exports and bounded old-client
execution are supporting evidence, not a complete substitute for every required
actual-library type association and delta disposition.

The compatibility base is pristine
`2f623c6607c078d1069dac23e9d8af9cb146e771`; the follow-up starts at
`5acb895208d584c81779262fb5ec478e00d06035`. The contract covers the base's
header-declared interfaces and reachable types on separately qualified x86-64
Linux GCC/CPU and oneAPI/SYCL profiles. Cross-compiler interchangeability,
other platforms and unpublished intermediate campaign ABIs are excluded.

## Repairs and evidence boundaries

Commit `eef488983` restores `common_speculative_draft_params` through its
original final `seed` field. Request identity, proposal identity and recorded
probabilities live behind the existing opaque `common_speculative` object.
Additive accessors update both consumers. The legacy parameter reference and
`result_q` output remain available. Public `llama_batch`, context parameters and
backend vtables receive no appended fields in this follow-up.

The retained GDB evidence shows the original record is 48 bytes, alignment eight,
with offsets 0, 4, 8, 12, 16, 24, 32, 40 and 44 for drafting, n_max, pos0,
id_last, prompt, result, result_q, temp and seed. The restored candidate matches.
See [layout evidence](/home/svnbjrn/wt/kmic68-proof-evidence/abi-layout/final-layout-results.json).
This concrete layout comparison does not stand for all types in the SDK.

An unsuppressed export comparison found a second real issue: the GCC candidate
had lost the D1/D2 destructor exports for the installed
`std::vector<common_ngram_map>` specialization. Explicit instantiation restores
those providers. The old-client fixture contains undefined references that
cannot disappear through inline destruction and verifies the actual provider.
The oneAPI baseline exports D2 only; its fixture requires only D2. Hardcoded
aliases are restricted to the qualified Linux/x86-64/libstdc++ compiler families.
The ordinary-vector alias probe also excludes `_GLIBCXX_DEBUG`.
Ordinary client checks remain available outside that alias probe.

The base-built GCC client runs unchanged with replacement libraries and verified
runtime mappings. It exercises caller-owned records, vector allocation/stride,
copy/move/destruction, two speculative sequences, draft/accept/reset, null/error
paths and sampler callbacks. The original alias-binding result is retained in
[abi-destructor-repair](/home/svnbjrn/wt/kmic68-proof-evidence/abi-destructor-repair/report.md).
Fresh final fixtures compiled against pristine headers/libraries pass unchanged
against replacement libraries for both profiles, with verified providers and
runtime maps. See the [final GCC client report](/home/svnbjrn/wt/kmic68-proof-evidence/final-client-cpu/report.md)
and [final oneAPI client report](/home/svnbjrn/wt/kmic68-proof-evidence/final-client-sycl/report.md).
OneAPI traces show no `/dev/dri` opens in these host-only checks. These clients
do not establish GPU operation or exhaustive interface coverage.

## Compiler and debug profiles

Both revisions use shared optimized unstripped libraries, no LTO, native CPU
optimization disabled, downloads disabled, and both UI asset options disabled.
Compiler launchers are empty and ccache is off. Test assertions are retained
through a test-only overlay; production Release/NDEBUG code generation remains.

| Profile | Debug qualification |
| --- | --- |
| GCC 16 CPU | O3, NDEBUG, DWARF4, fno-eliminate-unused-debug-types; actual ELF debug sections and representative GDB layouts retained. |
| oneAPI SYCL | O3, NDEBUG, standalone host DWARF4; `-Xarch_device -g0` disables device debug information only. Exact compiler frontend traces distinguish host and device settings. |

The first full-device-debug SYCL attempt was terminated by earlyoom during
`sycl-post-link`, with approximately 32 GiB RSS. Its log is preserved. The new
host-debug profile successfully built the pristine libraries with no source
change, no OOM-policy change and no device execution. Host C++ ABI evidence does
not require or claim device-debug coverage. See
[baseline SYCL report](/home/svnbjrn/wt/kmic68-proof-evidence/baseline-sycl-hostdwarf/report.md).

The normalized matched SYCL compile-command inventory has 455 common translation
units and zero differences after source/build-path normalization. Hashes,
dependencies, exports, flags and actual child exits are retained for each build.
The pristine offline server executable has a separate missing-resolver link
failure; its successful library builds must not be described as a server pass.

## Inventory and finite proof boundary

The configured installation supplies 57 headers. The inventory adds the explicit
innerq and staging surfaces, then follows exported-type references and relevant
CPU headers. Nine compiler-profile-specific witnesses cover 139 distinct source
headers. They use the actual target standards and ABI macros: ggml C++17,
llama/common GCC-default C++20, and the C witness's C standard. Artificial
all-header include collisions and the superseded C++17 aggregate are retained
as failed/inadequate attempts, not compatibility failures.

The [finite canonical ledger](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/canonical/v2/finite-canonical-ledger.json)
records these bounded results:

- All 6,209 project-owned, per-group direct DWARF type-node shapes match. Member
  and base locations, callback conventions and vtable metadata are included.
- All 48,756 baseline compiler class-label/size/alignment tuples remain in the
  matched candidate groups. Repeated and anonymous labels are not a unique
  exported-type count or a complete anonymous-type identity proof.
- The distinct nested-vector forward-only witness gap has complete 192-bit
  definitions in both actual common/server corpora. That recovers definition
  availability; it does not by itself prove recursive actual-library equality.
- 2,326 named actual-library type associations are established in both arms.
  The exact residual ledger lists 3,394 association candidates: 2,879 unmatched
  canonical names, 41 incomplete-definition aliases and 474 anonymous rows.
  These are association gaps, not 3,394 observed ABI breaks.

The [coverage report](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/report.md)
and [residual names](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/canonical/v2/finite-unresolved-association-ledger.json)
state the remaining distinctions between inline/template instantiations,
disabled backend declarations, alternate compiler spelling, opaque boundaries
and genuinely exposed types. None was removed by blanket mangled-name filtering.
An empty successful `abidw` result on an ET_REL witness was explicitly rejected
as coverage evidence; direct DWARF inspection supplied the bounded comparisons.

The original GCC inventory includes 13,704 library/export pairs across seven
libraries. After restoring the vector destructors, removed weak constructors
remain for two implementation-only classes: the MTP implementation in
`common/speculative.cpp` and `server_slot` in `server-context.cpp`. Their absence
from base headers places those constructor exports outside the selected
header contract; their removal is still explicitly recorded. Opaque pointee
layout changes and standard-library alias-representative changes require their
own dispositions, rather than a global suppression. Raw `abidiff` exit 12 is
retained and is not relabeled an acceptance result.

The ten reported `libggml-base` unreachable alias deltas are individually
resolved as comparison representative-identity mismatches: every exact old
specialization remains in the candidate with an equal full recursive type graph.
The [per-alias evidence](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/canonical/v2/ten-alias-dispositions.json)
retains type IDs, corpus hashes and source excerpts. No ABI attributes or type
references were suppressed. This closes those ten deltas, not the residual
header-to-library association obligations.

One exact alternate-DWARF disposition recovers `common_download_remove`: both
objects contain its precise linkage-name signature, but no address-bearing DIE
association; both linked functions have the same three instruction bytes. The
[evidence](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/download-remove-qualified-signature.json)
is specific to that symbol and cannot explain unrelated missing associations.

## Checker and acceptance gate

Commit `384e1dc7a` adds explicit unsuppressed C++ mode and reviewed-manifest
coverage to the existing ABI checker. Generation requires complete debug data,
exact manifest/library keys and fresh generation markers. Tool errors and unknown
exit bits fail closed. The checker does not invent header completeness from
whatever types happen to survive a dump. Its tests include a deliberate layout
break and stripped-debug rejection, alongside the equal-layout positive case.

The pre-existing installed SDK also omits the httplib/subprocess dependency
headers and fails to propagate `LLAMA_SUBPROCESS` in the installed imported
common target. Source-tree qualification uses the recorded matching include and
macro profile. It does not establish default installed-SDK usability.

| Obligation | Disposition |
| --- | --- |
| ABI-01 pristine baseline and matching optimized profiles | Established for recorded library builds; pristine offline server separately fails. |
| ABI-02 restore base draft record and consumers | Established by source review, layout observations and bounded CPU tests. |
| ABI-03 retain contracted dynamic providers | Identified vector exports repaired; all raw removals retained with scope dispositions. Exhaustive typed association still incomplete. |
| ABI-04 complete reachable type/layout/signature coverage | Incomplete; finite association and actual-library delta ledgers remain. |
| ABI-05 unchanged base-built clients with verified maps | Established for bounded final GCC and oneAPI clients, with exact bytes and runtime mappings; host-only. |
| ABI-06 checker detects a deliberate break and tool errors | Established within fixture scope. |
| ABI-07 exhaustive delta adjudication and independent acceptance | Source review complete; full ABI claim remains incomplete. |

No SONAME bump is used to hide a break. No claim is made that the known repairs
alone establish drop-in compatibility for every contracted client. The finite
residual ledger is the remaining proof boundary.

## Final finite delta audit

The [actual-library ledger](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/canonical/v2/all-library-deltas/report.md)
adjudicates 168 of 169 changed-type rows across seven CPU libraries: 157 retain
exact recursive graphs, three retain both real const-qualified variants, and
eight retain old declarations with only enumerated additive nested declarations.
One common-library alias reaches three declaration-only baseline types whose
sizes cannot be recovered from that corpus. Candidate definitions alone cannot
supply the missing baseline evidence. A subsequent
[direct object search](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/remaining-common-object-types/report.md)
inventoried 105 common objects and confirms that baseline occurrences are
declaration-only; the candidate sampling object supplies the complete definitions.

All eleven changed-function findings have individual dispositions. Direct
[input-object evidence](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/vector-signature/report.md)
resolves `vector<int>::_M_realloc_append<int const&>`: both defining objects in
each arm retain the explicit const-int-reference argument and identical 201-byte
function sections. The XML omission is a tool interpretation/association limit;
its exact parser mechanism is not claimed. The other ten findings have explicit
opaque/source-boundary dispositions.

Independent header/export witnesses establish 1,306 exact observed symbol pairs
and a conservative closure reaching 272 of the 3,394 association candidates.
The other 3,122 are not thereby proven irrelevant: missing unused declaration
DIEs make this an observed lower bound, not an exhaustive header-contract root
inventory. Neither candidate count is an ABI failure count.

The eight-library SYCL raw comparison has its own
[command/exit ledger](/home/svnbjrn/wt/kmic68-proof-evidence/abi-sycl-comparison/results.json).
The [terminal report](/home/svnbjrn/wt/kmic68-proof-evidence/abi-sycl-comparison/report.md)
records fourteen successful dumps across seven library pairs. Both
`libggml-sycl` dumps reached the 900-second timeout (exit 124), retaining partial
XML without comparison. Six completed raw diffs return 12; `libggml` returns 0.
These oneAPI deltas require independent adjudication; GCC dispositions do not
cover them. All sixteen input library hashes remain unchanged. Partial dumps,
timeouts and raw incompatibility bits cannot pass the compatibility gate.

Compiled C++ source is anchored at `502ba7eae`; the subsequent
`c1de964f7` change touches only the Python response-validation harness. The final CPU fixture rebuild refreshed
generated commit identifiers in ggml-base/common; its exact before/after ELF
identities are preserved. Existing raw ABI corpora remain bound to their original
ELFs. All production SYCL ELF bytes remained unchanged during final fixture
refreshes. See the [follow-up ledger](kmic68-proof-follow-up-2026-10-11.md) for
source correspondence and final evidence identities.

The final [both-profile export ledger](/home/svnbjrn/wt/kmic68-proof-evidence/final-dynamic-export-ledger.json)
includes weak/template symbols without filtering. In oneAPI, the old float
`uniform_real_distribution< float >::operator()<mt19937>` provider disappears.
The exact-symbol [disposition](/home/svnbjrn/wt/kmic68-proof-evidence/abi-header-coverage/removed-uniform-report.md)
traces its only definition to sampling.cpp and both callers to the existing
rejection helper; the RNG types are absent from all 261 scanned base project
headers and remain behind cpp-defined opaque sampler state. Vendored httplib
includes `<random>`, so generic standard declarations are transitively available;
it does not declare or use this exact specialization. It is an incidental
implementation provider outside the selected header-declared contract. Its
removal is recorded, not repaired or hidden. Compatibility with arbitrary
clients binding every incidental ELF export is not established.
