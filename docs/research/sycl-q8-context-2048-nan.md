# SYCL Q8 KV: context-2048 perplexity NaNs

Status: open, reproduced before and after PR 51 integration. Root cause unknown.
This note tracks the defect because repository issues are disabled.

On the Arc A770 with oneAPI 2026.0, Llama-3.1-8B-Instruct Q4_K_M and
Wikitext-2 raw test data, Q8_0 K/V with flash attention produces NaN perplexity
at context 2048. The isolated pre-integration build (tree `fdfa74483`) also
produced `[1]-nan,[2]-nan,` at context/batch/ubatch 2048. The integrated build
reproduced it with both 512-token and 2048-token batches. These observations
establish that this particular failure predates the integration; they do not
identify the failing kernel or prove all context-2048 configurations fail.

Reproduce with the affected SYCL build and local model/corpus paths:

```bash
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_CACHE_PERSISTENT=1 GGML_SYCL_DISABLE_GRAPHS=1
timeout 180 /home/svnbjrn/build-pr51-sycl/bin/llama-perplexity \
  -m /mnt/mrgr/models/llama31-8b-q4km/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -f /home/svnbjrn/.cache/pr51-integration/wikitext-2-raw/wiki.test.raw \
  -t 4 -tb 4 -ngl 99 -fa on -fit off --device SYCL0 \
  -c 2048 -b 2048 -ub 2048 --chunks 2 -ctk q8_0 -ctv q8_0
```

The context/batch/ubatch-512 eight-chunk Q8 control is finite: PPL 7.9861 in
both builds. Saved logits from the NaN run are invalid KLD reference data.
The synthetic oracle passes but does not cover this real-model failure.

Evidence retained under `/home/svnbjrn/.cache/pr51-integration/`:
`baseline-sycl-probe.sh`, `ppl-baseline-sycl-q8-2048.log`,
`ppl-q8-2048.log`, and `ppl-q8-2048-singlebatch.log`.

Follow-up: localize the first non-finite tensor, compare FA against non-FA and
Q8 against F16 at the same context, and add a reproducer for the responsible
operation. Close only after the failing real-model case is finite and its
quality agrees with a finite control. No fix or throughput result is claimed.
