#!/usr/bin/env bash
# Runs test-moe-pack on 8-expert qwen3moe, qwen35moe, smallthinker and grok fixtures. With LLAMA_BENCH,
# also checks that llama-bench reports the pack that was built, not the one requested.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    printf 'usage: %s TEST_LLAMA_ARCHS TEST_MOE_PACK [LLAMA_BENCH]\n' "$0" >&2
    exit 2
fi

archs_bin=$1
pack_bin=$2
bench_bin=${3:-}

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-moe-pack.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

# smallthinker gates its experts with ReLU and grok with GELU: a pack must follow the layer's activation
"$archs_bin" -o "$work_dir" -a '^(qwen3moe|qwen35moe|smallthinker|grok)$' -s 42 --n-expert 8 --n-expert-used 2 \
    > "$work_dir/archs.log" 2>&1

# without --n-expert the expert count follows the arch: deepseek2 fixtures have 64, the others 2
mkdir -p "$work_dir/default-count"
"$archs_bin" -o "$work_dir/default-count" -a '^deepseek2$' -s 42 --n-expert-used 3 > "$work_dir/default-count.log" 2>&1 || {
    tail -n 3 "$work_dir/default-count.log" >&2
    printf 'FAIL: --n-expert-used 3 must be accepted for the 64-expert deepseek2 fixture\n' >&2
    exit 1
}
if "$archs_bin" -o "$work_dir/default-count" -a '^qwen3moe$' -s 42 --n-expert-used 3 > "$work_dir/default-count.log" 2>&1; then
    printf 'FAIL: --n-expert-used 3 must be refused for the 2-expert qwen3moe fixture\n' >&2
    exit 1
fi
grep -q "exceeds the 2 experts of the qwen3moe fixture" "$work_dir/default-count.log" || {
    printf 'FAIL: missing diagnostic for --n-expert-used above the fixture default\n' >&2
    exit 1
}
printf 'ok: --n-expert-used checked against the fixture default\n'

for arch in qwen3moe qwen35moe smallthinker grok; do
    case_dir="$work_dir/$arch"
    mkdir -p "$case_dir"
    printf '== %s\n' "$arch"
    # the synthetic profiles name qwen3moe; other arches get their arch line rewritten
    if [ "$arch" != qwen3moe ]; then
        export LLAMA_MOE_PACK_TEST_ARCH="$arch"
    fi
    "$pack_bin" "$work_dir/$arch-moe.gguf" "$case_dir"

    if [ -z "$bench_bin" ] || [ ! -x "$bench_bin" ]; then
        printf 'skip: llama-bench pack field (no llama-bench binary)\n'
        continue
    fi
    # test-moe-pack left the fixture without expert scales and its profiles in case_dir;
    # arch.csv names another arch, so that arm loads without a pack. CPU only: the packs
    # come from LLAMA_MOE_PACK_TEST_CPU, and OpenVINO fails qwen35moe at -ngl 0 even without a pack
    out=$(LLAMA_MOE_PACK_TEST_CPU=1 "$bench_bin" -m "$case_dir/f32.gguf" -dev none -ngl 0 -p 0 -n 1 -r 1 -t 2 -o jsonl \
        --moe-pack "off,$case_dir/mixed.csv:2,$case_dir/arch.csv:2" 2> "$case_dir/bench.err") || {
        cat "$case_dir/bench.err" >&2
        printf 'FAIL: llama-bench --moe-pack failed\n' >&2
        exit 1
    }
    got=$(printf '%s\n' "$out" | grep -o '"moe_pack": *"[^"]*"' | sed 's/^"moe_pack": *"//; s/"$//' | tr '\n' ' ')
    want="off $case_dir/mixed.csv:2 off "
    if [ "$got" != "$want" ]; then
        printf 'FAIL: llama-bench moe_pack fields [%s], expected [%s]\n' "$got" "$want" >&2
        exit 1
    fi
    if ! grep -q "moe-pack $case_dir/arch.csv:2 built no pack" "$case_dir/bench.err"; then
        printf 'FAIL: llama-bench did not warn about the arm without a pack\n' >&2
        exit 1
    fi
    printf 'ok: llama-bench reports the pack that was built\n'
done
