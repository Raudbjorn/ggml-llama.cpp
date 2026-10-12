#!/usr/bin/env bash
# Runs test-moe-pack on 8-expert qwen3moe and qwen35moe fixtures.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    printf 'usage: %s TEST_LLAMA_ARCHS TEST_MOE_PACK\n' "$0" >&2
    exit 2
fi

archs_bin=$1
pack_bin=$2

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-moe-pack.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

"$archs_bin" -o "$work_dir" -a '^(qwen3moe|qwen35moe)$' -s 42 --n-expert 8 --n-expert-used 2 > "$work_dir/archs.log" 2>&1

for arch in qwen3moe qwen35moe; do
    case_dir="$work_dir/$arch"
    mkdir -p "$case_dir"
    printf '== %s\n' "$arch"
    # the synthetic profiles name qwen3moe; other arches get their arch line rewritten
    if [ "$arch" != qwen3moe ]; then
        export LLAMA_MOE_PACK_TEST_ARCH="$arch"
    fi
    "$pack_bin" "$work_dir/$arch-moe.gguf" "$case_dir"
done
