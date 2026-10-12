#!/usr/bin/env bash
# Runs test-schema-decision's token-level oracle checks on a dense (llama) and a hybrid
# (qwen35moe, recurrent layers: padded branches) test-llama-archs fixture.
set -euo pipefail

if [ "$#" -ne 2 ]; then
    printf 'usage: %s TEST_LLAMA_ARCHS TEST_SCHEMA_DECISION\n' "$0" >&2
    exit 2
fi

archs_bin=$1
test_bin=$2

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/llama-schema-decision.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

"$archs_bin" -o "$work_dir" -a '^(llama|qwen35moe)$' -s 42 > "$work_dir/archs.log" 2>&1

shopt -s nullglob
models=("$work_dir"/llama-dense.gguf "$work_dir"/qwen35moe-*.gguf)
if [ "${#models[@]}" -lt 2 ]; then
    printf 'fixtures missing in %s:\n' "$work_dir" >&2
    ls "$work_dir" >&2
    exit 1
fi
"$test_bin" --model "${models[@]}"
