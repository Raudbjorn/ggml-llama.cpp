# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

# add_tool_subdir tolerates pruned trees: the generated lib branch
# (scripts/prune-to-lib.sh) keeps only a subset of the tool directories.
function(add_tool_subdir dir)
    # IS_DIRECTORY, not EXISTS: a stray file with a tool's name must not be
    # handed to add_subdirectory. Paths are quoted so a source tree containing
    # spaces does not split into multiple arguments.
    if (IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
        add_subdirectory("${dir}")
    endif()
endfunction()

if (EMSCRIPTEN)
else()
    add_tool_subdir(batched-bench)
    add_tool_subdir(gguf-split)
    add_tool_subdir(imatrix)
    add_tool_subdir(llama-bench)
    add_tool_subdir(completion)
    add_tool_subdir(perplexity)
    add_tool_subdir(quantize)
    if (LLAMA_BUILD_SERVER)
        add_tool_subdir(ui)
        add_tool_subdir(cli)
        add_tool_subdir(server)
    endif()
    add_tool_subdir(tokenize)
    add_tool_subdir(parser)
    add_tool_subdir(tts)
    add_tool_subdir(mtmd)
    if (NOT GGML_BACKEND_DL AND GGML_CPU)
        # these tools use backends directly (no dynamic loading) and depend on CPU backend symbols
        add_tool_subdir(cvector-generator)
        add_tool_subdir(export-lora)
    endif()
    add_tool_subdir(fit-params)
    add_tool_subdir(results)
endif()
