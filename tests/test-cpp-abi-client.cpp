#include "common.h"
#include "speculative.h"
#include "ngram-map.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(__linux__) && defined(__x86_64__) && defined(__GLIBCXX__) && \
    !defined(_GLIBCXX_DEBUG) && \
    ((!defined(__clang__) && defined(__GNUC__)) || defined(__INTEL_LLVM_COMPILER))
#include <dlfcn.h>

// The qualified libstdc++ baselines export D2; only the GCC baseline also exports D1.
// Explicit external references prevent an inlined client destructor from hiding a lost export.
#if !defined(__INTEL_LLVM_COMPILER)
extern "C" void base_ngram_vector_d1(void *) asm("_ZNSt6vectorI16common_ngram_mapSaIS0_EED1Ev");
#endif
extern "C" void base_ngram_vector_d2(void *) asm("_ZNSt6vectorI16common_ngram_mapSaIS0_EED2Ev");
#endif

static void require(bool ok, const char * message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static void exported_ngram_vector_destructors() {
#if defined(__linux__) && defined(__x86_64__) && defined(__GLIBCXX__) && \
    !defined(_GLIBCXX_DEBUG) && \
    ((!defined(__clang__) && defined(__GNUC__)) || defined(__INTEL_LLVM_COMPILER))
    const struct { const char * symbol; void (*destroy)(void *); } destructors[] = {
#if !defined(__INTEL_LLVM_COMPILER)
        {"_ZNSt6vectorI16common_ngram_mapSaIS0_EED1Ev", base_ngram_vector_d1},
#endif
        {"_ZNSt6vectorI16common_ngram_mapSaIS0_EED2Ev", base_ngram_vector_d2},
    };
    for (const auto & item : destructors) {
        void * address = dlsym(RTLD_DEFAULT, item.symbol);
        Dl_info provider{};
        require(address && dladdr(address, &provider) && provider.dli_fname, "resolve old vector destructor provider");
        require(std::string(provider.dli_fname).find("libllama-common.so") != std::string::npos,
                "old vector destructor must be supplied by shared llama-common");
        using map_vector = std::vector<common_ngram_map>;
        alignas(map_vector) unsigned char storage[sizeof(map_vector)];
        auto * maps = new (storage) map_vector;
        maps->emplace_back(2, 2, false, 1);
        maps->front().keys.resize(2);
        maps->front().key_map[12] = 34;
        require(maps->size() == 1 && maps->front().keys.size() == 2 && maps->front().key_map[12] == 34,
                "caller constructs populated installed-type container");
        item.destroy(maps);
        std::printf("BOUND %s address=%p provider=%s\n", item.symbol, address, provider.dli_fname);
    }
#endif
}

int main() {
    // Compile this client against base headers, then run the unchanged binary with replacement libraries.
    exported_ngram_vector_destructors();
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE };
    params.ngram_simple.size_n = 2;
    params.ngram_simple.size_m = 2;
    auto copied_params = params;
    auto moved_params = std::move(copied_params);
    common_speculative_ptr spec(common_speculative_init(moved_params, 2));
    require(bool(spec), "opaque speculative initialization");
    require(common_speculative_n_max(spec.get()) == 2, "configuration across library boundary");

    const llama_tokens prompt{9, 1, 2, 3, 4, 8, 1};
    llama_tokens result[2];
    std::vector<std::vector<llama_token_data>> q[2];
    std::vector<common_speculative_draft_params> owned(2);
    for (int round = 0; round < 2; ++round) {
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            common_speculative_begin(spec.get(), seq, prompt);
            result[seq].clear();
            owned[seq] = {true, 2, 7, 2, &prompt, &result[seq], &q[seq], 0.8f, uint32_t(123 + seq)};
        }
        auto copied = owned;
        auto moved = std::move(copied);
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            common_speculative_get_draft_params(spec.get(), seq) = moved[seq];
        }
        common_speculative_draft(spec.get());
        for (llama_seq_id seq = 0; seq < 2; ++seq) {
            require(result[seq] == llama_tokens({3, 4}), "draft writes through caller-owned objects for both sequences");
            const auto returned = common_speculative_get_draft_params(spec.get(), seq);
            require(!returned.drafting && returned.seed == uint32_t(123 + seq) && returned.result_q == &q[seq],
                    "copy returned parameter record with original field offsets");
            common_speculative_accept(spec.get(), seq, 1);
            std::vector<uint8_t> state;
            require(!common_speculative_get_state(spec.get(), seq, state), "unsupported state reports false");
            common_speculative_set_state(spec.get(), seq, state);
        }
    }
    common_speculative_begin(nullptr, 0, prompt);
    llama_batch empty{};
    require(common_speculative_process(nullptr, empty), "null legacy batch path");
    bool threw = false;
    try {
        common_speculative_types_from_names({"abi-client-invalid-type"});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    require(threw, "exception crosses the shared-library boundary");

    int callbacks = 0;
    llama_sampler_i iface{};
    iface.apply = [](llama_sampler * s, llama_token_data_array * row) {
        ++*static_cast<int *>(s->ctx);
        row->selected = 0;
    };
    iface.accept = [](llama_sampler * s, llama_token) { ++*static_cast<int *>(s->ctx); };
    iface.reset = [](llama_sampler * s) { ++*static_cast<int *>(s->ctx); };
    iface.free = [](llama_sampler * s) { ++*static_cast<int *>(s->ctx); };
    llama_sampler * sampler = llama_sampler_init(&iface, &callbacks);
    llama_token_data data{3, 0.0f, 1.0f};
    llama_token_data_array row{&data, 1, -1, false};
    llama_sampler_apply(sampler, &row);
    require(row.selected == 0, "caller-owned callback interface and row");
    llama_sampler_accept(sampler, 3);
    llama_sampler_reset(sampler);
    llama_sampler_free(sampler);
    require(callbacks == 4, "apply/accept/reset/free callbacks");

#ifdef __linux__
    std::ifstream maps("/proc/self/maps");
    require(maps.is_open(), "runtime library mappings readable");
    for (std::string line; std::getline(maps, line); ) {
        if (line.find(".so") != std::string::npos) { std::printf("MAP %s\n", line.c_str()); }
    }
#endif
    std::printf("PASS C++ ABI client: params_size=%zu params_align=%zu; two sequences, copy/move, reset, callbacks, error\n",
                sizeof(common_speculative_draft_params), alignof(common_speculative_draft_params));
}
