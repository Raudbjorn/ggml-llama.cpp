#pragma once

#include "speculative.h"

struct driver_graph_observation {
    struct execution {
        int64_t input_rows;
        int64_t output_width;
        int64_t output_rows;
    };

    int64_t input_rows = 0;
    std::vector<execution> executions;

    static bool callback(ggml_tensor * tensor, bool ask, void * data) {
        const std::string name(tensor->name);
        const bool input = tensor->op == GGML_OP_GET_ROWS && name.find("mtp_tok_embd-") == 0;
        const bool output = name == "result_output";
        if (ask) {
            return input || output;
        }
        auto & observed = *static_cast<driver_graph_observation *>(data);
        if (input) {
            observed.input_rows = tensor->ne[1];
        }
        if (output) {
            // Post-evaluation callbacks exclude graph construction and reserve-only probes.
            observed.executions.push_back({observed.input_rows, tensor->ne[0], tensor->ne[1]});
            observed.input_rows = 0;
        }
        return true;
    }
};

// Exercise the public driver against the synthetic models from test-qwen4exp-mtp.cpp.
static std::vector<llama_tokens> driver_drafts(
        llama_model * target_model, llama_model * draft_model,
        bool chained, int n_seq, int draft_ubatch, bool adaptive, int prompt_length = 1, bool at_limit = false,
        driver_graph_observation * observed = nullptr, bool fail_first_draft = false) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 128 * n_seq;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_seq_max = n_seq;
    cp.n_outputs_max = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // the --q8-kv run covers the driver paths too, for both contexts
    cp.type_k = cp.type_v = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    llama_context_ptr target(llama_init_from_model(target_model, cp));
    require(bool(target), "driver target context");
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    // as the server does: a head without its own tables borrows them from the target
    cp.ctx_other = target.get();
    if (observed) {
        cp.cb_eval = driver_graph_observation::callback;
        cp.cb_eval_user_data = observed;
    }
    cp.n_ubatch = draft_ubatch;
    llama_context_ptr draft(llama_init_from_model(draft_model, cp));
    require(bool(draft), "driver draft context");
    if (at_limit) {
        prompt_length = int(llama_n_ctx(draft.get())) - 2;
    }

    common_params_speculative params;
    params.types = {adaptive ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE : COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.draft.ctx_tgt = target.get();
    params.draft.ctx_dft = draft.get();
    params.draft.n_max = 4;
    params.draft.n_min_adaptive = 1;
    params.draft.backend_sampling = false;
    params.draft.chain = chained;
    common_speculative_ptr spec(common_speculative_init(params, n_seq));
    require(bool(spec), "initialize public MTP driver");

    std::vector<llama_tokens> all_drafts;
    const std::vector<int> expected_lengths = adaptive ? std::vector<int>{1, 1, 2, 2, 2, 2, 3, 1}
                                                     : std::vector<int>{at_limit ? 2 : 4};
    for (size_t round = 0; round < expected_lengths.size(); ++round) {
        // Rewind to an identical prompt while keeping acceptance feedback in the driver.
        // This isolates depth changes from target-model acceptance variability.
        llama_memory_clear(llama_get_memory(target.get()), true);
        llama_memory_clear(llama_get_memory(draft.get()), true);
        llama_batch batch = llama_batch_init(32, 0, 1);
        std::vector<llama_tokens> prompts(n_seq), results(n_seq);
        for (int seq = 0; seq < n_seq; ++seq) {
            prompts[seq].assign(prompt_length, llama_token(3 + seq));
        }
        for (int pos = 0; pos < prompt_length; ) {
            common_batch_clear(batch);
            const int end = std::min(prompt_length, pos + 32 / n_seq);
            for (int seq = 0; seq < n_seq; ++seq) {
                for (int i = pos; i < end; ++i) {
                    common_batch_add(batch, prompts[seq][i], i, {seq}, true);
                }
            }
            require(llama_decode(target.get(), batch) == 0, "driver target prefill");
            require(common_speculative_process(spec.get(), batch), "driver processes target hidden states");
            pos = end;
        }
        llama_batch_free(batch);

        auto request_drafts = [&]() {
            for (int seq = 0; seq < n_seq; ++seq) {
                auto & dp = common_speculative_get_draft_params(spec.get(), seq);
                dp.drafting = true;
                dp.pos0 = prompt_length;
                dp.id_last = 5 + seq;
                dp.prompt = &prompts[seq];
                dp.result = &results[seq];
                dp.n_max = at_limit ? 2 : adaptive && round + 1 == expected_lengths.size() ? 1 : -1;
            }
        };
        for (int seq = 0; seq < n_seq; ++seq) {
            common_speculative_begin(spec.get(), seq, prompts[seq]);
        }
        request_drafts();
        if (fail_first_draft && round == 0) {
            // A backend sampler makes a chain decode return -1 before it touches the cache.
            llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
            require(llama_set_sampler(draft.get(), 0, sampler.get()), "backend sampler attaches to the draft context");
            common_speculative_draft(spec.get());
            require(llama_set_sampler(draft.get(), 0, nullptr), "backend sampler detaches");
            require(results[0].empty(), "a failed draft decode drafts nothing");
            request_drafts();
        }
        if (observed) {
            observed->input_rows = 0;
            observed->executions.clear();
        }
        common_speculative_draft(spec.get());
        for (int seq = 0; seq < n_seq; ++seq) {
            require(results[seq].size() == size_t(expected_lengths[round]),
                    "driver respects configured/adaptive depth and remaining-context cap");
            all_drafts.push_back(results[seq]);
            common_speculative_accept(spec.get(), seq, results[seq].size());
        }
    }
    return all_drafts;
}

static void test_driver(llama_model * target, llama_model * head) {
    for (const auto & config : std::vector<std::pair<int, int>>{{1, 2}, {2, 8}}) {
        const auto sequential = driver_drafts(target, head, false, config.first, config.second, false);
        const auto chained = driver_drafts(target, head, true, config.first, config.second, false);
        require(chained == sequential, "driver fallback preserves sequential greedy drafts");
        fprintf(stderr, "PASS driver fallback sequences=%d ubatch=%d\n", config.first, config.second);
    }
    for (int ubatch : {2, 8}) {
        const auto sequential = driver_drafts(target, head, false, 1, ubatch, true);
        const auto chained = driver_drafts(target, head, true, 1, ubatch, true);
        require(chained == sequential, "adaptive driver preserves greedy drafts through depth and cap changes");
        fprintf(stderr, "PASS adaptive driver depth changes and context cap ubatch=%d\n", ubatch);
    }
    // With two catch-up rows and depth four, ubatches 4/5 must flush the prefix;
    // ubatch 6 can merge it into one chain exactly at capacity.
    for (int ubatch : {4, 5, 6}) {
        const auto sequential = driver_drafts(target, head, false, 1, ubatch, false, 2);
        driver_graph_observation observed;
        const auto chained = driver_drafts(target, head, true, 1, ubatch, false, 2, false, &observed);
        require(chained == sequential, "catch-up admission preserves full-capacity drafts");
        int packed_graphs = 0;
        for (const auto & execution : observed.executions) {
            if (execution.output_width == 2 && execution.output_rows == 4) {
                ++packed_graphs;
                require(execution.input_rows == (ubatch == 6 ? 6 : 4),
                        "executed chain includes catch-up only when the combined batch fits");
            }
        }
        require(packed_graphs == 1, "full-capacity drafting executes one packed chain graph, without silent fallback");
        if (ubatch == 6) {
            require(observed.executions.size() == 1, "two catch-up rows and four draft rows execute in one merged graph");
        }
        fprintf(stderr, "PASS driver catch-up capacity ubatch=%d\n", ubatch);
    }
    const auto sequential = driver_drafts(target, head, false, 1, 8, false, 1, true);
    const auto chained = driver_drafts(target, head, true, 1, 8, false, 1, true);
    require(chained == sequential, "driver honors supplied cap with an occupied context near its limit");
    fprintf(stderr, "PASS driver supplied cap near occupied context limit\n");
}

// A head handed to a non-MTP draft type gets an ordinary context. The public
// draft-simple driver must run it; its greedy drafts follow the head's own logits.
static void test_ordinary_draft_driver(llama_model * target_model, llama_model * head) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.type_k = cp.type_v = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    llama_context_ptr target(llama_init_from_model(target_model, cp));
    llama_context_ptr draft(llama_init_from_model(head, cp));
    llama_context_ptr reference(llama_init_from_model(head, cp));
    require(target && draft && reference, "ordinary draft driver contexts");

    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    params.draft.ctx_tgt = target.get();
    params.draft.ctx_dft = draft.get();
    params.draft.n_max = 4;
    params.draft.backend_sampling = false;
    common_speculative_ptr spec(common_speculative_init(params, 1));
    require(bool(spec), "initialize draft-simple driver on an MTP-only head");

    llama_tokens prompt = {3, 4};
    llama_tokens result;
    llama_batch batch = llama_batch_init(32, 0, 1);
    for (size_t i = 0; i < prompt.size(); ++i) {
        common_batch_add(batch, prompt[i], i, {0}, true);
    }
    require(llama_decode(target.get(), batch) == 0, "ordinary draft driver target prefill");
    require(common_speculative_process(spec.get(), batch), "draft-simple driver mirrors the target batch");

    common_speculative_begin(spec.get(), 0, prompt);
    auto & dp = common_speculative_get_draft_params(spec.get(), 0);
    dp.drafting = true;
    dp.pos0 = prompt.size();
    dp.id_last = 5;
    dp.prompt = &prompt;
    dp.result = &result;
    dp.n_max = -1;
    common_speculative_draft(spec.get());

    // Without a trunk each draft token depends on its predecessor alone.
    llama_tokens expected;
    llama_token token = dp.id_last;
    for (int i = 0; i < params.draft.n_max; ++i) {
        common_batch_clear(batch);
        common_batch_add(batch, token, i, {0}, true);
        require(llama_decode(reference.get(), batch) == 0, "ordinary draft reference decode");
        const float * logits = llama_get_logits_ith(reference.get(), -1);
        token = std::max_element(logits, logits + n_vocab) - logits;
        expected.push_back(token);
    }
    llama_batch_free(batch);
    require(result == expected, "draft-simple drafts follow the trunkless head");
    fprintf(stderr, "PASS draft-simple driver on an MTP-only head\n");
}

// fork: environment changes are scoped to this synthetic production-driver fixture.
struct distribution_test_env {
    std::string name;
    std::string previous;
    bool existed;
    distribution_test_env(const char * key, const char * value) : name(key), existed(std::getenv(key) != nullptr) {
        if (existed) { previous = std::getenv(key); }
        set(value);
    }
    void set(const char * value) {
#ifdef _WIN32
        _putenv_s(name.c_str(), value ? value : "");
#else
        if (value) { setenv(name.c_str(), value, 1); }
        else { unsetenv(name.c_str()); }
#endif
    }
    ~distribution_test_env() { set(existed ? previous.c_str() : nullptr); }
};

static void test_distribution_driver(llama_model * target_model, llama_model * draft_model) {
    distribution_test_env temperature("LLAMA_SPEC_SAMPLE_TEMP", "0.8");
    distribution_test_env top_p("LLAMA_SPEC_DRAFT_TOPP", "0.7");
    distribution_test_env logging("LLAMA_SPEC_LOG", nullptr);
    distribution_test_env profiling("LLAMA_SPEC_PROFILE", "0");
    distribution_test_env chain_env("LLAMA_SPEC_CHAIN", "0");
    for (bool chained : {false, true}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 32;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = 32;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = q8_kv ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
        llama_context_ptr target(llama_init_from_model(target_model, cp));
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.ctx_other = target.get();
        llama_context_ptr draft(llama_init_from_model(draft_model, cp));
        require(target && draft, "P06 synthetic contexts");
        common_params_speculative params;
        params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        params.draft.ctx_tgt = target.get();
        params.draft.ctx_dft = draft.get();
        params.draft.n_max = 3;
        params.draft.chain = chained;
        params.draft.backend_sampling = true; // P06 must explicitly suppress offload
        common_speculative_ptr spec(common_speculative_init(params, 1));
        llama_tokens prompt{3, 4}, tokens;
        std::vector<std::vector<llama_token_data>> compatibility_q;
        common_batch batch(target.get());
        batch.add(3, 0, 0, true);
        batch.add(4, 1, 0, true);
        require(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "P06 prefill");
        require(common_speculative_process(spec.get(), batch), "P06 target hidden state handoff");
        common_speculative_begin(spec.get(), 0, prompt);
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true; dp.pos0 = 2; dp.id_last = 5; dp.prompt = &prompt;
        dp.result = &tokens; dp.seed = 1201; dp.n_max = 2; dp.result_q = &compatibility_q;
        common_speculative_set_request_id(spec.get(), 0, 17);
        common_speculative_draft(spec.get());
        const auto * record = common_speculative_cycle_get(spec.get(), 0);
        const uint64_t proposal_id = common_speculative_get_proposal_id(spec.get(), 0);
        require(record && record->request_id == 17 && record->proposal_id == proposal_id && record->proposal == tokens && tokens.size() == 2,
                "P06 has finalized P04 identity without logging and honors the final cap");
        require(!record->draft_phase && record->attempts.empty() && record->verify_phases.empty(),
                "identity-only mode records no timing/attempt instrumentation");
        if (chained) {
            require(record->dists.empty() && record->sampling_reason == "chained_draft", "chained argmax is ineligible");
        } else {
            require(record->dists.size() == tokens.size() && compatibility_q.size() == tokens.size(), "q and tokens clamp together");
            for (size_t i = 0; i < tokens.size(); ++i) {
                require(record->dists[i].size() <= LLAMA_DRAFT_TOP_K &&
                        common_sampler_dist_valid(record->dists[i], n_vocab, tokens[i]), "recorded sampled q is normalized and keyed");
            }
            require(llama_get_sampled_token_ith(draft.get(), -1) == LLAMA_TOKEN_NULL, "P06 draft backend sampler is disabled");
        }
        batch.clear();
        batch.add(5, 2, 0, true);
        for (size_t i = 0; i < tokens.size(); ++i) { batch.add(tokens[i], 3 + i, 0, true); }
        require(llama_process(target.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "P06 verification rows");
        common_params_sampling sp;
        sp.seed = 4321; sp.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
        common_sampler_ptr smpl(common_sampler_init(target_model, sp));
        common_sampler_sample(smpl.get(), target.get(), 0, true);
        const auto * original = common_sampler_get_candidates(smpl.get(), false);
        common_sampler_ptr clone(common_sampler_clone(smpl.get()));
        auto * copied = common_sampler_get_candidates(clone.get(), false);
        require(original->size == copied->size && original->data != copied->data, "clone owns copied candidate storage");
        const float probability = original->data[0].p;
        copied->data[0].p = -123;
        require(original->data[0].p == probability, "clone candidate mutation cannot alias live sampler");
        common_sampler_copy(smpl.get(), clone.get());
        require(common_sampler_get_candidates(clone.get(), false)->data[0].p == probability, "copy repairs independent candidates");
        const std::vector<int> indices{0, 1, 2};
        bool used_distribution = false;
        const auto result = common_speculative_sample_and_accept(spec.get(), 0, smpl.get(), target.get(), indices,
                tokens, proposal_id, false, true, &used_distribution);
        require(used_distribution == !chained, "repair eligibility follows this actual acceptance attempt");
        require(!result.empty() && result.size() <= 3 && record->sampling_consumed, "P06 production acceptance consumes identity");
        require(record->sampling_mode == (chained ? "exact_match" : "distribution"), "explicit effective sampling mode");
        common_sampler_copy(smpl.get(), clone.get());
        const auto replayed = common_speculative_sample_and_accept(spec.get(), 0, smpl.get(), target.get(), indices,
                tokens, proposal_id, false, true, &used_distribution);
        require(!used_distribution, "new consumed-proposal attempt remains exact, not cache repair");
        require(replayed == common_sampler_sample_and_accept_n(clone.get(), target.get(), indices, tokens, true),
                "consumed identity falls back with grammar_first preserved");
        auto replaced = tokens;
        replaced.back() = (replaced.back() + 1) % n_vocab;
        common_speculative_invalidate(spec.get(), 0);
        require(compatibility_q.empty() && record->proposal == tokens, "replay invalidation keeps immutable snapshot");
        common_sampler_copy(smpl.get(), clone.get());
        require(common_speculative_sample_and_accept(spec.get(), 0, smpl.get(), target.get(), indices,
                    replaced, proposal_id, true, true, &used_distribution) ==
                common_sampler_sample_and_accept_n(clone.get(), target.get(), indices, replaced, true),
                "same-length replay uses exact match");
        require(!used_distribution, "new invalidated-proposal attempt cannot start distribution repair");
        common_speculative_cycle_finish(spec.get(), 0, proposal_id, 0);
        require(!common_speculative_cycle_get(spec.get(), 0), "identity-only completion releases proposal");
    }
    fprintf(stderr, "PASS P06 synthetic production MTP sampling, chained fallback, ownership and replay\n");
}
