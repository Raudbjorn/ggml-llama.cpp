#pragma once
#ifndef LLAMA_TEST_SPEC_CONSUMER
#error "consumer fixtures must never be included in production builds"
#endif

#include "../src/llama-batch.h"

#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "speculative-cycle.h"

#include <cmath>
#include <cstdlib>

// Synthetic consumer-boundary data, not a trained-model distribution campaign.
// The ordinary wrapper still consumes its real proposal once. Only this test build
// replaces that decision using the production distribution transaction on a clone
// of its pre-decision sampler. The committed replacement is then left to the real
// consumer's checkpoint, batching, stopping and release code.
struct test_spec_consumer {
    using row = std::vector<llama_token_data>;
    std::string name = std::getenv("LLAMA_TEST_SPEC_CONSUMER_CASE") ?
            std::getenv("LLAMA_TEST_SPEC_CONSUMER_CASE") : "";
    bool injected = false;
    bool awaiting = false;
    bool stopping = false;
    bool cancel_sent = false;
    size_t attempts = 0;
    size_t saved_attempts = 0;
    uint64_t proposal = 0;
    llama_token seed = LLAMA_TOKEN_NULL;
    llama_tokens selected;
    common_sampler_ptr committed;
    llama_context * target = nullptr;

    void event(const char * kind, common_json data = common_json::object()) const {
        if (name.empty()) { return; }
        data["event"] = kind;
        data["case"] = name;
        data["synthetic"] = true;
        data["proposal"] = proposal;
        const auto line = data.dump();
        fprintf(stderr, "P06_CONSUMER %s\n", line.c_str());
        fflush(stderr);
    }

    static llama_token chain_draw(common_sampler * sampler) {
        row choices = {{4, std::log(.5f), .5f}, {5, std::log(.5f), .5f}};
        llama_token_data_array candidates{choices.data(), choices.size(), -1, false};
        llama_sampler_apply(common_sampler_get(sampler), &candidates);
        GGML_ASSERT(candidates.selected >= 0);
        return candidates.data[candidates.selected].id;
    }

    static llama_tokens draw(common_sampler * sampler, const llama_tokens & draft,
            const std::vector<row> & p, const std::vector<row> & q) {
        std::vector<int> indices(p.size());
        for (size_t i = 0; i < indices.size(); ++i) { indices[i] = int(i); }
        common_sampler_dist_source source{
            32, [](int) { return false; },
            [&](common_sampler * work, int i, bool, row & out) {
                // Advance the ordinary sampler RNG as the host distribution adapter
                // does; the returned probabilities are explicitly fixture data.
                chain_draw(work);
                out = p.at(i);
                return true;
            },
            [](common_sampler *, bool) -> llama_tokens {
                GGML_ABORT("consumer fixture unexpectedly used exact fallback");
            },
        };
        std::string reason;
        auto result = common_sampler_sample_and_accept_n_dist(sampler, source, indices, draft, q, reason);
        GGML_ASSERT(reason == "distribution");
        return result;
    }

    void same_sampler(common_sampler * live) const {
        GGML_ASSERT(committed);
        common_sampler_ptr a(common_sampler_clone(live));
        common_sampler_ptr b(common_sampler_clone(committed.get()));
        GGML_ASSERT(common_sampler_last(a.get()) == common_sampler_last(b.get()));
        GGML_ASSERT(common_sampler_prev_str(a.get(), target, 32) == common_sampler_prev_str(b.get(), target, 32));
        // Probe both independent RNG domains without advancing either live state.
        for (int i = 0; i < 32; ++i) {
            GGML_ASSERT(chain_draw(a.get()) == chain_draw(b.get()));
            const std::vector<row> p(2, {{4, 0, .5f}, {5, 0, .5f}});
            GGML_ASSERT(draw(a.get(), {4}, p, {{{4, 0, 1}}}) ==
                        draw(b.get(), {4}, p, {{{4, 0, 1}}}));
        }
    }

    void attempt() { GGML_ASSERT(!awaiting); ++attempts; }
    void fresh_draft() { GGML_ASSERT(!awaiting); event("fresh_draft"); }

    void decision(llama_tokens & ids, common_sampler * live, const common_sampler * before,
            const llama_tokens & draft, llama_token last, uint64_t id, bool used_distribution,
            llama_context * ctx, const llama_vocab * vocab, bool ignore) {
        if (name.empty() || injected || draft.size() != 3) { return; }
        GGML_ASSERT(used_distribution && before);
        injected = true;
        target = ctx;
        proposal = id;
        seed = last;
        GGML_ASSERT(proposal != 0 && llama_vocab_eos(vocab) == 31);
        llama_tokens fixture_draft = draft;
        const bool early = name == "eog_early";
        const bool final = name == "eog_final";
        const bool correction_eog = name == "eog_correction";
        const bool full = early || final;
        int prefix = name == "repair1" ? 1 : name == "repair2" ? 2 : 0;
        GGML_ASSERT(full || correction_eog || name == "repair0" || name == "repair1" ||
                    name == "repair2" || name == "cancel");
        // EOG cases force the acceptor's input vector as well as its p/q. Their
        // evidence concerns consumer stopping, not agreement with decoded logits.
        if (full) {
            fixture_draft = {4, 5, 6};
            fixture_draft[early ? 0 : 2] = 31;
        }
        if (correction_eog) { fixture_draft = {4, 5, 6}; }
        std::vector<row> q, p;
        for (auto token : fixture_draft) {
            q.push_back({{token, 0, 1}});
            p.push_back({{token, 0, 1}});
        }
        p.push_back({{7, 0, 1}});
        if (!full) {
            const llama_token correction = correction_eog ? 31 : (fixture_draft[prefix] == 4 ? 5 : 4);
            GGML_ASSERT(correction != fixture_draft[prefix]);
            p[prefix] = {{correction, 0, 1}};
        }
        common_sampler_copy(before, live);
        ids = draw(live, fixture_draft, p, q);
        GGML_ASSERT(ids.size() == (full ? 4u : size_t(prefix + 1)));
        selected = ids;
        committed.reset(common_sampler_clone(live));
        saved_attempts = attempts;
        awaiting = !full && (!correction_eog || ignore);
        event("decision", {{"draft", draft}, {"fixture_draft", fixture_draft}, {"selected", ids},
                {"seed", seed}, {"original_draft", draft.size()}, {"attempts", attempts}});
    }

    void prepared(const common_speculative_repair & repair, const llama_tokens & draft, common_sampler * live) {
        if (!awaiting) { return; }
        GGML_ASSERT(repair.pending && repair.n_draft == 3 && draft == selected);
        GGML_ASSERT(repair.n_decode(draft) + 1 == selected.size());
        same_sampler(live);
        event("prepared", {{"selected", draft}, {"original_draft", repair.n_draft}, {"sampler_preserved", true}});
    }

    void decode(const llama_batch_ext * batch, llama_seq_id seq, bool repairing) {
        if (name.empty()) { return; }
        GGML_ASSERT(!stopping);
        GGML_ASSERT(batch);
        llama_tokens tokens;
        // Inspect the rendered decode slice, including every row's sequence membership.
        for (const auto & token : batch->tokens) {
            if (token.seq_ids.count(seq)) {
                GGML_ASSERT(token.id != LLAMA_TOKEN_NULL);
                tokens.push_back(token.id);
            }
        }
        if (tokens.empty()) { return; }
        if (repairing && awaiting) {
            llama_tokens expected{seed};
            expected.insert(expected.end(), selected.begin(), selected.end() - 1);
            GGML_ASSERT(tokens == expected && attempts == saved_attempts);
        }
        event("decode", {{"tokens", tokens}, {"repairing", repairing}});
    }

    void repaired(const llama_tokens & ids, common_sampler * live, size_t original,
            const common_speculative_cycle_record * record) {
        if (!awaiting) { return; }
        GGML_ASSERT(ids == selected && original == 3 && attempts == saved_attempts);
        GGML_ASSERT(record && record->proposal_id == proposal && record->verify_phases.size() == 2);
        GGML_ASSERT(record->distribution_attempts == 1 && record->exact_match_attempts == 0);
        same_sampler(live);
        awaiting = false;
        event("repaired", {{"selected", ids}, {"sampler_preserved", true}, {"original_draft", original},
                {"attempts", attempts}, {"verify_phases", record->verify_phases.size()}});
    }

    void policy(bool stops, size_t effective, bool ignore) {
        stopping = stops;
        if (injected) { event("policy", {{"stops", stops}, {"effective", effective}, {"ignore_eos", ignore}}); }
    }
    void visit(llama_token token) { if (injected) { event("visit", {{"token", token}}); } }
    void reset(bool pending) {
        event("reset", {{"pending_before", pending}, {"awaiting_before", awaiting}});
        awaiting = stopping = false;
        committed.reset();
    }
    bool cancel() {
        if (name != "cancel" || !awaiting || cancel_sent) { return false; }
        cancel_sent = true;
        event("cancel_queued");
        return true;
    }
};
