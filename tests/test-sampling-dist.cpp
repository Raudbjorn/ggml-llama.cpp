// fork: production transaction and a predeclared output-stream distribution corpus.
#include "sampling.h"
#include "speculative.h"
#include "../src/llama-model.h"

#undef NDEBUG
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>

using distribution = std::vector<llama_token_data>;
using distributions = std::vector<distribution>;

static distribution probabilities(std::initializer_list<float> values) {
    distribution row;
    for (float p : values) { row.push_back({llama_token(row.size()), 0, p}); }
    return row;
}

static common_sampler_ptr sampler(llama_model & model, uint32_t seed) {
    common_params_sampling params;
    params.seed = seed;
    params.temp = 1;
    params.no_perf = true;
    params.samplers.clear();
    return common_sampler_ptr(common_sampler_init(&model, params));
}

static llama_token native_sample(common_sampler * smpl, distribution & row) {
    for (auto & item : row) { item.logit = item.p > 0 ? std::log(item.p) : -INFINITY; }
    llama_token_data_array candidates{row.data(), row.size(), -1, false};
    llama_sampler_apply(common_sampler_get(smpl), &candidates);
    assert(candidates.selected >= 0 && size_t(candidates.selected) < candidates.size);
    row.resize(candidates.size);
    return candidates.data[candidates.selected].id;
}

struct fixture {
    int vocab = 2;
    distributions p;
    llama_tokens draft;
    int backend = -1;
    int corrupt = -1;
    int reads = 0;
    int exact_calls = 0;
    bool grammar_first_seen = false;

    common_sampler_dist_source source() {
        return {vocab,
            [this](int i) { return i == backend; },
            [this](common_sampler * smpl, int i, bool first, distribution & row) {
                ++reads;
                grammar_first_seen = first;
                row = p.at(i);
                native_sample(smpl, row);
                if (i == corrupt) { row.back().p = std::numeric_limits<float>::quiet_NaN(); }
                return true;
            },
            [this](common_sampler * smpl, bool first) {
                ++exact_calls;
                grammar_first_seen = first;
                llama_tokens result;
                for (size_t i = 0; i <= draft.size(); ++i) {
                    auto row = p.at(i);
                    const auto id = native_sample(smpl, row);
                    common_sampler_accept(smpl, id, true);
                    result.push_back(id);
                    if (i < draft.size() && draft[i] != id) { break; }
                }
                return result;
            }};
    }

    llama_tokens run(common_sampler * smpl, const distributions & q, std::string & reason, bool first = false) {
        std::vector<int> idxs(draft.size() + 1);
        std::iota(idxs.begin(), idxs.end(), 0);
        return common_sampler_sample_and_accept_n_dist(smpl, source(), idxs, draft, q, reason, first);
    }
};

static void canonical_proposal_draw() {
    const auto row = probabilities({0, .25f, 0, .75f, 0});
    assert(common_sampler_dist_draw(row, 0) == 1);
    assert(common_sampler_dist_draw(row, std::nextafter(.25, 0.0)) == 1);
    assert(common_sampler_dist_draw(row, .25) == 3);
    assert(common_sampler_dist_draw(row, std::nextafter(1.0, 0.0)) == 3);
    const auto rounded = probabilities({.1f, .2f, .7f});
    const double mass = double(rounded[0].p) + rounded[1].p + rounded[2].p;
    assert(mass != 1.0);
    const double boundary = double(rounded[0].p) / mass;
    assert(common_sampler_dist_draw(rounded, std::nextafter(boundary, 0.0)) == 0);
    assert(common_sampler_dist_draw(rounded, std::nextafter(boundary, 1.0)) == 1);
    std::puts("PASS: canonical recorded-q draw normalizes float mass and skips zero CDF boundaries");
}

static void transactions(llama_model & model) {
    const auto half = probabilities({.5f, .5f});
    const distributions q(3, half);
    fixture f{2, distributions(4, half), {0, 1, 0}};
    auto smpl = sampler(model, 1201);
    std::string reason;
    auto full = f.run(smpl.get(), q, reason);
    assert(full.size() == 4 && reason == "distribution" && f.reads == 4 && f.exact_calls == 0);

    for (int bad = 0; bad < 10; ++bad) {
        auto invalid = q;
        f.backend = -1;
        f.corrupt = -1;
        switch (bad) {
            case 0: invalid.pop_back(); break;
            case 1: invalid.back().clear(); break;
            case 2: invalid.back()[0].p = 0; break;
            case 3: invalid.back()[0].p = INFINITY; break;
            case 4: invalid.back()[1].id = invalid.back()[0].id; break;
            case 5: invalid.back()[0].p = -.5f; break;
            case 6: invalid.back()[0].id = 2; break;
            case 7: invalid.back() = probabilities({0, 1}); break;
            case 8: f.corrupt = 3; break; // late failure after all clone chain draws
            case 9: std::swap(invalid[1], invalid[2]); invalid[2] = probabilities({0, 1}); break;
        }
        auto control = common_sampler_ptr(common_sampler_clone(smpl.get()));
        f.reads = f.exact_calls = 0;
        const auto result = f.run(smpl.get(), invalid, reason, true);
        assert(reason != "distribution" && f.exact_calls == 1 && f.grammar_first_seen);
        assert(f.reads == (bad == 8 ? 4 : 0));
        assert(result == f.source().exact(control.get(), true));
        f.corrupt = -1;
        // Both RNG domains must still agree after fallback, not just its first result.
        for (int j = 0; j < 8; ++j) { assert(f.run(smpl.get(), q, reason) == f.run(control.get(), q, reason)); }
    }
    for (int idx = 0; idx < 4; ++idx) {
        f.backend = idx;
        f.reads = 0;
        auto control = common_sampler_ptr(common_sampler_clone(smpl.get()));
        assert(f.run(smpl.get(), q, reason) == f.source().exact(control.get(), false));
        assert(reason == "backend_target_token" && f.reads == 0);
    }
    f.backend = -1;
    auto copy = sampler(model, 1201);
    common_sampler_copy(smpl.get(), copy.get());
    auto clone = common_sampler_ptr(common_sampler_clone(smpl.get()));
    for (int i = 0; i < 30; ++i) {
        const auto expected = f.run(smpl.get(), q, reason);
        assert(f.run(copy.get(), q, reason) == expected && f.run(clone.get(), q, reason) == expected);
    }
    common_sampler_reset(smpl.get());
    common_sampler_reset(copy.get());
    common_sampler_reset(clone.get());
    auto fresh = sampler(model, 1201);
    for (int i = 0; i < 30; ++i) {
        const auto expected = f.run(fresh.get(), q, reason);
        assert(f.run(smpl.get(), q, reason) == expected);
        assert(f.run(copy.get(), q, reason) == expected && f.run(clone.get(), q, reason) == expected);
    }

    // A supported chain is an allowlist, not absence of known bad parameters.
    static llama_sampler_i unknown = [] {
        llama_sampler_i result{};
        result.name = [](const llama_sampler *) { return "unknown-stateful"; };
        result.apply = [](llama_sampler *, llama_token_data_array *) {};
        return result;
    }();
    llama_sampler_chain_add(common_sampler_get(smpl.get()), llama_sampler_init(&unknown, nullptr));
    f.reads = 0;
    f.run(smpl.get(), q, reason);
    assert(reason == "unsupported_target_sampler" && f.reads == 0);
    for (int stateful = 0; stateful < 8; ++stateful) {
        common_params_sampling sp;
        sp.seed = 1;
        switch (stateful) {
            case 0: sp.penalty_repeat = 1.1f; break;
            case 1: sp.penalty_freq = .1f; break;
            case 2: sp.penalty_present = .1f; break;
            case 3: sp.mirostat = 2; break;
            case 4: sp.reasoning_budget_tokens = 4; break;
            case 5: sp.samplers = {COMMON_SAMPLER_TYPE_ADAPTIVE_P}; break;
            case 6: sp.samplers = {COMMON_SAMPLER_TYPE_DRY}; sp.dry_multiplier = 1; sp.dry_sequence_breakers.clear(); break;
            case 7: sp.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z]*"}; break;
        }
        auto other = common_sampler_ptr(common_sampler_init(&model, sp));
        assert(!common_sampler_dist_supported(other.get()));
        auto source = f.source();
        int fallbacks = 0;
        source.exact = [&](common_sampler * original, bool first) {
            assert(original == other.get() && first);
            ++fallbacks;
            return llama_tokens{1};
        };
        assert(common_sampler_sample_and_accept_n_dist(other.get(), source, {0, 1, 2, 3}, f.draft,
                    q, reason, true) == llama_tokens{1});
        assert(fallbacks == 1 && reason == "unsupported_target_sampler");
    }

    // Zero target probability and disjoint support cannot return the draft token.
    fixture disjoint{2, {probabilities({0, 1}), half}, {0}};
    for (int i = 0; i < 100; ++i) {
        assert(disjoint.run(fresh.get(), {probabilities({1, 0})}, reason) == llama_tokens{1});
    }
    // Force rejection on a tiny q-only token. At 1e-30 the clipped residual rounds
    // to zero in double precision, exercising the required p fallback without RNG hunting.
    for (float tiny : {1e-7f, 1e-30f}) {
        assert(disjoint.run(fresh.get(), {probabilities({tiny, 1 - tiny})}, reason) == llama_tokens{1});
        assert(reason == "distribution");
    }
    // Full acceptance includes the bonus even when a draft position is EOG.
    for (const llama_tokens draft : {llama_tokens{1, 0, 0}, llama_tokens{0, 0, 1}}) {
        distributions point;
        for (auto id : draft) { point.push_back(id ? probabilities({0, 1}) : probabilities({1, 0})); }
        auto target = point;
        target.push_back(probabilities({1, 0}));
        fixture eog{2, target, draft};
        const auto tokens = eog.run(fresh.get(), point, reason);
        assert(tokens.size() == 4);
        const auto is_eog = [](llama_token id) { return id == 1; };
        const size_t prefix = common_sampler_eog_prefix(tokens, false, is_eog);
        assert(prefix == (draft[0] == 1 ? 1 : 3));
        assert(common_sampler_eog_prefix(tokens, true, is_eog) == tokens.size());
    }
}

static void checkpoint_repair(llama_model & model) {
    const auto half = probabilities({.5f, .5f});
    const auto point_a = probabilities({1, 0});
    const auto point_b = probabilities({0, 1});
    auto smpl = sampler(model, 9017);
    std::string reason;

    // G=1, q(A)=1, p(A)=p(B)=1/2: re-drawing a rejected B would give
    // P(A)=1/2+(1/2)(1/2)=3/4. Materialization must leave both branches intact.
    assert(.5 + .5*.5 == .75 && .75 != .5);
    fixture counterexample{2, {half, half}, {0}};
    uint32_t first[2] = {}, emitted[2] = {};
    for (int iteration = 0; iteration < 256; ++iteration) {
        const auto selected = counterexample.run(smpl.get(), {point_a}, reason);
        assert(reason == "distribution");
        ++first[selected.front()];
        auto committed = common_sampler_ptr(common_sampler_clone(smpl.get()));
        common_speculative_repair repair;
        auto draft = counterexample.draft;
        llama_tokens result = selected;
        if (selected.size() == 1) {
            assert(selected.front() == 1 && repair.prepare(draft, selected, false));
            assert(repair.pending && repair.n_draft == 1 && !draft.empty());
            assert(repair.n_decode(draft) == 0); // seed only, correction stays unprocessed
            const int reads = counterexample.reads, exact_calls = counterexample.exact_calls;
            result = repair.finish(draft); // the same helper used by BOTH consumers
            assert(result == selected && !repair.pending && repair.n_draft == 0);
            assert(counterexample.reads == reads && counterexample.exact_calls == exact_calls);
        }
        ++emitted[result.front()];
        assert(common_sampler_last(smpl.get()) == common_sampler_last(committed.get()));
        // Compare future work, not only the last token: both RNG domains survive repair.
        assert(counterexample.run(smpl.get(), {point_a}, reason) ==
                counterexample.run(committed.get(), {point_a}, reason));
    }
    assert(first[0] > 0 && first[1] > 0 && first[0] == emitted[0] && first[1] == emitted[1]);

    // Forced rejection after zero, one, or two accepted drafts, including a
    // replacement whose length equals the original proposal. No q survives it.
    for (size_t prefix = 0; prefix < 3; ++prefix) {
        fixture f{2, distributions(prefix + 2, point_a), llama_tokens(prefix + 1, 0)};
        f.p[prefix] = point_b;
        distributions q(f.draft.size(), point_a);
        common_speculative_cycle_recorder recorder(1, nullptr, true);
        const auto id = recorder.begin(0, 23, 4);
        recorder.sampling_request(0, "pending");
        auto draft_phase = recorder.phase_begin(nullptr, {{id, 0}});
        recorder.attach_draft(0, draft_phase);
        recorder.phase_end(nullptr, draft_phase, true);
        recorder.finalize(0, f.draft, true, q);
        assert(recorder.consume(0, id, f.draft, false));
        auto decision_phase = recorder.phase_begin(nullptr, {{id, 0}});
        recorder.attach_verify(0, decision_phase);
        const auto selected = f.run(smpl.get(), q, reason);
        recorder.phase_end(nullptr, decision_phase, false);
        assert(selected.size() == prefix + 1 && selected.back() == 1);
        recorder.sampling_decision(0, reason);
        const auto original = f.draft;
        common_speculative_repair repair;
        assert(repair.prepare(f.draft, selected, false));
        recorder.invalidate(0);
        q.clear();
        assert(repair.n_draft == original.size() && repair.n_decode(f.draft) == prefix);
        assert(!f.draft.empty()); // consumer draft creation remains bypassed, even at prefix=0
        assert(recorder.get(0)->proposal == original);
        auto phase = recorder.phase_begin(nullptr, {{id, 0}});
        recorder.attach_verify(0, phase);
        // Cache work replays [seed, accepted-prefix], never the correction/bonus.
        llama_tokens materialized{7};
        materialized.insert(materialized.end(), f.draft.begin(), f.draft.begin() + repair.n_decode(f.draft));
        assert(materialized.size() == prefix + 1 && materialized.back() == (prefix ? 0 : 7));
        const auto result = repair.finish(f.draft);
        recorder.phase_end(nullptr, phase, false);
        assert(result == selected && result.size() - 1 == prefix);
        assert(recorder.get(0)->distribution_attempts == 1 && recorder.get(0)->exact_match_attempts == 0);
        assert(recorder.get(0)->verify_phases.size() == 2);
        // A genuinely new consumed-proposal attempt is still unusable for q.
        assert(!recorder.consume(0, id, original, false));
        recorder.finish(0, id, prefix);

        // Abort/release/reset clears repair state before the slot is reused.
        f.draft = original;
        assert(repair.prepare(f.draft, selected, false));
        repair.clear();
        assert(!repair.pending && repair.n_draft == 0);
        assert(repair.n_decode(original) == original.size());
        f.draft = original;
        assert(repair.prepare(f.draft, selected, false)); // next request gets fresh repair state
        assert(repair.finish(f.draft) == selected);
    }

    // The active stopping policy prevents repair at early/final EOG. Ignore-EOG
    // permits the same prefix materialization; no correction is decoded either way.
    const auto is_eog = [](llama_token token) { return token == 1; };
    for (const llama_tokens selected : {llama_tokens{1, 0}, llama_tokens{0, 1}}) {
        for (bool ignore : {false, true}) {
            llama_tokens draft{0, 0, 0};
            const auto effective = common_sampler_eog_prefix(selected, ignore, is_eog);
            const bool stops = !ignore && is_eog(selected[effective - 1]);
            common_speculative_repair repair;
            assert(repair.prepare(draft, selected, stops) == ignore);
            if (ignore) {
                assert(repair.n_decode(draft) == selected.size() - 1);
                assert(repair.finish(draft) == selected);
            } else {
                assert(!repair.pending && draft == llama_tokens({0, 0, 0}));
            }
        }
    }
    common_speculative_repair full;
    llama_tokens draft{0};
    assert(!full.prepare(draft, {0, 1}, false)); // G+1 full acceptance needs no repair
    std::puts("PASS: P06 checkpoint repair preserves decisions/RNG, counts, phases and EOG policy");
}

static void lifecycle() {
    common_speculative_cycle_recorder recorder(1, nullptr, false);
    const distributions q{probabilities({.8f, .2f}), probabilities({.5f, .5f})};
    const auto id = recorder.begin(0, 23, 4);
    recorder.sampling_request(0, "pending");
    recorder.finalize(0, {1, 0}, true, q);
    assert(!recorder.enabled() && !recorder.phase_begin(nullptr, {{id, 0}}));
    auto snapshot = *recorder.get(0);
    assert(recorder.consume(0, id, {1, 0}, false));
    assert(!recorder.consume(0, id, {1, 0}, false));
    assert(recorder.get(0)->proposal == snapshot.proposal && recorder.get(0)->dists.size() == 2);
    recorder.finish(0, id, 1);
    assert(!recorder.get(0)); // identity-only finish must release storage with no phase
    for (int mode = 0; mode < 4; ++mode) {
        const auto next = recorder.begin(0, 23, 4);
        assert(next > id);
        recorder.finalize(0, {1, 0}, true, q);
        if (mode == 0) { recorder.invalidate(0); }
        assert(!recorder.consume(0, mode == 1 ? id : next, mode == 2 ? llama_tokens{1, 1} : llama_tokens{1, 0}, mode == 0 || mode == 3));
        assert(!recorder.consume(0, next, {1, 0}, false));
        recorder.finish(0, next, 0);
    }
    // q records keep sampled token mass distinct from top-1 confidence.
    common_speculative_cycle_recorder logged(1, nullptr, true);
    const auto lid = logged.begin(0, 1, 0);
    auto phase = logged.phase_begin(nullptr, {{lid, 0}});
    logged.attach_draft(0, phase);
    logged.attempt(0, 1, .8f, true, 0, .2f);
    logged.attempt(0, 0, .5f, true, 0, .5f);
    logged.attempt(0, 0, .1f, false, 0, .1f);
    logged.phase_end(nullptr, phase, true);
    logged.finalize(0, {1}, true, {q[0]});
    const auto row = logged.row(*logged.get(0));
    assert(row["attempts"].size() == 3 && row["q_records"].size() == 1);
    assert(row["attempts"][0]["token"] != row["attempts"][0]["top1_token"]);
    assert(row["q_records"][0]["draft_token"] == 1 && row["q_records"][0]["proposal_id"] == lid);
    logged.finish(0, lid, 1);
}

static void environment() {
    common_speculative_sampling_config config;
    std::string error;
    assert(common_speculative_sampling_parse(nullptr, nullptr, config, error) && config.temperature == 0 && config.top_p == 1);
    assert(common_speculative_sampling_parse("0", "1", config, error) && config.temperature == 0);
    assert(common_speculative_sampling_parse("0.8", ".7", config, error) && config.temperature > 0 && config.top_p == .7f);
    for (const char * bad : {"", "nan", "inf", "-1", "0.8x", "1e999", "1e-999"}) {
        assert(!common_speculative_sampling_parse(bad, nullptr, config, error));
        assert(error.find("LLAMA_SPEC_SAMPLE_TEMP") != std::string::npos && error.find(bad) != std::string::npos);
    }
    for (const char * bad : {"", "nan", "inf", "-1", "0", "1.1", ".9x"}) {
        assert(!common_speculative_sampling_parse("0", bad, config, error));
        assert(error.find("LLAMA_SPEC_DRAFT_TOPP") != std::string::npos);
    }
}

// Regularized upper incomplete gamma: bounded series / continued fraction (double).
static double chi_square_tail(double x, int df) {
    assert(x >= 0 && df > 0);
    if (x == 0) { return 1; }
    const double a = .5 * df, z = .5 * x;
    constexpr double eps = 1e-13, tiny = 1e-300;
    const double factor = std::exp(a * std::log(z) - z - std::lgamma(a));
    if (z < a + 1) {
        double term = 1 / a, sum = term;
        for (int i = 1; i <= 10000; ++i) {
            term *= z / (a + i);
            sum += term;
            if (std::abs(term) <= std::abs(sum) * eps) { return std::clamp(1 - sum * factor, 0.0, 1.0); }
        }
    } else {
        double b = z + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
        for (int i = 1; i <= 10000; ++i) {
            const double an = -double(i) * (i - a);
            b += 2;
            d = an * d + b;
            if (std::abs(d) < tiny) { d = tiny; }
            c = b + an / c;
            if (std::abs(c) < tiny) { c = tiny; }
            d = 1 / d;
            const double delta = d * c;
            h *= delta;
            if (std::abs(delta - 1) < eps) { return std::clamp(factor * h, 0.0, 1.0); }
        }
    }
    assert(false && "chi-square tail did not converge");
    return 0;
}

static void statistics_self_check() {
    for (double x : {.01, .5, 2., 10., 80.}) {
        assert(std::abs(chi_square_tail(x, 1) - std::erfc(std::sqrt(x / 2))) < 1e-11);
        assert(std::abs(chi_square_tail(x, 2) - std::exp(-x / 2)) < 1e-11);
        assert(std::abs(chi_square_tail(x, 4) - std::exp(-x / 2) * (1 + x / 2)) < 1e-11);
    }
    assert(std::abs(chi_square_tail(9.487729036781154, 4) - .05) < 1e-11);
    assert(std::abs(chi_square_tail(124.34211340400407, 100) - .05) < 1e-10);
}

struct corpus_case {
    int vocab;
    int kind; // 0 conditional dense/sparse; 1 p=q; 2 disjoint; 3 near-zero residual; 4 rare bins
    std::string name() const { return "v" + std::to_string(vocab) + "_k" + std::to_string(kind); }
    distribution row(bool proposal, int parity) const {
        distribution result;
        double sum = 0;
        for (int id = 0; id < vocab; ++id) {
            double w = 1 + ((id * 7 + parity * 3) % 11);
            if (kind == 1 && id >= std::min(vocab, 10)) { w = 0; }
            if (kind == 2 && ((proposal && id != 0) || (!proposal && id == 0))) { w = 0; }
            if (kind == 3) { w = 1 + (proposal && id == 0 ? 1e-6 : 0); }
            if (kind == 4 && id == vocab - 1) { w = 1e-6; }
            if (proposal && id >= 10) { w = 0; }
            if (proposal && kind == 0) { w = id < 10 ? 1 + ((id * 3 + parity * 5) % 13) : 0; }
            result.push_back({id, 0, float(w)});
            sum += w;
        }
        for (auto & item : result) { item.p /= sum; }
        if (proposal) {
            result.erase(std::remove_if(result.begin(), result.end(), [](const auto & item) { return item.p == 0; }), result.end());
        }
        return result;
    }
};

static llama_token draw(const distribution & row, std::mt19937 & rng) {
    std::vector<double> weights;
    for (const auto & item : row) { weights.push_back(item.p); }
    return row[std::discrete_distribution<size_t>(weights.begin(), weights.end())(rng)].id;
}

// Each observation is four consecutive output tokens. The first three parity bits and
// the exact fourth ID form the bin: 8*v bins test conditional prefixes and token support.
// Episodes restart the conditional prefix, never RNG. Exactly 100000 output tokens/arm.
static std::vector<uint64_t> collect(llama_model & model, const corpus_case & c, bool enabled, uint32_t seed) {
    constexpr int outputs = 100000;
    auto smpl = sampler(model, seed);
    std::mt19937 draft_rng(seed ^ 0xa511e9b3u);
    std::vector<uint64_t> counts(8 * c.vocab);
    for (int episode = 0; episode < outputs / 4; ++episode) {
        llama_tokens emitted;
        while (emitted.size() < 4) {
            int parity = emitted.empty() ? 0 : emitted.back() % 2;
            fixture f;
            f.vocab = c.vocab;
            distributions q;
            for (int i = 0; i < 3; ++i) {
                f.p.push_back(c.row(false, parity));
                q.push_back(c.row(true, parity));
                f.draft.push_back(draw(q.back(), draft_rng));
                parity = f.draft.back() % 2;
            }
            f.p.push_back(c.row(false, parity));
            std::string reason;
            const auto result = enabled ? f.run(smpl.get(), q, reason) : f.source().exact(smpl.get(), false);
            if (enabled) { assert(reason == "distribution"); }
            for (auto id : result) {
                const auto target = c.row(false, emitted.empty() ? 0 : emitted.back() % 2);
                assert(id >= 0 && id < c.vocab && target[id].p > 0);
                emitted.push_back(id);
                if (emitted.size() == 4) { break; } // declared N counts outputs, never extra G+1 suffixes
            }
        }
        const int prefix = (emitted[0] % 2) * 4 + (emitted[1] % 2) * 2 + emitted[2] % 2;
        ++counts[prefix * c.vocab + emitted[3]];
    }
    return counts;
}

static std::vector<double> expected(const corpus_case & c) {
    std::vector<double> result(8 * c.vocab);
    auto parity_mass = [&](int previous, int next) {
        const auto row = c.row(false, previous);
        double sum = 0, mass = 0;
        for (auto item : row) { sum += item.p; if (item.id % 2 == next) { mass += item.p; } }
        return mass / sum;
    };
    for (int prefix = 0; prefix < 8; ++prefix) {
        const int a = prefix / 4, b = (prefix / 2) % 2, d = prefix % 2;
        const double prior = parity_mass(0, a) * parity_mass(a, b) * parity_mass(b, d);
        const auto row = c.row(false, d);
        double sum = 0;
        for (auto item : row) { sum += item.p; }
        for (auto item : row) { result[prefix * c.vocab + item.id] = prior * item.p / sum; }
    }
    return result;
}

struct hypothesis { std::string name; double p; };

static void compare(const corpus_case & c, const std::vector<uint64_t> & on,
        const std::vector<uint64_t> & off, std::vector<hypothesis> & family) {
    const auto truth = expected(c);
    // Preserve every unpooled bin, including zero support, for independent reanalysis.
    std::printf("P06 counts corpus=%s on=[", c.name().c_str());
    for (size_t i = 0; i < on.size(); ++i) { std::printf("%s%llu", i ? "," : "", (unsigned long long) on[i]); }
    std::printf("] off=[");
    for (size_t i = 0; i < off.size(); ++i) { std::printf("%s%llu", i ? "," : "", (unsigned long long) off[i]); }
    std::printf("] expected=[");
    for (size_t i = 0; i < truth.size(); ++i) { std::printf("%s%.17g", i ? "," : "", truth[i]); }
    std::printf("]\n");
    const double n = std::accumulate(on.begin(), on.end(), uint64_t(0));
    assert(n == 25000 && std::accumulate(off.begin(), off.end(), uint64_t(0)) == 25000);
    std::vector<int> order;
    for (size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == 0) { assert(on[i] == 0 && off[i] == 0); }
        else { order.push_back(i); }
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return truth[a] != truth[b] ? truth[a] < truth[b] : a < b; });
    // Predeclared deterministic pooling: ascending expected mass (ID tie-break), one other bin.
    // Also satisfy >=5 for the homogeneity test's pooled observed expectation in each arm.
    size_t pooled = 0;
    double other_expected = 0, other_on = 0, other_off = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        const int id = order[i];
        if (truth[id] * n < 5 || (on[id] + off[id]) / 2.0 < 5) { pooled = i + 1; }
    }
    for (size_t i = 0; i < pooled; ++i) {
        other_expected += truth[order[i]] * n;
        other_on += on[order[i]];
        other_off += off[order[i]];
    }
    while (pooled && pooled < order.size() && (other_expected < 5 || (other_on + other_off) / 2 < 5)) {
        const int id = order[pooled++];
        other_expected += truth[id] * n; other_on += on[id]; other_off += off[id];
    }
    double gof_on = 0, gof_off = 0, homogeneous = 0;
    int bins = 0;
    auto add = [&](double e, double a, double b) {
        assert(e >= 5 && (a + b) / 2 >= 5);
        gof_on += (a-e)*(a-e)/e; gof_off += (b-e)*(b-e)/e;
        homogeneous += (a-b)*(a-b)/(a+b);
        ++bins;
    };
    if (pooled) { add(other_expected, other_on, other_off); }
    for (size_t i = pooled; i < order.size(); ++i) { int id = order[i]; add(truth[id]*n, on[id], off[id]); }
    // Print every mapping, including identity bins, so the acceptance artifact is reproducible.
    std::printf("P06 corpus=%s observations=25000 outputs_per_arm=100000 pooled_other=[", c.name().c_str());
    for (size_t i = 0; i < pooled; ++i) { std::printf("%s%d", i ? "," : "", order[i]); }
    std::printf("] individual=[");
    for (size_t i = pooled; i < order.size(); ++i) { std::printf("%s%d", i != pooled ? "," : "", order[i]); }
    double tv = 0, max_error = 0;
    for (size_t i = 0; i < truth.size(); ++i) {
        const double err = std::abs(on[i]/n - truth[i]); tv += .5*err; max_error = std::max(max_error, err);
    }
    std::printf("] TV=%.9g maxerror=%.9g\n", tv, max_error);
    for (const auto & test : std::vector<std::pair<const char *, double>>{{"on_p", gof_on}, {"off_p", gof_off}, {"on_off", homogeneous}}) {
        family.push_back({c.name() + "/" + test.first, bins > 1 ? chi_square_tail(test.second, bins - 1) : 1});
    }
}

static void corpus(llama_model & model) {
    std::vector<corpus_case> cases;
    for (int vocab = 2; vocab <= 64; ++vocab) { cases.push_back({vocab, 0}); }
    for (int vocab : {2, 7, 10, 64}) {
        for (int kind : {1, 2, 4}) { cases.push_back({vocab, kind}); }
    }
    cases.push_back({2, 3}); cases.push_back({7, 3});
    std::vector<hypothesis> family;
    for (size_t i = 0; i < cases.size(); ++i) {
        // Fixed independent arm seeds; no retries, seed search, or omission of failed cases.
        const uint32_t on_seed = 0x426b1021u + uint32_t(i) * 7919u;
        const uint32_t off_seed = 0xd31780a5u + uint32_t(i) * 104729u;
        std::printf("P06 seeds %s on=%u off=%u\n", cases[i].name().c_str(), on_seed, off_seed);
        const auto on = collect(model, cases[i], true, on_seed);
        const auto off = collect(model, cases[i], false, off_seed);
        compare(cases[i], on, off, family);
    }
    std::sort(family.begin(), family.end(), [](const auto & a, const auto & b) { return a.p < b.p; });
    bool passed = true;
    std::printf("P06 family=conditional-prefix-v1 cases=%zu tests=%zu alpha=0.01\n", cases.size(), family.size());
    bool testing = true;
    for (size_t i = 0; i < family.size(); ++i) {
        const double threshold = .01 / (family.size() - i);
        std::printf("P06 Holm %s p=%.17g threshold=%.17g\n", family[i].name.c_str(), family[i].p, threshold);
        if (testing && family[i].p <= threshold) { passed = false; }
        else { testing = false; }
    }
    assert(passed && "predeclared Pearson/Holm family rejected distribution parity");
}

void test_speculative_sampling() {
    // The base is abstract; the factory provides an unloaded concrete owner for sampler-only tests.
    llama_model_ptr model_owner(llama_model_create(LLM_ARCH_LLAMA, llama_model_default_params()));
    assert(model_owner);
    llama_model & model = *model_owner;
    environment();
    lifecycle();
    canonical_proposal_draw();
    transactions(model);
    checkpoint_repair(model);
    statistics_self_check();
    corpus(model);
    std::puts("PASS: P06 transactional sampling, lifecycle, EOG prefix and Pearson/Holm corpus");
}
