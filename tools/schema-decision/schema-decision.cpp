#include "schema-decision.h"

#include "chat.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>

namespace schema_decision {

namespace {

// prompt tokens re-tokenized together with each value; see plan_tokens
constexpr size_t max_tail_tokens = 16;
constexpr size_t window_tokens   = 8;

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

// owns a llama_batch for the duration of one decode
struct batch_holder {
    llama_batch batch;

    explicit batch_holder(int32_t n_tokens) : batch(llama_batch_init(n_tokens, 0, 1)) {}
    ~batch_holder() { llama_batch_free(batch); }
    batch_holder(const batch_holder &)             = delete;
    batch_holder & operator=(const batch_holder &) = delete;
};

failure fail_with(error_code code, std::string message) {
    return { code, std::move(message) };
}

failure decode_error(int rc, const char * what) {
    if (rc == 1) {
        return fail_with(error_code::no_memory, std::string("no free KV cache space for the decision ") + what);
    }
    if (rc == 2) {
        return fail_with(error_code::cancelled, std::string("the decision ") + what + " decode was aborted");
    }
    return fail_with(error_code::internal, std::string("llama_decode failed on the decision ") + what + " (" + std::to_string(rc) + ")");
}

failure cancelled() {
    return fail_with(error_code::cancelled, "the decision was cancelled");
}

size_t common_prefix(const tokens_t & a, const tokens_t & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

bool aborted(const options & opt) {
    return opt.should_abort && opt.should_abort();
}

// One field's token trie. Paths are prefix-free (checked when the field is built), so a path
// never ends while another active path continues.
struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active; // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix; // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // greedy: follow single-path edges for free; return the options at the next divergence
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth < paths[i].size() && std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            if (opts.empty()) {
                break; // unreachable for prefix-free paths
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (depth < paths[i].size() && paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth < paths[i].size() && std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (depth < paths[i].size() && paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    size_t max_path() const {
        size_t n = 0;
        for (const auto & p : paths) {
            n = std::max(n, p.size());
        }
        return n;
    }

    size_t tree_rows() const {
        size_t n = 0;
        for (const auto & p : node_prefix) {
            n += suffix.size() + p.size();
        }
        return n;
    }

    // greedy: the most token rows the walk of options() and select() can decode, over every
    // value it can end on (each divergence replays the suffix and the chosen tokens)
    size_t greedy_max_rows() const {
        std::vector<int> all;
        for (int i = 0; i < (int) paths.size(); ++i) {
            all.push_back(i);
        }
        return greedy_max_rows(std::move(all), 0);
    }

    size_t greedy_max_rows(std::vector<int> act, size_t depth) const {
        while (act.size() > 1) {
            std::vector<std::pair<llama_token, std::vector<int>>> next;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    continue;
                }
                const llama_token t  = paths[i][depth];
                auto              it = std::find_if(next.begin(), next.end(), [t](const auto & g) { return g.first == t; });
                if (it == next.end()) {
                    next.push_back({ t, { i } });
                } else {
                    it->second.push_back(i);
                }
            }
            if (next.size() > 1) {
                size_t worst = 0;
                for (auto & g : next) {
                    worst = std::max(worst, greedy_max_rows(std::move(g.second), depth + 1));
                }
                return suffix.size() + depth + worst;
            }
            if (next.empty()) {
                break;
            }
            act = std::move(next[0].second);
            ++depth;
        }
        return 0;
    }

    // Constrained distribution: log-softmax at each node over its allowed tokens, summed along
    // every candidate path, normalised over candidates. node_scores holds finite logits.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<double>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp((double) x - mx);
            }
            const double lz = mx + std::log(z);
            std::vector<double> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<double> path_lp(paths.size(), 0.0);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size() && n < node_logp.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it != opts.end()) {
                    path_lp[i] += node_logp[n][it - opts.begin()];
                }
            }
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        double z = 0;
        for (double x : path_lp) {
            z += std::exp(x - path_lp[best]);
        }
        probs.clear();
        for (double x : path_lp) {
            probs.push_back((float) (std::exp(x - path_lp[best]) / z));
        }
        winner       = best;
        path_score   = probs[best];
        scored_nodes = (int) node_prefix.size();
    }
};

const char * mode_names[] = { "auto", "tree", "greedy" };

} // namespace

const char * mode_name(score_mode mode) {
    return mode_names[(int) mode];
}

// ---------------------------------------------------------------- token plan

namespace {

tokens_t tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special) {
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

bool is_partition_token(const llama_vocab * vocab, llama_token t) {
    // the tokenizer splits the text at these before any merge, so no merge crosses them
    return (llama_vocab_get_attr(vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED)) != 0;
}

} // namespace

namespace {

bool starts_with(const tokens_t & a, const tokens_t & prefix) {
    return a.size() >= prefix.size() && std::equal(prefix.begin(), prefix.end(), a.begin());
}

// How the end of one context continues into the field values. The trunk keeps the context's
// tokens up to the cut; every value continues with tokenize(left + rest + suffix + value + "\n")
// minus the tokens of left, where left is the text of a few tokens before the cut (it only
// steers the tokenizer) and rest is the text after the cut.
struct cut_plan {
    size_t      cut  = 0;
    std::string left;
    std::string rest;
    size_t      n_left = 0; // tokens of left
};

// the fields' tokens after a cut; false when a value merges into the left context
bool tokenize_fields(const llama_vocab * vocab, const cut_plan & cp, const std::vector<field_input> & inputs,
                     const tokens_t & left_tokens, std::vector<std::vector<tokens_t>> & seqs) {
    seqs.clear();
    for (const auto & in : inputs) {
        std::vector<tokens_t> field;
        for (const auto & c : in.candidates) {
            tokens_t t = tokenize_text(vocab, cp.left + cp.rest + in.suffix + c + "\n", false);
            if (!starts_with(t, left_tokens) || t.size() == left_tokens.size()) {
                return false;
            }
            field.emplace_back(t.begin() + left_tokens.size(), t.end());
        }
        seqs.push_back(std::move(field));
    }
    return true;
}

} // namespace

// Splits every context's tokens into the cached prefix and a trunk, and tokenizes each field
// value together with the end of the context. Tokenizing the prompt and a value apart gives
// tokens that the joint text does not tokenize to on many vocabularies: '{\n' + '  "a"' merges
// on GPT-2 style BPE (SmolLM2, Starcoder), '{' + '\n  "a"' on Llama 3 and Qwen, and SPM adds a
// space prefix to every separately tokenized string.
//
// Preferred cut: after the last special token near the end of the context. The tokenizer
// partitions the text at special tokens before any merge, so the values tokenized from there
// are exactly the joint tokenization. Otherwise the cut moves back from the end of the context
// until the values tokenize after a few tokens of left context without changing it, which is
// local, not proven: tests/test-schema-decision.cpp checks it against the joint tokenization on
// the vocabularies in models/.
outcome<token_plan> plan_tokens(const llama_vocab * vocab, const std::string & shared_text,
                                const std::vector<std::string> & contexts, const std::vector<field_input> & inputs) {
    outcome<token_plan> out;
    token_plan & plan = out.value;

    const tokens_t shared = tokenize_text(vocab, shared_text, true);
    std::vector<tokens_t> full;
    size_t n_shared = shared.size();
    for (const auto & text : contexts) {
        full.push_back(tokenize_text(vocab, shared_text + text, true));
        n_shared = std::min(n_shared, common_prefix(shared, full.back()));
    }
    plan.shared.assign(shared.begin(), shared.begin() + n_shared);

    std::map<std::string, int> tail_index; // left + '\0' + rest -> tail
    for (size_t i = 0; i < full.size(); ++i) {
        const tokens_t & f = full[i];
        if (f.size() <= n_shared) {
            out.error = "context " + std::to_string(i) + " adds no tokens to the prompt";
            return out;
        }
        std::vector<cut_plan> candidates;
        for (size_t k = f.size(); k > n_shared && f.size() - k <= max_tail_tokens; --k) {
            if (is_partition_token(vocab, f[k - 1])) {
                cut_plan cp;
                cp.cut  = k;
                cp.rest = common_detokenize(vocab, tokens_t(f.begin() + k, f.end()), false);
                if (tokenize_text(vocab, cp.rest, false) == tokens_t(f.begin() + k, f.end())) {
                    candidates.push_back(cp);
                }
                break;
            }
        }
        for (size_t c = f.size(); c > n_shared && f.size() - c <= window_tokens; --c) {
            cut_plan cp;
            cp.cut  = c;
            cp.left = common_detokenize(vocab, tokens_t(f.begin() + (c > window_tokens ? c - window_tokens : 0), f.begin() + c), true);
            cp.rest = common_detokenize(vocab, tokens_t(f.begin() + c, f.end()), true);
            candidates.push_back(cp);
        }

        bool found = false;
        std::vector<std::vector<tokens_t>> seqs;
        cut_plan chosen;
        for (auto & cp : candidates) {
            const tokens_t left_tokens = cp.left.empty() ? tokens_t() : tokenize_text(vocab, cp.left, false);
            // the token before the cut must come out of the left context unchanged
            if (!cp.left.empty() && (left_tokens.empty() || left_tokens.back() != f[cp.cut - 1])) {
                continue;
            }
            cp.n_left = left_tokens.size();
            // the fields' tokens depend only on left and rest: a known tail already passed
            if (tail_index.count(cp.left + std::string(1, '\0') + cp.rest) != 0 ||
                tokenize_fields(vocab, cp, inputs, left_tokens, seqs)) {
                chosen = cp;
                found  = true;
                break;
            }
        }
        if (!found) {
            out.error = "no stable token boundary at the end of context " + std::to_string(i) +
                        ": the field values merge with the prompt for every cut tried";
            return out;
        }
        plan.trunks.emplace_back(f.begin() + n_shared, f.begin() + chosen.cut);

        const std::string key = chosen.left + std::string(1, '\0') + chosen.rest;
        auto it = tail_index.find(key);
        if (it != tail_index.end()) {
            plan.tail_of.push_back(it->second);
            continue;
        }
        tail_index.emplace(key, (int) plan.tails.size());
        plan.tail_of.push_back((int) plan.tails.size());

        std::vector<field_tokens> fields;
        for (size_t fi = 0; fi < inputs.size(); ++fi) {
            const auto & fs = seqs[fi];
            // split at the longest token prefix shared by every value, leaving each path a token
            size_t common = fs[0].size();
            for (const auto & sq : fs) {
                common = std::min({ common, common_prefix(sq, fs[0]), sq.size() - 1 });
            }
            field_tokens ft;
            ft.suffix.assign(fs[0].begin(), fs[0].begin() + common);
            for (const auto & sq : fs) {
                ft.paths.emplace_back(sq.begin() + common, sq.end());
            }
            if (ft.suffix.empty()) {
                out.error = "field " + std::to_string(fi) + " has no token before its values";
                return out;
            }
            for (size_t a2 = 0; a2 < ft.paths.size(); ++a2) {
                if (ft.paths[a2].size() > max_path_tokens) {
                    out.error = "field " + std::to_string(fi) + ": value " + std::to_string(a2) + " tokenizes to " +
                                std::to_string(ft.paths[a2].size()) + " tokens, more than " + std::to_string(max_path_tokens);
                    return out;
                }
                for (size_t b2 = 0; b2 < a2; ++b2) {
                    const size_t m = std::min(ft.paths[a2].size(), ft.paths[b2].size());
                    if (std::equal(ft.paths[a2].begin(), ft.paths[a2].begin() + m, ft.paths[b2].begin())) {
                        out.error = "field " + std::to_string(fi) + ": values " + std::to_string(b2) + " and " +
                                    std::to_string(a2) + " tokenize to colliding paths";
                        return out;
                    }
                }
            }
            fields.push_back(std::move(ft));
        }
        plan.tails.push_back(std::move(fields));
    }
    return out;
}

// ---------------------------------------------------------------- engine

engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1),
      pad_branches(llama_model_is_recurrent(llama_get_model(ctx)) || llama_model_is_hybrid(llama_get_model(ctx))) {}

outcome<std::unique_ptr<engine>> engine::create(llama_context * ctx, llama_seq_id seq_base, int n_seqs) {
    outcome<std::unique_ptr<engine>> out;
    out.code = error_code::internal; // a misconfigured caller, not a request
    if (ctx == nullptr || llama_get_memory(ctx) == nullptr) {
        out.error = "a decision engine needs a context with memory";
        return out;
    }
    if (n_seqs < 3) {
        out.error = "a decision engine needs at least 3 sequences (cached prefix, trunk, branch), got " + std::to_string(n_seqs);
        return out;
    }
    const int64_t n_seq_max = llama_n_seq_max(ctx);
    if (seq_base < 0 || (int64_t) seq_base + n_seqs > n_seq_max) {
        out.error = "decision sequences [" + std::to_string(seq_base) + ", " + std::to_string((int64_t) seq_base + n_seqs) +
                    ") exceed the context's " + std::to_string(n_seq_max) + " sequences";
        return out;
    }
    out.value.reset(new engine(ctx, seq_base, n_seqs));
    return out;
}

void engine::clear_pool() {
    for (llama_seq_id s = seq_pool; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
}

void engine::reset() {
    llama_memory_seq_rm(mem, seq_snap, -1, -1);
    clear_pool();
    cached.clear();
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
failure engine::decode_parts(const std::vector<prompt_part> & parts, const options & opt, int & n_decodes) {
    const int    n_batch = (int) llama_n_batch(ctx);
    batch_holder b(n_batch);
    const auto flush = [&]() -> failure {
        if (b.batch.n_tokens == 0) {
            return {};
        }
        if (aborted(opt)) {
            return cancelled();
        }
        const int rc = llama_decode(ctx, b.batch);
        ++n_decodes;
        common_batch_clear(b.batch);
        return rc == 0 ? failure{} : decode_error(rc, "prompt");
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (b.batch.n_tokens == n_batch) {
                failure err = flush();
                if (err) {
                    return err;
                }
            }
            common_batch_add(b.batch, (*p.toks)[i], p.pos0 + (llama_pos) i, { p.seq }, false);
        }
    }
    return flush();
}

// Restore (or build) the cached static prefix on seq_snap. Only this engine's own sequences are
// touched, so it can share a context with other users (e.g. server slots).
failure engine::prepare_prefix(const tokens_t & shared, const options & opt, bool & hit, int & n_decodes) {
    // pos_max alone: recurrent memory reports the state's position as pos_min too
    hit = opt.allow_cache && !shared.empty() && shared == cached &&
          llama_memory_seq_pos_max(mem, seq_snap) == (llama_pos) cached.size() - 1;
    if (hit) {
        return {};
    }
    reset();
    if (shared.empty()) {
        return {};
    }
    failure err = decode_parts({ { &shared, 0, seq_snap } }, opt, n_decodes);
    if (err) {
        reset();
        return err;
    }
    cached = shared;
    return {};
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
//
// Recurrent layers take a ubatch only when every sequence in it holds the same number of tokens
// (split_equal), so on recurrent/hybrid models uneven branches would split one decode into many
// passes. There each group is right-padded to its longest branch (longest first, so padding stays
// small) and bounded by n_ubatch, so that it is one ubatch. Every layer is causal, so the logits
// read at a branch's last real token do not see the padding, and the padded cells are removed
// with the branch.
failure engine::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free,
                                   const options & opt, std::vector<std::vector<float>> & scores, int & n_decodes) {
    scores.assign(branches.size(), {});
    if (n_free < 1) {
        return fail_with(error_code::internal, "no free sequence for the decision branches");
    }
    std::vector<size_t> order(branches.size());
    for (size_t b = 0; b < order.size(); ++b) {
        order[b] = b;
    }
    if (pad_branches) {
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return branches[a].toks.size() > branches[b].toks.size();
        });
    }
    const int max_rows = (int) (pad_branches ? llama_n_ubatch(ctx) : llama_n_batch(ctx));
    size_t start = 0;
    while (start < order.size()) {
        const int width = (int) branches[order[start]].toks.size(); // padded length of this group
        size_t end  = start;
        int    rows = 0;
        while (end < order.size() && (int) (end - start) < n_free) {
            const int need = pad_branches ? width : (int) branches[order[end]].toks.size();
            if (rows + need > max_rows) {
                break;
            }
            rows += need;
            ++end;
        }
        if (end == start) {
            return fail_with(error_code::invalid_request, "a decision branch of " + std::to_string(width) + " tokens exceeds the " +
                             (pad_branches ? "ubatch" : "batch") + " size " + std::to_string(max_rows));
        }
        const auto clear_group = [&]() {
            for (size_t k = start; k < end; ++k) {
                llama_memory_seq_rm(mem, first + (llama_seq_id) (k - start), -1, -1);
            }
        };
        batch_holder     b(rows);
        std::vector<int> out_idx;
        for (size_t k = start; k < end; ++k) {
            const auto &       br  = branches[order[k]];
            const llama_seq_id seq = first + (llama_seq_id) (k - start);
            llama_memory_seq_rm(mem, seq, -1, -1);
            llama_memory_seq_cp(mem, br.trunk, seq, -1, -1);
            const int n = pad_branches ? width : (int) br.toks.size();
            for (int i = 0; i < n; ++i) {
                const bool last = i + 1 == (int) br.toks.size();
                if (last) {
                    out_idx.push_back(b.batch.n_tokens);
                }
                common_batch_add(b.batch, br.toks[std::min(i, (int) br.toks.size() - 1)], br.pos0 + (llama_pos) i, { seq }, last);
            }
        }
        if (aborted(opt)) {
            clear_group();
            return cancelled();
        }
        const int rc = llama_decode(ctx, b.batch);
        ++n_decodes;
        if (rc != 0) {
            clear_group();
            return decode_error(rc, "branches");
        }
        for (size_t k = start; k < end; ++k) {
            const float * logits = llama_get_logits_ith(ctx, out_idx[k - start]);
            if (logits == nullptr) {
                clear_group();
                return fail_with(error_code::internal, "no logits for a decision branch");
            }
            for (llama_token t : branches[order[k]].cands) {
                if (!std::isfinite(logits[t])) {
                    clear_group();
                    return fail_with(error_code::internal, "non-finite logit " + std::to_string(logits[t]) + " for token " +
                                     std::to_string(t) + " in a decision branch");
                }
                scores[order[k]].push_back(logits[t]);
            }
        }
        clear_group();
        start = end;
    }
    return {};
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt, failure & error) {
    error = {};
    std::string err;
    error_code  code = error_code::invalid_request;
    try {
        if (contexts.empty() || contexts.size() > max_contexts) {
            err = "a decision needs 1-" + std::to_string(max_contexts) + " contexts";
        } else if (inputs.empty() || inputs.size() > max_fields) {
            err = "a decision needs 1-" + std::to_string(max_fields) + " fields";
        } else {
            for (const auto & in : inputs) {
                if (in.candidates.empty() || in.candidates.size() > max_values) {
                    err = "each field needs 1-" + std::to_string(max_values) + " allowed values";
                    break;
                }
            }
        }
        if (err.empty()) {
            auto planned = plan_tokens(vocab, shared_text, contexts, inputs);
            if (planned.ok()) {
                return decide_plan(planned.value, opt, error);
            }
            err  = planned.error;
            code = planned.code;
        }
    } catch (const std::exception & e) {
        err  = std::string("unexpected error: ") + e.what();
        code = error_code::internal;
    }
    error = { code, err };
    LOG_ERR("%s: %s (contexts %zu, fields %zu, shared %zu bytes)\n", __func__, err.c_str(), contexts.size(),
            inputs.size(), shared_text.size());
    return {};
}

batch_result engine::decide_plan(const token_plan & plan, const options & opt, failure & error) {
    batch_result out;
    error = {};
    const auto t_start = std::chrono::steady_clock::now();

    // trunks and branches never outlive a call; the cached prefix survives only a success
    struct pool_guard {
        engine & e;
        bool     ok = false;
        ~pool_guard() {
            if (ok) {
                e.clear_pool();
            } else {
                e.reset();
            }
        }
    } guard{ *this };

    const size_t n_ctx_items = plan.trunks.size();
    const auto fail = [&](failure f) {
        error = std::move(f);
        LOG_ERR("%s: %s (contexts %zu, tails %zu, mode %s, tree_max %d, cache %d, prefix %zu tokens)\n", __func__,
                error.message.c_str(), n_ctx_items, plan.tails.size(), mode_name(opt.mode), opt.tree_max,
                opt.allow_cache ? 1 : 0, plan.shared.size());
        return batch_result{};
    };
    const auto invalid = [&](std::string msg) {
        return fail(fail_with(error_code::invalid_request, std::move(msg)));
    };

    size_t n_fields = 0;
    try {
        if (n_ctx_items == 0 || n_ctx_items > max_contexts || plan.tail_of.size() != n_ctx_items) {
            return invalid("a decision needs 1-" + std::to_string(max_contexts) + " contexts, each with a tail");
        }
        if (opt.tree_max < 0 || opt.tree_max > max_tree_max) {
            return invalid("tree_max must be 0-" + std::to_string(max_tree_max));
        }
        const llama_token n_vocab = llama_vocab_n_tokens(vocab);
        const auto bad_tokens = [&](const tokens_t & toks) {
            return std::any_of(toks.begin(), toks.end(), [&](llama_token t) { return t < 0 || t >= n_vocab; });
        };
        if (bad_tokens(plan.shared)) {
            return invalid("the cached prefix holds a token outside the vocabulary");
        }
        for (size_t i = 0; i < n_ctx_items; ++i) {
            if (plan.tail_of[i] < 0 || (size_t) plan.tail_of[i] >= plan.tails.size() || bad_tokens(plan.trunks[i])) {
                return invalid("context " + std::to_string(i) + " has an invalid tail or token");
            }
        }
        for (const auto & tf : plan.tails) {
            if (tf.empty() || tf.size() > max_fields || (n_fields != 0 && tf.size() != n_fields)) {
                return invalid("every tail needs the same 1-" + std::to_string(max_fields) + " fields");
            }
            n_fields = tf.size();
            for (size_t f = 0; f < tf.size(); ++f) {
                const auto & ft = tf[f];
                if (ft.suffix.empty() || bad_tokens(ft.suffix) || ft.paths.empty() || ft.paths.size() > max_values) {
                    return invalid("field " + std::to_string(f) + " needs a suffix and 1-" + std::to_string(max_values) + " values");
                }
                for (size_t a = 0; a < ft.paths.size(); ++a) {
                    if (ft.paths[a].empty() || ft.paths[a].size() > max_path_tokens || bad_tokens(ft.paths[a])) {
                        return invalid("field " + std::to_string(f) + ": value " + std::to_string(a) + " needs 1-" +
                                    std::to_string(max_path_tokens) + " valid tokens");
                    }
                    for (size_t b = 0; b < a; ++b) {
                        const size_t m = std::min(ft.paths[a].size(), ft.paths[b].size());
                        if (std::equal(ft.paths[a].begin(), ft.paths[a].begin() + m, ft.paths[b].begin())) {
                            return invalid("field " + std::to_string(f) + ": values " + std::to_string(b) + " and " +
                                        std::to_string(a) + " have colliding token paths");
                        }
                    }
                }
            }
        }

        // per tail: the fields as tries; per context the round-1 branches and rows
        std::vector<std::vector<decision_field>> tail_fields;
        std::vector<size_t> tail_branches, tail_max_branch;
        for (const auto & tf : plan.tails) {
            std::vector<decision_field> fields;
            size_t rows = 0, branches = 0, max_branch = 0;
            for (size_t f = 0; f < tf.size(); ++f) {
                decision_field field(tf[f].suffix, tf[f].paths);
                field.build_nodes();
                const size_t n = tf[f].paths.size();
                field.use_tree = opt.mode == score_mode::tree ? true : opt.mode == score_mode::greedy ? false : n <= (size_t) opt.tree_max;
                rows      += field.use_tree ? field.tree_rows() : field.greedy_max_rows();
                branches  += field.use_tree ? field.node_prefix.size() : (n > 1 ? 1 : 0);
                max_branch = std::max(max_branch, field.suffix.size() + field.max_path());
                fields.push_back(std::move(field));
            }
            if (rows > max_rows_per_context) {
                return invalid("the schema scores up to " + std::to_string(rows) + " token rows per context, more than " +
                            std::to_string(max_rows_per_context));
            }
            tail_fields.push_back(std::move(fields));
            tail_branches.push_back(branches);
            tail_max_branch.push_back(max_branch);
        }

        const size_t n_ctx_seq = llama_n_ctx_seq(ctx);
        size_t branches = 0;
        for (size_t i = 0; i < n_ctx_items; ++i) {
            const size_t t    = (size_t) plan.tail_of[i];
            const size_t need = plan.shared.size() + plan.trunks[i].size() + tail_max_branch[t];
            if (need > n_ctx_seq) {
                return invalid("context " + std::to_string(i) + " needs " + std::to_string(need) +
                            " positions with its longest branch, the context holds " + std::to_string(n_ctx_seq));
            }
            branches = std::max(branches, tail_branches[t]);
        }

        out.shared_tokens = plan.shared.size();
        out.items.resize(n_ctx_items);

        const auto t0 = std::chrono::steady_clock::now();
        failure err = prepare_prefix(plan.shared, opt, out.cache_hit, out.decodes);
        out.prefill_ms += ms_since(t0);
        if (err) {
            return fail(err);
        }

        // each context in a group holds one trunk sequence; the rest of the pool scores branches
        const size_t per_group = std::clamp<size_t>((size_t) n_pool / (1 + std::max<size_t>(branches, 1)), 1, n_ctx_items);
        for (size_t g0 = 0; g0 < n_ctx_items; g0 += per_group) {
            const size_t n_group = std::min(per_group, n_ctx_items - g0);

            const auto tp = std::chrono::steady_clock::now();
            std::vector<prompt_part> parts;
            for (size_t i = 0; i < n_group; ++i) {
                const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                llama_memory_seq_rm(mem, trunk, -1, -1);
                if (!plan.shared.empty()) {
                    llama_memory_seq_cp(mem, seq_snap, trunk, -1, -1);
                }
                parts.push_back({ &plan.trunks[g0 + i], (llama_pos) plan.shared.size(), trunk });
            }
            err = decode_parts(parts, opt, out.decodes);
            if (err) {
                return fail(err);
            }
            llama_synchronize(ctx); // llama_decode is asynchronous: bill the prefill to the prefill time
            out.prefill_ms += ms_since(tp);

            // round 1 carries every tree node and each greedy field's first step; later rounds
            // only continue greedy fields that are still open
            const auto ts = std::chrono::steady_clock::now();
            std::vector<std::vector<decision_field>> state;
            for (size_t i = 0; i < n_group; ++i) {
                state.push_back(tail_fields[plan.tail_of[g0 + i]]);
            }
            std::vector<size_t> scored(n_group, 0); // token rows submitted per context
            bool first = true;
            while (true) {
                std::vector<branch> todo;
                std::vector<std::pair<size_t, size_t>> owner; // (context in group, field)
                for (size_t i = 0; i < n_group; ++i) {
                    const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                    const llama_pos    pos0  = (llama_pos) (plan.shared.size() + plan.trunks[g0 + i].size());
                    for (size_t f = 0; f < state[i].size(); ++f) {
                        auto & fd = state[i][f];
                        if (fd.use_tree) {
                            if (first) {
                                for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                                    tokens_t ids = fd.suffix;
                                    ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                                    scored[i] += ids.size();
                                    todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                                    owner.push_back({ i, f });
                                }
                            }
                        } else {
                            tokens_t opts = fd.options();
                            if (!opts.empty()) {
                                tokens_t ids = fd.suffix;
                                ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                                scored[i] += ids.size();
                                todo.push_back({ trunk, pos0, ids, opts });
                                owner.push_back({ i, f });
                            }
                        }
                    }
                }
                if (todo.empty()) {
                    break;
                }
                std::vector<std::vector<float>> scores;
                err = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group, opt, scores, out.decodes);
                if (err) {
                    return fail(err);
                }
                out.rounds += 1;
                std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group);
                for (size_t i = 0; i < n_group; ++i) {
                    tree_scores[i].resize(state[i].size());
                }
                for (size_t row = 0; row < owner.size(); ++row) {
                    const auto [i, f] = owner[row];
                    auto & fd = state[i][f];
                    if (fd.use_tree) {
                        tree_scores[i][f].push_back(scores[row]);
                    } else {
                        const auto & s    = scores[row];
                        const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                        double denom = 0;
                        for (float x : s) {
                            denom += std::exp((double) x - s[best]);
                        }
                        fd.select(todo[row].cands[best], (float) (1.0 / denom));
                    }
                }
                if (first) {
                    for (size_t i = 0; i < n_group; ++i) {
                        for (size_t f = 0; f < state[i].size(); ++f) {
                            if (state[i][f].use_tree) {
                                state[i][f].finish_tree(tree_scores[i][f]);
                            }
                        }
                    }
                }
                first = false;
            }
            for (size_t i = 0; i < n_group; ++i) {
                llama_memory_seq_rm(mem, seq_pool + (llama_seq_id) i, -1, -1);
                result & r = out.items[g0 + i];
                r.context_tokens = plan.trunks[g0 + i].size();
                r.rows           = (int) scored[i];
                out.rows        += (int) scored[i];
                for (auto & fd : state[i]) {
                    if (fd.use_tree && fd.probs.empty()) {
                        fd.finish_tree({});
                    }
                    r.fields.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
                }
            }
            out.scoring_ms += ms_since(ts);
        }
    } catch (const std::exception & e) {
        return fail(fail_with(error_code::internal, std::string("unexpected error: ") + e.what()));
    }

    guard.ok = true;
    LOG_INF("%s: %zu context(s), %zu field(s): %d rows in %d round(s), %d decode(s), cached prefix %zu tokens (%s), "
            "prefill %.1f ms, scoring %.1f ms, total %.1f ms\n", __func__, n_ctx_items, n_fields, out.rows,
            out.rounds, out.decodes, out.shared_tokens, out.cache_hit ? "hit" : "miss", out.prefill_ms, out.scoring_ms,
            ms_since(t_start));
    return out;
}

// ---------------------------------------------------------------- schema compiler

namespace {

// decimals that print x so that it reads back as the same double; -1 if more than 9. A relative
// tolerance would let 999999999999999.5 print as 1000000000000000.
int decimal_places(double x) {
    for (int k = 0; k <= 9; ++k) {
        char buf[64];
        const int len = std::snprintf(buf, sizeof(buf), "%.*f", k, x);
        if (len > 0 && len < (int) sizeof(buf) && std::strtod(buf, nullptr) == x) {
            return k;
        }
    }
    return -1;
}

std::string json_text(const std::string & s) {
    return common_json::make(s).dump();
}

// a JSON integer as long long; false when it is not an integer or out of range
bool read_integer(const common_json & j, long long & v) {
    if (!j.is_number_integer()) {
        return false;
    }
    v = j.get<long long>();
    const std::string text = j.dump();
    return (v < 0) == (!text.empty() && text[0] == '-'); // a wrapped unsigned value flips the sign
}

bool read_number(const common_json & j, double & v) {
    if (!j.is_number()) {
        return false;
    }
    v = j.get<double>();
    return std::isfinite(v);
}

// keys a field spec may carry; anything else (exclusiveMinimum, const, pattern, anyOf, ...)
// would restrict the values in a way the compiler does not implement
bool known_field_key(const std::string & key, bool json_schema) {
    static const char * common_keys[] = { "type", "description", "enum", "minimum", "maximum", "aggregate",
                                          "x-aggregate", "title", "$comment" };
    for (const char * k : common_keys) {
        if (key == k) {
            return true;
        }
    }
    if (json_schema) {
        return key == "multipleOf" || key == "default" || key == "examples";
    }
    return key == "choices" || key == "step";
}

outcome<field_spec> make_field(const std::string & name, const std::string & type, const std::string & description,
                               const common_json & spec, bool json_schema) {
    outcome<field_spec> out;
    field_spec & f = out.value;
    f.name        = name;
    f.description = description;
    const auto bad = [&](const std::string & msg) {
        out.error = "field \"" + name + "\": " + msg;
        return out;
    };
    for (const auto & e : spec.items()) {
        if (!known_field_key(e.key(), json_schema)) {
            return bad("unsupported keyword \"" + e.key() + "\"");
        }
    }
    // a value constraint the type does not implement would let values outside the schema through
    // ({"type": "boolean", "enum": [true]} could return false); integers report a step themselves
    const bool known_type = type == "boolean" || type == "enum" || type == "string" || type == "integer" || type == "number";
    for (const char * key : { "enum", "choices", "minimum", "maximum", "step", "multipleOf" }) {
        const std::string k = key;
        const bool applies  = k == "enum" || k == "choices"      ? type == "enum" || type == "string" :
                              k == "minimum" || k == "maximum"   ? type == "integer" || type == "number" :
                                                                   type == "integer" || type == "number";
        if (known_type && spec.contains(k) && !applies) {
            return bad("\"" + k + "\" does not apply to " + type + " fields");
        }
    }

    if (type == "boolean") {
        f.type    = "boolean";
        f.values  = { common_json(true), common_json(false) };
        f.encoded = { "true", "false" };
    } else if (type == "enum" || type == "string") {
        const char * key = spec.contains("enum") ? "enum" : "choices";
        if (!spec.contains(key) || !spec.at(key).is_array()) {
            return bad("enum fields need a list of choices");
        }
        if (spec.at(key).size() > max_values) {
            return bad("more than " + std::to_string(max_values) + " choices");
        }
        f.type = "enum";
        for (const auto & c : spec.at(key)) {
            if (!c.is_string()) {
                return bad("enum choices must be strings");
            }
            const std::string v = c.get<std::string>();
            if (v.size() > max_value_bytes) {
                return bad("an enum choice is longer than " + std::to_string(max_value_bytes) + " bytes");
            }
            f.values.push_back(common_json(v));
            f.encoded.push_back(json_text(v));
        }
    } else if (type == "integer") {
        long long lo = 0, hi = 0;
        if (!spec.contains("minimum") || !spec.contains("maximum") ||
            !read_integer(spec.at("minimum"), lo) || !read_integer(spec.at("maximum"), hi)) {
            return bad("integer fields need integer minimum and maximum within the 64-bit range");
        }
        if (spec.contains("multipleOf") || spec.contains("step")) {
            return bad("integer fields take no step; list the values as an enum");
        }
        // width in unsigned arithmetic: hi - lo cannot overflow there when hi >= lo
        if (hi < lo || (unsigned long long) hi - (unsigned long long) lo > max_values - 1) {
            return bad("integer bounds must define 1-" + std::to_string(max_values) + " values");
        }
        const unsigned long long width = (unsigned long long) hi - (unsigned long long) lo;
        f.type        = "integer";
        f.number_base = (double) lo;
        for (unsigned long long i = 0; i <= width; ++i) {
            const long long v = (long long) ((unsigned long long) lo + i);
            f.values.push_back(common_json(v));
            f.encoded.push_back(std::to_string(v));
            f.numbers.push_back((double) i); // a double holds a 64-bit value only up to 2^53
        }
    } else if (type == "number") {
        const char * step_key = json_schema ? "multipleOf" : "step";
        double lo = 0, hi = 0, step = 0;
        if (!spec.contains("minimum") || !spec.contains("maximum") || !spec.contains(step_key) ||
            !read_number(spec.at("minimum"), lo) || !read_number(spec.at("maximum"), hi) ||
            !read_number(spec.at(step_key), step)) {
            return bad(std::string("number fields need finite minimum, maximum and ") + step_key);
        }
        constexpr double max_abs = 1e15; // grid values stay exact in a double and short in print
        if (!(step > 0) || !(hi >= lo) || std::fabs(lo) > max_abs || std::fabs(hi) > max_abs || step > max_abs) {
            return bad(std::string("number needs ordered bounds within 1e15 and a positive ") + step_key);
        }
        const int p_lo = decimal_places(lo), p_hi = decimal_places(hi), p_step = decimal_places(step);
        if (p_step < 0 || (!json_schema && (p_lo < 0 || p_hi < 0))) {
            return bad("number grid values need at most 9 decimals");
        }
        // compact "step": the grid starts at minimum and must end at maximum. JSON Schema
        // "multipleOf": the values are the multiples of the step within the bounds.
        double first = lo;
        long long n = 0; // index of the last grid value
        int places = std::max({ p_lo, p_hi, p_step });
        if (json_schema) {
            const double k_lo = std::ceil(lo / step - 1e-9), k_hi = std::floor(hi / step + 1e-9);
            if (!(k_hi >= k_lo) || k_hi - k_lo > (double) (max_values - 1)) {
                return bad("the multiples of multipleOf within the bounds must be 1-" + std::to_string(max_values) + " values");
            }
            first  = k_lo * step;
            n      = (long long) (k_hi - k_lo);
            places = p_step;
        } else {
            const double count = (hi - lo) / step;
            if (count > (double) (max_values - 1) || std::fabs(count - std::nearbyint(count)) > 1e-7) {
                return bad("the number grid must include both ends and hold 1-" + std::to_string(max_values) + " values");
            }
            n = (long long) std::nearbyint(count);
        }
        f.type = "number";
        for (long long i = 0; i <= n; ++i) {
            char buf[64];
            double v = first + (double) i * step;
            if (std::fabs(v) < 0.5 * std::pow(10.0, -places)) {
                v = 0.0; // never print -0
            }
            const int len = std::snprintf(buf, sizeof(buf), "%.*f", places, v);
            if (len <= 0 || len >= (int) sizeof(buf)) {
                return bad("a grid value does not print in 63 characters");
            }
            // the bounds print exactly at these decimals, so a grid value on a bound reads back
            // as that bound
            const double back = std::strtod(buf, nullptr);
            if (back < lo || back > hi) {
                if (json_schema && (i == 0 || i == n)) {
                    continue; // a multiple just past a bound that is not itself a multiple
                }
                return bad(std::string("grid value ") + buf + " falls outside the bounds");
            }
            f.values.push_back(common_json(back));
            f.encoded.push_back(buf); // fixed decimals: every value has the same shape
            f.numbers.push_back(back);
        }
    } else {
        return bad("supported types are boolean, enum, integer and number");
    }
    if (f.values.empty() || f.values.size() > max_values) {
        return bad("needs 1-" + std::to_string(max_values) + " allowed values");
    }
    for (size_t a = 0; a < f.encoded.size(); ++a) {
        for (size_t b = 0; b < a; ++b) {
            if (f.encoded[a] == f.encoded[b]) {
                return bad("duplicate allowed value " + f.encoded[a]);
            }
        }
    }
    std::string agg = "mode";
    for (const char * key : { "aggregate", "x-aggregate" }) {
        if (spec.contains(key)) {
            if (!spec.at(key).is_string()) {
                return bad(std::string("\"") + key + "\" must be a string");
            }
            agg = spec.at(key).get<std::string>();
            break;
        }
    }
    const bool numeric = f.type == "integer" || f.type == "number";
    if (agg != "mode" && !(numeric && (agg == "median" || agg == "mean"))) {
        return bad("aggregate must be mode, or median/mean for numeric fields");
    }
    f.aggregate = agg;
    return out;
}

bool known_schema_key(const std::string & key) {
    static const char * keys[] = { "type", "properties", "required", "additionalProperties", "title",
                                   "description", "$schema", "$comment" };
    for (const char * k : keys) {
        if (key == k) {
            return true;
        }
    }
    return false;
}

} // namespace

outcome<compiled_schema> compile_schema(const common_json & schema, const std::string & instructions) {
    outcome<compiled_schema> out;
    try {
        if (!schema.is_object()) {
            out.error = "\"schema\" must be an object";
            return out;
        }
        if (instructions.size() > max_instructions_bytes) {
            out.error = "instructions are longer than " + std::to_string(max_instructions_bytes) + " bytes";
            return out;
        }
        compiled_schema & cs = out.value;
        const bool json_schema = schema.contains("properties");
        if (json_schema) {
            for (const auto & e : schema.items()) {
                if (!known_schema_key(e.key())) {
                    out.error = "unsupported schema keyword \"" + e.key() + "\"";
                    return out;
                }
            }
            if (schema.contains("type") && !(schema.at("type").is_string() && schema.at("type").get<std::string>() == "object")) {
                out.error = "a JSON Schema decision must be of type object";
                return out;
            }
        }
        const common_json & props = json_schema ? schema.at("properties") : schema;
        if (!props.is_object() || props.size() < 1 || props.size() > max_fields) {
            out.error = "the schema must define 1-" + std::to_string(max_fields) + " fields";
            return out;
        }
        for (const auto & e : props.items()) {
            const common_json & spec = e.value();
            if (e.key().empty() || e.key().size() > max_name_bytes) {
                out.error = "field names must be 1-" + std::to_string(max_name_bytes) + " bytes";
                return out;
            }
            if (!spec.is_object()) {
                out.error = "field \"" + e.key() + "\" must be an object";
                return out;
            }
            std::string type;
            if (spec.contains("type")) {
                if (!spec.at("type").is_string()) {
                    out.error = "field \"" + e.key() + "\": \"type\" must be a string";
                    return out;
                }
                type = spec.at("type").get<std::string>();
            }
            if (spec.contains("enum") && (type.empty() || type == "string")) {
                type = "enum";
            }
            std::string description;
            if (spec.contains("description")) {
                if (!spec.at("description").is_string()) {
                    out.error = "field \"" + e.key() + "\": \"description\" must be a string";
                    return out;
                }
                description = spec.at("description").get<std::string>();
            }
            if (description.size() > max_description_bytes) {
                out.error = "field \"" + e.key() + "\": description longer than " + std::to_string(max_description_bytes) + " bytes";
                return out;
            }
            if (!json_schema && description.empty()) {
                out.error = "field \"" + e.key() + "\" needs a description";
                return out;
            }
            auto f = make_field(e.key(), type, description, spec, json_schema);
            if (!f.ok()) {
                out.error = f.error;
                return out;
            }
            cs.specs.push_back(std::move(f.value));
        }

        std::string catalog;
        for (const auto & f : cs.specs) {
            // the values' common leading characters are fixed in the suffix; only the rest is scored
            std::string common = f.encoded[0];
            for (const auto & v : f.encoded) {
                size_t c = 0;
                while (c < std::min(common.size(), v.size()) && common[c] == v[c]) {
                    ++c;
                }
                common.resize(c);
            }
            field_input in;
            in.suffix = "  " + json_text(f.name) + ": " + common;
            for (const auto & v : f.encoded) {
                in.candidates.push_back(v.substr(common.size()));
            }
            cs.inputs.push_back(in);

            std::string allowed;
            for (size_t i = 0; i < f.encoded.size(); ++i) {
                allowed += (i ? ", " : "") + f.encoded[i];
            }
            catalog += (catalog.empty() ? "" : "\n") + json_text(f.name) + (f.description.empty() ? "" : ": " + f.description) +
                       "\nAllowed values: " + allowed;
        }
        cs.system_text = "Select the requested field value from its allowed values, based on the context. "
                         "Respond with the JSON value only.\n\nFields:\n" + catalog + "\n" + instructions;
    } catch (const std::exception & e) {
        out = {};
        out.error = std::string("invalid schema: ") + e.what();
    }
    return out;
}

outcome<rendered_prompt> render_prompt(const common_chat_templates * tmpls, bool use_jinja, const std::string & system_text) {
    outcome<rendered_prompt> out;
    if (tmpls == nullptr) {
        out.value = { system_text + "\nContext:\n", "\nOutput:\n{\n" };
        return out;
    }
    static const std::string sentinel = "\x1f<<decision-context>>\x1f";
    if (system_text.find(sentinel) != std::string::npos) {
        out.error = "the instructions contain the decision placeholder";
        return out;
    }
    try {
        common_chat_templates_inputs in;
        in.use_jinja             = use_jinja;
        in.add_generation_prompt = true;
        in.enable_thinking       = false;
        common_chat_msg sys;
        sys.role    = "system";
        sys.content = system_text;
        common_chat_msg usr;
        usr.role    = "user";
        usr.content = sentinel;
        in.messages = { sys, usr };
        const std::string prompt = common_chat_templates_apply(tmpls, in).prompt;
        const size_t at = prompt.find(sentinel);
        if (at == std::string::npos || prompt.find(sentinel, at + 1) != std::string::npos) {
            out.error = "the chat template must render the user message exactly once and unchanged";
            out.code  = error_code::internal;
            return out;
        }
        out.value = { prompt.substr(0, at), prompt.substr(at + sentinel.size()) + "{\n" };
    } catch (const std::exception & e) {
        out.error = std::string("the chat template failed: ") + e.what();
        out.code  = error_code::internal;
    }
    return out;
}

outcome<common_json> assemble(const compiled_schema & cs, const result & r) {
    outcome<common_json> out;
    if (r.fields.size() != cs.specs.size()) {
        out.error = "the result does not match the schema";
        return out;
    }
    common_json decision = common_json::object();
    common_json fields   = common_json::object();
    for (size_t i = 0; i < cs.specs.size(); ++i) {
        const auto & sp = cs.specs[i];
        const auto & fr = r.fields[i];
        int idx = fr.winner;
        common_json f = common_json::object();
        const bool numeric = !sp.numbers.empty();
        if (numeric && fr.probs.size() == sp.values.size()) {
            std::vector<int> order(sp.values.size());
            for (size_t k = 0; k < order.size(); ++k) {
                order[k] = (int) k;
            }
            std::sort(order.begin(), order.end(), [&](int a, int b) { return sp.numbers[a] < sp.numbers[b]; });
            const auto quantile = [&](double q) {
                double acc = 0;
                for (int k : order) {
                    acc += fr.probs[k];
                    if (acc >= q) {
                        return k;
                    }
                }
                return order.back();
            };
            double mean = 0;
            for (size_t k = 0; k < sp.numbers.size(); ++k) {
                mean += sp.numbers[k] * fr.probs[k];
            }
            if (sp.aggregate == "median") {
                idx = quantile(0.5);
            } else if (sp.aggregate == "mean") {
                idx = 0; // the grid value nearest to the mean
                for (size_t k = 1; k < sp.numbers.size(); ++k) {
                    if (std::fabs(sp.numbers[k] - mean) < std::fabs(sp.numbers[idx] - mean)) {
                        idx = (int) k;
                    }
                }
                f["mean"] = sp.number_base + mean; // a statistic: rounds to a double
            }
            common_json interval = common_json::array();
            interval.push_back(sp.values[quantile(0.1)]);
            interval.push_back(sp.values[quantile(0.9)]);
            f["interval_p10_p90"] = interval;
        }
        if (idx < 0 || idx >= (int) sp.values.size()) {
            out.error = "field \"" + sp.name + "\" has no selected value";
            return out;
        }
        decision[sp.name] = sp.values[idx];
        f["value"]        = sp.values[idx];
        f["probability"]  = (double) (fr.probs.size() == sp.values.size() ? fr.probs[idx] : fr.path_score);
        f["scored_nodes"] = fr.scored_nodes;
        f["tree"]         = fr.tree;
        fields[sp.name]   = f;
    }
    out.value             = common_json::object();
    out.value["decision"] = decision;
    out.value["fields"]   = fields;
    return out;
}

// ---------------------------------------------------------------- request and response

outcome<request> parse_request(const common_json & body) {
    outcome<request> out;
    try {
        if (!body.is_object()) {
            out.error = "the request must be a JSON object";
            return out;
        }
        request & req = out.value;
        if (body.contains("instructions")) {
            if (!body.at("instructions").is_string()) {
                out.error = "\"instructions\" must be a string";
                return out;
            }
            req.instructions = body.at("instructions").get<std::string>();
        }
        if (!body.contains("schema")) {
            out.error = "\"schema\" is required";
            return out;
        }
        auto cs = compile_schema(body.at("schema"), req.instructions);
        if (!cs.ok()) {
            out.error = cs.error;
            return out;
        }
        req.schema = std::move(cs.value);

        if (!body.contains("contexts") || !body.at("contexts").is_array()) {
            out.error = "\"contexts\" must be an array of strings";
            return out;
        }
        const common_json & contexts = body.at("contexts");
        if (contexts.empty() || contexts.size() > max_contexts) {
            out.error = "\"contexts\" must hold 1-" + std::to_string(max_contexts) + " strings";
            return out;
        }
        size_t total = 0;
        for (const auto & c : contexts) {
            if (!c.is_string()) {
                out.error = "\"contexts\" must be an array of strings";
                return out;
            }
            std::string text = c.get<std::string>();
            if (text.empty() || text.size() > max_context_bytes) {
                out.error = "each context must be 1-" + std::to_string(max_context_bytes) + " bytes";
                return out;
            }
            total += text.size();
            req.contexts.push_back(std::move(text));
        }
        if (total > max_request_bytes) {
            out.error = "the contexts hold more than " + std::to_string(max_request_bytes) + " bytes";
            return out;
        }

        if (body.contains("mode")) {
            const common_json & m = body.at("mode");
            const std::string mode = m.is_string() ? m.get<std::string>() : std::string();
            if (mode == "auto") {
                req.opt.mode = score_mode::automatic;
            } else if (mode == "tree") {
                req.opt.mode = score_mode::tree;
            } else if (mode == "greedy") {
                req.opt.mode = score_mode::greedy;
            } else {
                out.error = "\"mode\" must be auto, tree or greedy";
                return out;
            }
        }
        if (body.contains("tree_max")) {
            long long v = 0;
            if (!read_integer(body.at("tree_max"), v) || v < 0 || v > max_tree_max) {
                out.error = "\"tree_max\" must be an integer 0-" + std::to_string(max_tree_max);
                return out;
            }
            req.opt.tree_max = (int) v;
        }
        if (body.contains("cache_prompt")) {
            if (!body.at("cache_prompt").is_boolean()) {
                out.error = "\"cache_prompt\" must be a boolean";
                return out;
            }
            req.opt.allow_cache = body.at("cache_prompt").get<bool>();
        }
    } catch (const std::exception & e) {
        out = {};
        out.error = std::string("invalid request: ") + e.what();
    }
    return out;
}

outcome<common_json> run(engine & eng, const common_chat_templates * tmpls, bool use_jinja, const request & req) {
    outcome<common_json> out;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        auto rp = render_prompt(tmpls, use_jinja, req.schema.system_text);
        if (!rp.ok()) {
            out.error = rp.error;
            out.code  = rp.code;
            return out;
        }
        std::vector<std::string> texts;
        for (const auto & c : req.contexts) {
            texts.push_back(c + rp.value.after);
        }
        const std::string & shared = rp.value.shared;

        failure err;
        const batch_result b = eng.decide_batch(shared, texts, req.schema.inputs, req.opt, err);
        if (err) {
            out.error = err.message;
            out.code  = err.code;
            return out;
        }

        common_json results = common_json::array();
        size_t context_tokens = 0;
        for (const auto & item : b.items) {
            auto a = assemble(req.schema, item);
            if (!a.ok()) {
                out.error = a.error;
                out.code  = error_code::internal;
                return out;
            }
            common_json usage = common_json::object();
            usage["context_tokens"] = item.context_tokens;
            usage["scored_rows"]    = item.rows;
            a.value["usage"]        = usage;
            results.push_back(a.value);
            context_tokens += item.context_tokens;
        }
        const double total_ms = ms_since(t0);

        common_json usage = common_json::object();
        usage["prompt_tokens"]  = b.shared_tokens + context_tokens;
        usage["cached_tokens"]  = b.cache_hit ? b.shared_tokens : 0;
        usage["context_tokens"] = context_tokens;
        usage["scored_rows"]    = b.rows;

        common_json timings = common_json::object();
        timings["prefill_ms"]      = b.prefill_ms;
        timings["scoring_ms"]      = b.scoring_ms;
        timings["total_ms"]        = total_ms;
        timings["rounds"]          = b.rounds;
        timings["decodes"]         = b.decodes;
        timings["per_decision_ms"] = total_ms / (double) b.items.size();

        out.value            = common_json::object();
        out.value["object"]  = "decision";
        out.value["mode"]    = mode_name(req.opt.mode);
        out.value["results"] = results;
        out.value["usage"]   = usage;
        out.value["timings"] = timings;
    } catch (const std::exception & e) {
        out = {};
        out.error = std::string("unexpected error: ") + e.what();
        out.code  = error_code::internal;
    }
    return out;
}

} // namespace schema_decision
