#include "speculative-cycle.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>

static int64_t cycle_sync(llama_context * ctx, bool enabled) {
    if (!enabled || !ctx) {
        return 0;
    }
    const int64_t start = ggml_time_us();
    llama_synchronize(ctx);
    return ggml_time_us() - start;
}

common_speculative_cycle_recorder::common_speculative_cycle_recorder(size_t n_seq, const char * path, bool profile)
    : profile(profile), cycles(n_seq, 0), pending(n_seq) {
    static std::atomic<uint64_t> launches{uint64_t(std::chrono::system_clock::now().time_since_epoch().count())};
    launch_id = ++launches;
    if (path && *path) {
        file = std::fopen(path, "a");
        if (!file) {
            GGML_ABORT("LLAMA_SPEC_LOG: cannot open %s", path);
        }
        if (std::fflush(file) != 0) {
            GGML_ABORT("LLAMA_SPEC_LOG: cannot flush %s", path);
        }
    }
}

common_speculative_cycle_recorder::~common_speculative_cycle_recorder() {
    for (size_t seq = 0; seq < pending.size(); ++seq) {
        cancel(seq);
    }
    flush();
    if (file && std::fclose(file) != 0) {
        GGML_ABORT("LLAMA_SPEC_LOG: close failed");
    }
}

std::shared_ptr<common_speculative_cycle_phase> common_speculative_cycle_recorder::phase_begin(
        llama_context * ctx, const std::vector<common_speculative_cycle_participant> & participants) {
    if (!enabled() || participants.empty()) {
        return {};
    }
    auto phase = std::make_shared<common_speculative_cycle_phase>();
    phase->id = ++next_phase;
    phase->participants = participants;
    phase->start_us = ggml_time_us();
    phase->synchronization_us = cycle_sync(ctx, profile);
    return phase;
}

void common_speculative_cycle_recorder::phase_end(llama_context * ctx,
        const std::shared_ptr<common_speculative_cycle_phase> & phase, bool draft) {
    if (!phase) {
        return;
    }
    GGML_ASSERT(!phase->end_us);
    phase->synchronization_us += cycle_sync(ctx, profile);
    phase->end_us = ggml_time_us();
    auto & mean = draft ? draft_mean : verify_mean;
    ++mean.count;
    mean.wall_us += phase->end_us - phase->start_us;
    mean.synchronization_us += phase->synchronization_us;
    flush();
}

uint64_t common_speculative_cycle_recorder::begin(llama_seq_id seq, int64_t request, llama_pos pos) {
    cancel(seq);
    auto record = std::make_unique<common_speculative_cycle_record>();
    record->launch_id = launch_id;
    record->request_id = request;
    record->proposal_id = ++next_proposal;
    record->sequence_id = seq;
    record->cycle_id = ++cycles.at(seq);
    record->starting_position = pos;
    LOG_INF("spec_cycle_start: launch=%llu request=%lld proposal=%llu sequence=%d cycle=%llu\n",
            (unsigned long long) launch_id, (long long) request, (unsigned long long) record->proposal_id,
            seq, (unsigned long long) record->cycle_id);
    pending.at(seq) = std::move(record);
    return next_proposal;
}

void common_speculative_cycle_recorder::attempt(llama_seq_id seq, llama_token token, float probability, bool emitted) {
    GGML_ASSERT(std::isfinite(probability) && probability >= 0 && probability <= 1);
    auto & record = *pending.at(seq);
    GGML_ASSERT(!record.finalized);
    record.attempts.push_back({ token, probability, emitted });
}

void common_speculative_cycle_recorder::finalize(llama_seq_id seq, const std::vector<llama_token> & proposal, bool selected) {
    auto & record = *pending.at(seq);
    GGML_ASSERT(!record.finalized);
    record.finalized = true;
    record.selected = selected;
    if (selected) {
        record.proposal = proposal;
    } else {
        finish(seq, record.proposal_id, 0);
    }
}

const common_speculative_cycle_record * common_speculative_cycle_recorder::get(llama_seq_id seq) const {
    return pending.at(seq).get();
}

void common_speculative_cycle_recorder::attach_draft(llama_seq_id seq,
        const std::shared_ptr<common_speculative_cycle_phase> & phase) {
    pending.at(seq)->draft_phase = phase;
}

void common_speculative_cycle_recorder::attach_verify(llama_seq_id seq,
        const std::shared_ptr<common_speculative_cycle_phase> & phase) {
    auto & record = *pending.at(seq);
    GGML_ASSERT(record.finalized && record.selected);
    record.verify_phases.push_back(phase);
}

void common_speculative_cycle_recorder::finish(llama_seq_id seq, uint64_t proposal, size_t accepted) {
    auto & record = pending.at(seq);
    if (!record || record->proposal_id != proposal || record->sequence_id != seq) {
        GGML_ABORT("spec cycle: proposal identity mismatch for sequence %d", seq);
    }
    GGML_ASSERT(record->finalized && accepted <= record->proposal.size());
    record->accepted_count = record->selected ? accepted : 0;
    record->completion_us = ggml_time_us();
    record->completed = true;
    completed.push_back(std::move(record));
    flush();
}

void common_speculative_cycle_recorder::cancel(llama_seq_id seq) {
    auto & record = pending.at(seq);
    if (!record) {
        return;
    }
    record->selected = false;
    record->finalized = true;
    record->cancelled = true;
    record->proposal.clear();
    // Retain actual invocations, including an open shared phase whose end will flush this row.
    finish(seq, record->proposal_id, 0);
}

static common_json phase_json(const common_speculative_cycle_phase & phase) {
    common_json participants = common_json::array();
    for (const auto & item : phase.participants) {
        participants.push_back(common_json{{"proposal_id", item.proposal_id}, {"sequence_id", item.sequence_id}});
    }
    return common_json{
        {"phase_id", phase.id}, {"participants", participants},
        {"participant_count", phase.participants.size()},
        {"timing_scope", phase.participants.size() == 1 ? "single_sequence" : "shared_batch"},
        {"start_us", phase.start_us}, {"end_us", phase.end_us},
        {"wall_us", phase.end_us - phase.start_us}, {"synchronization_us", phase.synchronization_us},
    };
}

common_json common_speculative_cycle_recorder::row(const common_speculative_cycle_record & record) const {
    common_json attempts = common_json::array();
    common_json probabilities = common_json::array();
    common_json proposal = common_json::array();
    common_json verifies = common_json::array();
    for (const auto & attempt : record.attempts) {
        attempts.push_back(common_json{{"token", attempt.token}, {"top1_probability", attempt.top1_probability},
                                      {"emitted", attempt.emitted}});
        probabilities.push_back(attempt.top1_probability);
    }
    for (auto token : record.proposal) {
        proposal.push_back(token);
    }
    for (const auto & phase : record.verify_phases) {
        verifies.push_back(phase_json(*phase));
    }
    const auto & draft = *record.draft_phase;
    common_json result{
        {"schema_version", 1}, {"launch_id", record.launch_id}, {"request_id", record.request_id},
        {"proposal_id", record.proposal_id}, {"sequence_id", record.sequence_id}, {"cycle_id", record.cycle_id},
        {"starting_position", record.starting_position}, {"selected", record.selected},
        {"outcome", record.cancelled ? "cancelled" : "completed"},
        {"drafted_count", record.proposal.size()}, {"attempted_count", record.attempts.size()},
        {"accepted_count", record.accepted_count}, {"attempts", attempts}, {"top1_probabilities", probabilities},
        {"proposal_tokens", proposal}, {"draft_phase", phase_json(draft)}, {"verify_phases", verifies},
        {"draft_wall_us", draft.end_us - draft.start_us},
        {"cycle_wall_us", record.completion_us - draft.start_us},
        {"inter_phase_wait_us", nullptr}, {"verify_sample_wall_us", nullptr}, {"additive_residual_us", nullptr},
        {"timing_mode", profile ? "synchronized" : "host_unsynchronized"},
    };
    if (!record.verify_phases.empty()) {
        result["inter_phase_wait_us"] = record.verify_phases.front()->start_us - draft.end_us;
        // Multiple phases (checkpoint replay) are listed individually, never presented as one invocation.
        if (record.verify_phases.size() == 1) {
            const auto & verify = *record.verify_phases.front();
            result["verify_sample_wall_us"] = verify.end_us - verify.start_us;
            if (!record.cancelled && draft.participants.size() == 1 && verify.participants.size() == 1) {
                result["additive_residual_us"] = record.completion_us - draft.start_us -
                        (draft.end_us - draft.start_us) - (verify.end_us - verify.start_us);
            }
        }
    }
    return result;
}

void common_speculative_cycle_recorder::flush() {
    for (auto it = completed.begin(); it != completed.end();) {
        auto & record = **it;
        if (!record.draft_phase || !record.draft_phase->end_us ||
                std::any_of(record.verify_phases.begin(), record.verify_phases.end(),
                    [](const auto & phase) { return !phase->end_us; })) {
            ++it;
            continue;
        }
        if (record.verify_phases.size() == 1 && record.verify_phases[0]->participants.size() == 1) {
            record.completion_us = std::max(record.completion_us, record.verify_phases[0]->end_us);
        }
        if (file) {
            const std::string line = row(record).dump() + "\n";
            if (std::fwrite(line.data(), 1, line.size(), file) != line.size() || std::fflush(file) != 0) {
                GGML_ABORT("LLAMA_SPEC_LOG: append or flush failed");
            }
        }
        ++cycle_mean.count;
        cycle_mean.wall_us += record.completion_us - record.draft_phase->start_us;
        it = completed.erase(it);
    }
    const uint64_t window = std::min({draft_mean.count, verify_mean.count, cycle_mean.count}) / 64;
    if (profile && window > reports) {
        reports = window;
        LOG_INF("spec_profile: draft_phases=%llu draft_mean_us=%.3f draft_sync_mean_us=%.3f "
                "verify_phases=%llu verify_mean_us=%.3f verify_sync_mean_us=%.3f "
                "sequence_cycles=%llu cycle_mean_us=%.3f\n",
                (unsigned long long) draft_mean.count, draft_mean.wall_us / draft_mean.count,
                draft_mean.synchronization_us / draft_mean.count,
                (unsigned long long) verify_mean.count, verify_mean.wall_us / verify_mean.count,
                verify_mean.synchronization_us / verify_mean.count,
                (unsigned long long) cycle_mean.count, cycle_mean.wall_us / cycle_mean.count);
    }
}
