#pragma once

#include "llama.h"
#include "json.h"

#include <cstdio>
#include <memory>
#include <vector>

// fork: cycle identity is independent of mutable server replay buffers.
struct common_speculative_cycle_attempt {
    llama_token token;
    float top1_probability;
    bool emitted;
};

struct common_speculative_cycle_participant {
    uint64_t proposal_id;
    llama_seq_id sequence_id;
};

struct common_speculative_cycle_phase {
    uint64_t id = 0;
    std::vector<common_speculative_cycle_participant> participants;
    int64_t start_us = 0;
    int64_t end_us = 0;
    int64_t synchronization_us = 0;
};

struct common_speculative_cycle_record {
    uint64_t launch_id = 0;
    int64_t request_id = -1;
    uint64_t proposal_id = 0;
    llama_seq_id sequence_id = -1;
    uint64_t cycle_id = 0;
    llama_pos starting_position = 0;
    bool selected = false;
    bool finalized = false;
    bool completed = false;
    bool cancelled = false;
    size_t accepted_count = 0;
    int64_t completion_us = 0;
    std::vector<common_speculative_cycle_attempt> attempts;
    std::vector<llama_token> proposal;
    std::shared_ptr<common_speculative_cycle_phase> draft_phase;
    std::vector<std::shared_ptr<common_speculative_cycle_phase>> verify_phases;
};

// Internal recorder, also exercised by synthetic tests without model contexts.
class common_speculative_cycle_recorder {
public:
    common_speculative_cycle_recorder(size_t n_seq, const char * path, bool profile);
    ~common_speculative_cycle_recorder();
    bool enabled() const { return file || profile; }
    bool profiling() const { return profile; }
    std::shared_ptr<common_speculative_cycle_phase> phase_begin(
            llama_context * ctx, const std::vector<common_speculative_cycle_participant> & participants);
    void phase_end(llama_context * ctx, const std::shared_ptr<common_speculative_cycle_phase> & phase, bool draft);
    uint64_t begin(llama_seq_id seq, int64_t request, llama_pos pos);
    void attempt(llama_seq_id seq, llama_token token, float probability, bool emitted);
    void finalize(llama_seq_id seq, const std::vector<llama_token> & proposal, bool selected);
    const common_speculative_cycle_record * get(llama_seq_id seq) const;
    void finish(llama_seq_id seq, uint64_t proposal, size_t accepted);
    void cancel(llama_seq_id seq);
    void attach_draft(llama_seq_id seq, const std::shared_ptr<common_speculative_cycle_phase> & phase);
    void attach_verify(llama_seq_id seq, const std::shared_ptr<common_speculative_cycle_phase> & phase);
    common_json row(const common_speculative_cycle_record & record) const;
    void flush();

private:
    FILE * file = nullptr;
    bool profile;
    uint64_t launch_id;
    uint64_t next_proposal = 0;
    uint64_t next_phase = 0;
    std::vector<uint64_t> cycles;
    std::vector<std::unique_ptr<common_speculative_cycle_record>> pending;
    std::vector<std::unique_ptr<common_speculative_cycle_record>> completed;
    struct mean {
        uint64_t count = 0;
        double wall_us = 0;
        double synchronization_us = 0;
    } draft_mean, verify_mean, cycle_mean;
    uint64_t reports = 0;
};
