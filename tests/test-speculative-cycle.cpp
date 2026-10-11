#include "speculative-cycle.h"
#include "log.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <string>

// R04.7/R04.11: finalized tokens may shrink; attempted diagnostics may not.
static void attempts_and_selection() {
    common_speculative_cycle_recorder recorder(3, nullptr, true);
    auto a = recorder.begin(0, 7, 20);
    auto b = recorder.begin(1, 8, 30);
    auto phase = recorder.phase_begin(nullptr, {{a, 0}, {b, 1}});
    recorder.attach_draft(0, phase);
    recorder.attach_draft(1, phase);
    for (int seq : {0, 1}) {
        recorder.attempt(seq, 10, .9f, true);
        recorder.attempt(seq, 11, .8f, true);
        recorder.attempt(seq, 12, .1f, false);
    }
    recorder.phase_end(nullptr, phase, true);
    recorder.finalize(0, {10}, true); // common n_max clamp
    const auto * selected = recorder.get(0);
    assert(selected->proposal.size() == 1 && selected->attempts.size() == 3);
    assert(!selected->attempts.back().emitted);
    assert(recorder.get(1)->attempts.size() == 3);
    recorder.finalize(1, {}, false); // n_min clear / unselected implementation
    assert(!recorder.get(1));

    auto c = recorder.begin(2, 9, 40);
    auto c_draft = recorder.phase_begin(nullptr, {{c, 2}});
    recorder.attach_draft(2, c_draft);
    recorder.attempt(2, 20, .7f, true);
    recorder.phase_end(nullptr, c_draft, true);
    recorder.finalize(2, {20}, true);

    // R04.10: draft {A,B}, verify {A,C}; same shared verify metadata.
    auto verify = recorder.phase_begin(nullptr, {{a, 0}, {c, 2}});
    recorder.attach_verify(0, verify);
    recorder.attach_verify(2, verify);
    assert(recorder.get(0)->draft_phase == phase);
    assert(recorder.get(2)->draft_phase == c_draft);
    assert(recorder.get(0)->verify_phases[0] == recorder.get(2)->verify_phases[0]);
    assert(phase->id != verify->id);
    auto a_copy = *recorder.get(0);
    auto c_copy = *recorder.get(2);
    recorder.finish(0, a, 1);
    recorder.finish(2, c, 0);
    recorder.phase_end(nullptr, verify, false);
    assert(recorder.row(a_copy)["additive_residual_us"].is_null());
    assert(recorder.row(a_copy)["verify_phases"].dump() == recorder.row(c_copy)["verify_phases"].dump());
}

// R04.9: exercise the production fail-fast path in a subprocess.
static void bad_identity() {
    common_speculative_cycle_recorder recorder(1, nullptr, true);
    const auto id = recorder.begin(0, 1, 0);
    auto phase = recorder.phase_begin(nullptr, {{id, 0}});
    recorder.attach_draft(0, phase);
    recorder.phase_end(nullptr, phase, true);
    recorder.finalize(0, {42}, true);
    recorder.finish(0, id + 1, 0);
}

static void replay_snapshot() {
    common_speculative_cycle_recorder recorder(1, nullptr, true);
    const auto id = recorder.begin(0, 1, 0);
    auto draft = recorder.phase_begin(nullptr, {{id, 0}});
    recorder.attach_draft(0, draft);
    recorder.attempt(0, 42, .8f, true);
    recorder.attempt(0, 43, .7f, true);
    recorder.attempt(0, 44, .6f, true);
    recorder.phase_end(nullptr, draft, true);
    std::vector<llama_token> mutable_draft{42, 43, 44};
    recorder.finalize(0, mutable_draft, true);
    auto verify = recorder.phase_begin(nullptr, {{id, 0}});
    recorder.attach_verify(0, verify);
    recorder.phase_end(nullptr, verify, false);
    mutable_draft = {42, 100}; // checkpoint replay replaces the server vector
    assert(recorder.get(0)->proposal.size() == 3);
    auto replay = recorder.phase_begin(nullptr, {{id, 0}});
    recorder.attach_verify(0, replay);
    const auto copy = *recorder.get(0);
    assert(copy.proposal_id == id && copy.verify_phases.size() == 2);
    recorder.finish(0, id, 1); // final replay-adjusted acceptance
    recorder.phase_end(nullptr, replay, false);
    assert(recorder.row(copy)["verify_sample_wall_us"].is_null());
    assert(recorder.row(copy)["additive_residual_us"].is_null());
}

// Cancellation must preserve every counted invocation, even when another participant completes.
static void cancellation_fixture(const char * path, int count) {
    common_speculative_cycle_recorder recorder(2, path, true);
    for (int i = 0; i < count; ++i) {
        const auto a = recorder.begin(0, i, i);
        std::vector<common_speculative_cycle_participant> participants{{a, 0}};
        uint64_t b = 0;
        if (i % 3 == 2) {
            b = recorder.begin(1, i, i);
            participants.push_back({b, 1});
        }
        auto draft = recorder.phase_begin(nullptr, participants);
        for (const auto & item : participants) {
            recorder.attach_draft(item.sequence_id, draft);
            recorder.attempt(item.sequence_id, 42, .75f, true);
        }
        recorder.phase_end(nullptr, draft, true);
        for (const auto & item : participants) {
            recorder.finalize(item.sequence_id, {42}, true);
        }
        auto verify = recorder.phase_begin(nullptr, participants);
        for (const auto & item : participants) {
            recorder.attach_verify(item.sequence_id, verify);
        }
        if (i % 3 == 1) {
            recorder.phase_end(nullptr, verify, false);
            auto replay = recorder.phase_begin(nullptr, participants);
            recorder.attach_verify(0, replay);
            recorder.phase_end(nullptr, replay, false);
            recorder.cancel(0); // both verification invocations already ended
        } else {
            recorder.cancel(0); // server release precedes the outer decode catch's phase_end
            if (b) {
                recorder.finish(1, b, 1); // same shared phase, different outcome
            }
            recorder.phase_end(nullptr, verify, false);
        }
        assert(!recorder.get(0) && !recorder.get(1));
    }
}

static int run(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--bad-identity") {
        bad_identity();
        return 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--write") {
        common_speculative_cycle_recorder recorder(1, argv[2], false);
        auto id = recorder.begin(0, 1, 0);
        auto phase = recorder.phase_begin(nullptr, {{id, 0}});
        recorder.attach_draft(0, phase);
        recorder.phase_end(nullptr, phase, true);
        recorder.finalize(0, {}, false);
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--profile-fixture") {
        common_speculative_cycle_recorder recorder(1, argv[2], true);
        for (int i = 0; i < 64; ++i) {
            auto id = recorder.begin(0, i, i);
            auto draft = recorder.phase_begin(nullptr, {{id, 0}});
            recorder.attach_draft(0, draft);
            recorder.attempt(0, 42, .75f, true);
            recorder.attempt(0, 43, .1f, false);
            recorder.phase_end(nullptr, draft, true);
            recorder.finalize(0, {42}, true);
            auto verify = recorder.phase_begin(nullptr, {{id, 0}});
            recorder.attach_verify(0, verify);
            recorder.finish(0, id, 1);
            recorder.phase_end(nullptr, verify, false);
        }
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--cancellation-fixture") {
        cancellation_fixture(argv[2], 64);
        return 0;
    }
    common_speculative_cycle_recorder disabled(1, nullptr, false);
    assert(!disabled.enabled());
    assert(!disabled.phase_begin(nullptr, {{1, 0}}));
    attempts_and_selection();
    replay_snapshot();
    cancellation_fixture(nullptr, 3);
    std::puts("PASS: speculative cycle attempts, selection, shared phases and replay");
    return 0;
}

int main(int argc, char ** argv) {
    const int result = run(argc, argv);
    // Recorder rows are synchronous, but their identity/profile logs use a worker thread.
    common_log_flush(common_log_main());
    return result;
}
