#include "../common/mtp-adaptive.h"

#include <cassert>
#include <cstdio>
#include <limits>

static void simulate(common_mtp_adaptive & c, int best, int count) {
    for (int i = 0; i < count; ++i) {
        const int depth = c.choose(6);
        // All depths accept fully, but only one has a good total round cost.
        c.observe(depth, depth, (depth + 1) * (depth == best ? 1.0 : 4.0));
    }
}

int main() {
    common_mtp_adaptive c;
    assert(c.choose(6) == 6);
    assert(c.choose(6) == 6); // no observations: stay at maximum
    for (int i = 0; i < 3; ++i) {
        assert(c.choose(6) == 6);
        c.observe(6, 0, 10);
    }
    assert(c.choose(6) == 4);
    assert(std::abs(c.reward(6) - 0.1) < 1e-12); // includes the target token
    c.observe(4, 3, 20);
    assert(std::abs(c.reward(4) - 0.2) < 1e-12);
    const auto before = c.rounds;
    c.observe(4, 4, 0);
    c.observe(4, 4, std::numeric_limits<double>::infinity());
    c.observe(4, 4, std::numeric_limits<double>::quiet_NaN());
    c.observe(4, 5, 10);
    assert(c.rounds == before);

    assert(common_mtp_adaptive::limit(6, -1) == 6);
    assert(common_mtp_adaptive::limit(6, 4) == 4);
    assert(common_mtp_adaptive::limit(4, 6) == 4);
    assert(common_mtp_adaptive::limit(6, 0) == 0);
    assert(common_mtp_adaptive::limit(0, -1) == 0);
    for (int cap = 0; cap <= 9; ++cap) {
        c.reset();
        assert(c.choose(cap) == std::min(6, cap));
        for (int i = 0; i < 100; ++i) {
            int depth = c.choose(cap);
            assert(depth >= 0 && depth <= cap && depth <= 6);
            c.observe(depth, depth, 10);
        }
    }

    c.reset();
    for (int i = 0; i < 100; ++i) {
        int depth = c.choose(6, 3);
        assert(depth == 3 || depth == 4 || depth == 6);
        c.observe(depth, depth, 10);
    }

    c.reset();
    simulate(c, 2, 200);
    assert(c.preferred == 2);
    simulate(c, 6, 400); // exploration recovers after the workload changes
    assert(c.preferred == 6);
    simulate(c, 4, 400);
    assert(c.preferred == 4);

    c.reset();
    for (int i = 0; i < 100; ++i) {
        int depth = c.choose(6);
        c.observe(depth, depth, (depth + 1) / (depth == 2 ? 1.05 : 1.0));
    }
    assert(c.preferred == 6); // less than 10% improvement: no switch
    common_mtp_adaptive other;
    assert(other.choose(6) == 6 && other.rounds == 0);
    c.reset();
    assert(c.choose(6) == 6 && c.rounds == 0 && c.stats[2].count == 0);
    simulate(c, 2, 100);
    assert(c.choose(3) == 3 && c.rounds == 0); // cap change discards incomparable costs

    common_mtp_round round;
    round.start(6, 32768, 1000);
    round.finish_draft(4, 3000);
    assert(round.finish_accept(2, false, 11000));
    assert(round.draft_us == 2000 && round.round_us == 10000);
    assert(round.length == 4 && round.context == 32768);
    assert(!round.finish_accept(2, false, 12000)); // no duplicate observations
    round.start(6, 20, 1000);
    round.finish_draft(6, 2000);
    const auto mtp_rounds = c.rounds;
    if (round.finish_accept(6, true, 9000)) {
        c.observe(round.depth, 6, round.round_us / 1000.0);
    }
    assert(c.rounds == mtp_rounds); // ngram cannot feed MTP
    assert(!round.finish_accept(6, false, 10000));
    round.start(2, 20, 1000);
    round.finish_draft(0, 2000);
    assert(!round.finish_accept(0, false, 9000));
    round.start(2, 20, 1000);
    round.finish_draft(2, 2000);
    round = {}; // begin / old-session restore cancels a pending round
    assert(!round.finish_accept(2, false, 9000));

    std::puts("mtp-adaptive: all tests passed");
}
