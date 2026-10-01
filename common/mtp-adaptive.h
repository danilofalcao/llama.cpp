#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

// Reward is emitted tokens / full round time, not acceptance rate / draft time.
struct common_mtp_adaptive {
    struct observation {
        unsigned count = 0;
        double tokens = 0;
        double ms = 0;
    };

    std::array<observation, 7> stats{};
    int cap = 0;
    int floor = 0;
    int preferred = 0;
    uint64_t rounds = 0;
    uint64_t last_switch = 0;
    unsigned explore = 0;

    void reset() { *this = common_mtp_adaptive{}; }

    static int limit(int configured, int context_cap) {
        return std::max(0, context_cap >= 0 ? std::min(configured, context_cap) : configured);
    }

    int choose(int maximum, int minimum = 0) {
        maximum = std::max(0, std::min(6, maximum));
        minimum = std::max(0, std::min(maximum, minimum));
        if (cap != maximum || floor != minimum) {
            reset();
            cap = preferred = maximum;
            floor = minimum;
        }
        if (cap == 0) {
            return 0;
        }

        int depths[3];
        int n = 0;
        for (int d : {2, 4, 6}) {
            d = std::min(std::max(d, floor), cap);
            if (n == 0 || depths[n - 1] != d) {
                depths[n++] = d;
            }
        }

        // Start at the maximum. Collect three observations per depth before comparing.
        if (stats[cap].count < 3) {
            return cap;
        }
        for (int i = n - 2; i >= 0; --i) {
            if (stats[depths[i]].count < 3) {
                return depths[i];
            }
        }

        int best = preferred;
        for (int i = 0; i < n; ++i) {
            if (reward(depths[i]) > reward(best)) {
                best = depths[i];
            }
        }
        if (rounds - last_switch >= 8 && reward(best) > 1.10 * reward(preferred)) {
            preferred = best;
            last_switch = rounds;
        }
        if (rounds % 8 == 0) {
            return depths[explore++ % n];
        }
        return preferred;
    }

    double reward(int depth) const {
        return stats[depth].ms > 0 ? stats[depth].tokens / stats[depth].ms : 0;
    }

    void observe(int depth, int accepted, double round_ms) {
        if (depth <= 0 || depth > cap || accepted < 0 || accepted > depth ||
                !std::isfinite(round_ms) || round_ms <= 0) {
            return;
        }
        auto & s = stats[depth];
        const double alpha = s.count == 0 ? 1.0 : 0.25;
        s.tokens += alpha * (1.0 + accepted - s.tokens);
        s.ms += alpha * (round_ms - s.ms);
        if (s.count < 3) {
            ++s.count;
        }
        ++rounds;
    }
};

struct common_mtp_round {
    int depth = 0;
    int length = 0;
    int context = 0;
    int64_t start_us = 0;
    int64_t draft_us = 0;
    int64_t round_us = 0;
    bool pending = false;

    void start(int chosen, int ctx, int64_t now) {
        *this = common_mtp_round{};
        depth = chosen;
        context = ctx;
        start_us = now;
    }

    void finish_draft(int generated, int64_t now) {
        length = generated;
        draft_us = now - start_us;
        pending = length > 0;
    }

    bool finish_accept(int accepted, bool is_other, int64_t now) {
        const bool valid = pending && !is_other && accepted >= 0 && accepted <= length && now > start_us;
        pending = false;
        round_us = now - start_us;
        return valid;
    }
};
