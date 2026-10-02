// PATCH(keyed-sampling): tests for the position-keyed Gumbel-max sampler and the draft-side selection
//  1. exactness  : keyed-dist picks follow softmax(logit / T), i.e. the same distribution as `dist`
//  2. determinism: same seed + position -> same token; clone keeps the position; reset restarts it
//  3. agreement  : given the same logits and history, common_keyed_draft_select over the top-10 candidates
//                  picks the token the real target chain (penalties, top-k, top-p, temp, keyed) samples
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#undef NDEBUG
#include <cassert>

static int n_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); n_fail++; } } while (0)

static llama_token sample_once(llama_sampler * chain, std::vector<llama_token_data> & buf, const std::vector<float> & logits) {
    buf.resize(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        buf[i] = { (llama_token) i, logits[i], 0.0f };
    }
    llama_token_data_array arr = { buf.data(), buf.size(), -1, false };
    llama_sampler_apply(chain, &arr);
    assert(arr.selected >= 0 && (size_t) arr.selected < arr.size);
    return arr.data[arr.selected].id;
}

static void test_exactness() {
    const std::vector<float> logits = { 2.0f, 1.2f, 0.5f, 0.0f, -0.7f, -2.0f };
    const float T = 0.6f;
    const int N = 400000;

    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_temp(T));
    llama_sampler_chain_add(chain, common_sampler_init_keyed(42));

    std::vector<int> cnt(logits.size(), 0);
    std::vector<llama_token_data> buf;
    for (int i = 0; i < N; ++i) {
        const llama_token t = sample_once(chain, buf, logits);
        cnt[t]++;
        llama_sampler_accept(chain, t); // advances the position
    }

    double s = 0.0;
    for (float l : logits) { s += std::exp((l - logits[0]) / T); }
    double max_err = 0.0, chi2 = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double p = std::exp((logits[i] - logits[0]) / T) / s;
        const double f = (double) cnt[i] / N;
        max_err = std::max(max_err, std::fabs(f - p));
        chi2 += (cnt[i] - N * p) * (cnt[i] - N * p) / (N * p);
        std::printf("  token %zu: expected %.4f observed %.4f\n", i, p, f);
    }
    // 5 degrees of freedom: chi2 > 20.5 has p < 0.001
    std::printf("exactness: max |freq - softmax| = %.5f, chi2(5 dof) = %.2f\n", max_err, chi2);
    CHECK(max_err < 0.004, "frequencies deviate from the softmax: %.5f", max_err);
    CHECK(chi2 < 20.5, "chi-square too large: %.2f", chi2);
    llama_sampler_free(chain);
}

static void test_determinism() {
    const std::vector<float> logits = { 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f };
    std::vector<llama_token_data> buf;

    auto make = [](uint32_t seed) {
        llama_sampler * c = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(c, llama_sampler_init_temp(1.0f));
        llama_sampler_chain_add(c, common_sampler_init_keyed(seed));
        return c;
    };
    llama_sampler * a = make(7);
    llama_sampler * b = make(7);
    llama_sampler * d = make(8);

    std::vector<llama_token> sa, sb, sd;
    llama_sampler * cl = nullptr;
    std::vector<llama_token> scl;
    for (int i = 0; i < 200; ++i) {
        if (i == 100) { cl = llama_sampler_clone(a); }
        const llama_token ta = sample_once(a, buf, logits); llama_sampler_accept(a, ta); sa.push_back(ta);
        const llama_token tb = sample_once(b, buf, logits); llama_sampler_accept(b, tb); sb.push_back(tb);
        const llama_token td = sample_once(d, buf, logits); llama_sampler_accept(d, td); sd.push_back(td);
    }
    for (int i = 100; i < 200; ++i) {
        const llama_token t = sample_once(cl, buf, logits); llama_sampler_accept(cl, t); scl.push_back(t);
    }
    CHECK(sa == sb, "same seed gave different sequences");
    CHECK(sa != sd, "different seeds gave the same sequence");
    CHECK(std::equal(scl.begin(), scl.end(), sa.begin() + 100), "clone did not continue at the same position");

    llama_sampler_reset(a);
    std::vector<llama_token> sr;
    for (int i = 0; i < 50; ++i) { const llama_token t = sample_once(a, buf, logits); llama_sampler_accept(a, t); sr.push_back(t); }
    CHECK(std::equal(sr.begin(), sr.end(), sa.begin()), "reset did not restart the position");
    std::printf("determinism: ok (same seed equal, other seed differs, clone continues, reset restarts)\n");

    llama_sampler_free(a); llama_sampler_free(b); llama_sampler_free(d); llama_sampler_free(cl);
}

static void test_agreement() {
    const int V = 3000;
    const int L = 300;
    const int TRIALS = 3000;

    common_params_sampling sp;
    sp.penalty_last_n  = 64;
    sp.penalty_repeat  = 1.0f;
    sp.penalty_freq    = 0.0f;
    sp.penalty_present = 1.0f;
    sp.top_k    = 20;
    sp.top_p    = 0.95f;
    sp.min_p    = 0.0f;
    sp.temp     = 0.6f;
    sp.min_keep = 0;

    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> small_tok(0, 150);
    std::uniform_int_distribution<int> n_boost(1, 6);
    std::uniform_real_distribution<float> boost(4.0f, 9.0f);
    std::uniform_int_distribution<int> n_drafted(0, 5);

    int agree = 0, outside = 0, n_multi = 0;
    std::vector<llama_token_data> buf;
    for (int t = 0; t < TRIALS; ++t) {
        const uint32_t seed = 1000 + t;

        // history drawn from a small set so that the penalties hit candidates
        llama_tokens prompt(L - 1);
        for (auto & x : prompt) { x = small_tok(rng); }
        const llama_token id_last = small_tok(rng);
        llama_tokens drafted(n_drafted(rng));
        for (auto & x : drafted) { x = small_tok(rng); }

        // peaked logits: noise plus a few boosted tokens, many of them in the history
        std::vector<float> logits(V);
        for (auto & l : logits) { l = 1.5f * nd(rng); }
        const int nb = n_boost(rng);
        for (int k = 0; k < nb; ++k) { logits[small_tok(rng)] += boost(rng); }

        // target: the production chain order, history accepted first
        llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(V, sp.penalty_last_n, sp.penalty_repeat, sp.penalty_freq, sp.penalty_present));
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(sp.top_k));
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(sp.top_p, sp.min_keep));
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(sp.min_p, sp.min_keep));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(sp.temp));
        llama_sampler_chain_add(chain, common_sampler_init_keyed(seed));
        for (llama_token x : prompt)  { llama_sampler_accept(chain, x); }
        llama_sampler_accept(chain, id_last);
        for (llama_token x : drafted) { llama_sampler_accept(chain, x); }
        const llama_token target = sample_once(chain, buf, logits);
        llama_sampler_free(chain);

        // draft: only its top-10 raw candidates, same position key
        std::vector<llama_token_data> cand(V);
        for (int i = 0; i < V; ++i) { cand[i] = { i, logits[i], 0.0f }; }
        std::partial_sort(cand.begin(), cand.begin() + 10, cand.end(),
                          [](const llama_token_data & a, const llama_token_data & b) { return a.logit > b.logit; });
        const int64_t pos = (int64_t) prompt.size() + 1 + (int64_t) drafted.size();
        float p = 0.0f;
        const llama_token draft = common_keyed_draft_select(cand.data(), 10, nullptr, sp, seed, pos, prompt, id_last, drafted, &p);

        bool in_top10 = false;
        for (int i = 0; i < 10; ++i) { in_top10 |= cand[i].id == target; }
        outside += !in_top10;
        agree   += draft == target;

        // how often the target did NOT pick the top post-penalty token, i.e. where an argmax draft fails
        std::vector<float> pen = logits;
        for (int i = 0; i < V; ++i) {
            int c = 0;
            const int n_hist = (int) prompt.size() + 1 + (int) drafted.size();
            for (int k = std::max(0, n_hist - sp.penalty_last_n); k < n_hist; ++k) {
                const llama_token h = k < (int) prompt.size() ? prompt[k] : (k == (int) prompt.size() ? id_last : drafted[k - prompt.size() - 1]);
                c += h == i;
            }
            if (c > 0) { pen[i] -= sp.penalty_present; }
        }
        n_multi += (int) (std::max_element(pen.begin(), pen.end()) - pen.begin()) != target;
    }
    // the draft only sees its top-10 while the target keeps top-k 20: when the target samples outside the
    // draft's candidates no draft can match, so the invariant is agreement on the reachable trials
    const double rate      = (double) agree / TRIALS;
    const double reachable = (double) agree / std::max(1, TRIALS - outside);
    std::printf("agreement: draft == target in %d / %d trials = %.2f%%; on the %d trials where the target's token is among "
                "the draft's candidates: %.2f%% (an argmax draft would miss %d = %.1f%% of all trials)\n",
                agree, TRIALS, 100.0 * rate, TRIALS - outside, 100.0 * reachable, n_multi, 100.0 * n_multi / TRIALS);
    CHECK(reachable >= 0.99, "draft/target agreement on reachable trials too low: %.4f", reachable);
}

int main() {
    test_exactness();
    test_determinism();
    test_agreement();
    if (n_fail) {
        std::printf("%d check(s) FAILED\n", n_fail);
        return 1;
    }
    std::printf("all keyed-sampling tests passed\n");
    return 0;
}
