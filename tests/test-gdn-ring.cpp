// PATCH(rs-ring) bit-exactness check: ring fold vs the K-snapshot path of ggml_gated_delta_net.
// For a pass A of T1 tokens followed by acceptance of m tokens and a pass B of T2 tokens:
//   snapshot route: A with K snapshots -> state after m tokens -> B
//   ring route    : A in ring mode (stores its tokens) -> B in ring mode with n_commit = m
// The state after m tokens and B's attention output must be bit-identical.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const int S = 128, HK = 16, HV = 48;

struct inputs { std::vector<float> q, k, v, g, b; int T; };

static inputs make(int T, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1, 1), ug(-3, -0.01f), ub(0, 1), uv(-0.3f, 2.0f);
    inputs x; x.T = T;
    x.q.resize(S*HK*T); x.k.resize(S*HK*T); x.v.resize(S*HV*T); x.g.resize(HV*T); x.b.resize(HV*T);
    for (auto * a : {&x.q, &x.k}) {
        for (auto & f : *a) f = u(rng);
        for (int i = 0; i < HK*T; ++i) { double n = 0; for (int j = 0; j < S; ++j) n += (*a)[i*S+j]*(*a)[i*S+j];
            for (int j = 0; j < S; ++j) (*a)[i*S+j] /= (float) std::sqrt(n); }
    }
    for (auto & f : x.v) f = uv(rng);
    for (auto & f : x.g) f = ug(rng);
    for (auto & f : x.b) f = ub(rng);
    return x;
}

struct runner {
    ggml_backend_t be;
    explicit runner(ggml_backend_t be) : be(be) {}

    // runs one op; returns dst (attn | states). ring/ctl used when R > 0 (ring updated in place)
    std::vector<float> run(const inputs & x, const std::vector<float> & s0, int K, int R,
                           std::vector<float> * ring, const int32_t * ctl) {
        ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        const int T = x.T;
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HK, T, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, HV, T, 1);
        ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, T, 1);
        ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, HV, T, 1);
        ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, HV, 1);
        ggml_tensor * rg = nullptr, * c = nullptr, * out;
        const int64_t TS = (int64_t) S*HK + HV + (int64_t) S*HV;
        if (R > 0) {
            rg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R*TS, 2);
            c  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
            out = ggml_gated_delta_net_ring(ctx, q, k, v, g, b, s, rg, c, R);
        } else {
            out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);
        }
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
        ggml_backend_tensor_set(q, x.q.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k, x.k.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v, x.v.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(g, x.g.data(), 0, ggml_nbytes(g));
        ggml_backend_tensor_set(b, x.b.data(), 0, ggml_nbytes(b));
        ggml_backend_tensor_set(s, s0.data(), 0, ggml_nbytes(s));
        if (R > 0) {
            ring->resize(ggml_nelements(rg));
            ggml_backend_tensor_set(rg, ring->data(), 0, ggml_nbytes(rg));
            ggml_backend_tensor_set(c, ctl, 0, ggml_nbytes(c));
        }
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(1); }
        std::vector<float> res(ggml_nelements(out));
        ggml_backend_tensor_get(out, res.data(), 0, ggml_nbytes(out));
        if (R > 0) ggml_backend_tensor_get(rg, ring->data(), 0, ggml_nbytes(rg));
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return res;
    }
};

static int n_fail = 0;
static void cmp(const char * what, const float * a, const float * b, size_t n) {
    size_t diff = 0; double maxd = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) { ++diff; maxd = std::max(maxd, (double) std::fabs(a[i]-b[i])); }
    }
    printf("  %-34s %s (%zu/%zu differ, max |d| %.3g)\n", what, diff ? "DIFF" : "bit-identical", diff, n, maxd);
    if (diff) ++n_fail;
}

int main(int argc, char ** argv) {
    const char * dev = argc > 1 ? argv[1] : "CUDA0";
    ggml_backend_load_all();
    ggml_backend_t be = strcmp(dev, "CPU") == 0 ? ggml_backend_cpu_init() : ggml_backend_init_by_name(dev, nullptr);
    if (!be) { fprintf(stderr, "no backend %s\n", dev); return 1; }
    printf("backend %s\n", ggml_backend_name(be));
    runner rn(be);
    std::mt19937 rng(1234);
    const size_t D = (size_t) S*S*HV;

    struct cs { int T1, m, T2, R; };
    for (cs c : std::vector<cs>{{7,1,7,7},{7,3,7,7},{7,7,7,7},{7,0,1,7},{1,1,7,7},{20,15,7,7},{20,13,6,7},{40,17,7,40},{256,230,7,40},{100,90,40,40}}) {
        printf("T1=%d accept m=%d T2=%d R=%d\n", c.T1, c.m, c.T2, c.R);
        std::vector<float> s0(D); std::uniform_real_distribution<float> u(-0.5f, 0.5f);
        for (auto & f : s0) f = u(rng);
        inputs A = make(c.T1, rng), B = make(c.T2, rng);
        const int b1 = c.T1 > c.R ? c.T1 - c.R : 0; // ring route: state after b1 tokens, ring holds the rest
        const int keep = c.m;                       // tokens of A accepted (m >= b1)
        if (keep < b1) { printf("  skip\n"); continue; }

        // snapshot route
        const int K = std::min(c.T1, c.R) + 1;
        std::vector<float> oa = rn.run(A, s0, K, 0, nullptr, nullptr);
        const size_t attnA = (size_t) S*HV*c.T1;
        std::vector<float> s_m;
        if (keep == 0) s_m = s0; else s_m.assign(oa.begin() + attnA + (size_t)(c.T1 - keep)*D, oa.begin() + attnA + (size_t)(c.T1 - keep + 1)*D);
        std::vector<float> ob = rn.run(B, s_m, 1, 0, nullptr, nullptr);

        // ring route
        std::vector<float> ring(0);
        int32_t ctl1[3] = {0, 0, 1};
        std::vector<float> ra = rn.run(A, s0, 1, c.R, &ring, ctl1);
        std::vector<float> s_cache(ra.begin() + attnA, ra.begin() + attnA + D);
        cmp("pass A attention", ra.data(), oa.data(), attnA);
        if (b1 > 0) {
            cmp("state after b tokens (direct part)", s_cache.data(), oa.data() + attnA + (size_t)(c.T1 - b1)*D, D);
        }
        int32_t ctl2[3] = {keep - b1, 1, 0};
        std::vector<float> rb = rn.run(B, s_cache, 1, c.R, &ring, ctl2);
        const size_t attnB = (size_t) S*HV*c.T2;
        cmp("pass B attention after fold", rb.data(), ob.data(), attnB);
        if (c.T2 <= c.R) {
            cmp("folded state written by pass B", rb.data() + attnB, s_m.data(), D);
        }
        // fold all of B and compare with B's final state
        std::vector<float> s_cache2(rb.begin() + attnB, rb.begin() + attnB + D);
        inputs Z = make(1, rng);
        int32_t ctl3[3] = {c.T2 <= c.R ? c.T2 : c.R, 0, 1};
        if (c.T2 <= c.R) {
            std::vector<float> rz = rn.run(Z, s_cache2, 1, c.R, &ring, ctl3);
            cmp("state after folding all of B", rz.data() + (size_t) S*HV, ob.data() + attnB, D);
        }
    }
    printf(n_fail ? "FAILED: %d comparisons differ\n" : "ALL BIT-IDENTICAL\n", n_fail);
    ggml_backend_free(be);
    return n_fail ? 1 : 0;
}
