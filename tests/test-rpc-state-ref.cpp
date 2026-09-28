// Synthetic test of the RPC state-ref save/restore path (state-ref-split + state-dedup).
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-rpc.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>
#include <utility>

typedef int (*take_misses_t)(void);

int main(int argc, char ** argv) {
    const char * ep = argc > 1 ? argv[1] : "127.0.0.1:50099";
    ggml_backend_t be = ggml_backend_rpc_init(ep, 0);
    if (!be) { printf("no backend\n"); return 1; }
    auto take = (take_misses_t) ggml_backend_reg_get_proc_address(ggml_backend_rpc_reg(), "ggml_backend_rpc_state_ref_take_misses");
    const size_t N = 32u << 20;
    ggml_init_params ip = { 4 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k   = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, N); ggml_set_name(k, "cache_k_l3");
    ggml_tensor * chk = ggml_view_1d(ctx, k, N, 0);             ggml_set_name(chk, "check_view");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { printf("alloc failed\n"); return 1; }
    std::mt19937_64 rng(42);
    std::vector<uint8_t> orig(N);
    for (auto & b : orig) b = (uint8_t) rng();
    int fails = 0;
    // scenarios: saved ranges (offset, len) and destination runs (offset, len); sums must match
    struct scen { const char * name; std::vector<std::pair<size_t,size_t>> saved, dst; };
    const size_t M = 1u << 20;
    std::vector<scen> S = {
        {"same layout",      {{0, 10*M}, {12*M, 8*M}},              {{0, 10*M}, {12*M, 8*M}}},
        {"one region split", {{0, 10*M}},                           {{1*M, 3*M}, {6*M, 7*M}}},
        {"split in 3 runs",  {{2*M, 12*M}, {20*M, 4*M}},            {{0, 1*M}, {3*M, 2*M}, {8*M, 13*M}}},
        {"odd sizes",        {{100000, 3000000}, {5000000, 777777}},{{7000000, 123457}, {9000000, 3654320}}},
        {"grown version",    {{0, 11*M}},                           {{0, 11*M}}},   // shares chunks with scenario 2 (dedup)
    };
    for (auto & sc : S) {
        ggml_backend_tensor_set(k, orig.data(), 0, N);             // real bytes (no marker in random data)
        size_t total = 0; for (auto & r : sc.saved) total += r.second;
        std::vector<uint8_t> stream(total), expect(total);
        size_t pos = 0;
        for (auto & r : sc.saved) {
            ggml_backend_tensor_get(k, stream.data() + pos, r.first, r.second);   // marker + zero filler
            memcpy(expect.data() + pos, orig.data() + r.first, r.second);
            pos += r.second;
        }
        const bool has_marker = memcmp(stream.data(), "GGML-RPC-REF-v1", 15) == 0;
        std::vector<uint8_t> zero(N, 0);
        ggml_backend_tensor_set(chk, zero.data(), 0, N);
        take();
        pos = 0;
        for (auto & d : sc.dst) { ggml_backend_tensor_set(k, stream.data() + pos, d.first, d.second); pos += d.second; }
        const int misses = take();
        std::vector<uint8_t> back(N);
        ggml_backend_tensor_get(chk, back.data(), 0, N);
        pos = 0; size_t bad = 0;
        for (auto & d : sc.dst) { if (memcmp(back.data() + d.first, expect.data() + pos, d.second) != 0) bad++; pos += d.second; }
        printf("%-18s marker=%d misses=%d bad_runs=%zu -> %s\n", sc.name, has_marker, misses, bad, (misses == 0 && bad == 0 && has_marker) ? "OK" : "FAIL");
        fails += !(misses == 0 && bad == 0 && has_marker);
    }
    printf(fails ? "FAILED %d\n" : "ALL OK\n", fails);
    return fails;
}
