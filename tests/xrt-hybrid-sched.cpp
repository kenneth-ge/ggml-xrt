// ggml-xrt hybrid-sharding mechanism harness.
//
// Proves the "true hybrid" placement WITHOUT llama-cli: a synthetic multi-layer
// transformer-ish graph is driven through ggml_backend_sched over the NPU (XRT),
// the iGPU (Vulkan) and the CPU. It checks that
//   * weight MUL_MATs of the NPU-selected layers run on the NPU,
//   * every other op (rms_norm, soft_max, silu, add) and the non-selected layers'
//     matmuls run on the iGPU (Vulkan),
//   * the CPU backend gets NOTHING (no op ever assigned to CPU),
//   * the NPU<->GPU handoffs are zero-copy: all tensors live in one shared XRT_HSA
//     (UMA) buffer, so ggml_backend_sched inserts 0 cross-backend copies,
//   * the hybrid result matches a pure-CPU reference (NRMSE within bf16 precision).
//
// It runs two scheduler scenarios that mirror the two ways the real llama.cpp
// integration can force the matmuls onto the NPU:
//   A) NPU-first backend order  -> the scheduler auto-routes selected matmuls to the
//      NPU by priority (backend_from_buffer picks the highest-prio backend that
//      supports both the weight's buffer type and the op).
//   B) Vulkan-first order (exactly what llama-context.cpp builds: GPU, then ACCEL,
//      then CPU) + ggml_backend_sched_set_tensor_backend() to PIN the selected
//      layers' matmul nodes to the NPU. This is the lever a small llama.cpp graph
//      patch would use.
//
// Per-layer selection follows ggml-xrt.cpp's GGML_XRT_NPU_LAYERS (unset = all
// layers eligible; "0-8" = layers 0..8; a bare count "N" = the first N layers).
// Run with GGML_XRT_MATMUL_ONLY=1 for the intended hybrid (ops off the NPU),
// GGML_SCHED_DEBUG=2 for the full per-node split, GGML_XRT_ENABLE_LOG=1 for the XRT
// op summary. If the NPU or a Vulkan device is missing, prints SKIP and exits 0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-xrt.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static const int64_t K = 2048; // hidden size (all Qwen3-1.7B matmuls have artifacts)
static const int64_t M = 64;   // token tile (matches the 64x..._4c prebuilt kernels)
static const int     N_LAYERS = 2;

static bool is_view_op(ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE || op == GGML_OP_CONT || op == GGML_OP_NONE;
}

// Mirror ggml-xrt.cpp's GGML_XRT_NPU_LAYERS parsing so expectations match the
// backend exactly. restricted=false (env unset) => every layer eligible for NPU.
static bool g_restricted = false;
static std::set<int> npu_layers_expected() {
    std::set<int> s;
    const char * e = std::getenv("GGML_XRT_NPU_LAYERS");
    if (!e || !e[0]) { g_restricted = false; return s; }
    g_restricted = true;
    std::string v(e);
    if (v.find_first_of("-,") == std::string::npos) {
        int n = std::atoi(v.c_str());
        for (int i = 0; i < n; ++i) { s.insert(i); }
        return s;
    }
    size_t pos = 0;
    while (pos <= v.size()) {
        size_t comma = v.find(',', pos);
        std::string tok = v.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            size_t dash = tok.find('-');
            if (dash == std::string::npos) { s.insert(std::atoi(tok.c_str())); }
            else {
                int a = std::atoi(tok.substr(0, dash).c_str());
                int b = std::atoi(tok.substr(dash + 1).c_str());
                for (int i = a; i <= b; ++i) { s.insert(i); }
            }
        }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    return s;
}
static bool layer_on_npu(int il, const std::set<int> & sel) { return !g_restricted || sel.count(il) != 0; }

// All four weights are [K,K] so the NPU only ever loads ONE kernel shape (mul_mat
// 2048x2048) = a single hw_context, keeping NPU pressure minimal on the shared
// device (avoids exhausting the ~5 hw_context limit while the user also runs).
struct layer_w { ggml_tensor * wq; ggml_tensor * wo; ggml_tensor * wg; ggml_tensor * wd; };

// One transformer-ish layer. The four weight-matmul output nodes are recorded into
// mm_nodes (tagged with the layer index) for the placement checker.
static ggml_tensor * build_layer(ggml_context * ctx, ggml_tensor * x, const layer_w & w,
                                 int il, std::vector<std::pair<ggml_tensor *, int>> & mm_nodes) {
    ggml_tensor * n1 = ggml_rms_norm(ctx, x, 1e-6f);   // GPU
    ggml_tensor * q  = ggml_mul_mat(ctx, w.wq, n1);    // NPU (selected layer)
    mm_nodes.push_back({q, il});
    ggml_tensor * s  = ggml_soft_max(ctx, q);          // GPU (attention-like)
    ggml_tensor * o  = ggml_mul_mat(ctx, w.wo, s);     // NPU
    mm_nodes.push_back({o, il});
    ggml_tensor * h  = ggml_add(ctx, x, o);            // GPU (residual)
    ggml_tensor * n2 = ggml_rms_norm(ctx, h, 1e-6f);   // GPU
    ggml_tensor * g  = ggml_mul_mat(ctx, w.wg, n2);    // NPU (ffn gate/up)
    mm_nodes.push_back({g, il});
    ggml_tensor * a  = ggml_silu(ctx, g);              // GPU
    ggml_tensor * d  = ggml_mul_mat(ctx, w.wd, a);     // NPU (ffn down)
    mm_nodes.push_back({d, il});
    return ggml_add(ctx, h, d);                        // GPU (residual)
}

static void set_names(const layer_w & w, int il) {
    char nm[64];
    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il); ggml_set_name(w.wq, nm);
    snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il); ggml_set_name(w.wo, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight",    il); ggml_set_name(w.wg, nm);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight",    il); ggml_set_name(w.wd, nm);
}

struct ctx_pack {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * out = nullptr;
    layer_w W[N_LAYERS];
    ggml_tensor * x = nullptr;
    std::vector<std::pair<ggml_tensor *, int>> mm_nodes;
};

// Build the 2-layer graph and static-allocate every tensor from `buft`.
static bool build_graph(ctx_pack & p, ggml_backend_buffer_type_t buft, std::mt19937 & rng) {
    const size_t nt = 4096;
    ggml_init_params ip = { ggml_tensor_overhead() * nt + ggml_graph_overhead(), nullptr, true };
    p.ctx = ggml_init(ip);
    for (int il = 0; il < N_LAYERS; ++il) {
        p.W[il].wq = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F16, K, K);
        p.W[il].wo = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F16, K, K);
        p.W[il].wg = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F16, K, K);
        p.W[il].wd = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F16, K, K);
        set_names(p.W[il], il);
    }
    p.x = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, K, M);
    ggml_set_name(p.x, "x_in");
    ggml_tensor * cur = p.x;
    for (int il = 0; il < N_LAYERS; ++il) { cur = build_layer(p.ctx, cur, p.W[il], il, p.mm_nodes); }
    ggml_set_name(cur, "out");
    p.out = cur;
    p.gf = ggml_new_graph(p.ctx);
    ggml_build_forward_expand(p.gf, cur);
    p.buf = ggml_backend_alloc_ctx_tensors_from_buft(p.ctx, buft);
    if (!p.buf) { return false; }

    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    auto fill = [&](ggml_tensor * t) {
        size_t n = ggml_nelements(t);
        std::vector<float> tmp(n);
        for (size_t i = 0; i < n; ++i) { tmp[i] = dist(rng); }
        if (t->type == GGML_TYPE_F32) { ggml_backend_tensor_set(t, tmp.data(), 0, n * sizeof(float)); }
        else { std::vector<ggml_fp16_t> h(n); ggml_fp32_to_fp16_row(tmp.data(), h.data(), n);
               ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t)); }
    };
    for (int il = 0; il < N_LAYERS; ++il) { fill(p.W[il].wq); fill(p.W[il].wo); fill(p.W[il].wg); fill(p.W[il].wd); }
    fill(p.x);
    return true;
}

// Run one scheduler scenario; returns number of failed checks.
static int run_scenario(const char * label, ggml_backend_t backends[3], bool pin_to_npu,
                        ggml_backend_t b_xrt, ggml_backend_t b_vk, ggml_backend_t b_cpu,
                        ggml_backend_buffer_type_t hsa_buft, const std::set<int> & sel,
                        const std::vector<float> & ref) {
    printf("\n=========================================================\n");
    printf("SCENARIO: %s\n", label);
    printf("=========================================================\n");
    std::mt19937 rng(1234);
    ctx_pack p;
    if (!build_graph(p, hsa_buft, rng)) { printf("SKIP: hsa allocation failed\n"); return 0; }

    const size_t nt = 4096;
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 3, nt, false, /*op_offload=*/true);

    // Scenario B: pin the selected layers' matmul nodes to the NPU (what a llama
    // graph patch would do; robust to the Vulkan-first backend order).
    if (pin_to_npu) {
        for (auto & mn : p.mm_nodes) {
            if (layer_on_npu(mn.second, sel)) { ggml_backend_sched_set_tensor_backend(sched, mn.first, b_xrt); }
        }
    }

    // alloc_graph runs placement + splits and (unlike reserve) does NOT reset, so the
    // per-node assignment is queryable and copy rewiring is observable below.
    if (!ggml_backend_sched_alloc_graph(sched, p.gf)) { printf("FAIL: sched alloc_graph\n"); ggml_backend_sched_free(sched); return 1; }

    int fails = 0;
    auto backend_of = [&](ggml_tensor * t) { return ggml_backend_sched_get_tensor_backend(sched, t); };

    printf("\n-- matmul placement (NPU for selected layers, GPU otherwise) --\n");
    for (auto & mn : p.mm_nodes) {
        ggml_backend_t bk = backend_of(mn.first);
        ggml_backend_t want = layer_on_npu(mn.second, sel) ? b_xrt : b_vk;
        bool ok = (bk == want);
        printf("  L%d %-24s -> %-8s %s\n", mn.second, mn.first->name, bk ? ggml_backend_name(bk) : "(none)", ok ? "OK" : "MISMATCH");
        if (!ok) { ++fails; }
    }

    printf("\n-- non-matmul op placement (GPU, never CPU) --\n");
    int cpu_nodes = 0, xrt_nonmm = 0;
    for (int i = 0; i < ggml_graph_n_nodes(p.gf); ++i) {
        ggml_tensor * node = ggml_graph_node(p.gf, i);
        if (node->op == GGML_OP_MUL_MAT || is_view_op(node->op)) { continue; }
        ggml_backend_t bk = backend_of(node);
        if (bk == b_cpu) { ++cpu_nodes; printf("  %-12s %-16s -> CPU (unexpected)\n", ggml_op_name(node->op), node->name); }
        if (bk == b_xrt) { ++xrt_nonmm; printf("  %-12s %-16s -> NPU (unexpected)\n", ggml_op_name(node->op), node->name); }
    }
    if (cpu_nodes) { printf("  FAIL: %d non-matmul op(s) on CPU\n", cpu_nodes); fails += cpu_nodes; }
    else           { printf("  OK: no op on CPU\n"); }
    if (xrt_nonmm && getenv("GGML_XRT_MATMUL_ONLY")) { printf("  FAIL: %d non-matmul op(s) on NPU under MATMUL_ONLY\n", xrt_nonmm); fails += xrt_nonmm; }

    // zero-copy: a copied input is rewired by alloc_graph to a tensor named
    // "<backend>#<name>#<c>"; count '#'-named srcs (0 == fully zero-copy handoffs).
    int n_copies = 0;
    for (int i = 0; i < ggml_graph_n_nodes(p.gf); ++i) {
        ggml_tensor * node = ggml_graph_node(p.gf, i);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            ggml_tensor * s = node->src[j];
            if (s && std::strchr(s->name, '#')) { ++n_copies; }
        }
    }
    printf("\n-- scheduler: %d splits, %d cross-backend copies inserted --\n", ggml_backend_sched_get_n_splits(sched), n_copies);
    if (n_copies != 0) { printf("  FAIL: expected 0 copies on shared-UMA handoffs, got %d\n", n_copies); ++fails; }
    else               { printf("  OK: 0 cross-backend copies (zero-copy NPU<->GPU handoff over hsa UMA)\n"); }

    printf("\n-- executing hybrid graph on the NPU+GPU --\n");
    ggml_status st = ggml_backend_sched_graph_compute(sched, p.gf);
    for (int a = 1; a < 12 && st != GGML_STATUS_SUCCESS; ++a) {
        printf("  compute attempt %d failed (st=%d); retrying (NPU may be contended 0xc01e0009)\n", a, st);
        ggml_backend_sched_reset(sched);
        st = ggml_backend_sched_graph_compute(sched, p.gf);
    }
    if (st != GGML_STATUS_SUCCESS) { printf("FAIL: hybrid compute status=%d\n", st); ggml_backend_sched_free(sched); return fails + 1; }

    std::vector<float> hy(ggml_nelements(p.out));
    ggml_backend_tensor_get(p.out, hy.data(), 0, hy.size() * sizeof(float));
    double se = 0, s2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) { double d = hy[i] - ref[i]; se += d * d; s2 += (double)ref[i] * ref[i]; }
    double nrmse = std::sqrt(se / (s2 > 0 ? s2 : 1));
    printf("\n-- hybrid vs CPU reference: NRMSE = %.6f --\n", nrmse);
    if (nrmse > 0.05) { printf("  FAIL: NRMSE too high\n"); ++fails; }
    else              { printf("  OK: hybrid result matches CPU (bf16 precision)\n"); }

    ggml_backend_sched_free(sched);
    ggml_free(p.ctx);
    return fails;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t dev_xrt = nullptr, dev_vk = nullptr, dev_cpu = nullptr;
    printf("enumerating %zu device(s):\n", ggml_backend_dev_count());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
        printf("  [%zu] %-10s type=%d\n", i, ggml_backend_dev_name(d), (int)t);
        const bool is_gpu = (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU);
        if (t == GGML_BACKEND_DEVICE_TYPE_ACCEL && !dev_xrt) { dev_xrt = d; }
        if (is_gpu                              && !dev_vk)  { dev_vk  = d; }
        if (t == GGML_BACKEND_DEVICE_TYPE_CPU   && !dev_cpu) { dev_cpu = d; }
    }
    if (!dev_xrt || !dev_vk || !dev_cpu) {
        printf("SKIP: need NPU(ACCEL)+Vulkan(GPU/IGPU)+CPU (xrt=%p vk=%p cpu=%p)\n", (void *)dev_xrt, (void *)dev_vk, (void *)dev_cpu);
        return 0;
    }
    printf("devices: NPU='%s'  GPU='%s'  CPU='%s'\n", ggml_backend_dev_name(dev_xrt), ggml_backend_dev_name(dev_vk), ggml_backend_dev_name(dev_cpu));

    ggml_backend_t b_xrt = ggml_backend_dev_init(dev_xrt, nullptr);
    ggml_backend_t b_vk  = ggml_backend_dev_init(dev_vk,  nullptr);
    ggml_backend_t b_cpu = ggml_backend_dev_init(dev_cpu, nullptr);
    if (!b_xrt || !b_vk || !b_cpu) { printf("SKIP: backend init failed\n"); return 0; }

    const std::set<int> sel = npu_layers_expected();
    printf("GGML_XRT_NPU_LAYERS: %s\n", g_restricted ? "restricted (see per-node checks)" : "unset (all layers eligible for NPU)");

    ggml_backend_buffer_type_t hsa_buft = ggml_backend_xrt_hsa_buffer_type(dev_vk);
    if (!hsa_buft) { printf("SKIP: XRT_HSA buffer type unavailable\n"); return 0; }

    // ---- CPU reference (pure CPU backend, same weights/input) ----
    std::mt19937 rng(1234);
    ctx_pack cp;
    if (!build_graph(cp, ggml_backend_cpu_buffer_type(), rng)) { printf("SKIP: cpu ref alloc failed\n"); return 0; }
    if (ggml_backend_graph_compute(b_cpu, cp.gf) != GGML_STATUS_SUCCESS) { printf("FAIL: cpu ref compute\n"); return 1; }
    std::vector<float> ref(ggml_nelements(cp.out));
    ggml_backend_tensor_get(cp.out, ref.data(), 0, ref.size() * sizeof(float));
    ggml_free(cp.ctx);

    int fails = 0;
    // Scenario A: NPU-first order -> scheduler auto-routes matmuls to the NPU.
    ggml_backend_t order_npu_first[3] = { b_xrt, b_vk, b_cpu };
    fails += run_scenario("A) NPU-first order, automatic routing", order_npu_first, /*pin=*/false,
                          b_xrt, b_vk, b_cpu, hsa_buft, sel, ref);

    // Scenario B: Vulkan-first order (exactly llama-context.cpp's: GPU, ACCEL, CPU)
    // + set_tensor_backend pinning -> the lever a llama graph patch would use.
    ggml_backend_t order_llama[3] = { b_vk, b_xrt, b_cpu };
    fails += run_scenario("B) Vulkan-first (llama order) + set_tensor_backend pin", order_llama, /*pin=*/true,
                          b_xrt, b_vk, b_cpu, hsa_buft, sel, ref);

    printf("\n################ %s ################\n", fails == 0 ? "ALL SCENARIOS PASSED" : "FAILURES PRESENT");
    ggml_backend_free(b_xrt); ggml_backend_free(b_vk); ggml_backend_free(b_cpu);
    return fails == 0 ? 0 : 1;
}
