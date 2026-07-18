// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// ggml-xrt: Windows/Linux backend that dispatches AIE (XDNA NPU) kernels through
// the XRT runtime. Kernels are precompiled ahead-of-time to `.xclbin` (+ a control
// instruction sequence) by the mlir-aie / IRON toolchain (see src/ggml-hsa/kernels
// and src/ggml-xrt/kernels), and loaded here at runtime via xrt::hw_context /
// xrt::kernel.
//
// Design (see docs/ggml-xrt-plan.md):
//   * Unified memory: tensor buffers are XRT host-visible bo's, so the same
//     allocation is visible to CPU and NPU with no copy (buffer type reports
//     is_host = true). GPU (Vulkan) zero-copy requires external-memory import on
//     the Vulkan side (Windows work).
//   * AOT-only: supports_op returns true for an op ONLY if a matching precompiled
//     xclbin exists (there is no JIT on Windows). The ggml scheduler then routes
//     everything else to the GPU/CPU. This isolates the AOT limitation to the NPU.
//   * Dynamic M: MUL_MAT dispatch tiles the token dimension M on the host over a
//     fixed small-M kernel (mlir-aie bakes M into the DMA descriptors).
//
// HARDWARE-VALIDATION NOTE: this file compiles against the XRT C++ API (identical
// on Windows and Linux) but has NOT been executed on an NPU (none is present in the
// dev/WSL environment). Points that require on-hardware validation are marked
// TODO(hw): kernel arg layout / b-col-major, bo memory groups, and the exact
// instruction-file format.

#include "ggml-xrt.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_module.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

// GGML_XRT_ENABLE_LOG: 0/unset = off, 1/true/on = normal (the per-graph op summary
// is COLLAPSED — decode issues the same graph every token, so an unchanged summary
// is printed once and repeats are counted), 2 = verbose (every graph prints).
static int ggml_xrt_log_level() {
    static const int level = [] {
        const char * env = std::getenv("GGML_XRT_ENABLE_LOG");
        if (env == nullptr) { return 0; }
        if (std::strcmp(env, "2") == 0) { return 2; }
        if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
            std::strcmp(env, "on") == 0) { return 1; }
        return 0;
    }();
    return level;
}

static bool ggml_xrt_logging_enabled() {
    return ggml_xrt_log_level() > 0;
}

#define GGML_XRT_LOG_INFO(...)                          \
    do {                                                \
        if (ggml_xrt_logging_enabled()) {               \
            fprintf(stderr, "[ggml-xrt] " __VA_ARGS__); \
            fprintf(stderr, "\n");                      \
        }                                               \
    } while (0)

#define GGML_XRT_LOG_WARN(...)                              \
    do {                                                    \
        fprintf(stderr, "[ggml-xrt][warn] " __VA_ARGS__);   \
        fprintf(stderr, "\n");                              \
    } while (0)

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

namespace {

// XRT kernel name inside the mlir-aie xclbins (constant across designs).
constexpr const char * GGML_XRT_KERNEL_NAME = "MLIR_AIE";

// Directory holding precompiled `*.xclbin` + `*_insts.*` artifacts.
std::string ggml_xrt_kernel_dir() {
    if (const char * env = std::getenv("GGML_XRT_KERNEL_DIR")) {
        return std::string(env);
    }
    return std::string();
}

// AIE arch string for the current device family. aie2 = Phoenix/Hawk (NPU1).
constexpr const char * GGML_XRT_ARCH = "aie2";

} // namespace

// ---------------------------------------------------------------------------
// Kernel cache: resolve an op+shape to a precompiled xclbin, load it once.
// ---------------------------------------------------------------------------

struct ggml_xrt_kernel {
    xrt::hw_context        context;
    xrt::kernel            kernel;
    std::vector<uint8_t>   instr;          // control instruction sequence
    size_t                 instr_words = 0;
    std::optional<xrt::bo> instr_bo;       // instruction bo (group 1), uploaded once
};

// Build the deterministic prefix used to locate a MUL_MAT artifact for a given
// (K, N) and dtypes. The M tile and column count are matched by glob (any tier),
// since M is handled by host tiling.
static std::string ggml_xrt_mul_mat_prefix(const char * dti, const char * dto) {
    std::ostringstream oss;
    oss << "mul_mat_" << GGML_XRT_ARCH << "_" << dti << "_" << dto << "_";
    // filenames are <prefix>256x{K}x{N}_4c.xclbin (prefill) or 32x{K}x{N}_1c / 128x..._4c (decode)
    return oss.str();
}

// Locate an xclbin for MUL_MAT(K,N,dtypes). Picks the kernel whose M tile best
// fits the actual token count `m_want`: the LARGEST tile <= m_want (so a prefill
// of M tokens uses one big-tile launch instead of many small ones), falling back
// to the smallest tile when m_want is below every tile (decode, M=1). Empty if
// none found. Pass m_want <= 0 to just probe existence (returns smallest tile).
static std::filesystem::path ggml_xrt_find_mul_mat_xclbin(int64_t K, int64_t N,
                                                          const char * dti, const char * dto,
                                                          int64_t m_want, int * out_m_tile) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty() || !fs::exists(dir)) {
        return {};
    }
    const std::string prefix = ggml_xrt_mul_mat_prefix(dti, dto);
    const std::string kn = "x" + std::to_string(K) + "x" + std::to_string(N) + "_";

    // Parse the AIE column count from the "_<n>c" suffix (e.g. _4c/_2c/_1c); higher
    // is preferred (the full 4-column array is fastest — 4c beats 2c ~2.3x, and
    // beats two concurrent 2c on disjoint partitions; benchmarked). Default 1.
    auto parse_cols = [](const std::string & fn) -> int {
        auto c = fn.rfind('c');                       // "..._4c.xclbin" -> the 'c' before ".xclbin"
        if (c == std::string::npos || c < 1) { return 1; }
        // walk back over digits after a '_'
        size_t e = c, s = e;
        while (s > 0 && fn[s-1] >= '0' && fn[s-1] <= '9') { --s; }
        if (s < e && s > 0 && fn[s-1] == '_') { return std::atoi(fn.substr(s, e-s).c_str()); }
        return 1;
    };

    fs::path best_le;  int best_le_m  = 0;          int best_le_cols  = 0;  // largest tile <= m_want, prefer more cols
    fs::path best_min; int best_min_m = INT32_MAX;  int best_min_cols = 0;  // smallest tile (fallback), prefer more cols
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        const std::string fn = p.filename().string();
        if (fn.rfind(prefix, 0) != 0) { continue; }        // must start with prefix
        if (fn.find(kn) == std::string::npos) { continue; } // must contain xKxN_
        if (fn.find("_gemv") != std::string::npos) { continue; } // gemv kernels use a
        // different ABI (untransposed A@grp3, no M loop) — not the tiled matmul path.
        // They need the dedicated M==1 gemv branch (TODO); until then the tiled path
        // must not pick them up (would run them with the wrong transposed layout).
        // parse the leading M tile: <prefix><M>x<K>x<N>...
        const std::string tail = fn.substr(prefix.size());
        int m = std::atoi(tail.c_str());
        if (m <= 0) { continue; }
        const int cols = parse_cols(fn);
        // largest M tile <= m_want; on an equal tile, prefer the higher column count (4c).
        if (m <= m_want && (m > best_le_m || (m == best_le_m && cols > best_le_cols))) {
            best_le_m = m; best_le_cols = cols; best_le = p;
        }
        // fallback: smallest M tile; on an equal tile, prefer the higher column count.
        if (m < best_min_m || (m == best_min_m && cols > best_min_cols)) {
            best_min_m = m; best_min_cols = cols; best_min = p;
        }
    }
    if (!best_le.empty()) { if (out_m_tile) { *out_m_tile = best_le_m;  } return best_le; }
    if (!best_min.empty()){ if (out_m_tile) { *out_m_tile = best_min_m; } return best_min; }
    return {};
}

// Locate the dedicated M=1 decode gemv xclbin for (K,N), named
// mul_mat_<arch>_bf16_f32_1x<K>x<N>_gemv.xclbin. Empty if none.
static std::filesystem::path ggml_xrt_find_gemv_xclbin(int64_t K, int64_t N) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty() || !fs::exists(dir)) { return {}; }
    const std::string needle = std::string("mul_mat_") + GGML_XRT_ARCH + "_bf16_f32_1x"
                             + std::to_string(K) + "x" + std::to_string(N) + "_gemv";
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        if (p.filename().string().rfind(needle, 0) == 0) { return p; }
    }
    return {};
}

// ---------------------------------------------------------------------------
// Native-quant decode gemv (Q4_0 / Q4_K / Q6_K): the NPU dequantizes on-chip, so
// the host uploads the RAW QUANTIZED weight instead of a BF16 copy. This removes
// the ~4x BF16 expansion that the tiled path's weight cache holds resident (the
// whole point — see docs/ggml-xrt-linux-kernel-wishlist.md section 7).
//
// The kernels take ONE weight buffer of fixed-size "records", one per quant block
// per output row: row-major [N][K/blk][rec]. A record is the ggml block with its
// f16 scale(s) widened to f32 (and, for Q4_K, the fields reordered) — raw ggml
// blocks are DMA-hostile (18/144/210-byte strides) and separate scale streams
// would exceed the shim's 2 read-DMA channels.
//
// All three repack layouts and the dequant math are HARDWARE-VALIDATED against a
// CPU reference (NRMSE 0.00000 on all nine Qwen3-1.7B decode kernels) via
// C:\dev\xrt-sdk\work\q4_gemv_check.cpp.
// ---------------------------------------------------------------------------

#pragma pack(push, 1)
struct ggml_xrt_blk_q4_0 { uint16_t d; uint8_t qs[16]; };                              // 18
struct ggml_xrt_blk_q4_K { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; }; // 144
struct ggml_xrt_blk_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; }; // 210
#pragma pack(pop)
static_assert(sizeof(ggml_xrt_blk_q4_0) == 18,  "block_q4_0 layout drift");
static_assert(sizeof(ggml_xrt_blk_q4_K) == 144, "block_q4_K layout drift");
static_assert(sizeof(ggml_xrt_blk_q6_K) == 210, "block_q6_K layout drift");

// Filename dtype token for a natively-supported quant weight, or nullptr.
static const char * ggml_xrt_quant_token(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_K: return "q4k";
        case GGML_TYPE_Q6_K: return "q6k";
        default:             return nullptr;
    }
}

// GGML_XRT_NATIVE_QUANT: feed quantized weights to the NPU raw (on-chip dequant).
static bool ggml_xrt_native_quant_enabled() {
    static const bool en = []() {
        const char * e = std::getenv("GGML_XRT_NATIVE_QUANT");
        return e && e[0] && e[0] != '0';
    }();
    return en;
}

// Repacked record size in bytes per quant block (0 if unsupported).
static size_t ggml_xrt_quant_rec_bytes(ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_0: return 20;   // qs[16] + f32 d
        case GGML_TYPE_Q4_K: return 148;  // qs[128] + scales[12] + f32 d + f32 dmin
        case GGML_TYPE_Q6_K: return 212;  // ql[128] + qh[64] + scales[16] + f32 d
        default:             return 0;
    }
}

static inline float ggml_xrt_f16_bits_to_f32(uint16_t bits) {
    ggml_fp16_t h;
    std::memcpy(&h, &bits, sizeof(h));
    return ggml_fp16_to_fp32(h);
}

// Repack a ggml quantized weight [N,K] into the kernel's [N][K/blk][rec] buffer.
// Weight stays in ggml-native [N,K] order — the quant gemv does NOT transpose.
static void ggml_xrt_repack_quant_weight(ggml_type t, const void * src, uint8_t * dst,
                                         int64_t N, int64_t K) {
    const int64_t blk     = ggml_blck_size(t);
    const int64_t nblocks = K / blk;
    const size_t  src_row = ggml_row_size(t, K);
    const size_t  rec     = ggml_xrt_quant_rec_bytes(t);

    for (int64_t n = 0; n < N; ++n) {
        const uint8_t * s = (const uint8_t *) src + (size_t) n * src_row;
        uint8_t       * d = dst + (size_t) n * nblocks * rec;
        for (int64_t b = 0; b < nblocks; ++b) {
            uint8_t * r = d + (size_t) b * rec;
            switch (t) {
                case GGML_TYPE_Q4_0: {
                    const auto * x = (const ggml_xrt_blk_q4_0 *) s + b;
                    std::memcpy(r, x->qs, 16);
                    const float fd = ggml_xrt_f16_bits_to_f32(x->d);
                    std::memcpy(r + 16, &fd, 4);
                } break;
                case GGML_TYPE_Q4_K: {
                    const auto * x = (const ggml_xrt_blk_q4_K *) s + b;
                    std::memcpy(r,       x->qs,     128);
                    std::memcpy(r + 128, x->scales,  12);
                    const float fd  = ggml_xrt_f16_bits_to_f32(x->d);
                    const float fdm = ggml_xrt_f16_bits_to_f32(x->dmin);
                    std::memcpy(r + 140, &fd,  4);
                    std::memcpy(r + 144, &fdm, 4);
                } break;
                case GGML_TYPE_Q6_K: {
                    const auto * x = (const ggml_xrt_blk_q6_K *) s + b;
                    std::memcpy(r,       x->ql,     128);
                    std::memcpy(r + 128, x->qh,      64);
                    std::memcpy(r + 192, x->scales,  16);
                    const float fd = ggml_xrt_f16_bits_to_f32(x->d);
                    std::memcpy(r + 208, &fd, 4);
                } break;
                default: GGML_ABORT("ggml-xrt: unsupported native-quant type");
            }
        }
    }
}

// Locate the native-quant PREFILL fused matmul xclbin for (K,N,qtype), named
// mul_mat_<arch>_<qtok>_f32_32x<K>x<N>_mm.xclbin. M is baked at 32 per dispatch.
static std::filesystem::path ggml_xrt_find_quant_mm_xclbin(int64_t K, int64_t N,
                                                           const char * qtok) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty() || !qtok || !fs::exists(dir)) { return {}; }
    const std::string needle = std::string("mul_mat_") + GGML_XRT_ARCH + "_" + qtok + "_f32_32x"
                             + std::to_string(K) + "x" + std::to_string(N) + "_mm";
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        if (p.filename().string().rfind(needle, 0) == 0) { return p; }
    }
    return {};
}

// QUARANTINE: prefill mm shapes that are built but FAIL hardware validation.
// Reported upstream to the kernel side; until a fixed xclbin lands, these fall
// back to the (correct) host-BF16-dequant tiled path rather than produce garbage.
//
//   q6k 6144x2048 - rows 16..31 of every M=32 tile are wrong (per-row NRMSE ~0.60
//   vs ~0.003 for rows 0..15; overall NRMSE 0.425, max_abs_err 113). Reproduced on
//   seeds {1,3,9} and a uniform activation. q4_0/q4k at the same 6144x2048 shape
//   and q6k at K=2048 all pass, so it is specific to (q6k, K=6144).
static bool ggml_xrt_quant_mm_quarantined(const char * qtok, int64_t K, int64_t N) {
    if (!qtok) { return false; }
    if (std::strcmp(qtok, "q6k") == 0 && K == 6144 && N == 2048) { return true; }
    return false;
}

// ---------------------------------------------------------------------------
// Decode overlay + per-shape ELF modules (fixes hw_context thrashing)
//
// Phoenix allows only ~5 concurrent hw_contexts, and one xclbin == one context, so
// a model touching more distinct decode kernels than the LRU cap (default 4) evicts
// and re-registers an xclbin on EVERY layer — the dominant cost once several shapes
// are live. Qwen3-1.7B-Q4_K_M decode alone wants ~8 distinct kernels.
//
// The fix (see wishlist section 8): the gemv core's loop bounds depend on K only,
// not on the output N, so all shapes sharing a (dtype,K) share one overlay xclbin
// and differ only in their instruction sequence. Register ONE context per (dtype,K)
// and load each shape as a cheap xrt::module ELF into it:
//     overlays/<dtype>_k<K>_overlay.xclbin   register once per (dtype,K)
//     overlays/<dtype>_1x{K}x{N}_gemv.elf    per-shape instruction module
// Qwen3-1.7B decode then needs 2 contexts (K=2048, K=6144) for ALL projections
// instead of one per shape — no eviction, no thrash. Dispatch differs from the
// xclbin path: instructions come from the module, so args are (opcode, 0, 0, bo...).
// ---------------------------------------------------------------------------

// A shape kernel bound to a shared overlay context. Holds the module alive.
struct ggml_xrt_module_kernel {
    xrt::elf         elf;
    xrt::module      mod;
    xrt::ext::kernel kernel;
    std::shared_ptr<xrt::hw_context> ctx;   // keeps the shared context alive
    // declaration order == construction order: elf -> module(elf) -> kernel(ctx, mod)
    ggml_xrt_module_kernel(const std::string & elf_path, const xrt::hw_context & c,
                           const char * name)
        : elf(elf_path), mod(elf), kernel(c, mod, name) {}
};

static std::filesystem::path ggml_xrt_overlay_dir() {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty()) { return {}; }
    // overlays/ lives at the root of the prebuilt tree; also accept a direct point-at.
    fs::path root(dir);
    std::error_code ec;
    if (fs::exists(root / "overlays", ec)) { return root / "overlays"; }
    if (root.filename() == "overlays" && fs::exists(root, ec)) { return root; }
    return {};
}

// Resolve the (overlay, elf) pair for a decode gemv shape, or empty if not built.
static bool ggml_xrt_find_overlay_shape(const char * dt, int64_t K, int64_t N,
                                        std::filesystem::path & overlay,
                                        std::filesystem::path & elf) {
    namespace fs = std::filesystem;
    const fs::path od = ggml_xrt_overlay_dir();
    if (od.empty() || !dt) { return false; }
    overlay = od / (std::string(dt) + "_k" + std::to_string(K) + "_overlay.xclbin");
    elf     = od / (std::string(dt) + "_1x" + std::to_string(K) + "x" + std::to_string(N) + "_gemv.elf");
    std::error_code ec;
    return fs::exists(overlay, ec) && fs::exists(elf, ec);
}

// Locate the native-quant M=1 decode gemv xclbin for (K,N,qtype), named
// mul_mat_<arch>_<qtok>_f32_1x<K>x<N>_gemv.xclbin. Empty if none.
static std::filesystem::path ggml_xrt_find_quant_gemv_xclbin(int64_t K, int64_t N,
                                                             const char * qtok) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty() || !qtok || !fs::exists(dir)) { return {}; }
    const std::string needle = std::string("mul_mat_") + GGML_XRT_ARCH + "_" + qtok + "_f32_1x"
                             + std::to_string(K) + "x" + std::to_string(N) + "_gemv";
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        if (p.filename().string().rfind(needle, 0) == 0) { return p; }
    }
    return {};
}

// ---------------------------------------------------------------------------
// XRT device layer (lazy, exception-guarded, single device for the scaffold)
// ---------------------------------------------------------------------------

namespace {

struct ggml_xrt_device {
    int32_t                    index       = 0;
    bool                       probed      = false;
    bool                       available   = false;
    std::string                description = "AMD XDNA NPU (XRT)";
    std::optional<xrt::device> device;
    std::mutex                 mutex;

    // op-shape -> loaded kernel, with an LRU cap: the NPU allows only a small
    // number of concurrent hw_contexts (Phoenix/XRT 2.21 = 5; the 6th create
    // fails 0xc01e0009). A full model touches more distinct kernels than that, so
    // we evict the least-recently-used context when the cap is hit and reload on
    // demand. Cap via GGML_XRT_MAX_CONTEXTS (default 4, leaving headroom under 5).
    //
    // IMPORTANT: `max_contexts` is the budget for ALL hw_contexts this backend
    // holds, i.e. `kernels` PLUS `overlay_ctxs` (see below), not for `kernels`
    // alone. Counting only the LRU pool let prefill fill the budget with
    // per-shape kernels, after which the first decode overlay context failed to
    // create (0xc01e0009) and silently fell back to the thrashing path.
    //
    // The default stays 4. A pre-fix hardware run did hold 5 contexts (4 kernels
    // + 1 overlay) successfully and only failed on the 6th, so 5 looks reachable,
    // but that measured this process alone: the limit is per-device, shared with
    // anything else on the NPU, so 4 keeps a slot of headroom. Raise it with
    // GGML_XRT_MAX_CONTEXTS=5 to trade that headroom for one more resident
    // context (worth it when a model spans >2 distinct decode K values).
    std::unordered_map<std::string, std::shared_ptr<ggml_xrt_kernel>> kernels;
    std::list<std::string> kernel_lru;   // front = most-recently-used
    size_t max_contexts = []() {
        const char * e = std::getenv("GGML_XRT_MAX_CONTEXTS");
        int v = e ? std::atoi(e) : 0;
        return (size_t)(v > 0 ? v : 4);
    }();

    // weight src data ptr -> dequantized+transposed BF16 device bo (weights are
    // constant, so this is built once per weight tensor and reused every token).
    std::mutex weight_mutex;
    std::unordered_map<const void *, std::shared_ptr<xrt::bo>> weight_bos;
    // gemv (M=1 decode) weights: same key but UNtransposed [N,K] layout (the gemv
    // kernel wants A in ggml-native order), so a separate cache from weight_bos.
    std::unordered_map<const void *, std::shared_ptr<xrt::bo>> gemv_weight_bos;
    // native-quant gemv weights: repacked RAW QUANT records [N][K/blk][rec], no BF16
    // expansion. Keyed by host ptr **plus dtype and shape**, NOT the ptr alone: a
    // freed weight's address can be recycled by a DIFFERENT weight, and a pointer-only
    // key then returns a buffer sized for the old shape, so the kernel reads past its
    // end (observed on hardware as NaN output). Weights are constant within a model,
    // so this stays a one-time repack per weight.
    // NOTE: the two BF16 caches above are still pointer-keyed and carry the same
    // latent risk; they have not been reworked because that code is validated and
    // in-model weight lifetimes make recycling unlikely. Worth unifying later.
    std::unordered_map<std::string, std::shared_ptr<xrt::bo>> quant_weight_bos;

    // Decode overlay contexts, keyed "<dtype>_k<K>". Never EVICTED by the LRU:
    // there is one per distinct K (2 for Qwen3-1.7B), which is the whole point —
    // they must stay resident or we are back to per-layer re-registration. Shape
    // modules are cheap and hang off the context they were built against.
    // They are, however, COUNTED against `max_contexts`: each live overlay context
    // permanently reserves a slot, shrinking the LRU pool's effective cap to
    // `max_contexts - overlay_ctxs.size()`.
    std::unordered_map<std::string, std::shared_ptr<xrt::hw_context>>          overlay_ctxs;
    std::unordered_map<std::string, std::shared_ptr<ggml_xrt_module_kernel>>   overlay_kernels;
    std::unordered_map<std::string, int>                                       overlay_shapes_per_ctx;

    // Fetch (or create) the shape kernel for a decode gemv, loading its overlay
    // context on first use. Returns nullptr if the artifacts are absent or XRT
    // rejects them — callers then fall back to the per-shape xclbin path.
    std::shared_ptr<ggml_xrt_module_kernel> load_overlay_kernel(const char * dt, int64_t K, int64_t N) {
        // GGML_XRT_OVERLAY=0 forces the per-shape xclbin path (A/B + escape hatch).
        static const bool overlay_enabled = []() {
            const char * e = std::getenv("GGML_XRT_OVERLAY");
            return !(e && e[0] == '0');
        }();
        if (!overlay_enabled) { return nullptr; }
        std::filesystem::path overlay, elf;
        if (!ggml_xrt_find_overlay_shape(dt, K, N, overlay, elf)) { return nullptr; }

        const std::string skey = std::string(dt) + "_1x" + std::to_string(K) + "x" + std::to_string(N);
        std::lock_guard<std::mutex> lock(mutex);
        // Several shape modules coexisting on one context is fine — verified on
        // hardware with 2 modules on the K=2048 context across repeated cycles.
        if (auto it = overlay_kernels.find(skey); it != overlay_kernels.end()) { return it->second; }
        if (!available || !device) { return nullptr; }

        const std::string okey = std::string(dt) + "_k" + std::to_string(K);
        try {
            std::shared_ptr<xrt::hw_context> ctx;
            if (auto it = overlay_ctxs.find(okey); it != overlay_ctxs.end()) {
                ctx = it->second;
            } else {
                // A new overlay context needs a slot out of the shared budget.
                // Overlay contexts are long-lived and shared across every layer,
                // so they outrank per-shape xclbin kernels: evict from the LRU to
                // make room. Keep at least one slot for the LRU pool, otherwise
                // load_kernel could never succeed and ordinary ops would fail
                // outright instead of falling back.
                if (overlay_ctxs.size() + 1 > max_overlay_contexts()) {
                    GGML_XRT_LOG_INFO("overlay budget exhausted for %s (%zu overlay contexts, "
                                      "total cap %zu) - using per-shape xclbin",
                                      okey.c_str(), overlay_ctxs.size(), max_contexts);
                    return nullptr;
                }
                if (kernels.size() + overlay_ctxs.size() + 1 > max_contexts) {
                    GGML_XRT_LOG_INFO("reserving a context slot for overlay %s", okey.c_str());
                    // room for the new overlay ctx => kernels must fit in
                    // max_contexts - (overlay_ctxs.size() + 1).
                    evict_kernels_locked(max_contexts - (overlay_ctxs.size() + 1), "overlay reserve");
                }
                if (kernels.size() + overlay_ctxs.size() + 1 > max_contexts) {
                    GGML_XRT_LOG_INFO("no free context slot for overlay %s - using per-shape xclbin",
                                      okey.c_str());
                    return nullptr;
                }
                xrt::xclbin xcl(overlay.string());
                auto uuid = device->register_xclbin(xcl);
                ctx = std::make_shared<xrt::hw_context>(*device, uuid);
                overlay_ctxs.emplace(okey, ctx);
                GGML_XRT_LOG_INFO("registered overlay %s (1 context serves every N at this K)",
                                  okey.c_str());
            }
            auto mk = std::make_shared<ggml_xrt_module_kernel>(elf.string(), *ctx, "MLIR_AIE");
            mk->ctx = ctx;
            overlay_kernels.emplace(skey, mk);
            ++overlay_shapes_per_ctx[okey];
            GGML_XRT_LOG_INFO("loaded shape module %s", skey.c_str());
            return mk;
        } catch (const std::exception & ex) {
            GGML_XRT_LOG_INFO("overlay path unavailable for %s (%s) - using per-shape xclbin",
                              skey.c_str(), ex.what());
            return nullptr;
        }
    }

    // Pool of reusable activation/output bo's, keyed by (kernel key + role). Unlike
    // the weight bo the contents change every call, but the ALLOCATION (size/group)
    // is fixed per shape, so repeated matmuls of the same shape reuse the buffer
    // instead of re-allocating + re-mapping it on every dispatch/tile.
    std::mutex io_mutex;
    std::unordered_map<std::string, std::shared_ptr<xrt::bo>> io_bos;

    // Fetch (or allocate on first use) a pooled bo for `key` of exactly `size`
    // bytes in `group`. Caller uses it serially (graph_compute is not re-entrant
    // per backend), matching the existing weight-bo reuse contract.
    std::shared_ptr<xrt::bo> get_io_bo(const std::string & key, size_t size,
                                       xrt::bo::flags flags, int group) {
        std::lock_guard<std::mutex> lk(io_mutex);
        auto it = io_bos.find(key);
        if (it != io_bos.end()) { return it->second; }
        auto bo = std::make_shared<xrt::bo>(*device, size, flags, group);
        io_bos.emplace(key, bo);
        return bo;
    }

    explicit ggml_xrt_device(int32_t i) : index(i) {}

    bool ensure_open() {
        std::lock_guard<std::mutex> lock(mutex);
        if (probed) { return available; }
        probed = true;
        try {
            device.emplace(static_cast<unsigned int>(index));
            try { description = device->get_info<xrt::info::device::name>(); }
            catch (const std::exception &) {}
            available = true;
            GGML_XRT_LOG_INFO("opened device %d: %s", index, description.c_str());
        } catch (const std::exception & ex) {
            available = false;
            GGML_XRT_LOG_INFO("device %d unavailable: %s", index, ex.what());
        }
        return available;
    }

    // Load (and cache) the kernel for a resolved xclbin path.
    std::shared_ptr<ggml_xrt_kernel> load_kernel(const std::string & key,
                                                 const std::filesystem::path & xclbin,
                                                 const std::filesystem::path & insts) {
        std::lock_guard<std::mutex> lock(mutex);
        if (auto it = kernels.find(key); it != kernels.end()) {
            kernel_lru.remove(key); kernel_lru.push_front(key);   // promote to MRU
            return it->second;
        }
        if (!available || !device) { return nullptr; }
        // Evict least-recently-used contexts until there's room to create a new one.
        // Safe because dispatches are serial (each op finishes run.wait() before the
        // next load_kernel), so an evicted kernel is not in flight; dropping the map's
        // shared_ptr destroys its hw_context and frees the NPU context slot.
        // Live overlay contexts occupy slots out of the same budget, so the LRU
        // pool's effective cap is what is left over (at least 1, see
        // max_overlay_contexts()).
        evict_kernels_locked(lru_cap(), "context cap");
        try {
            auto k = std::make_shared<ggml_xrt_kernel>();
            // NPU (aie2) path: register the xclbin and open a hw_context on its
            // uuid. device.load_xclbin() maps to the legacy load_axlf ioctl which
            // the XDNA/NPU shim rejects ("load_axlf: not supported").
            xrt::xclbin xcl(xclbin.string());
            auto uuid   = device->register_xclbin(xcl);
            k->context  = xrt::hw_context(*device, uuid);
            k->kernel   = xrt::kernel(k->context, GGML_XRT_KERNEL_NAME);
            k->instr    = ggml_xrt_read_instrs(insts, &k->instr_words);
            // Upload the (constant) instruction sequence once and cache the bo, so
            // every dispatch skips a fresh alloc + memcpy + sync of the instrs.
            k->instr_bo.emplace(*device, k->instr.size(), xrt::bo::flags::cacheable,
                                k->kernel.group_id(1));
            std::memcpy(k->instr_bo->map<void *>(), k->instr.data(), k->instr.size());
            k->instr_bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            kernels[key] = k;
            kernel_lru.push_front(key);
            GGML_XRT_LOG_INFO("loaded kernel %s (%zu instr words)", key.c_str(), k->instr_words);
            return k;
        } catch (const std::exception & ex) {
            GGML_XRT_LOG_WARN("failed to load kernel %s: %s", key.c_str(), ex.what());
            return nullptr;
        }
    }

  private:
    // Most overlay contexts we will hold: always leave one slot for the LRU pool
    // so load_kernel can still make progress (an op with no context available
    // fails outright, whereas an overlay with no context merely falls back).
    size_t max_overlay_contexts() const {
        return max_contexts > 1 ? max_contexts - 1 : 0;
    }

    // Effective cap on the LRU xclbin-kernel pool, given the slots currently
    // reserved by live overlay contexts. Never 0 (see max_overlay_contexts()).
    size_t lru_cap() const {
        return max_contexts > overlay_ctxs.size() ? max_contexts - overlay_ctxs.size() : 1;
    }

    // Evict least-recently-used xclbin kernels until `kernels.size() < cap`, i.e.
    // until there is room for one more context. Safe because dispatches are
    // serial (each op finishes run.wait() before the next load), so an evicted
    // kernel is not in flight; dropping the map's shared_ptr destroys its
    // hw_context and frees the NPU context slot.
    // PRECONDITION: `mutex` is ALREADY held by the caller (both load_kernel and
    // load_overlay_kernel lock it) — this must not lock, that would deadlock.
    void evict_kernels_locked(size_t cap, const char * why) {
        while (kernels.size() >= cap && !kernel_lru.empty()) {
            const std::string victim = kernel_lru.back();
            kernel_lru.pop_back();
            kernels.erase(victim);
            GGML_XRT_LOG_INFO("evicted kernel %s (%s; total budget %zu, %zu overlay contexts)",
                              victim.c_str(), why, max_contexts, overlay_ctxs.size());
        }
    }

    // Read an instruction file. The toolchain emits either a raw uint32 blob or
    // whitespace-separated hex words, and it does not use the extension
    // consistently (some `_insts.txt` files are actually the raw binary blob).
    // So sniff by content: only parse as hex text if every byte is printable
    // ASCII / whitespace; otherwise use the raw bytes verbatim.
    static std::vector<uint8_t> ggml_xrt_read_instrs(const std::filesystem::path & path,
                                                     size_t * out_words) {
        std::ifstream in(path, std::ios::binary);
        std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());

        bool looks_text = !raw.empty();
        for (uint8_t b : raw) {
            if (!(b == '\t' || b == '\n' || b == '\r' || (b >= 0x20 && b <= 0x7e))) {
                looks_text = false;
                break;
            }
        }

        std::vector<uint8_t> bytes;
        if (looks_text) {
            std::istringstream ss(std::string(raw.begin(), raw.end()));
            std::string tok;
            std::vector<uint32_t> words;
            while (ss >> tok) {
                try {
                    words.push_back(static_cast<uint32_t>(std::stoul(tok, nullptr, 16)));
                } catch (const std::exception &) { looks_text = false; break; }
            }
            if (looks_text) {
                bytes.resize(words.size() * sizeof(uint32_t));
                std::memcpy(bytes.data(), words.data(), bytes.size());
            }
        }
        if (!looks_text) { bytes = std::move(raw); }
        if (out_words) { *out_words = bytes.size() / sizeof(uint32_t); }
        return bytes;
    }
};

ggml_xrt_device & ggml_xrt_get_device(int32_t index) {
    static ggml_xrt_device dev0(0);
    GGML_ASSERT(index == 0 && "ggml-xrt scaffold currently supports a single device");
    return dev0;
}

int32_t ggml_xrt_probe_device_count() {
    return ggml_xrt_get_device(0).ensure_open() ? 1 : 0;
}

} // namespace

int32_t ggml_backend_xrt_get_device_count(void) { return ggml_xrt_probe_device_count(); }

void ggml_backend_xrt_get_device_description(int32_t device, char * description, size_t description_size) {
    auto & dev = ggml_xrt_get_device(device);
    dev.ensure_open();
    snprintf(description, description_size, "%s", dev.description.c_str());
}

void ggml_backend_xrt_get_device_memory(int32_t device, size_t * free, size_t * total) {
    GGML_UNUSED(device);
    // TODO(hw): query the real NPU/shared-memory budget via XRT info.
    if (free)  { *free  = 0; }
    if (total) { *total = 0; }
}

// ---------------------------------------------------------------------------
// Unified-memory buffers
//
// A tensor buffer is an XRT host-visible bo. On the Phoenix APU this lives in
// shared system memory, so the same allocation is visible to CPU and NPU with no
// copy. get_base returns the mapped host pointer. When no device is present (e.g.
// WSL) we fall back to aligned host memory so the backend remains testable.
// ---------------------------------------------------------------------------

// Alignment for Vulkan host-pointer import (VK_EXT_external_memory_host):
// minImportedHostPointerAlignment on the 780M is 4096, and Vulkan requires the
// imported allocationSize to be a multiple of it. The bo base is already
// page-aligned; we round the size up so the whole buffer is importable, enabling
// zero-copy NPU<->Vulkan sharing of the same host allocation.
#define GGML_XRT_IMPORT_ALIGN 4096
static inline size_t ggml_xrt_round_up(size_t n, size_t a) { return (n + a - 1) & ~(a - 1); }

struct ggml_backend_xrt_buffer_context {
    size_t                     size = 0;        // logical size requested by ggml
    size_t                     import_size = 0; // rounded up to GGML_XRT_IMPORT_ALIGN
    std::optional<xrt::bo>     bo;      // device-visible allocation (preferred)
    void *                     host = nullptr; // fallback host allocation
    void *                     base = nullptr; // mapped/usable pointer

    explicit ggml_backend_xrt_buffer_context(size_t s)
        : size(s), import_size(ggml_xrt_round_up(s ? s : 1, GGML_XRT_IMPORT_ALIGN)) {
        auto & dev = ggml_xrt_get_device(0);
        if (dev.ensure_open() && dev.device) {
            try {
                // host_only => shared/host-visible memory (unified). group 0 is the
                // default shared bank. Allocate import_size so the buffer is safe to
                // import into Vulkan (see GGML_XRT_IMPORT_ALIGN).
                bo.emplace(*dev.device, import_size, xrt::bo::flags::host_only, /*group=*/0);
                base = bo->map<void *>();
                return;
            } catch (const std::exception & ex) {
                GGML_XRT_LOG_WARN("bo alloc failed (%s), using host memory", ex.what());
            }
        }
        host = ggml_aligned_malloc(import_size);
        base = host;
    }

    ~ggml_backend_xrt_buffer_context() {
        if (host) { ggml_aligned_free(host, import_size); }
        // xrt::bo frees itself
    }
};

static void ggml_backend_xrt_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_backend_xrt_buffer_context *>(buffer->context);
}

static void * ggml_backend_xrt_buffer_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_xrt_buffer_context *>(buffer->context)->base;
}

static void ggml_backend_xrt_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void ggml_backend_xrt_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void ggml_backend_xrt_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                               void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static bool ggml_backend_xrt_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src,
                                               ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_xrt_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<ggml_backend_xrt_buffer_context *>(buffer->context);
    std::memset(ctx->base, value, ctx->size);
}

static const ggml_backend_buffer_i ggml_backend_xrt_buffer_interface = {
    /* .free_buffer   = */ ggml_backend_xrt_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_xrt_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_xrt_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_xrt_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_xrt_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_xrt_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_xrt_buffer_clear,
    /* .reset         = */ nullptr,
};

static const char * ggml_backend_xrt_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XRT_NAME;
}

static ggml_backend_buffer_t ggml_backend_xrt_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                       size_t size) {
    auto * ctx = new ggml_backend_xrt_buffer_context(size);
    if (ctx->base == nullptr && size > 0) {
        delete ctx;
        GGML_XRT_LOG_WARN("failed to allocate %zu bytes", size);
        return nullptr;
    }
    return ggml_backend_buffer_init(buft, ggml_backend_xrt_buffer_interface, ctx, size);
}

static size_t ggml_backend_xrt_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XRT_IMPORT_ALIGN; // 4096: page-align so buffers are Vulkan-importable
}

static bool ggml_backend_xrt_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // Unified memory: the allocation is host-visible, so the CPU and other
    // host-memory-aware backends can read it directly (no copy). This is the key
    // lever that lets the ggml scheduler avoid CPU<->NPU copies.
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_xrt_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_xrt_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_xrt_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_xrt_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ ggml_backend_xrt_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int32_t device) {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_xrt_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    GGML_UNUSED(device);
    return &buft;
}

// ---------------------------------------------------------------------------
// Shared "hsa" buffer: one XRT host_only bo, usable zero-copy by ALL THREE of the
// CPU, the NPU (XRT) and the iGPU (Vulkan) — a true unified-memory allocation.
//
// This is the is_host=TRUE variant. The allocation is REAL host memory:
//   * get_base returns the real XRT host pointer P (= bo.map()), so tensor->data
//     = P + offset is genuine host memory and the buffer type reports is_host=true.
//     The CPU backend operates on hsa tensors in place (NO scheduler copy), and the
//     XRT dispatch reads/writes P + offset directly (ggml_xrt_tensor_host_ptr
//     collapses to tensor->data since get_base == P).
//   * The buffer uses the CPU buffer interface (ggml_backend_cpu_buffer_from_ptr):
//     get_base / set_tensor / get_tensor / memset / clear are plain host memcpy on
//     the shared pages — correct for CPU and XRT, coherent for Vulkan (the imported
//     memory is HOST_COHERENT).
//   * Vulkan cannot use its sentinel offset math on a real pointer, so we import P
//     into the Vulkan device and register it in ggml-vulkan's pinned-host-memory
//     table (via get_proc_address hooks). On a UMA device every Vulkan op resolves
//     hsa tensors through ggml_vk_host_get(tensor->data) -> (imported VkBuffer,
//     offset), for BOTH read (src) and write (dst) sites, so Vulkan reads/writes the
//     same physical pages, zero-copy.
//   * The xrt::bo owns the pages and is kept alive for the buffer's lifetime; the
//     Vulkan import + pinned registration is torn down on free.
// ---------------------------------------------------------------------------

static const char * ggml_backend_xrt_hsa_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XRT_HSA_NAME;
}

// Resolved once from the Vulkan backend registry (kept decoupled: no link dep).
typedef bool (*ggml_vk_register_host_ptr_fn)(void *, size_t);
typedef void (*ggml_vk_unregister_host_ptr_fn)(void *);

// Per-buffer bookkeeping for a shared hsa allocation.
struct ggml_xrt_hsa_entry {
    std::shared_ptr<xrt::bo>       bo;                    // owns the host pages
    void *                         host_base = nullptr;   // P: real mapped host ptr
    ggml_vk_unregister_host_ptr_fn vk_unregister = nullptr;
};

static std::mutex                                                    g_hsa_mutex;
static std::unordered_map<ggml_backend_buffer_t, ggml_xrt_hsa_entry> g_hsa_registry;

static void * ggml_xrt_hsa_host_base(ggml_backend_buffer_t buffer) {
    if (!buffer) { return nullptr; }
    std::lock_guard<std::mutex> lk(g_hsa_mutex);
    auto it = g_hsa_registry.find(buffer);
    return it == g_hsa_registry.end() ? nullptr : it->second.host_base;
}

static void ggml_backend_xrt_hsa_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_xrt_hsa_entry entry;
    {
        std::lock_guard<std::mutex> lk(g_hsa_mutex);
        auto it = g_hsa_registry.find(buffer);
        if (it != g_hsa_registry.end()) { entry = it->second; g_hsa_registry.erase(it); }
    }
    // Tear down the Vulkan import + pinned registration, then release the xrt pages.
    if (entry.vk_unregister && entry.host_base) { entry.vk_unregister(entry.host_base); }
    // entry.bo (shared_ptr) releases the xrt::bo host pages here.
    // (the cpu-from-ptr buffer has a NULL free_buffer: nothing else to free)
}

struct ggml_backend_xrt_hsa_buft_context {
    ggml_backend_dev_t             import_dev    = nullptr; // Vulkan device to import into
    ggml_vk_register_host_ptr_fn   vk_register   = nullptr;
    ggml_vk_unregister_host_ptr_fn vk_unregister = nullptr;
    bool                           resolved      = false;
};

// Resolve the ggml-vulkan pinned-registration hooks from the import device's reg.
static void ggml_xrt_hsa_resolve_vk_hooks(ggml_backend_xrt_hsa_buft_context * bctx) {
    if (bctx->resolved || !bctx->import_dev) { return; }
    bctx->resolved = true;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(bctx->import_dev);
    if (!reg) { return; }
    bctx->vk_register = (ggml_vk_register_host_ptr_fn)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_register_host_ptr");
    bctx->vk_unregister = (ggml_vk_unregister_host_ptr_fn)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_vk_unregister_host_ptr");
}

static ggml_backend_buffer_t ggml_backend_xrt_hsa_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * bctx = static_cast<ggml_backend_xrt_hsa_buft_context *>(buft->context);
    if (!bctx || !bctx->import_dev) {
        GGML_XRT_LOG_WARN("hsa buft: no Vulkan import device configured");
        return nullptr;
    }
    ggml_xrt_hsa_resolve_vk_hooks(bctx);
    if (!bctx->vk_register || !bctx->vk_unregister) {
        GGML_XRT_LOG_WARN("hsa buft: Vulkan pinned-registration hooks unavailable "
                          "(need ggml-vulkan with get_proc_address host-ptr support)");
        return nullptr;
    }
    auto & dev = ggml_xrt_get_device(0);
    if (!dev.ensure_open() || !dev.device) {
        GGML_XRT_LOG_WARN("hsa buft: XRT device unavailable");
        return nullptr;
    }

    const size_t import_size = ggml_xrt_round_up(size ? size : 1, GGML_XRT_IMPORT_ALIGN);
    std::shared_ptr<xrt::bo> bo;
    void * P = nullptr;
    try {
        bo = std::make_shared<xrt::bo>(*dev.device, import_size, xrt::bo::flags::host_only, /*group=*/0);
        P  = bo->map<void *>();
    } catch (const std::exception & ex) {
        GGML_XRT_LOG_WARN("hsa buft: bo alloc failed: %s", ex.what());
        return nullptr;
    }

    // Import P into Vulkan + register it in the pinned-host table so Vulkan ops
    // resolve the real pointer to the imported VkBuffer (VK_EXT_external_memory_host).
    if (!bctx->vk_register(P, import_size)) {
        GGML_XRT_LOG_WARN("hsa buft: Vulkan import/pin of host ptr %p (size %zu) failed", P, import_size);
        return nullptr;
    }

    // Wrap the REAL host pointer as a CPU buffer: get_base == P, host memcpy I/O,
    // is_host == true. Re-tag the buffer TYPE so both accelerators' supports_buft
    // key on it, and wrap free to unregister Vulkan + release the xrt bo.
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(P, import_size);
    if (!buffer) {
        bctx->vk_unregister(P);
        GGML_XRT_LOG_WARN("hsa buft: cpu_buffer_from_ptr failed");
        return nullptr;
    }

    ggml_xrt_hsa_entry entry;
    entry.bo            = bo;
    entry.host_base     = P;
    entry.vk_unregister = bctx->vk_unregister;
    {
        std::lock_guard<std::mutex> lk(g_hsa_mutex);
        g_hsa_registry[buffer] = entry;
    }
    buffer->buft              = buft;
    buffer->iface.free_buffer = ggml_backend_xrt_hsa_free_buffer;
    GGML_XRT_LOG_INFO("hsa buft: allocated shared bo %p (size %zu, import %zu) as host buffer %p "
                      "[is_host=true, Vulkan-pinned]", P, size, import_size, (void *) buffer);
    return buffer;
}

static size_t ggml_backend_xrt_hsa_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // 4096: page-align each tensor so the buffer stays Vulkan-importable and every
    // tensor offset also satisfies Vulkan's storage-buffer offset alignment.
    return GGML_XRT_IMPORT_ALIGN;
}

static bool ggml_backend_xrt_hsa_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // TRUE unified memory: the allocation is real host memory (get_base == P), so
    // the CPU backend can operate on hsa tensors in place with no scheduler copy.
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_xrt_hsa_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_xrt_hsa_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_xrt_hsa_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_xrt_hsa_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ ggml_backend_xrt_hsa_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_xrt_hsa_buffer_type(ggml_backend_dev_t import_dev) {
    static ggml_backend_xrt_hsa_buft_context bctx;
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_xrt_hsa_buffer_type_interface,
        /* .device  = */ nullptr,
        /* .context = */ &bctx,
    };
    if (import_dev != bctx.import_dev) { bctx.import_dev = import_dev; bctx.resolved = false; }
    return &buft;
}

bool ggml_backend_buffer_is_xrt_hsa(ggml_backend_buffer_t buffer) {
    return buffer && buffer->buft &&
           buffer->buft->iface.get_name == ggml_backend_xrt_hsa_buffer_type_get_name;
}

// Real host pointer for a tensor's data, in BOTH the normal-XRT and shared-hsa
// cases. For is_host=true hsa buffers get_base == P, so this collapses to
// tensor->data; kept as a single routing point for the XRT dispatch.
void * ggml_xrt_tensor_host_ptr(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buf = tensor->buffer;
    void * P = ggml_xrt_hsa_host_base(buf);
    if (P) {
        void * base = ggml_backend_buffer_get_base(buf);  // == P for is_host=true
        return (char *) P + ((const char *) tensor->data - (char *) base);
    }
    return tensor->data;
}

// ---------------------------------------------------------------------------
// Backend (stream)
// ---------------------------------------------------------------------------

struct ggml_backend_xrt_context {
    int32_t     device = 0;
    std::string name;
    explicit ggml_backend_xrt_context(int32_t dev)
        : device(dev), name(GGML_XRT_NAME + std::to_string(dev)) {}
};

static const char * ggml_backend_xrt_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_xrt_context *>(backend->context)->name.c_str();
}

static void ggml_backend_xrt_free(ggml_backend_t backend) {
    delete static_cast<ggml_backend_xrt_context *>(backend->context);
    delete backend;
}

// Map a ggml_type to the dtype token used in kernel filenames.
static const char * ggml_xrt_dtype_token(ggml_type t) {
    switch (t) {
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "bf16"; // reinterpreted as bf16 on AIE
        default:             return nullptr;
    }
}

// The NPU kernels consume BF16 inputs. Any ggml type with a to_float trait (F16,
// BF16, and every quant format) can be host-dequantized to BF16 first, so the NPU
// only ever sees BF16 (plan C1). F32 is handled directly (no to_float trait).
static bool ggml_xrt_bf16_convertible(ggml_type t) {
    if (t == GGML_TYPE_F32) { return true; }
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    return tr && tr->to_float != nullptr;
}

// Dequantize/convert a contiguous run of `n` elements of type `t` to f32.
static void ggml_xrt_to_f32(ggml_type t, const void * src, float * dst, int64_t n) {
    if (t == GGML_TYPE_F32) {
        std::memcpy(dst, src, (size_t)n * sizeof(float));
        return;
    }
    ggml_get_type_traits(t)->to_float(src, dst, n);
}

// Is there a precompiled MUL_MAT xclbin for this op? (K = src0->ne[0], N = src0->ne[1])
// The weight (src0) and activation (src1) may be any BF16-convertible type; they
// are host-dequantized to BF16 in the dispatch. Output must be F32.
static bool ggml_xrt_have_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0]; // weight [K, N]
    const ggml_tensor * src1 = op->src[1]; // activation [K, M]
    if (!src0 || !src1) { return false; }
    const char * dto = ggml_xrt_dtype_token(op->type);
    if (!dto || op->type != GGML_TYPE_F32) { return false; }
    if (!ggml_xrt_bf16_convertible(src0->type) || !ggml_xrt_bf16_convertible(src1->type)) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) { return false; }

    // PREFILL (M>1) with a native-quant weight: DON'T claim it. The M>1 quant path
    // uses the scalar `mm` kernels (baked M=32, ~1.6-8.5 ms... actually SECONDS per
    // op on the scalar matvec), so a single prefill pass = ~196 ops x seconds =
    // minutes before the first token. Declining here lets the scheduler route
    // prefill matmuls to the GPU (Vulkan handles dynamic M in milliseconds) while
    // decode (M==1) stays on the NPU's fast vectorized gemv. Net: NPU does decode,
    // GPU does prefill — the intended hybrid split (plan doc section 9).
    // Only applies when native-quant is on AND the weight is a native-quant type;
    // bf16/f16 prefill still runs the (working, vectorized) tiled matmul on the NPU.
    // Remove once the prefill `mm` kernels are vectorized (then M>1 quant is fast).
    const int64_t M = src1->ne[1];
    if (ggml_xrt_native_quant_enabled() && M > 1 && ggml_xrt_quant_token(src0->type) != nullptr) {
        return false;
    }

    int m_tile = 0;
    auto path = ggml_xrt_find_mul_mat_xclbin(src0->ne[0], src0->ne[1], "bf16", dto,
                                             src1->ne[1], &m_tile);
    return !path.empty();
}

// Dispatch a MUL_MAT node to the NPU, tiling the token dimension M on the host.
// C[M,N] = A[M,K] * B[K,N]; A = src1 (activation), B = src0 (weight). The kernel
// ABI (opcode, instr, ninstr, A, B, C at bo groups 1/3/4/5) and the B transpose
// are hardware-validated against the CPU reference (bit-exact for BF16 inputs).
// Weight and activation are host-dequantized to BF16 here.
static bool ggml_backend_xrt_mul_mat(ggml_backend_xrt_context & ctx, ggml_tensor * op) {
    auto & dev = ggml_xrt_get_device(ctx.device);
    if (!dev.available || !dev.device) { return false; }

    // GGML_XRT_LOW_MEM: dequantize weights into a per-SHAPE pooled bo (re-done each
    // call) instead of caching a BF16 copy per WEIGHT. The NPU only consumes BF16,
    // so the default per-weight cache holds ~4x the Q4_K weight size resident; this
    // trades that RAM (down to ~shape-count buffers) for recompute. Much slower —
    // for memory-constrained one-off runs (e.g. correctness checks).
    static const bool low_mem = []() {
        const char * e = std::getenv("GGML_XRT_LOW_MEM");
        return e && e[0] && e[0] != '0';
    }();

    const ggml_tensor * src0 = op->src[0]; // weight  [K, N]
    const ggml_tensor * src1 = op->src[1]; // activation [K, M]
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const char * dto = ggml_xrt_dtype_token(op->type);
    if (!dto) { return false; }

    // -----------------------------------------------------------------------
    // M==1 decode, NATIVE QUANT (GGML_XRT_NATIVE_QUANT=1): if the weight is a
    // natively-supported quant type (Q4_0/Q4_K/Q6_K) and a matching quant gemv
    // artifact exists, upload the REPACKED RAW QUANT weight and let the NPU
    // dequantize on-chip. Same gemv ABI as the bf16 gemv (A=weight untransposed
    // @grp3, B=activation bf16 @grp4, C=f32 out @grp5, one launch) — only the A
    // operand's layout differs.
    //
    // This is the MEMORY path: the resident weight copy is the quantized size
    // (+2.8% for Q4_K / +11% for Q4_0 record padding) instead of ~4x it for BF16.
    // For Qwen3-1.7B-Q4_K_M that is roughly 1 GB resident instead of ~2.8 GB.
    //
    // OFF BY DEFAULT for the same reason as GGML_XRT_USE_GEMV: these are gemv
    // designs built on the SCALAR matvec, so they run ~single-lane and are slower
    // than letting decode fall through to the vectorized tiled kernel (which pads
    // M=1 up to its tile but keeps ~100% AIE util). Enable when RAM matters more
    // than decode latency; revisit the default once a VECTORIZED gemv exists.
    // Numerics are hardware-validated (NRMSE 0 on all nine Qwen3-1.7B kernels).
    // -----------------------------------------------------------------------
    static const bool native_quant = []() {
        const char * e = std::getenv("GGML_XRT_NATIVE_QUANT");
        return e && e[0] && e[0] != '0';
    }();
    if (M == 1 && native_quant) {
        // One-shot diagnostic: dump the tensor shapes/strides the gemv actually
        // receives in-model, to catch a layout the isolated harnesses don't replicate
        // (the gemv is correct in every standalone test but garbage in real decode).
        if (ggml_xrt_logging_enabled()) {
            static int dbg = 0;
            if (dbg < 8) {
                ++dbg;
                GGML_XRT_LOG_INFO("gemv[%d] w:ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu] t=%s | "
                    "act:ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu] t=%s cont=%d | "
                    "out:ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu] t=%s",
                    dbg,
                    (long long)src0->ne[0],(long long)src0->ne[1],(long long)src0->ne[2],(long long)src0->ne[3],
                    src0->nb[0],src0->nb[1], ggml_type_name(src0->type),
                    (long long)src1->ne[0],(long long)src1->ne[1],(long long)src1->ne[2],(long long)src1->ne[3],
                    src1->nb[0],src1->nb[1], ggml_type_name(src1->type), (int)ggml_is_contiguous(src1),
                    (long long)op->ne[0],(long long)op->ne[1],(long long)op->ne[2],(long long)op->ne[3],
                    op->nb[0],op->nb[1], ggml_type_name(op->type));
            }
        }
        const char * qtok = ggml_xrt_quant_token(src0->type);
        // Prefer the shared-overlay module (one context per K, no LRU eviction);
        // fall back to the per-shape xclbin if the overlay set isn't built.
        auto ovl = qtok ? dev.load_overlay_kernel(qtok, K, N) : nullptr;
        auto qpath = (!ovl && qtok) ? ggml_xrt_find_quant_gemv_xclbin(K, N, qtok)
                                    : std::filesystem::path{};
        // K must be a whole number of quant blocks for the repack to be well-defined.
        if ((ovl || !qpath.empty()) && (K % ggml_blck_size(src0->type)) == 0) {
            std::shared_ptr<ggml_xrt_kernel> qkern;
            if (!ovl) {
                auto qinsts = qpath; qinsts.replace_extension(); qinsts += "_insts.bin";
                if (!std::filesystem::exists(qinsts)) { qinsts = qpath; qinsts.replace_extension(); qinsts += "_insts.txt"; }
                std::ostringstream qk; qk << "qgemv_" << qtok << "_" << K << "x" << N;
                qkern = dev.load_kernel(qk.str(), qpath, qinsts);
            }
            std::ostringstream qkk; qkk << "qgemv_" << qtok << "_" << K << "x" << N;
            const std::string qkey = qkk.str();
            // group ids are identical for both paths (same kernel signature)
            auto group_id = [&](int i) {
                return ovl ? ovl->kernel.group_id(i) : qkern->kernel.group_id(i);
            };
            if (ovl || (qkern && qkern->instr_bo)) {
                const size_t rec      = ggml_xrt_quant_rec_bytes(src0->type);
                const int64_t nblocks = K / ggml_blck_size(src0->type);
                const size_t a_bytes  = (size_t) N * nblocks * rec;

                // A = repacked quant weight, cached per weight host ptr (weights are
                // constant). LOW_MEM re-packs into a per-shape pooled bo each call.
                const void * w_host = ggml_xrt_tensor_host_ptr(src0);
                auto fill_quant_weight = [&](xrt::bo & bo) {
                    ggml_xrt_repack_quant_weight(src0->type, w_host, bo.map<uint8_t *>(), N, K);
                    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                };
                // WEIGHT CACHE — keyed on the tensor NAME, not the data pointer.
                // The scheduler stages each weight through a REUSED buffer before the
                // NPU matmul, so src0->data (w_host) is the SAME address for every
                // weight (confirmed: all weights logged w_host=...E1C60000). Keying the
                // cache on that pointer collapsed every same-(dtype,shape) weight onto
                // ONE entry — e.g. up_proj reused gate_proj's repacked weight verbatim,
                // corrupting the whole layer. The tensor NAME (e.g. blk.7.ffn_up.weight)
                // is stable and unique, so it is the correct key. If the name is empty
                // (can't identify the weight), fall back to repack-fresh into a pooled
                // per-shape bo — correct because the staged buffer holds the CURRENT
                // weight at dispatch time (serial execution; run.wait before the next).
                std::shared_ptr<xrt::bo> a_ptr;
                const bool has_name = src0->name[0] != '\0';
                if (low_mem || !has_name) {
                    a_ptr = dev.get_io_bo(qkey + "_w", a_bytes,
                                          xrt::bo::flags::host_only, group_id(3));
                    fill_quant_weight(*a_ptr);   // fresh from the staged weight each call
                } else {
                    std::string wkey = std::string(src0->name) + "|" + qtok + "|"
                                     + std::to_string(K) + "x" + std::to_string(N);
                    std::lock_guard<std::mutex> lk(dev.weight_mutex);
                    auto it = dev.quant_weight_bos.find(wkey);
                    if (it != dev.quant_weight_bos.end() && it->second->size() >= a_bytes) {
                        a_ptr = it->second;
                    } else {
                        a_ptr = std::make_shared<xrt::bo>(*dev.device, a_bytes,
                                    xrt::bo::flags::host_only, group_id(3));
                        fill_quant_weight(*a_ptr);
                        dev.quant_weight_bos[wkey] = a_ptr;
                        GGML_XRT_LOG_INFO("native-quant %s '%s' %lldx%lld: %zu KiB repacked (vs %lld KiB bf16)",
                                          qtok, src0->name, (long long) K, (long long) N, a_bytes / 1024,
                                          (long long) ((size_t) N * K * sizeof(uint16_t) / 1024));
                    }
                }

                // per-op timing (host prep + kernel wait + output copy), aggregated/token
                const auto _t0 = std::chrono::steady_clock::now();
                // B = activation [K] bf16 (group 4), C = output [N] f32 (group 5).
                const size_t q_elt_in  = sizeof(uint16_t);
                const size_t q_elt_out = (op->type == GGML_TYPE_F32) ? 4 : 2;
                auto b_ptr = dev.get_io_bo(qkey + "_b", (size_t) K * q_elt_in,
                                           xrt::bo::flags::host_only, group_id(4));
                auto c_ptr = dev.get_io_bo(qkey + "_c", (size_t) N * q_elt_out,
                                           xrt::bo::flags::host_only, group_id(5));
                const char * b_src = (const char *) ggml_xrt_tensor_host_ptr(src1);
                if (src1->type == GGML_TYPE_BF16) {
                    std::memcpy(b_ptr->map<void *>(), b_src, (size_t) K * q_elt_in);
                } else {
                    std::vector<float> bf((size_t) K);
                    ggml_xrt_to_f32(src1->type, b_src, bf.data(), (int64_t) K);
                    ggml_fp32_to_bf16_row(bf.data(), b_ptr->map<ggml_bf16_t *>(), (int64_t) K);
                }
                b_ptr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                // Which resources does each call actually use? (diagnoses the up==gate
                // stale-output: same-shape gate/up must get DISTINCT weight bos.)
                if (ggml_xrt_logging_enabled()) {
                    static int rc = 0;
                    if (rc < 6) { ++rc;
                        GGML_XRT_LOG_INFO("qgemv#%d %s %lldx%lld w_host=%p a_bo=%p b_bo=%p c_bo=%p dst=%p",
                            rc, qtok, (long long)K, (long long)N, w_host,
                            (void*)a_ptr.get(), (void*)b_ptr.get(), (void*)c_ptr.get(),
                            ggml_xrt_tensor_host_ptr(op));
                    }
                }
                // Overlay path: instructions come from the ELF module, so the instr
                // bo/count args are 0. xclbin path: pass the instruction bo as before.
                auto run = ovl ? ovl->kernel(3u, 0, 0, *a_ptr, *b_ptr, *c_ptr)
                               : qkern->kernel(3u, *qkern->instr_bo, qkern->instr_words,
                                               *a_ptr, *b_ptr, *c_ptr);
                const auto _tw0 = std::chrono::steady_clock::now();
                run.wait();
                const auto _tw1 = std::chrono::steady_clock::now();
                c_ptr->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                std::memcpy((char *) ggml_xrt_tensor_host_ptr(op), c_ptr->map<void *>(), (size_t) N * q_elt_out);

                // Aggregate per-token (196 quant matmuls = 1 Qwen3-1.7B decode token):
                // total dispatch time, of which kernel-wait vs host (convert/sync/copy).
                if (ggml_xrt_logging_enabled()) {
                    auto _t1 = std::chrono::steady_clock::now();
                    auto d_ms = [](auto a, auto b){ return std::chrono::duration<double,std::milli>(b-a).count(); };
                    static double disp = 0, wait = 0; static int nc = 0;
                    disp += d_ms(_t0, _t1);   // whole dispatch (host prep + wait + copy)
                    wait += d_ms(_tw0, _tw1);  // kernel only
                    if (++nc % 196 == 0) {
                        GGML_XRT_LOG_INFO("per-token(196 qmatmul): dispatch=%.1f ms (kernel %.1f + host %.1f); "
                                          "token has this + attention/norms/lm_head/sampling on GPU/CPU",
                                          disp, wait, disp - wait);
                        disp = 0; wait = 0;
                    }
                }

                // IN-SITU SELF-CHECK: on the first few decode calls, recompute the
                // result on CPU from the SAME weight+activation this call just used
                // (real in-model inputs) and log NRMSE vs the NPU output. Every offline
                // harness passes; this catches whatever only the real graph triggers.
                if (ggml_xrt_logging_enabled() && op->type == GGML_TYPE_F32) {
                    static int sc = 0;
                    if (sc < 6) {
                        ++sc;
                        const float * npu = c_ptr->map<float *>();
                        // dequant weight rows, bf16-round (kernel holds bf16), dot bf16(act)
                        std::vector<float> wf((size_t) N * K);
                        ggml_xrt_to_f32(src0->type, w_host, wf.data(), (int64_t) N * K);
                        std::vector<ggml_bf16_t> wbf((size_t) N * K);
                        ggml_fp32_to_bf16_row(wf.data(), wbf.data(), (int64_t) N * K);
                        std::vector<float> af((size_t) K);
                        ggml_xrt_to_f32(src1->type, (const char *) ggml_xrt_tensor_host_ptr(src1), af.data(), (int64_t) K);
                        std::vector<ggml_bf16_t> abf((size_t) K);
                        ggml_fp32_to_bf16_row(af.data(), abf.data(), (int64_t) K);
                        auto b2f = [](ggml_bf16_t b){ uint32_t u=(uint32_t)b.bits<<16; float f; std::memcpy(&f,&u,4); return f; };
                        double sse=0, ref=0, maxe=0;
                        for (int64_t n = 0; n < N; ++n) {
                            double acc = 0;
                            const ggml_bf16_t * wr = wbf.data() + (size_t) n * K;
                            for (int64_t k = 0; k < K; ++k) acc += (double) b2f(wr[k]) * b2f(abf[k]);
                            double e = (double) npu[n] - acc; if (std::fabs(e)>maxe) maxe=std::fabs(e);
                            sse += e*e; ref += acc*acc;
                        }
                        GGML_XRT_LOG_INFO("SELFCHECK[%d] %s %lldx%lld in-model NRMSE=%.5f max=%.4f  npu[0..2]=[%.3f %.3f %.3f]",
                            sc, qtok, (long long)K, (long long)N,
                            std::sqrt(sse/(ref>0?ref:1)), maxe, npu[0], npu[1], npu[2]);
                    }
                }
                return true;
            }
        }
        // no quant gemv artifact (or unsupported type/K) -> fall through below.
    }

    // -----------------------------------------------------------------------
    // M>1 prefill, NATIVE QUANT: fused tiled matmul with on-chip dequant.
    // Same operand framing as the dense tiled path (A=activation@grp3,
    // B=weight@grp4, C@grp5), but B is the RAW repacked quant weight — no host
    // BF16 dequant and no BF16 cache, so prefill gets the same memory win decode
    // already has. M is baked at 32 per dispatch, so the host chunks tokens into
    // 32-row groups and zero-pads the final partial chunk.
    //
    // The core zeroes C then accumulates over all k-tiles, so C is the complete
    // result for the chunk (not an accumulate-into-existing) and is plain
    // row-major [32,N] — the same layout the dense tiled path copies back.
    //
    // Hardware-validated (8 of 9 built shapes) at NRMSE 0.0033-0.0040. Unlike the
    // gemv this is NOT bit-exact: the kernel keeps dequant(W) as bf16 in on-chip
    // scratch, so bf16 rounding of the weight is expected. One shape is
    // quarantined as broken — see ggml_xrt_quant_mm_quarantined.
    // -----------------------------------------------------------------------
    if (M > 1 && native_quant) {
        const char * qtok = ggml_xrt_quant_token(src0->type);
        auto mpath = (qtok && !ggml_xrt_quant_mm_quarantined(qtok, K, N))
                   ? ggml_xrt_find_quant_mm_xclbin(K, N, qtok)
                   : std::filesystem::path{};
        if (!mpath.empty() && (K % ggml_blck_size(src0->type)) == 0) {
            auto minsts = mpath; minsts.replace_extension(); minsts += "_insts.bin";
            if (!std::filesystem::exists(minsts)) { minsts = mpath; minsts.replace_extension(); minsts += "_insts.txt"; }
            std::ostringstream mk; mk << "qmm_" << qtok << "_" << K << "x" << N;
            const std::string mkey = mk.str();
            auto mkern = dev.load_kernel(mkey, mpath, minsts);
            if (mkern && mkern->instr_bo) {
                const int64_t MM_TILE = 32;               // baked into the xclbin
                const size_t  rec     = ggml_xrt_quant_rec_bytes(src0->type);
                const int64_t nblocks = K / ggml_blck_size(src0->type);
                const size_t  w_bytes = (size_t) N * nblocks * rec;
                const size_t  elt_in  = sizeof(uint16_t);
                const size_t  elt_out = (op->type == GGML_TYPE_F32) ? 4 : 2;

                // B = repacked quant weight @ group 4 (decode uses group 3; the
                // cache key carries the group so the two share a bo only when the
                // memory group actually matches).
                const int    w_group = mkern->kernel.group_id(4);
                const void * w_host  = ggml_xrt_tensor_host_ptr(src0);
                auto fill_quant_weight = [&](xrt::bo & bo) {
                    ggml_xrt_repack_quant_weight(src0->type, w_host, bo.map<uint8_t *>(), N, K);
                    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                };
                std::shared_ptr<xrt::bo> b_ptr;
                if (low_mem) {
                    b_ptr = dev.get_io_bo(mkey + "_w", w_bytes,
                                          xrt::bo::flags::host_only, w_group);
                    fill_quant_weight(*b_ptr);
                } else {
                    std::ostringstream wk;
                    wk << w_host << "_" << qtok << "_" << K << "x" << N << "_g" << w_group;
                    const std::string wkey = wk.str();
                    std::lock_guard<std::mutex> lk(dev.weight_mutex);
                    auto it = dev.quant_weight_bos.find(wkey);
                    if (it != dev.quant_weight_bos.end() && it->second->size() >= w_bytes) {
                        b_ptr = it->second;
                    } else {
                        b_ptr = std::make_shared<xrt::bo>(*dev.device, w_bytes,
                                    xrt::bo::flags::host_only, w_group);
                        fill_quant_weight(*b_ptr);
                        dev.quant_weight_bos[wkey] = b_ptr;
                        GGML_XRT_LOG_INFO("native-quant mm %s weight %lldx%lld: %zu KiB repacked (vs %lld KiB bf16)",
                                          qtok, (long long) K, (long long) N, w_bytes / 1024,
                                          (long long) ((size_t) N * K * sizeof(uint16_t) / 1024));
                    }
                }

                // A = activation [32,K] bf16 @ group 3, C = [32,N] f32 @ group 5.
                auto a_ptr = dev.get_io_bo(mkey + "_a", (size_t) MM_TILE * K * elt_in,
                                           xrt::bo::flags::host_only, mkern->kernel.group_id(3));
                auto c_ptr = dev.get_io_bo(mkey + "_c", (size_t) MM_TILE * N * elt_out,
                                           xrt::bo::flags::host_only, mkern->kernel.group_id(5));
                char * a_map = a_ptr->map<char *>();
                char * c_map = c_ptr->map<char *>();
                const char * src1_host = (const char *) ggml_xrt_tensor_host_ptr(src1);
                char       * dst_host  = (char *)       ggml_xrt_tensor_host_ptr(op);

                for (int64_t m0 = 0; m0 < M; m0 += MM_TILE) {
                    const int64_t rows = std::min<int64_t>(MM_TILE, M - m0);
                    if (rows < MM_TILE) {   // zero-pad the tail chunk
                        std::memset(a_map + (size_t) rows * K * elt_in, 0,
                                    (size_t)(MM_TILE - rows) * K * elt_in);
                    }
                    const char * a_src = src1_host + m0 * src1->nb[1];
                    if (src1->type == GGML_TYPE_BF16) {
                        std::memcpy(a_map, a_src, (size_t) rows * K * elt_in);
                    } else {
                        std::vector<float> af((size_t) rows * K);
                        ggml_xrt_to_f32(src1->type, a_src, af.data(), (int64_t) rows * K);
                        ggml_fp32_to_bf16_row(af.data(), reinterpret_cast<ggml_bf16_t *>(a_map),
                                              (int64_t) rows * K);
                    }
                    a_ptr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                    auto run = mkern->kernel(3u, *mkern->instr_bo, mkern->instr_words,
                                             *a_ptr, *b_ptr, *c_ptr);
                    run.wait();
                    c_ptr->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                    // drop the pad rows: copy back only `rows`
                    std::memcpy(dst_host + m0 * N * elt_out, c_map,
                                (size_t) rows * N * elt_out);
                }
                return true;
            }
        }
        // no mm artifact / quarantined / unsupported -> fall through to the
        // dense host-BF16-dequant tiled path below (still correct).
    }

    // -----------------------------------------------------------------------
    // M==1 decode: OPT-IN dedicated gemv kernel (GGML_XRT_USE_GEMV=1). Different
    // ABI from the tiled matmul: C[N] = A[N,K] . B[K], A = weight ggml-native
    // [N,K] (NO transpose) @ group 3, B = activation [K] @ group 4, C = output
    // [N] @ group 5, one launch; A cached UNtransposed.
    // OFF BY DEFAULT: the prebuilt gemv uses the SCALAR matvec (the vectorized
    // path is erroneous upstream — see wishlist), so it runs single-lane (~5% AIE
    // util) and is SLOWER than letting decode fall through to the tiled kernel,
    // which uses the vectorized aie::mac (~100% util) despite padding M=1 up to
    // its tile. Re-enable once a VECTORIZED gemv is built.
    // -----------------------------------------------------------------------
    static const bool use_gemv = []() {
        const char * e = std::getenv("GGML_XRT_USE_GEMV");
        return e && e[0] && e[0] != '0';
    }();
    if (M == 1 && use_gemv) {
        auto gpath = ggml_xrt_find_gemv_xclbin(K, N);
        if (!gpath.empty()) {
            auto ginsts = gpath; ginsts.replace_extension(); ginsts += "_insts.bin";
            if (!std::filesystem::exists(ginsts)) { ginsts = gpath; ginsts.replace_extension(); ginsts += "_insts.txt"; }
            std::ostringstream gk; gk << "gemv_" << K << "x" << N << "_" << dto;
            const std::string gkey = gk.str();
            auto gkern = dev.load_kernel(gkey, gpath, ginsts);
            if (gkern && gkern->instr_bo) {
                const size_t g_elt_in  = sizeof(uint16_t);                       // bf16
                const size_t g_elt_out = (op->type == GGML_TYPE_F32) ? 4 : 2;

                // A = weight, untransposed [N,K] bf16. Cached per real host ptr, or
                // (GGML_XRT_LOW_MEM) a per-shape pooled bo re-filled each call.
                const void * w_host = ggml_xrt_tensor_host_ptr(src0);
                auto fill_gemv_weight = [&](xrt::bo & bo) {
                    std::vector<float> wf((size_t)N * K);
                    ggml_xrt_to_f32(src0->type, w_host, wf.data(), (int64_t)N * K);
                    ggml_fp32_to_bf16_row(wf.data(), bo.map<ggml_bf16_t *>(), (int64_t)N * K);
                    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
                };
                std::shared_ptr<xrt::bo> a_ptr;
                if (low_mem) {
                    a_ptr = dev.get_io_bo(gkey + "_w", (size_t)N * K * g_elt_in,
                                          xrt::bo::flags::host_only, gkern->kernel.group_id(3));
                    fill_gemv_weight(*a_ptr);
                } else {
                    std::lock_guard<std::mutex> lk(dev.weight_mutex);
                    auto it = dev.gemv_weight_bos.find(w_host);
                    if (it != dev.gemv_weight_bos.end()) {
                        a_ptr = it->second;
                    } else {
                        a_ptr = std::make_shared<xrt::bo>(*dev.device, (size_t)N * K * g_elt_in,
                                    xrt::bo::flags::host_only, gkern->kernel.group_id(3));
                        fill_gemv_weight(*a_ptr);
                        dev.gemv_weight_bos[w_host] = a_ptr;
                    }
                }

                // B = activation [K] bf16 (group 4), C = output [N] (group 5); pooled.
                auto b_ptr = dev.get_io_bo(gkey + "_b", (size_t)K * g_elt_in,
                                           xrt::bo::flags::host_only, gkern->kernel.group_id(4));
                auto c_ptr = dev.get_io_bo(gkey + "_c", (size_t)N * g_elt_out,
                                           xrt::bo::flags::host_only, gkern->kernel.group_id(5));
                const char * b_src = (const char *) ggml_xrt_tensor_host_ptr(src1);
                if (src1->type == GGML_TYPE_BF16) {
                    std::memcpy(b_ptr->map<void *>(), b_src, (size_t)K * g_elt_in);
                } else {
                    std::vector<float> bf((size_t)K);
                    ggml_xrt_to_f32(src1->type, b_src, bf.data(), (int64_t)K);
                    ggml_fp32_to_bf16_row(bf.data(), b_ptr->map<ggml_bf16_t *>(), (int64_t)K);
                }
                b_ptr->sync(XCL_BO_SYNC_BO_TO_DEVICE);

                auto run = gkern->kernel(3u, *gkern->instr_bo, gkern->instr_words, *a_ptr, *b_ptr, *c_ptr);
                run.wait();
                c_ptr->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                std::memcpy((char *) ggml_xrt_tensor_host_ptr(op), c_ptr->map<void *>(), (size_t)N * g_elt_out);
                return true;
            }
        }
        // no gemv artifact (or load failed) -> fall through to the tiled path.
    }

    int m_tile = 0;
    auto xclbin = ggml_xrt_find_mul_mat_xclbin(K, N, "bf16", dto, M, &m_tile);
    if (xclbin.empty() || m_tile <= 0) { return false; }
    auto insts = xclbin; insts.replace_extension();
    insts += "_insts.txt";
    if (!std::filesystem::exists(insts)) { insts = xclbin; insts.replace_extension(); insts += "_insts.bin"; }

    std::ostringstream key; key << "mul_mat_" << K << "x" << N << "x" << m_tile << "_" << dto;
    const std::string kkey = key.str();
    auto kern = dev.load_kernel(kkey, xclbin, insts);
    if (!kern || !kern->instr_bo) { return false; }

    // Instruction bo (group 1): uploaded once in load_kernel and cached.
    xrt::bo & bo_instr = *kern->instr_bo;

    const size_t elt_in  = sizeof(uint16_t);            // bf16 activations/weights
    const size_t elt_out = (op->type == GGML_TYPE_F32) ? 4 : 2;

    // Weight bo (B): host-dequantize src0 (any BF16-convertible type) to BF16, and
    // transpose N x K -> K x N. The stock mlir-aie matmul expects B row-major K x N
    // (B[k,n] at k*N+n), but ggml stores the weight [K,N] as N x K row-major
    // (w[n,k] at n*K+k) = the transpose. Cached per weight data ptr (weights are
    // constant), so the dequant+transpose+upload happens only on first use.
    // Real host pointers (handles both normal-XRT and shared-hsa buffers). Note the
    // hsa case: tensor->data is the shared vk sentinel, so it is NOT a valid cache
    // key (all hsa buffers share the same sentinel base) -> key by the real ptr.
    const void * src0_host = ggml_xrt_tensor_host_ptr(src0);
    auto fill_weight_T = [&](xrt::bo & bo) {   // dequant src0 -> BF16, transpose N x K -> K x N
        std::vector<float> wf((size_t)K * N);
        ggml_xrt_to_f32(src0->type, src0_host, wf.data(), (int64_t)K * N);
        std::vector<ggml_bf16_t> wbf((size_t)K * N);
        ggml_fp32_to_bf16_row(wf.data(), wbf.data(), (int64_t)K * N);
        const uint16_t * s = reinterpret_cast<const uint16_t *>(wbf.data());
        uint16_t * bdst = bo.map<uint16_t *>();
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < K; ++k) { bdst[k * N + n] = s[n * K + k]; }
        }
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    };
    std::shared_ptr<xrt::bo> bo_b_ptr;
    if (low_mem) {
        // one BF16 weight bo per shape (re-filled each call) — see GGML_XRT_LOW_MEM.
        bo_b_ptr = dev.get_io_bo(kkey + "_w", (size_t)K * N * elt_in,
                                 xrt::bo::flags::host_only, kern->kernel.group_id(4));
        fill_weight_T(*bo_b_ptr);
    } else {
        std::lock_guard<std::mutex> lk(dev.weight_mutex);
        auto it = dev.weight_bos.find(src0_host);
        if (it != dev.weight_bos.end()) {
            bo_b_ptr = it->second;
        } else {
            bo_b_ptr = std::make_shared<xrt::bo>(*dev.device, (size_t)K * N * elt_in,
                                                 xrt::bo::flags::host_only,
                                                 kern->kernel.group_id(4));
            fill_weight_T(*bo_b_ptr);
            dev.weight_bos[src0_host] = bo_b_ptr;
        }
    }
    xrt::bo & bo_b = *bo_b_ptr;

    // Activation (A) and output (C) bo's: fixed size/group per kernel shape, so pull
    // them from the pool (allocated + mapped once, reused across tiles and calls)
    // instead of re-allocating on every dispatch. Contents are rewritten each tile.
    const size_t a_bytes = (size_t)m_tile * K * elt_in;
    const size_t c_bytes = (size_t)m_tile * N * elt_out;
    auto bo_a_ptr = dev.get_io_bo(kkey + "_a", a_bytes, xrt::bo::flags::host_only,
                                  kern->kernel.group_id(3));
    auto bo_c_ptr = dev.get_io_bo(kkey + "_c", c_bytes, xrt::bo::flags::host_only,
                                  kern->kernel.group_id(5));
    xrt::bo & bo_a = *bo_a_ptr;
    xrt::bo & bo_c = *bo_c_ptr;
    char * a_map = bo_a.map<char *>();
    char * c_map = bo_c.map<char *>();

    // Real host pointers for activation (read) and output (write).
    const char * src1_host = (const char *) ggml_xrt_tensor_host_ptr(src1);
    char *       dst_host  = (char *)       ggml_xrt_tensor_host_ptr(op);

    // Host-side M-tiling: iterate over ceil(M / m_tile) row blocks, zero-padding
    // the final (partial) block up to m_tile.
    for (int64_t m0 = 0; m0 < M; m0 += m_tile) {
        const int64_t rows = std::min<int64_t>(m_tile, M - m0);

        // Activation A: convert this row block (any BF16-convertible type) to BF16.
        // Only zero the padding tail when the tile is partial (rows < m_tile); a
        // full tile is entirely overwritten below, so skip the memset there.
        if (rows < m_tile) {
            std::memset(a_map + (size_t)rows * K * elt_in, 0,
                        (size_t)(m_tile - rows) * K * elt_in);
        }
        const char * a_src = src1_host + m0 * src1->nb[1];
        if (src1->type == GGML_TYPE_BF16) {
            std::memcpy(a_map, a_src, (size_t)rows * K * elt_in);
        } else {
            std::vector<float> af((size_t)rows * K);
            ggml_xrt_to_f32(src1->type, a_src, af.data(), (int64_t)rows * K);
            ggml_fp32_to_bf16_row(af.data(), reinterpret_cast<ggml_bf16_t *>(a_map),
                                  (int64_t)rows * K);
        }
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_a, bo_b, bo_c);
        run.wait();

        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(dst_host + m0 * N * elt_out,
                    c_map,
                    (size_t)rows * N * elt_out);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Elementwise / norm op kernels (SILU, GELU, RMS_NORM, ROPE)
//
// These are single-shape artifacts under <GGML_XRT_KERNEL_DIR>/ops/, named
// "<tag>_<size>_aie2.xclbin". Resolution is by (tag, element/row size); if no
// artifact matches the op's actual shape, supports_op returns false and the op is
// left to the GPU. Real deployment needs shape-parameterized op artifacts.
//
// TODO(hw): validate arg layouts. SILU/GELU/RMS_NORM are single-in/single-out.
// ROPE additionally needs position/frequency inputs whose binding is unvalidated.
// ---------------------------------------------------------------------------

static const char * ggml_xrt_op_tag(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_RMS_NORM: return "rms_norm";
        case GGML_OP_ROPE:     return "rope";
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU: return "silu";
                case GGML_UNARY_OP_GELU: return "gelu";
                default:                 return nullptr;
            }
        default: return nullptr;
    }
}

static std::filesystem::path ggml_xrt_find_op_xclbin(const char * tag, int64_t size) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (!tag || dir.empty() || !fs::exists(dir)) { return {}; }
    const std::string szmark = "_" + std::to_string(size) + "_";
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        const std::string fn = p.filename().string();
        if (fn.rfind(std::string(tag) + "_", 0) == 0 && fn.find(szmark) != std::string::npos) {
            return p;
        }
    }
    return {};
}

// "Row size" used to key a row-wise op artifact (RMS_NORM): last-dim length.
static int64_t ggml_xrt_op_size(const ggml_tensor * op) { return op->ne[0]; }

// Find any "<tag>_<len>_aie2.xclbin" and return the largest tile length (fewest
// host tiles). Used for elementwise ops (SILU/GELU) that tile over total elements.
static std::filesystem::path ggml_xrt_find_op_xclbin_any(const char * tag, int64_t * out_len) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (!tag || dir.empty() || !fs::exists(dir)) { return {}; }
    const std::string pfx = std::string(tag) + "_";
    fs::path best; int64_t best_len = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        const std::string fn = p.filename().string();
        if (fn.rfind(pfx, 0) != 0) { continue; }
        int64_t len = std::atoll(fn.c_str() + pfx.size());
        if (len > best_len) { best_len = len; best = p; }
    }
    if (best_len > 0 && out_len) { *out_len = best_len; }
    return best;
}

static bool ggml_xrt_have_op_kernel(const ggml_tensor * op) {
    const char * tag = ggml_xrt_op_tag(op);
    if (!tag) { return false; }
    if (op->op == GGML_OP_UNARY) {
        int64_t len = 0;                    // elementwise: any tile length works
        return !ggml_xrt_find_op_xclbin_any(tag, &len).empty();
    }
    return !ggml_xrt_find_op_xclbin(tag, ggml_xrt_op_size(op)).empty();
}

// Is there a precompiled RoPE xclbin matching this op, and is it a case this
// backend handles? Only NEOX mode with full-width rotation (n_dims == head_dim)
// and a matching rope_<head_dim> artifact is claimed; everything else -> GPU/CPU.
static bool ggml_xrt_have_rope(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1 || src1->type != GGML_TYPE_I32) { return false; }
    const int32_t * pp = (const int32_t *) op->op_params;
    const int n_dims = pp[1];
    const int mode   = pp[2];
    if (mode != GGML_ROPE_TYPE_NEOX) { return false; }
    const int64_t ne0 = op->ne[0];
    if (n_dims != ne0 || (ne0 % 2) != 0)          { return false; }
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_BF16) { return false; }
    if (!ggml_xrt_bf16_convertible(src0->type))   { return false; }
    return !ggml_xrt_find_op_xclbin("rope", ne0).empty();
}

// Single-in / single-out op dispatch (SILU, GELU, RMS_NORM). Iterates rows,
// running the fixed-size kernel once per row (last dim = row size).
static bool ggml_backend_xrt_op_rowwise(ggml_backend_xrt_context & ctx, ggml_tensor * op) {
    auto & dev = ggml_xrt_get_device(ctx.device);
    if (!dev.available || !dev.device) { return false; }
    const char * tag = ggml_xrt_op_tag(op);
    const int64_t cols = ggml_xrt_op_size(op);
    auto xclbin = ggml_xrt_find_op_xclbin(tag, cols);
    if (xclbin.empty()) { return false; }
    auto insts = xclbin; insts.replace_extension(); insts += "_insts.bin";
    if (!std::filesystem::exists(insts)) { insts = xclbin; insts.replace_extension(); insts += "_insts.txt"; }

    std::ostringstream key; key << tag << "_" << cols;
    auto kern = dev.load_kernel(key.str(), xclbin, insts);
    if (!kern) { return false; }

    const ggml_tensor * src = op->src[0];
    const char * src_host = (const char *) ggml_xrt_tensor_host_ptr(src);
    char *       dst_host = (char *)       ggml_xrt_tensor_host_ptr(op);
    const int64_t rows = ggml_nrows(op);
    const size_t elt = sizeof(uint16_t);  // kernel is BF16 in / BF16 out (aie2/rms_norm.cc)

    // The rmsnorm kernels are built with a fixed row tile (sequence_length=32); the
    // host tiles the row/token dimension over it, zero-padding the final block. The
    // kernel consumes/produces BF16, so convert the (typically F32) tensor rows.
    // TODO(hw): keep ROW_TILE in sync with the seq used to build the artifacts.
    // NOTE: kernel bakes epsilon=1e-5 (Qwen3 uses 1e-6) — negligible vs bf16 error.
    const int64_t ROW_TILE = 32;

    if (!kern->instr_bo) { return false; }
    xrt::bo & bo_instr = *kern->instr_bo;  // uploaded once, cached in load_kernel

    for (int64_t r0 = 0; r0 < rows; r0 += ROW_TILE) {
        const int64_t rr = std::min<int64_t>(ROW_TILE, rows - r0);
        const int64_t nel = rr * cols;
        xrt::bo bo_in (*dev.device, (size_t)ROW_TILE * cols * elt, xrt::bo::flags::host_only, kern->kernel.group_id(3));
        xrt::bo bo_out(*dev.device, (size_t)ROW_TILE * cols * elt, xrt::bo::flags::host_only, kern->kernel.group_id(4));

        // input rows -> BF16
        uint16_t * in = bo_in.map<uint16_t *>();
        std::memset(in, 0, (size_t)ROW_TILE * cols * elt);
        const char * src_rows = src_host + r0 * src->nb[1];
        if (src->type == GGML_TYPE_BF16) {
            std::memcpy(in, src_rows, (size_t)nel * elt);
        } else {
            std::vector<float> f((size_t)nel);
            ggml_xrt_to_f32(src->type, src_rows, f.data(), nel);
            ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(in), nel);
        }
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_in, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        // BF16 output -> op type
        const uint16_t * out = bo_out.map<uint16_t *>();
        char * dst_rows = dst_host + r0 * op->nb[1];
        if (op->type == GGML_TYPE_BF16) {
            std::memcpy(dst_rows, out, (size_t)nel * elt);
        } else {
            ggml_xrt_to_f32(GGML_TYPE_BF16, out, reinterpret_cast<float *>(dst_rows), nel);
        }
    }
    return true;
}

// Flat elementwise op dispatch (SILU, GELU): tile the total element count over a
// fixed-length kernel, zero-padding the final chunk (silu(0)=gelu(0)=0).
static bool ggml_backend_xrt_op_elementwise(ggml_backend_xrt_context & ctx, ggml_tensor * op) {
    auto & dev = ggml_xrt_get_device(ctx.device);
    if (!dev.available || !dev.device) { return false; }
    const char * tag = ggml_xrt_op_tag(op);
    int64_t tile = 0;
    auto xclbin = ggml_xrt_find_op_xclbin_any(tag, &tile);
    if (xclbin.empty() || tile <= 0) { return false; }
    auto insts = xclbin; insts.replace_extension(); insts += "_insts.bin";
    if (!std::filesystem::exists(insts)) { insts = xclbin; insts.replace_extension(); insts += "_insts.txt"; }

    std::ostringstream key; key << tag << "_" << tile;
    auto kern = dev.load_kernel(key.str(), xclbin, insts);
    if (!kern) { return false; }

    const ggml_tensor * src = op->src[0];
    const char * src_host = (const char *) ggml_xrt_tensor_host_ptr(src);
    char *       dst_host = (char *)       ggml_xrt_tensor_host_ptr(op);
    const size_t es_in  = ggml_type_size(src->type);
    const size_t es_out = ggml_type_size(op->type);
    const int64_t nelem = ggml_nelements(op);
    const size_t elt = sizeof(uint16_t);  // kernel is BF16 in / BF16 out (mlir-aie ml/{silu,gelu})

    if (!kern->instr_bo) { return false; }
    xrt::bo & bo_instr = *kern->instr_bo;  // uploaded once, cached in load_kernel

    for (int64_t off = 0; off < nelem; off += tile) {
        const int64_t n = std::min<int64_t>(tile, nelem - off);
        xrt::bo bo_in (*dev.device, (size_t)tile * elt, xrt::bo::flags::host_only, kern->kernel.group_id(3));
        xrt::bo bo_out(*dev.device, (size_t)tile * elt, xrt::bo::flags::host_only, kern->kernel.group_id(4));

        // input chunk -> BF16
        uint16_t * in = bo_in.map<uint16_t *>();
        std::memset(in, 0, (size_t)tile * elt);
        const char * src_off = src_host + off * es_in;
        if (src->type == GGML_TYPE_BF16) {
            std::memcpy(in, src_off, (size_t)n * elt);
        } else {
            std::vector<float> f((size_t)n);
            ggml_xrt_to_f32(src->type, src_off, f.data(), n);
            ggml_fp32_to_bf16_row(f.data(), reinterpret_cast<ggml_bf16_t *>(in), n);
        }
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_in, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        // BF16 output -> op type
        const uint16_t * out = bo_out.map<uint16_t *>();
        char * dst_off = dst_host + off * es_out;
        if (op->type == GGML_TYPE_BF16) {
            std::memcpy(dst_off, out, (size_t)n * elt);
        } else {
            ggml_xrt_to_f32(GGML_TYPE_BF16, out, reinterpret_cast<float *>(dst_off), n);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// ROPE (NEOX) dispatch
//
// The prebuilt rope kernel (mlir-aie programming_examples/ml/rope, aie2p/rope.cc)
// is a pure per-row elementwise op over BF16. For each ADJACENT lane pair
// (in[2p], in[2p+1]) it reads the per-pair (cos, sin) from a second input buffer
// ("LUT") at (lut[2p], lut[2p+1]) and writes:
//     out[2p]   = in[2p]*cos - in[2p+1]*sin
//     out[2p+1] = in[2p]*sin + in[2p+1]*cos
// The rotation math is identical to ggml's, but the kernel pairs ADJACENT lanes,
// which is ggml's GGML_ROPE_TYPE_NORMAL (GPT-J) layout. Qwen3 uses NEOX, which
// instead pairs the split element (x[j], x[j + n_dims/2]).
//
// Because the kernel takes a fully general cos/sin LUT as an input buffer (it does
// NOT bake positions or frequencies), NEOX is realized purely by HOST PERMUTATION:
// feed the NEOX pair (x[j], x[j+half]) as the kernel's adjacent lanes (2j, 2j+1),
// and un-permute the output (out[j]=y[2j], out[j+half]=y[2j+1]). The LUT for a row
// is exactly ggml's cos/sin cache (interleaved cos,sin,cos,sin, ...), so no LUT
// permutation is needed. cos/sin are computed on the host replicating ggml's
// ggml_rope_cache_init (incl. YaRN); all rotation arithmetic runs on the NPU in
// BF16. The kernel bakes seq tile = 32 rows, so the row/token dim is host-tiled.
//
// ABI (validated from the xclbin metadata + mlir-aie source): kernel "MLIR_AIE",
// opcode 3, args (instr@group1, ninstr, in@group3, lut@group4, out@group5), all
// BF16 in/out, row length = embedding_dim baked into the artifact (rope_<ne0>).
static bool ggml_backend_xrt_rope(ggml_backend_xrt_context & ctx, ggml_tensor * op) {
    auto & dev = ggml_xrt_get_device(ctx.device);
    if (!dev.available || !dev.device) { return false; }

    const ggml_tensor * src0 = op->src[0]; // activations [ne0=head_dim, ne1=heads, ne2=seq, ne3=batch]
    const ggml_tensor * src1 = op->src[1]; // positions   I32, length ne2
    const ggml_tensor * src2 = op->src[2]; // freq_factors F32 (optional)
    if (!src0 || !src1 || src1->type != GGML_TYPE_I32) { return false; }

    const int32_t * pp = (const int32_t *) op->op_params;
    const int n_dims     = pp[1];
    const int mode       = pp[2];
    const int n_ctx_orig = pp[4];
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    std::memcpy(&freq_base,   pp + 5,  sizeof(float));
    std::memcpy(&freq_scale,  pp + 6,  sizeof(float));
    std::memcpy(&ext_factor,  pp + 7,  sizeof(float));
    std::memcpy(&attn_factor, pp + 8,  sizeof(float));
    std::memcpy(&beta_fast,   pp + 9,  sizeof(float));
    std::memcpy(&beta_slow,   pp + 10, sizeof(float));

    if (mode != GGML_ROPE_TYPE_NEOX) { return false; } // only NEOX is wired up here
    const int64_t ne0 = op->ne[0];
    if (n_dims != ne0 || (ne0 % 2) != 0) { return false; } // full-width rotation only

    auto xclbin = ggml_xrt_find_op_xclbin("rope", ne0);
    if (xclbin.empty()) { return false; }
    auto insts = xclbin; insts.replace_extension(); insts += "_insts.bin";
    if (!std::filesystem::exists(insts)) { insts = xclbin; insts.replace_extension(); insts += "_insts.txt"; }

    std::ostringstream key; key << "rope_" << ne0;
    auto kern = dev.load_kernel(key.str(), xclbin, insts);
    if (!kern) { return false; }

    const int64_t ne1  = op->ne[1];
    const int64_t ne2  = op->ne[2];
    const int64_t ne3  = op->ne[3];
    const int64_t rows = ne1 * ne2 * ne3;
    const int64_t half = ne0 / 2;
    const size_t  elt  = sizeof(uint16_t);   // kernel is BF16 in / BF16 out
    const int64_t ROW_TILE = 32;             // baked seq tile (rope built with seq=32)

    const float theta_scale = powf(freq_base, -2.0f / n_dims);
    float corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);

    const int32_t * pos          = (const int32_t *) ggml_xrt_tensor_host_ptr(src1);
    const float   * freq_factors = src2 ? (const float *) ggml_xrt_tensor_host_ptr(src2) : nullptr;
    const char    * src0_host    = (const char *)     ggml_xrt_tensor_host_ptr(src0);
    char          * dst_host     = (char *)           ggml_xrt_tensor_host_ptr(op);

    xrt::bo bo_instr(*dev.device, kern->instr.size(), xrt::bo::flags::cacheable, kern->kernel.group_id(1));
    std::memcpy(bo_instr.map<void *>(), kern->instr.data(), kern->instr.size());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<float> rowf((size_t)ne0);       // activation row -> f32
    std::vector<float> in_perm((size_t)ne0);    // permuted (adjacent-pair) input, f32
    std::vector<float> lut_f((size_t)ne0);      // cos/sin cache, f32
    std::vector<float> out_perm((size_t)ne0);   // kernel output for a row, f32

    for (int64_t r0 = 0; r0 < rows; r0 += ROW_TILE) {
        const int64_t rr = std::min<int64_t>(ROW_TILE, rows - r0);
        xrt::bo bo_in (*dev.device, (size_t)ROW_TILE * ne0 * elt, xrt::bo::flags::host_only, kern->kernel.group_id(3));
        xrt::bo bo_lut(*dev.device, (size_t)ROW_TILE * ne0 * elt, xrt::bo::flags::host_only, kern->kernel.group_id(4));
        xrt::bo bo_out(*dev.device, (size_t)ROW_TILE * ne0 * elt, xrt::bo::flags::host_only, kern->kernel.group_id(5));
        uint16_t * in  = bo_in.map<uint16_t *>();
        uint16_t * lut = bo_lut.map<uint16_t *>();
        std::memset(in,  0, (size_t)ROW_TILE * ne0 * elt);
        std::memset(lut, 0, (size_t)ROW_TILE * ne0 * elt);

        for (int64_t rw = 0; rw < rr; ++rw) {
            const int64_t r  = r0 + rw;
            const int64_t i1 = r % ne1;             // head
            const int64_t i2 = (r / ne1) % ne2;     // token / seq position slot
            const int64_t i3 = r / (ne1 * ne2);     // batch
            const float   position = (float) pos[i2];

            // cos/sin cache for this token (replicates ggml_rope_cache_init).
            float theta = position;
            for (int64_t i = 0; i < ne0; i += 2) {
                const float ff = freq_factors ? freq_factors[i / 2] : 1.0f;
                const float theta_extrap = theta / ff;
                const float theta_interp = freq_scale * theta_extrap;
                float th     = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float y    = ((float)(i / 2) - corr_dims[0]) /
                                       std::max(0.001f, corr_dims[1] - corr_dims[0]);
                    const float ramp = (1.0f - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
                    th     = theta_interp * (1.0f - ramp) + theta_extrap * ramp;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                lut_f[i + 0] = cosf(th) * mscale;
                lut_f[i + 1] = sinf(th) * mscale;
                theta *= theta_scale;
            }

            // read activation row -> f32
            const char * src_row = src0_host +
                                   i3 * src0->nb[3] + i2 * src0->nb[2] + i1 * src0->nb[1];
            ggml_xrt_to_f32(src0->type, src_row, rowf.data(), ne0);

            // NEOX -> adjacent-pair permutation: pair j -> lanes (2j, 2j+1)
            for (int64_t j = 0; j < half; ++j) {
                in_perm[2 * j + 0] = rowf[j];
                in_perm[2 * j + 1] = rowf[j + half];
            }
            ggml_fp32_to_bf16_row(in_perm.data(), reinterpret_cast<ggml_bf16_t *>(in  + rw * ne0), ne0);
            ggml_fp32_to_bf16_row(lut_f.data(),   reinterpret_cast<ggml_bf16_t *>(lut + rw * ne0), ne0);
        }
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_lut.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_in, bo_lut, bo_out);
        run.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        const uint16_t * out = bo_out.map<uint16_t *>();
        for (int64_t rw = 0; rw < rr; ++rw) {
            const int64_t r  = r0 + rw;
            const int64_t i1 = r % ne1;
            const int64_t i2 = (r / ne1) % ne2;
            const int64_t i3 = r / (ne1 * ne2);
            ggml_xrt_to_f32(GGML_TYPE_BF16, out + rw * ne0, out_perm.data(), ne0);
            char * dst_row = dst_host +
                             i3 * op->nb[3] + i2 * op->nb[2] + i1 * op->nb[1];
            // inverse permutation: out[j]=y[2j], out[j+half]=y[2j+1]
            if (op->type == GGML_TYPE_F32) {
                float * d = reinterpret_cast<float *>(dst_row);
                for (int64_t j = 0; j < half; ++j) {
                    d[j]        = out_perm[2 * j + 0];
                    d[j + half] = out_perm[2 * j + 1];
                }
            } else { // BF16 output
                const uint16_t * yb = out + rw * ne0;
                uint16_t * d = reinterpret_cast<uint16_t *>(dst_row);
                for (int64_t j = 0; j < half; ++j) {
                    d[j]        = yb[2 * j + 0];
                    d[j + half] = yb[2 * j + 1];
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_xrt_compute_node(ggml_backend_xrt_context & ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return ggml_backend_xrt_mul_mat(ctx, node);
        case GGML_OP_RMS_NORM:
            // per-row reduction over the last dim
            return ggml_backend_xrt_op_rowwise(ctx, node);
        case GGML_OP_UNARY:
            // SILU / GELU: flat elementwise, host-tiled
            return ggml_backend_xrt_op_elementwise(ctx, node);
        case GGML_OP_ROPE:
            // NEOX RoPE: host permute + BF16 cos/sin LUT, rotation on NPU
            return ggml_backend_xrt_rope(ctx, node);
        default:
            return false;
    }
}

static ggml_status ggml_backend_xrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto & ctx = *static_cast<ggml_backend_xrt_context *>(backend->context);
    int n_mm = 0, n_rms = 0, n_un = 0, n_other = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_op_is_empty(node->op) || node->op == GGML_OP_NONE) { continue; }
        if (!ggml_backend_xrt_compute_node(ctx, node)) {
            GGML_XRT_LOG_WARN("no NPU kernel for op %s (node '%s')", ggml_op_name(node->op), node->name);
            return GGML_STATUS_FAILED;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:  ++n_mm;    break;
            case GGML_OP_RMS_NORM: ++n_rms;   break;
            case GGML_OP_UNARY:    ++n_un;    break;
            default:               ++n_other; break;
        }
    }
    // Per-graph summary of what the scheduler routed to the NPU (this backend only
    // sees its own assigned ops). Combine with GGML_SCHED_DEBUG=2 for the full split.
    //
    // Decode re-issues an identical graph for every token, so printing this per graph
    // buries everything else. At log level 1 an unchanged summary is collapsed: print
    // on change, then count repeats and flush the count periodically (and on change),
    // so generation stays readable but there is still proof of life. Level 2 prints
    // every graph.
    if (ggml_xrt_log_level() >= 2) {
        GGML_XRT_LOG_INFO("graph_compute: %d ops on NPU (mul_mat=%d rms_norm=%d unary=%d other=%d)",
                          n_mm + n_rms + n_un + n_other, n_mm, n_rms, n_un, n_other);
    } else if (ggml_xrt_logging_enabled()) {
        char buf[192];
        snprintf(buf, sizeof(buf),
                 "graph_compute: %d ops on NPU (mul_mat=%d rms_norm=%d unary=%d other=%d)",
                 n_mm + n_rms + n_un + n_other, n_mm, n_rms, n_un, n_other);
        static std::mutex   sum_mutex;
        static std::string  last_summary;
        static uint64_t     repeats = 0;
        std::lock_guard<std::mutex> lk(sum_mutex);
        if (last_summary == buf) {
            if (++repeats % 100 == 0) {
                GGML_XRT_LOG_INFO("  (same graph x%llu)", (unsigned long long) repeats);
            }
        } else {
            if (repeats > 0) {
                GGML_XRT_LOG_INFO("  (previous graph repeated %llu times total)",
                                  (unsigned long long) repeats);
                repeats = 0;
            }
            GGML_XRT_LOG_INFO("%s", buf);
            last_summary = buf;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_xrt_interface = {
    /* .get_name           = */ ggml_backend_xrt_get_name,
    /* .free               = */ ggml_backend_xrt_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_xrt_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static ggml_guid_t ggml_backend_xrt_guid() {
    static ggml_guid guid = { 0x9a, 0x1c, 0x4e, 0x33, 0x77, 0x2b, 0x48, 0x51,
                              0xb0, 0xde, 0xc5, 0x12, 0x6f, 0x84, 0xaa, 0x01 };
    return &guid;
}

bool ggml_backend_is_xrt(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_xrt_guid());
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

struct ggml_backend_xrt_device_context {
    int32_t     device = 0;
    std::string name;
    std::string description;
};

static const char * ggml_backend_xrt_device_get_name(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_xrt_device_context *>(dev->context)->name.c_str();
}

static const char * ggml_backend_xrt_device_get_description(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_xrt_device_context *>(dev->context)->description.c_str();
}

static void ggml_backend_xrt_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * ctx = static_cast<ggml_backend_xrt_device_context *>(dev->context);
    ggml_backend_xrt_get_device_memory(ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_xrt_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_xrt_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xrt_device_get_name(dev);
    props->description = ggml_backend_xrt_device_get_description(dev);
    props->type        = ggml_backend_xrt_device_get_type(dev);
    ggml_backend_xrt_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xrt_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    auto * dev_ctx = static_cast<ggml_backend_xrt_device_context *>(dev->context);
    return new ggml_backend{
        /* .guid    = */ ggml_backend_xrt_guid(),
        /* .iface   = */ ggml_backend_xrt_interface,
        /* .device  = */ dev,
        /* .context = */ new ggml_backend_xrt_context(dev_ctx->device),
    };
}

static ggml_backend_buffer_type_t ggml_backend_xrt_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_xrt_device_context *>(dev->context);
    ggml_backend_buffer_type_t buft = ggml_backend_xrt_buffer_type(dev_ctx->device);
    buft->device = dev;
    return buft;
}

// Every op class with a validated kernel (MUL_MAT, RMS_NORM, SILU/GELU, NEOX RoPE)
// is claimed by default, to maximize NPU coverage for layer sharding. AOT gating
// still applies: an op is only claimed if a matching artifact exists, otherwise
// the scheduler routes it to Vulkan/CPU.
// GGML_XRT_MATMUL_ONLY=1: claim only MUL_MAT on the NPU; route RMS_NORM/SILU/GELU/
// RoPE to the GPU/CPU. The NPU allows only ~5 concurrent hw_contexts, and the
// op kernels push the working set past that, so with all ops on the LRU cache
// thrashes (evict+reload every layer). Restricting to MUL_MAT keeps the NPU to
// its ~4 matmul shapes (within budget, no thrash) — and ops are faster on the
// GPU anyway. Temporary lever until the shared-context work (wishlist §8) lands.
static bool ggml_xrt_matmul_only() {
    static const bool en = []() {
        const char * e = std::getenv("GGML_XRT_MATMUL_ONLY");
        return e && e[0] && e[0] != '0';
    }();
    return en;
}

// Unified memory (UMA): advertise support for HOST buffers in supports_buft. The NPU shim
// reads host pages directly and the native-quant path repacks the weight from src0->data
// (a host pointer) regardless, so a CPU-resident weight needs no copy. Without this,
// ggml_backend_sched forces a NEW SPLIT at every offloaded matmul (ggml-backend.cpp:1282
// fires because the weight's buffer is "incompatible") -> one op per graph_compute ->
// ~196 NPU<->CPU transitions/token (the handoff tail). Claiming host buffers trips the
// escape hatch (!buffer_supported) so adjacent NPU matmuls GROUP into multi-op splits --
// the prerequisite for SwiGLU/residency fusion. DEFAULT ON: UMA is fully supported across
// CPU/GPU/NPU (the host pages are shared), validated in-model (coherent, tail 167->~99 ms,
// 1.5->2.3 t/s, graph_compute shows 2-op splits). Set GGML_XRT_UNIFIED_BUFT=0 to disable
// (falls back to the per-op offload path) only if a config ever regresses.
static bool ggml_xrt_unified_buft() {
    static const bool en = []() {
        const char * e = std::getenv("GGML_XRT_UNIFIED_BUFT");
        return !e || !e[0] || e[0] != '0';   // default ON; only an explicit "0" disables
    }();
    return en;
}

// GGML_XRT_NPU_LAYERS: restrict which transformer layers' weight matmuls run on
// the NPU (per-layer sharding). Everything not selected is declined by supports_op
// and the scheduler routes it to the GPU (Vulkan). Syntax:
//   "0-8"        layers 0..8 inclusive
//   "0,2,4"      layers 0, 2 and 4
//   "0-3,8,10-12" mixed ranges + singletons
//   "8"          a bare count -> layers 0..7 (the first 8 layers)
// Unset/empty  -> no restriction (every layer with an artifact is eligible).
// The layer index is parsed from the weight (src0) tensor name "blk.<N>." that
// llama.cpp assigns; a matmul whose weight has no "blk.<N>" (lm_head/output.weight,
// token_embd, attention KQ/KQV) is treated as "not a layer weight" and, when the
// filter is active, declined -> those run on the GPU (matching the hybrid goal:
// only per-layer weight matmuls on the NPU, lm_head/embeddings/attention on the GPU).
struct ggml_xrt_layer_filter {
    bool          restricted = false;
    std::set<int> layers;
    bool selected(int il) const { return !restricted || (il >= 0 && layers.count(il) != 0); }
};

static const ggml_xrt_layer_filter & ggml_xrt_npu_layers() {
    static const ggml_xrt_layer_filter filt = []() {
        ggml_xrt_layer_filter f;
        const char * e = std::getenv("GGML_XRT_NPU_LAYERS");
        if (!e || !e[0]) { return f; }
        f.restricted = true;
        const std::string s(e);
        // bare integer (no '-' or ',') -> treat as a count: layers 0..N-1.
        if (s.find_first_of("-,") == std::string::npos) {
            const int n = std::atoi(s.c_str());
            for (int i = 0; i < n; ++i) { f.layers.insert(i); }
            return f;
        }
        // comma-separated list of singletons and inclusive "a-b" ranges.
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t comma = s.find(',', pos);
            const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!tok.empty()) {
                const size_t dash = tok.find('-');
                if (dash == std::string::npos) {
                    f.layers.insert(std::atoi(tok.c_str()));
                } else {
                    const int a = std::atoi(tok.substr(0, dash).c_str());
                    const int b = std::atoi(tok.substr(dash + 1).c_str());
                    for (int i = a; i <= b; ++i) { f.layers.insert(i); }
                }
            }
            if (comma == std::string::npos) { break; }
            pos = comma + 1;
        }
        return f;
    }();
    return filt;
}

// Parse the layer index from a matmul's weight (src0) name "blk.<N>.*". Returns
// -1 when the weight is not a per-layer tensor (lm_head, token_embd) or the src is
// a non-weight (attention KQ/KQV, whose src0 is an activation, not "blk.N.*.weight").
static int ggml_xrt_op_layer(const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    if (!w) { return -1; }
    const char * p = std::strstr(w->name, "blk.");
    if (!p) { return -1; }
    p += 4;
    if (*p < '0' || *p > '9') { return -1; }
    return std::atoi(p);
}

// True if this MUL_MAT is one the NPU should run: a matching artifact exists AND
// (when GGML_XRT_NPU_LAYERS is set) its weight belongs to a selected layer.
static bool ggml_xrt_mul_mat_selected(const ggml_tensor * op) {
    if (!ggml_xrt_have_mul_mat(op)) { return false; }
    const ggml_xrt_layer_filter & f = ggml_xrt_npu_layers();
    if (!f.restricted) { return true; }
    return f.selected(ggml_xrt_op_layer(op));
}

static bool ggml_backend_xrt_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // AOT-only gating: claim an op ONLY if a matching precompiled xclbin exists.
    // Everything else is left to the GPU/CPU by the scheduler. This is what makes
    // hybrid NPU+GPU execution work without JIT (see docs/ggml-xrt-plan.md §9).
    if (ggml_op_is_empty(op->op)) { return true; }
    const bool mm_only = ggml_xrt_matmul_only();
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            // AOT artifact must exist; when GGML_XRT_NPU_LAYERS is set, only the
            // selected layers' weight matmuls are claimed (rest -> GPU).
            return ggml_xrt_mul_mat_selected(op);
        case GGML_OP_RMS_NORM:
            // validated on-device (unit harness vs CPU, NRMSE ~0.004); default-on
            return !mm_only && ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        case GGML_OP_UNARY:
            // SILU / GELU validated on-device (unit harness vs CPU); default-on
            return !mm_only && ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        case GGML_OP_ROPE:
            // NEOX RoPE validated on-device (unit harness vs CPU); default-on.
            // Requires contiguous src0 and a matching rope_<head_dim> artifact
            // (ggml_xrt_have_rope also restricts to full-width NEOX).
            return !mm_only && ggml_xrt_have_rope(op) &&
                   ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op);
        default:
            return false;
    }
}

static bool ggml_backend_xrt_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // Normal XRT (unified) buffers, plus the shared hsa buffer type (matched by
    // name; it is device-agnostic since the pages are host memory shared with the
    // iGPU). Accepting hsa here lets the scheduler place NPU ops on hsa tensors
    // with zero copy.
    if (buft->iface.get_name == ggml_backend_xrt_hsa_buffer_type_get_name) { return true; }
    // Unified memory: accept host buffers so the scheduler stops forcing a per-weight split
    // at every offloaded matmul (see ggml_xrt_unified_buft). Lets adjacent NPU matmuls group
    // into multi-op splits -> enables SwiGLU/residency fusion. The native-quant dispatch
    // already reads src0/src1 from their host pointers, so no copy is needed.
    if (ggml_xrt_unified_buft() && ggml_backend_buft_is_host(buft)) { return true; }
    return buft->iface.get_name == ggml_backend_xrt_buffer_type_get_name && buft->device == dev;
}

// Offload hint (consulted by ggml_backend_sched only when a supported op's WEIGHT
// is resident on a host buffer of the lowest-priority backend, i.e. the CPU). By
// returning true here the scheduler pulls those matmuls onto the NPU exactly like
// it offloads big GEMMs to BLAS — so a plain `-ngl 0` run (weights CPU/host
// resident) routes conformant matmuls to the NPU with no llama.cpp change. We only
// claim ops we actually support (delegates to supports_op, so it also honours
// GGML_XRT_MATMUL_ONLY and GGML_XRT_NPU_LAYERS). For the full CPU-bypass hybrid
// (non-matmul ops on the GPU) the layers are instead offloaded to Vulkan and the
// selected matmul weights placed in XRT_HSA buffers; see docs/ggml-xrt-plan.md.
static bool ggml_backend_xrt_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_backend_xrt_device_supports_op(dev, op);
}

static const ggml_backend_device_i ggml_backend_xrt_device_interface = {
    /* .get_name             = */ ggml_backend_xrt_device_get_name,
    /* .get_description      = */ ggml_backend_xrt_device_get_description,
    /* .get_memory           = */ ggml_backend_xrt_device_get_memory,
    /* .get_type             = */ ggml_backend_xrt_device_get_type,
    /* .get_props            = */ ggml_backend_xrt_device_get_props,
    /* .init_backend         = */ ggml_backend_xrt_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xrt_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_xrt_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xrt_device_supports_buft,
    /* .offload_op           = */ ggml_backend_xrt_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

struct ggml_backend_xrt_reg_context {
    std::vector<ggml_backend_device> devices;
};

static const char * ggml_backend_xrt_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_XRT_NAME;
}

static size_t ggml_backend_xrt_reg_get_device_count(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_xrt_reg_context *>(reg->context)->devices.size();
}

static ggml_backend_dev_t ggml_backend_xrt_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * ctx = static_cast<ggml_backend_xrt_reg_context *>(reg->context);
    GGML_ASSERT(index < ctx->devices.size());
    return &ctx->devices[index];
}

static const ggml_backend_reg_i ggml_backend_xrt_reg_interface = {
    /* .get_name         = */ ggml_backend_xrt_reg_get_name,
    /* .get_device_count = */ ggml_backend_xrt_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xrt_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_xrt_reg(void) {
    static ggml_backend_reg reg;
    static ggml_backend_xrt_reg_context reg_ctx;
    static bool initialized = false;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {
        reg = ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_VERSION,
            /* .iface       = */ ggml_backend_xrt_reg_interface,
            /* .context     = */ &reg_ctx,
        };
        const int32_t count = ggml_backend_xrt_get_device_count();
        for (int32_t i = 0; i < count; ++i) {
            char desc[256] = {0};
            ggml_backend_xrt_get_device_description(i, desc, sizeof(desc));
            auto * dev_ctx = new ggml_backend_xrt_device_context{
                /* .device      = */ i,
                /* .name        = */ std::string(GGML_XRT_NAME) + std::to_string(i),
                /* .description = */ std::string(desc),
            };
            reg_ctx.devices.push_back(ggml_backend_device{
                /* .iface   = */ ggml_backend_xrt_device_interface,
                /* .reg     = */ &reg,
                /* .context = */ dev_ctx,
            });
        }
        initialized = true;
        GGML_XRT_LOG_INFO("registered %d device(s)", count);
    }
    return &reg;
}

ggml_backend_t ggml_backend_xrt_init(int32_t device) {
    ggml_backend_reg_t reg = ggml_backend_xrt_reg();
    if (device < 0 || static_cast<size_t>(device) >= ggml_backend_xrt_reg_get_device_count(reg)) {
        GGML_XRT_LOG_WARN("init: no XRT device %d present", device);
        return nullptr;
    }
    return ggml_backend_xrt_device_init_backend(ggml_backend_xrt_reg_get_device(reg, device), nullptr);
}

GGML_BACKEND_DL_IMPL(ggml_backend_xrt_reg)
