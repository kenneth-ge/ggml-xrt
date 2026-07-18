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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static bool ggml_xrt_logging_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_XRT_ENABLE_LOG");
        return env != nullptr && (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 ||
                                  std::strcmp(env, "on") == 0);
    }();
    return enabled;
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
    xrt::hw_context      context;
    xrt::kernel          kernel;
    std::vector<uint8_t> instr;   // control instruction sequence
    size_t               instr_words = 0;
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

// Locate an xclbin for MUL_MAT(K,N,dtypes). Returns the best (smallest-M) match so
// decode paths prefer the small tile. Empty if none found.
static std::filesystem::path ggml_xrt_find_mul_mat_xclbin(int64_t K, int64_t N,
                                                          const char * dti, const char * dto,
                                                          int * out_m_tile) {
    namespace fs = std::filesystem;
    const std::string dir = ggml_xrt_kernel_dir();
    if (dir.empty() || !fs::exists(dir)) {
        return {};
    }
    const std::string prefix = ggml_xrt_mul_mat_prefix(dti, dto);
    const std::string kn = "x" + std::to_string(K) + "x" + std::to_string(N) + "_";

    fs::path best;
    int best_m = INT32_MAX;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(dir, ec);
         !ec && it != fs::recursive_directory_iterator(); ++it) {
        const auto & p = it->path();
        if (p.extension() != ".xclbin") { continue; }
        const std::string fn = p.filename().string();
        if (fn.rfind(prefix, 0) != 0) { continue; }        // must start with prefix
        if (fn.find(kn) == std::string::npos) { continue; } // must contain xKxN_
        // parse the leading M tile: <prefix><M>x<K>x<N>...
        const std::string tail = fn.substr(prefix.size());
        int m = std::atoi(tail.c_str());
        if (m > 0 && m < best_m) { best_m = m; best = p; }
    }
    if (!best.empty() && out_m_tile) { *out_m_tile = best_m; }
    return best;
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

    // op-shape -> loaded kernel
    std::unordered_map<std::string, std::shared_ptr<ggml_xrt_kernel>> kernels;

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
        if (auto it = kernels.find(key); it != kernels.end()) { return it->second; }
        if (!available || !device) { return nullptr; }
        try {
            auto k = std::make_shared<ggml_xrt_kernel>();
            auto uuid   = device->load_xclbin(xclbin.string());
            k->context  = xrt::hw_context(*device, uuid);
            k->kernel   = xrt::kernel(k->context, GGML_XRT_KERNEL_NAME);
            k->instr    = ggml_xrt_read_instrs(insts, &k->instr_words);
            kernels[key] = k;
            GGML_XRT_LOG_INFO("loaded kernel %s (%zu instr words)", key.c_str(), k->instr_words);
            return k;
        } catch (const std::exception & ex) {
            GGML_XRT_LOG_WARN("failed to load kernel %s: %s", key.c_str(), ex.what());
            return nullptr;
        }
    }

  private:
    // Read an instruction file. Supports the toolchain's `.bin` (raw uint32) and
    // `.txt` (whitespace-separated hex words) formats.
    static std::vector<uint8_t> ggml_xrt_read_instrs(const std::filesystem::path & path,
                                                     size_t * out_words) {
        std::vector<uint8_t> bytes;
        if (path.extension() == ".txt") {
            std::ifstream f(path);
            std::string tok;
            std::vector<uint32_t> words;
            while (f >> tok) {
                words.push_back(static_cast<uint32_t>(std::stoul(tok, nullptr, 16)));
            }
            bytes.resize(words.size() * sizeof(uint32_t));
            std::memcpy(bytes.data(), words.data(), bytes.size());
        } else {
            std::ifstream f(path, std::ios::binary);
            bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
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

struct ggml_backend_xrt_buffer_context {
    size_t                     size = 0;
    std::optional<xrt::bo>     bo;      // device-visible allocation (preferred)
    void *                     host = nullptr; // fallback host allocation
    void *                     base = nullptr; // mapped/usable pointer

    explicit ggml_backend_xrt_buffer_context(size_t s) : size(s) {
        auto & dev = ggml_xrt_get_device(0);
        if (dev.ensure_open() && dev.device) {
            try {
                // host_only => shared/host-visible memory (unified). group 0 is the
                // default shared bank. TODO(hw): confirm group vs kernel.group_id.
                bo.emplace(*dev.device, size, xrt::bo::flags::host_only, /*group=*/0);
                base = bo->map<void *>();
                return;
            } catch (const std::exception & ex) {
                GGML_XRT_LOG_WARN("bo alloc failed (%s), using host memory", ex.what());
            }
        }
        host = ggml_aligned_malloc(size);
        base = host;
    }

    ~ggml_backend_xrt_buffer_context() {
        if (host) { ggml_aligned_free(host, size); }
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
    return 64;
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

// Is there a precompiled MUL_MAT xclbin for this op? (K = src0->ne[0], N = src0->ne[1])
static bool ggml_xrt_have_mul_mat(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0]; // weight [K, N]
    const ggml_tensor * src1 = op->src[1]; // activation [K, M]
    if (!src0 || !src1) { return false; }
    const char * dti = ggml_xrt_dtype_token(src1->type);
    const char * dto = ggml_xrt_dtype_token(op->type);
    if (!dti || !dto) { return false; }
    int m_tile = 0;
    auto path = ggml_xrt_find_mul_mat_xclbin(src0->ne[0], src0->ne[1], "bf16", dto, &m_tile);
    return !path.empty();
}

// Dispatch a MUL_MAT node to the NPU, tiling the token dimension M on the host.
// C[M,N] = A[M,K] * B[K,N]; A = src1 (activation), B = src0 (weight).
// TODO(hw): validate the A/B/C arg order and b-col-major against the compiled
// kernel; validate bo group ids. Not executable in the dev environment.
static bool ggml_backend_xrt_mul_mat(ggml_backend_xrt_context & ctx, ggml_tensor * op) {
    auto & dev = ggml_xrt_get_device(ctx.device);
    if (!dev.available || !dev.device) { return false; }

    const ggml_tensor * src0 = op->src[0]; // weight  [K, N]
    const ggml_tensor * src1 = op->src[1]; // activation [K, M]
    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];
    const char * dto = ggml_xrt_dtype_token(op->type);
    if (!dto) { return false; }

    int m_tile = 0;
    auto xclbin = ggml_xrt_find_mul_mat_xclbin(K, N, "bf16", dto, &m_tile);
    if (xclbin.empty() || m_tile <= 0) { return false; }
    auto insts = xclbin; insts.replace_extension();
    insts += "_insts.txt";
    if (!std::filesystem::exists(insts)) { insts = xclbin; insts.replace_extension(); insts += "_insts.bin"; }

    std::ostringstream key; key << "mul_mat_" << K << "x" << N << "x" << m_tile << "_" << dto;
    auto kern = dev.load_kernel(key.str(), xclbin, insts);
    if (!kern) { return false; }

    // Instruction bo (group 1), cacheable.
    xrt::bo bo_instr(*dev.device, kern->instr.size(), xrt::bo::flags::cacheable,
                     kern->kernel.group_id(1));
    std::memcpy(bo_instr.map<void *>(), kern->instr.data(), kern->instr.size());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const size_t elt_in  = sizeof(uint16_t);            // bf16 activations/weights
    const size_t elt_out = (op->type == GGML_TYPE_F32) ? 4 : 2;

    // Weight bo (B), shared across all M-tiles.
    xrt::bo bo_b(*dev.device, (size_t)K * N * elt_in, xrt::bo::flags::host_only,
                 kern->kernel.group_id(4));
    std::memcpy(bo_b.map<void *>(), src0->data, (size_t)K * N * elt_in);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Host-side M-tiling: iterate over ceil(M / m_tile) row blocks, zero-padding
    // the final (partial) block up to m_tile.
    for (int64_t m0 = 0; m0 < M; m0 += m_tile) {
        const int64_t rows = std::min<int64_t>(m_tile, M - m0);

        xrt::bo bo_a(*dev.device, (size_t)m_tile * K * elt_in, xrt::bo::flags::host_only,
                     kern->kernel.group_id(3));
        xrt::bo bo_c(*dev.device, (size_t)m_tile * N * elt_out, xrt::bo::flags::host_only,
                     kern->kernel.group_id(5));

        char * a_map = bo_a.map<char *>();
        std::memset(a_map, 0, (size_t)m_tile * K * elt_in);
        std::memcpy(a_map,
                    static_cast<const char *>(src1->data) + m0 * K * elt_in,
                    (size_t)rows * K * elt_in);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_a, bo_b, bo_c);
        run.wait();

        bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(static_cast<char *>(op->data) + m0 * N * elt_out,
                    bo_c.map<char *>(),
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

// "Row size" used to key an op artifact: last-dim length for norm/unary ops.
static int64_t ggml_xrt_op_size(const ggml_tensor * op) { return op->ne[0]; }

static bool ggml_xrt_have_op_kernel(const ggml_tensor * op) {
    const char * tag = ggml_xrt_op_tag(op);
    if (!tag) { return false; }
    return !ggml_xrt_find_op_xclbin(tag, ggml_xrt_op_size(op)).empty();
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
    const size_t row_bytes_in  = cols * ggml_type_size(src->type);
    const size_t row_bytes_out = cols * ggml_type_size(op->type);
    const int64_t rows = ggml_nrows(op);

    xrt::bo bo_instr(*dev.device, kern->instr.size(), xrt::bo::flags::cacheable, kern->kernel.group_id(1));
    std::memcpy(bo_instr.map<void *>(), kern->instr.data(), kern->instr.size());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    for (int64_t r = 0; r < rows; ++r) {
        xrt::bo bo_in (*dev.device, row_bytes_in,  xrt::bo::flags::host_only, kern->kernel.group_id(3));
        xrt::bo bo_out(*dev.device, row_bytes_out, xrt::bo::flags::host_only, kern->kernel.group_id(4));
        std::memcpy(bo_in.map<void *>(), static_cast<const char *>(src->data) + r * row_bytes_in, row_bytes_in);
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        unsigned int opcode = 3;
        auto run = kern->kernel(opcode, bo_instr, kern->instr_words, bo_in, bo_out);
        run.wait();

        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::memcpy(static_cast<char *>(op->data) + r * row_bytes_out, bo_out.map<char *>(), row_bytes_out);
    }
    return true;
}

static bool ggml_backend_xrt_compute_node(ggml_backend_xrt_context & ctx, ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT:
            return ggml_backend_xrt_mul_mat(ctx, node);
        case GGML_OP_RMS_NORM:
        case GGML_OP_UNARY:
            // SILU / GELU / RMS_NORM: row-wise single-in/single-out.
            return ggml_backend_xrt_op_rowwise(ctx, node);
        // TODO(hw): GGML_OP_ROPE needs position/freq input binding — dispatch path
        // resolves the artifact but the arg layout is unvalidated; left to GPU.
        default:
            return false;
    }
}

static ggml_status ggml_backend_xrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto & ctx = *static_cast<ggml_backend_xrt_context *>(backend->context);
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (ggml_op_is_empty(node->op) || node->op == GGML_OP_NONE) { continue; }
        if (!ggml_backend_xrt_compute_node(ctx, node)) {
            GGML_XRT_LOG_WARN("no NPU kernel for op %s (node '%s')", ggml_op_name(node->op), node->name);
            return GGML_STATUS_FAILED;
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

static bool ggml_backend_xrt_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    // AOT-only gating: claim an op ONLY if a matching precompiled xclbin exists.
    // Everything else is left to the GPU/CPU by the scheduler. This is what makes
    // hybrid NPU+GPU execution work without JIT (see docs/ggml-xrt-plan.md §9).
    if (ggml_op_is_empty(op->op)) { return true; }
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return ggml_xrt_have_mul_mat(op);
        case GGML_OP_RMS_NORM:
            return ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        case GGML_OP_UNARY:
            // SILU / GELU (other unary ops have no artifact -> tag is null -> false)
            return ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        // ROPE dispatch is not enabled (unvalidated position/freq binding); the
        // scheduler routes it to the GPU.
        default:
            return false;
    }
}

static bool ggml_backend_xrt_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_xrt_buffer_type_get_name && buft->device == dev;
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
    /* .offload_op           = */ nullptr,
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
