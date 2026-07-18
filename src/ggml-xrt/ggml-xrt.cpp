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

#include <algorithm>
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
        while (kernels.size() >= max_contexts && !kernel_lru.empty()) {
            const std::string victim = kernel_lru.back();
            kernel_lru.pop_back();
            kernels.erase(victim);
            GGML_XRT_LOG_INFO("evicted kernel %s (context cap %zu)", victim.c_str(), max_contexts);
        }
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
    // M==1 decode: prefer the dedicated gemv kernel if one exists for (K,N).
    // Different ABI from the tiled matmul: C[N] = A[N,K] . B[K], with
    //   A = weight in ggml-native [N,K] layout (NO transpose) @ group 3,
    //   B = activation [K] @ group 4, C = output [N] @ group 5, one launch.
    // The weight A is cached UNtransposed (separate from the tiled transposed
    // cache). Falls through to the tiled path if no gemv artifact exists.
    // -----------------------------------------------------------------------
    if (M == 1) {
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
    GGML_XRT_LOG_INFO("graph_compute: %d ops on NPU (mul_mat=%d rms_norm=%d unary=%d other=%d)",
                      n_mm + n_rms + n_un + n_other, n_mm, n_rms, n_un, n_other);
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
            // validated on-device (unit harness vs CPU, NRMSE ~0.004); default-on
            return ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        case GGML_OP_UNARY:
            // SILU / GELU validated on-device (unit harness vs CPU); default-on
            return ggml_xrt_have_op_kernel(op) && ggml_is_contiguous(op);
        case GGML_OP_ROPE:
            // NEOX RoPE validated on-device (unit harness vs CPU); default-on.
            // Requires contiguous src0 and a matching rope_<head_dim> artifact
            // (ggml_xrt_have_rope also restricts to full-width NEOX).
            return ggml_xrt_have_rope(op) &&
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
