// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
// ggml-xrt: Windows/Linux backend that dispatches AIE (XDNA NPU) kernels through
// the XRT runtime. Kernels are precompiled ahead-of-time to `.xclbin` (+ a control
// `_insts.bin` instruction sequence) by the mlir-aie / IRON toolchain (see
// src/ggml-hsa/kernels), and loaded here at runtime via xrt::hw_context / xrt::kernel.
//
// This file is a SCAFFOLD. The backend registers cleanly and is safe to load on a
// machine with no NPU (device probing is lazy and exception-guarded). Actual op
// dispatch to the NPU is marked with TODO(xrt) and is intended to be completed on
// the Windows side against the Windows XRT SDK. The XRT C++ API used here is
// identical across Linux and Windows.

#include "ggml-xrt.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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

#define GGML_XRT_LOG_INFO(...)                    \
    do {                                          \
        if (ggml_xrt_logging_enabled()) {         \
            fprintf(stderr, "[ggml-xrt] " __VA_ARGS__); \
            fprintf(stderr, "\n");                \
        }                                         \
    } while (0)

#define GGML_XRT_LOG_WARN(...)                        \
    do {                                              \
        fprintf(stderr, "[ggml-xrt][warn] " __VA_ARGS__); \
        fprintf(stderr, "\n");                        \
    } while (0)

// ---------------------------------------------------------------------------
// XRT device layer
//
// A single lazily-opened xrt::device per index. Opening is exception-guarded so
// that backend registration never crashes on a machine without an NPU (e.g. WSL,
// where no /dev/accel device is exposed).
// ---------------------------------------------------------------------------

namespace {

// Directory that holds precompiled `<name>.xclbin` + `<name>_insts.bin` artifacts.
std::string ggml_xrt_kernel_dir() {
    const char * env = std::getenv("GGML_XRT_KERNEL_DIR");
    return env != nullptr ? std::string(env) : std::string();
}

struct ggml_xrt_device {
    int32_t                       index       = 0;
    bool                          probed      = false;
    bool                          available   = false;
    std::string                   description = "AMD XDNA NPU (XRT)";
    std::optional<xrt::device>    device;
    std::mutex                    mutex;

    explicit ggml_xrt_device(int32_t i) : index(i) {}

    // Attempt to open the physical device. Returns true if a device is present.
    // Safe to call anywhere: never throws.
    bool ensure_open() {
        std::lock_guard<std::mutex> lock(mutex);
        if (probed) {
            return available;
        }
        probed = true;
        try {
            device.emplace(static_cast<unsigned int>(index));
            try {
                description = device->get_info<xrt::info::device::name>();
            } catch (const std::exception &) {
                // name query is best-effort
            }
            available = true;
            GGML_XRT_LOG_INFO("opened device %d: %s", index, description.c_str());
        } catch (const std::exception & ex) {
            available = false;
            GGML_XRT_LOG_INFO("device %d unavailable: %s", index, ex.what());
        }
        return available;
    }
};

// Single-device scaffold. Extend to a vector when multi-NPU is needed.
ggml_xrt_device & ggml_xrt_get_device(int32_t index) {
    static ggml_xrt_device dev0(0);
    GGML_ASSERT(index == 0 && "ggml-xrt scaffold currently supports a single device");
    return dev0;
}

int32_t ggml_xrt_probe_device_count() {
    // Report 1 only if a real device can be opened, else 0. This keeps the backend
    // invisible (rather than crashing) on machines without an NPU.
    return ggml_xrt_get_device(0).ensure_open() ? 1 : 0;
}

} // namespace

int32_t ggml_backend_xrt_get_device_count(void) {
    return ggml_xrt_probe_device_count();
}

void ggml_backend_xrt_get_device_description(int32_t device, char * description, size_t description_size) {
    auto & dev = ggml_xrt_get_device(device);
    dev.ensure_open();
    snprintf(description, description_size, "%s", dev.description.c_str());
}

void ggml_backend_xrt_get_device_memory(int32_t device, size_t * free, size_t * total) {
    GGML_UNUSED(device);
    // TODO(xrt): query real NPU memory budget once available via XRT info.
    if (free)  { *free  = 0; }
    if (total) { *total = 0; }
}

// ---------------------------------------------------------------------------
// Buffer type / buffer
//
// Scaffold uses host memory for tensor storage so that set/get/clear work and the
// backend is testable without a device. When real NPU execution is wired up, these
// should allocate xrt::bo (device-visible buffers) and get_base should return
// bo.map(). Kept as host memory here to keep the scaffold runnable on any machine.
// ---------------------------------------------------------------------------

struct ggml_backend_xrt_buffer_context {
    void *      data = nullptr;
    size_t      size = 0;
    std::string name;

    explicit ggml_backend_xrt_buffer_context(size_t s) : size(s), name(GGML_XRT_NAME "_Buffer") {
        // 64-byte alignment mirrors typical AIE/DMA alignment expectations.
        data = ggml_aligned_malloc(size);
    }

    ~ggml_backend_xrt_buffer_context() {
        ggml_aligned_free(data, size);
    }
};

static void ggml_backend_xrt_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    delete static_cast<ggml_backend_xrt_buffer_context *>(buffer->context);
}

static void * ggml_backend_xrt_buffer_get_base(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_xrt_buffer_context *>(buffer->context)->data;
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
    std::memset(ctx->data, value, ctx->size);
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

// buffer type

static const char * ggml_backend_xrt_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_XRT_NAME;
}

static ggml_backend_buffer_t ggml_backend_xrt_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                       size_t size) {
    auto * ctx = new ggml_backend_xrt_buffer_context(size);
    if (ctx->data == nullptr && size > 0) {
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

static const ggml_backend_buffer_type_i ggml_backend_xrt_buffer_type_interface = {
    /* .get_name       = */ ggml_backend_xrt_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_xrt_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_xrt_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int32_t device) {
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ ggml_backend_xrt_buffer_type_interface,
        /* .device  = */ nullptr, // set on first use below
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

// Dispatch a single node to the NPU.
// TODO(xrt): implement real dispatch. The intended flow, using the XRT C++ API
// (identical on Windows and Linux):
//   1. Build a kernel key from the op + tensor shapes/dtypes (reuse the naming
//      scheme in src/ggml-hsa/kernel-discovery).
//   2. Load "<GGML_XRT_KERNEL_DIR>/<key>.xclbin" via device.load_xclbin(),
//      create an xrt::hw_context, and look up the xrt::kernel by name.
//   3. Read "<key>_insts.bin" into an xrt::bo (group_id from kernel.group_id).
//   4. Wrap src/dst tensor data in xrt::bo's (or use device buffers directly once
//      buffers are migrated to xrt::bo), sync inputs to the device.
//   5. run = kernel(opcode, insts_bo, insts_len, in_bo, out_bo, ...); run.wait();
//   6. Sync the output bo back to host.
static bool ggml_backend_xrt_compute_node(ggml_backend_xrt_context & ctx, ggml_tensor * node) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(node);
    return false; // not yet implemented
}

static ggml_status ggml_backend_xrt_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto & ctx = *static_cast<ggml_backend_xrt_context *>(backend->context);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        // Ops that carry no computation (views, reshapes, etc.) are no-ops here.
        if (ggml_op_is_empty(node->op) || node->op == GGML_OP_NONE) {
            continue;
        }

        if (!ggml_backend_xrt_compute_node(ctx, node)) {
            // The scheduler should not route unsupported ops here (see supports_op),
            // so reaching this point means a kernel is missing.
            GGML_XRT_LOG_WARN("no NPU kernel for op %s (node '%s') — not yet implemented",
                              ggml_op_name(node->op), node->name);
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
    // Arbitrary, stable GUID for the XRT backend.
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
    props->description  = ggml_backend_xrt_device_get_description(dev);
    props->type         = ggml_backend_xrt_device_get_type(dev);
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
    // Scaffold: only claim no-op tensors (views/reshapes/etc.). Real ops (starting
    // with MUL_MAT in bf16) are enabled here once ggml_backend_xrt_compute_node
    // gains a working xclbin dispatch path.
    return ggml_op_is_empty(op->op);
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
