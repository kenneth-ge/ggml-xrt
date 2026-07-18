#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XRT_NAME "XRT"
#define GGML_XRT_MAX_DEVICES 16

// Distinct buffer-type name for the shared NPU<->iGPU "hsa" buffer. Kept as a
// literal here so the Vulkan backend (which cannot include this header) can match
// it by strcmp in its supports_buft (see ggml-vulkan.cpp). Keep the two in sync.
#define GGML_XRT_HSA_NAME "XRT_HSA"

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_xrt_init(int32_t device);

GGML_BACKEND_API bool ggml_backend_is_xrt(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int32_t device);

// Shared "hsa" buffer type: allocations are XRT host_only bo's imported into
// ggml-vulkan (VK_EXT_external_memory_host), so ONE allocation is usable, with
// zero copy, by BOTH the NPU (XRT) and the iGPU (Vulkan). `import_dev` is the
// Vulkan device the pages are imported into; it is remembered by the (singleton)
// buffer type. Must be called with a valid Vulkan device before allocating.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_xrt_hsa_buffer_type(ggml_backend_dev_t import_dev);

// True if `buffer` was allocated from the shared hsa buffer type.
GGML_BACKEND_API bool ggml_backend_buffer_is_xrt_hsa(ggml_backend_buffer_t buffer);

// Real host pointer for a tensor's data: for a normal XRT buffer tensor->data is
// already the real host address; for a shared hsa buffer tensor->data is the
// Vulkan sentinel base + offset, which this translates back to the real mapped
// host address. Returns tensor->data unchanged for any other buffer.
GGML_BACKEND_API void * ggml_xrt_tensor_host_ptr(const struct ggml_tensor * tensor);

GGML_BACKEND_API int32_t ggml_backend_xrt_get_device_count(void);
GGML_BACKEND_API void ggml_backend_xrt_get_device_description(int32_t device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_xrt_get_device_memory(int32_t device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xrt_reg(void);

#ifdef  __cplusplus
}
#endif
