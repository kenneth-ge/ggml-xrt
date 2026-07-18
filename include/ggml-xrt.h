#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define GGML_XRT_NAME "XRT"
#define GGML_XRT_MAX_DEVICES 16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_xrt_init(int32_t device);

GGML_BACKEND_API bool ggml_backend_is_xrt(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_xrt_buffer_type(int32_t device);

GGML_BACKEND_API int32_t ggml_backend_xrt_get_device_count(void);
GGML_BACKEND_API void ggml_backend_xrt_get_device_description(int32_t device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_xrt_get_device_memory(int32_t device, size_t * free, size_t * total);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xrt_reg(void);

#ifdef  __cplusplus
}
#endif
