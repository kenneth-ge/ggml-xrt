//===- eltwise_add.cc ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Based on aie_kernels/aie2/add.cc (eltwise_vadd v16 idiom).
//
//===--------------------------------------------------------------------===//
//
// ggml-xrt RESIDUAL-ADD for the decode residual stream (single compute tile).
//
//   c[i] = a[i] + b[i]      (pure elementwise, f32)
//
// Used for the two per-layer residual adds (x += attn_out, x += ffn_out) so the
// residual stream stays resident on the NPU as part of the q..o block.
//
// f32 (the residual stream is f32, NOT bf16). DIM_N is n_embd (Qwen3-1.7B =
// 2048), a multiple of the v16 vector width so the loop is clean. No LUT, no
// accumulator/reduction: a pure store of add(load,load).
//
//===--------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef VEC_LEN
#define VEC_LEN 16 // f32 v16 store/load; DIM_N must be a multiple of this.
#endif

#ifndef DIM_N
#define DIM_N 2048 // n_embd (Qwen3-1.7B); MUST be a multiple of VEC_LEN.
#endif

using namespace aie;

static inline void add_f32_impl(float *restrict a, float *restrict b,
                                float *restrict c) {
  event0();
  for (int i = 0; i < DIM_N; i += VEC_LEN) {
    aie::vector<float, VEC_LEN> va = aie::load_v<VEC_LEN>(a + i);
    aie::vector<float, VEC_LEN> vb = aie::load_v<VEC_LEN>(b + i);
    aie::store_v(c + i, aie::add(va, vb));
  }
  event1();
}

extern "C" {

// padded n_embd via -DDIM_N (compile define, mirroring the DIM_N convention of
// the other ggml-xrt kernels).
void add_f32(float *restrict a, float *restrict b, float *restrict c) {
  add_f32_impl(a, b, c);
}

} // extern "C"
