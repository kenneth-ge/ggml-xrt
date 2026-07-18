//===- mv_bf16out.cc ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
//
// bf16-OUTPUT scalar matvec for the on-chip-CHAINED decode-attention kernel
// (attn_chain.py, Core A = QK^T). Computes C[M] = A[M,K] . B[K] and stores the
// result as bf16, because the *next* on-chip op is soft_max_ext_bf16 which
// consumes the scores vector as bf16 (see softmax_ext.cc). Keeping the QK^T
// output bf16 lets scores flow core->core through an object_fifo with NO
// f32<->bf16 repack and NO L3 round-trip.
//
// Accumulation is done in float (like mv.cc's bf16->f32 path) and only the
// final per-row sum is rounded to bf16, so this avoids the K-proportional
// bf16-accumulation bias that a naive bf16 accumulator would introduce.
//
// DIM_M / DIM_K are set at compile time (mirrors mv.cc). For QK^T:
//   DIM_M = m-tile of n_kv (32), DIM_K = head_dim (128).
//
// Compile-only validation on Linux/WSL; NOT executed on NPU.
#define NOCPP

#include <stdint.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 128
#endif

template <int M, int K>
void matvec_scalar_bf16bf16_impl(bfloat16 *restrict a, bfloat16 *restrict b,
                                 bfloat16 *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    float runningSum = 0.0f;
    for (int i = 0; i < K; i++) {
      // Promote to float BEFORE multiply (full-precision products), accumulate
      // in float, round to bf16 only at the end.
      runningSum += static_cast<float>(a[row * K + i]) * static_cast<float>(b[i]);
    }
    c[row] += (bfloat16)runningSum;
  }
  event1();
}

extern "C" {

void matvec_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out) {
  matvec_scalar_bf16bf16_impl<DIM_M, DIM_K>(a_in, b_in, c_out);
}

void zero_scalar_bf16(bfloat16 *c_out) { zero_scalar<bfloat16, DIM_M, 1>(c_out); }

} // extern "C"
