//===- mv_vt.cc -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
//
//===----------------------------------------------------------------------===//
//
// ZERO-COPY scores*V for the FLASH_ATTN_EXT decode kernel (attn_flash.py, Core C).
//
//   out[d] += sum_j V(d,j) * probs[j]        (d in [0,HD), j in [0,KC) per chunk)
//
// ggml flash V is NON-transposed: v->ne=[head_dim, n_kv, n_head_kv], contiguous
// dim = head_dim (element (d,j) at j*head_dim + d). So each n_kv position j owns a
// CONTIGUOUS head_dim vector V(:,j). This kernel consumes V in exactly that native
// layout -- a (KC rows of n_kv, HD cols of head_dim) row-major tile, V[j*HD + d] --
// and does a probs-weighted COLUMN accumulate (out += probs[j] * V(:,j)). That
// REMOVES the transpose-in-DMA that the row-major matvec (mv.cc) would have needed:
// mv.cc computes c[row]+=A[row*K+i]*b[i] (A row-major, contraction contiguous),
// which for out[d]=sum_j V(d,j)probs[j] would require V n_kv-contiguous, i.e. a
// bf16 element transpose -- ILLEGAL on the shim DMA (a 1-element bf16 stride = 2
// bytes is not divisible by 4). Reading V in its native head_dim-contiguous layout
// (this kernel) keeps V zero-copy with NO transpose on either K or V.
//
// Core C accumulates the full n_kv contraction across n_kv/KC calls (c[d]+=),
// zeroing the HD output once per token then summing the chunks. HD (head_dim) and
// KC (n_kv chunk) are compile-time (-DHD -DKC), mirroring mv.cc's DIM_M/DIM_K.
//
// Compile-only validation on Linux/WSL; NOT executed on NPU.
#define NOCPP

#include <stdint.h>
#include <type_traits>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

#include "zero.cc"

#ifndef HD
#define HD 128   // attention head_dim (out length)
#endif

#ifndef KC
#define KC 64    // n_kv chunk (rows of V per call); V tile = KC*HD*2 bytes
#endif

template <int hd, int kc>
void matvec_vt_scalar_impl(bfloat16 *restrict v, bfloat16 *restrict p,
                           float *restrict c) {
  event0();
  // out[d] += sum_j p[j] * V(d,j) ; V row-major (KC, HD) native flash layout.
  // Promote to float BEFORE multiply (full-precision products), accumulate in the
  // f32 output (matches mv.cc's bf16->f32 path; no bf16-rounding accumulation bias).
  for (int j = 0; j < kc; j++) {
    float pj = static_cast<float>(p[j]);
    bfloat16 *restrict vrow = v + j * hd;
    for (int d = 0; d < hd; d++) {
      c[d] += static_cast<float>(vrow[d]) * pj;
    }
  }
  event1();
}

extern "C" {

// V tile (KC, HD) bf16, probs chunk (KC) bf16, out (HD) f32 (accumulated).
void matvec_vt_scalar_bf16_f32(bfloat16 *v_in, bfloat16 *p_in, float *c_out) {
  matvec_vt_scalar_impl<HD, KC>(v_in, p_in, c_out);
}

// Zero the whole HD output accumulator once per token (before the chunk loop).
void zero_scalar_f32_hd(float *c_out) { zero_scalar<float, HD, 1>(c_out); }

} // extern "C"
