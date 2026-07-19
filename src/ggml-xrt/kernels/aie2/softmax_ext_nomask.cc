//===- softmax_ext_nomask.cc ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Based on softmax_ext.cc (max-subtract 3-pass, getExpBf16 natural-e LUT).
//
//===--------------------------------------------------------------------===//
//
// NO-MASK soft_max_ext for FLASH_ATTN_EXT decode (attn_flash.py, Core B).
//
//   probs[i] = exp(scores[i]*scale - m) / sum_j exp(scores[j]*scale - m)
//   scale = 1/sqrt(head_dim) (baked via -DSCALE) ; m = max over VALID lanes.
//
// WHY NO MASK: Qwen3-1.7B decode is FULL-CAUSAL (no sliding window). The ggml
// flash kq_mask is all-0 for every valid past position, so it adds nothing --
// and ggml HARD-forces that mask to F16 (aie2 has NO native f16). Dropping the
// mask entirely sidesteps the f16 read completely (no mask fifo/DMA at all).
//
// THE MASK'S OTHER ROLE = PADDING (handled here instead): the overlay is built
// for a fixed bucket DIM_N (512/1024/2048) >= the actual n_kv. Lanes
// [n_valid, DIM_N) are padded/uninitialized KV positions that must NOT
// contribute. n_valid is a RUNTIME scalar (passed as a 1-element int32 buffer,
// nv[0]); this kernel writes a large-negative sentinel into scores[n_valid:DIM_N)
// BEFORE the softmax, so those lanes never win the max and exp(sentinel-m) -> 0.
// This makes ANY bucket DIM_N >= n_valid correct for arbitrary (not necessarily
// 16-aligned) n_valid -- the sentinel fill is per-lane scalar; the 3 vector
// passes then run over the whole DIM_N.
//
// getExpBf16 is strictly 16-wide, so SM_VEC_LEN must be 16 and DIM_N a multiple
// of 16 (the bucket sizes are).
//
// TODO: attention sinks not handled (Qwen3-1.7B has none).
//
//===--------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#ifndef SM_VEC_LEN
#define SM_VEC_LEN 16 // getExpBf16 is v16-only; keep at 16.
#endif

#ifndef DIM_N
#define DIM_N 512 // bucket n_kv; MUST be a multiple of SM_VEC_LEN.
#endif

#ifndef SCALE
// 1/sqrt(128) = 0.088388347648...
#define SCALE 0.08838834764831843f
#endif

// large-negative sentinel for padded lanes (bf16-representable, exp(.)->0).
#ifndef SM_NEG_SENTINEL
#define SM_NEG_SENTINEL (-1.0e30f)
#endif

using namespace aie;

static inline void softmax_nomask_impl(bfloat16 *restrict scores,
                                       bfloat16 *restrict out, int n_valid) {
  event0();

  // ---- Pad: scores[n_valid:DIM_N) = -sentinel (mask's -inf tail role) -------
  // Clamp n_valid into [0, DIM_N]. Scalar per-lane fill (handles any n_valid).
  if (n_valid < 0)
    n_valid = 0;
  if (n_valid > DIM_N)
    n_valid = DIM_N;
  for (int i = n_valid; i < DIM_N; i++) {
    scores[i] = (bfloat16)SM_NEG_SENTINEL;
  }

  const int iters = DIM_N / SM_VEC_LEN;
  const aie::vector<bfloat16, SM_VEC_LEN> scale_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)SCALE);

  auto it_s1 = aie::cbegin_restrict_vector<SM_VEC_LEN>(scores);
  auto it_s2 = aie::cbegin_restrict_vector<SM_VEC_LEN>(scores);
  auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>(out);
  auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>(out);
  auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>(out);

  // ---- Pass 1: x = scores*scale ; running max -----------------------------
  float max_val = -3.0e38f;
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, SM_VEC_LEN> s = *it_s1++;
    aie::accum<accfloat, SM_VEC_LEN> acc = aie::mul(s, scale_vec); // s*scale
    float rmax = aie::reduce_max(acc.to_vector<bfloat16>());
    if (rmax > max_val)
      max_val = rmax;
  }
  const aie::vector<bfloat16, SM_VEC_LEN> max_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)max_val);

  // ---- Pass 2: exp(x - max) ; accumulate sum ; stash into out -------------
  aie::accum<accfloat, SM_VEC_LEN> sum_acc = aie::zeros<accfloat, SM_VEC_LEN>();
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, SM_VEC_LEN> s = *it_s2++;
    aie::accum<accfloat, SM_VEC_LEN> acc = aie::mul(s, scale_vec); // s*scale
    acc = aie::sub(acc, max_vec);                                  // - max
    aie::vector<bfloat16, SM_VEC_LEN> e =
        to_v16bfloat16(getExpBf16(acc.to_vector<bfloat16>()));
    sum_acc = aie::add(sum_acc, e);
    *it_exp_out++ = e;
  }
  aie::vector<float, SM_VEC_LEN> red = sum_acc.to_vector<float>();
  float sum = aie::reduce_add(red);
  bfloat16 inv_sum = (bfloat16)aie::inv(sum);

  // ---- Pass 3: divide by sum ---------------------------------------------
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, SM_VEC_LEN> e = *it_scale++;
    aie::accum<accfloat, SM_VEC_LEN> o = aie::mul(e, inv_sum);
    *it_soft_out++ = o.to_vector<bfloat16>();
  }

  event1();
}

extern "C" {

// scale baked via -DSCALE, bucket n_kv via -DDIM_N. n_valid is a runtime scalar
// (nv[0]) -> lanes [n_valid, DIM_N) get the -sentinel (padding), NO mask read.
void softmax_ext_nomask_bf16(bfloat16 *restrict scores, bfloat16 *restrict out,
                             int32_t *restrict nv) {
  softmax_nomask_impl(scores, out, nv[0]);
}

} // extern "C"
