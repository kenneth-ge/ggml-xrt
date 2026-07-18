//===- softmax_ext.cc ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// Based on aie2p/softmax.cc (max-subtract 3-pass) and aie2/softmax.cc /
// lut_based_ops.h (getExpBf16 natural-e LUT).
//
//===--------------------------------------------------------------------===//
//
// ggml soft_max_ext for ONE decode-attention step (one head).
//
//   probs[i] = exp(scores[i]*scale + mask[i] - m) / sum_j exp(scores[j]*scale + mask[j] - m)
//   scale = 1/sqrt(head_dim) = 1/sqrt(128)   (baked in via -DSCALE)
//   m     = max_i (scores[i]*scale + mask[i])   (numerical-stability shift)
//
// Numerically-safe max-subtract variant: after subtracting m every argument is
// <= 0, so getExpBf16 (which reads a 256-entry LUT via bfloat16_to_int(.,8) and
// truncates out-of-range) stays in its well-defined (0,1] output range.
//
// The natural-e LUT (getExpBf16) is used instead of aie::exp2, so NO log2e
// multiply is needed (base-e directly). getExpBf16 is strictly 16-wide, so
// SM_VEC_LEN must be 16.
//
// PAD REQUIREMENT (variable n_kv):
//   DIM_N is the *padded* n_kv and MUST be a multiple of SM_VEC_LEN (16).
//   The caller pads the tail lanes [n_kv, DIM_N) of BOTH scores and mask with a
//   large-negative sentinel (e.g. -1e30 bf16). Those lanes then never win the
//   max and exp(large_neg - m) -> 0, so they add nothing to the sum. (Padding
//   just the mask with -1e30 is also sufficient; padding both is safest.)
//
// TODO: attention sinks are NOT handled here (Qwen3-1.7B first pass has none).
//       When needed, extend the sum with the sink logits before the divide.
//
//===--------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#ifndef SM_VEC_LEN
#define SM_VEC_LEN 16 // getExpBf16 is v16-only; keep at 16.
#endif

#ifndef DIM_N
#define DIM_N 512 // padded n_kv; MUST be a multiple of SM_VEC_LEN.
#endif

#ifndef SCALE
// 1/sqrt(128) = 0.088388347648...
#define SCALE 0.08838834764831843f
#endif

using namespace aie;

static inline void softmax_ext_impl(bfloat16 *restrict scores,
                                    bfloat16 *restrict mask,
                                    bfloat16 *restrict out) {
  event0();

  const int iters = DIM_N / SM_VEC_LEN;
  const aie::vector<bfloat16, SM_VEC_LEN> scale_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)SCALE);

  auto it_s1 = aie::cbegin_restrict_vector<SM_VEC_LEN>(scores);
  auto it_m1 = aie::cbegin_restrict_vector<SM_VEC_LEN>(mask);
  auto it_s2 = aie::cbegin_restrict_vector<SM_VEC_LEN>(scores);
  auto it_m2 = aie::cbegin_restrict_vector<SM_VEC_LEN>(mask);
  auto it_exp_out = aie::begin_restrict_vector<SM_VEC_LEN>(out);
  auto it_scale = aie::cbegin_restrict_vector<SM_VEC_LEN>(out);
  auto it_soft_out = aie::begin_restrict_vector<SM_VEC_LEN>(out);

  // ---- Pass 1: x = scores*scale + mask ; running max ----------------------
  float max_val = -3.0e38f;
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, SM_VEC_LEN> s = *it_s1++;
    aie::vector<bfloat16, SM_VEC_LEN> mk = *it_m1++;
    aie::accum<accfloat, SM_VEC_LEN> acc = aie::mul(s, scale_vec); // s*scale
    acc = aie::add(acc, mk);                                       // + mask
    float rmax = aie::reduce_max(acc.to_vector<bfloat16>());
    if (rmax > max_val)
      max_val = rmax;
  }
  const aie::vector<bfloat16, SM_VEC_LEN> max_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)max_val);

  // ---- Pass 2: exp(x - max) ; accumulate sum ; stash into out ------------
  aie::accum<accfloat, SM_VEC_LEN> sum_acc = aie::zeros<accfloat, SM_VEC_LEN>();
  for (int i = 0; i < iters; i++) {
    aie::vector<bfloat16, SM_VEC_LEN> s = *it_s2++;
    aie::vector<bfloat16, SM_VEC_LEN> mk = *it_m2++;
    aie::accum<accfloat, SM_VEC_LEN> acc = aie::mul(s, scale_vec); // s*scale
    acc = aie::add(acc, mk);                                       // + mask
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

// scale is baked in via -DSCALE, padded n_kv via -DDIM_N (compile defines,
// mirroring the DIM_M/DIM_K convention of the other ggml-xrt kernels).
void softmax_ext_bf16(bfloat16 *restrict scores, bfloat16 *restrict mask,
                      bfloat16 *restrict out) {
  softmax_ext_impl(scores, mask, out);
}

} // extern "C"
