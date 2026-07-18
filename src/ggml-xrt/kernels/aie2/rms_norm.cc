//===- rms_norm.cc (aie2 / Phoenix) -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
// aie2 (Phoenix/NPU1) RMS-norm core. Adapted from aie_kernels/aie2p/rms_norm.cc:
// the aie2p version accumulated squares through an aie::accum<acc32> and assigned
// a float vector to it, which does not type-check for aie2. This version keeps the
// running sum-of-squares in a plain float vector, which compiles for aie2.
//
// NOTE: compiled for Phoenix but NOT executed on hardware (no NPU in the dev
// environment) — correctness is unverified. gamma (weight) is fixed to 1.0 here;
// the ggml RMS_NORM weight multiply is applied separately (host/other kernel).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

template <typename T, int N>
void rms_norm(const T *restrict input, T *restrict output, int32_t cols) {
  event0();
  constexpr float epsilon = 1e-5f;
  const float gamma = 1.0f;

  ::aie::vector<T, N> gamma_v = ::aie::broadcast<T, N>(static_cast<T>(gamma));
  ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

  const int vector_chunks = cols / N;

  // Sum of squares, accumulated in a float vector (lane-wise), then reduced.
  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    // mul_square returns an accumulator; materialize as a float vector and add.
    ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a).template to_vector<float>();
    add_res = ::aie::add(add_res, square_v);
  }
  float sum_sq = ::aie::reduce_add(add_res);

  const int remaining = cols % N;
  const int start_idx = vector_chunks * N;
  for (int i = 0; i < remaining; i++) {
    float val = static_cast<float>(input[start_idx + i]);
    sum_sq += val * val;
  }

  const float mean_sq = sum_sq / static_cast<float>(cols) + epsilon;
  // Scalar aie::invsqrt() emits sqrtf, which the aie2 peano runtime lacks. Use the
  // vector reciprocal-sqrt intrinsic (hardware, no libm) and extract lane 0.
  ::aie::vector<float, N> ms_v   = ::aie::broadcast<float, N>(mean_sq);
  ::aie::vector<float, N> irms_v = ::aie::invsqrt(ms_v);
  const float inv_rms = irms_v.get(0);
  ::aie::vector<T, N> inv_rms_v = ::aie::broadcast<T, N>(static_cast<T>(inv_rms));

  for (int i = 0; i < vector_chunks; i++) {
    ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
    ::aie::vector<T, N> norm_v = ::aie::mul(reg_a, inv_rms_v).template to_vector<T>();
    ::aie::vector<T, N> out_v  = ::aie::mul(norm_v, gamma_v).template to_vector<T>();
    ::aie::store_v(output + i * N, out_v);
  }

  for (int i = 0; i < remaining; i++) {
    float val = static_cast<float>(input[start_idx + i]);
    output[start_idx + i] = static_cast<T>(val * inv_rms * gamma);
  }
  event1();
}

extern "C" {
void rms_norm(bfloat16 *input, bfloat16 *output, int32_t cols) {
  rms_norm<bfloat16, 16>(input, output, cols);
}
}
