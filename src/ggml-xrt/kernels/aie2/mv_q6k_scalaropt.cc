//===- mv_q6k_scalaropt.cc (aie2 / Phoenix) ---------------*- C++ -*-===//
//
// Q6_K decode matvec — SCALAR, but with the obvious codegen fixes, to separate "bad
// scalar codegen" from "not vectorized" (per the decomposition round). Same ABI/repack as
// mv_q6k.cc; drop-in matvec_q6k_f32.
// Changes vs mv_q6k.cc:
//   * hoist d*scale: precompute dsc[16] = d * sc[j] once per row (was recomputed inline).
//   * break the single serial `sum +=` dependency chain into 4 independent accumulators
//     (one per sub-element q1..q4), so the scalar FPU can pipeline instead of stalling on
//     each dependent add.
//   * activation promoted to float once.
// Still fully scalar float — no aie::vector. If this alone closes most of the gap, the
// problem was codegen; if not, vectorization (mv_q6k_vec.cc) is required.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q6k_scalaropt(const uint8_t *restrict a, const bfloat16 *restrict b,
                          float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    float dsc[16];
    for (int j = 0; j < 16; j++)
      dsc[j] = d * (float)sc[j];

    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;  // 4 independent chains
    for (int ch = 0; ch < 2; ch++) {
      const uint8_t *qlc = ql + ch * 64;
      const uint8_t *qhc = qh + ch * 32;
      const int scb = ch * 8;
      const int base = ch * 128;
      for (int l = 0; l < 32; l++) {
        const int is = l >> 4;
        const float q1 = (float)((int)((qlc[l] & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32);
        const float q2 = (float)((int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32);
        const float q3 = (float)((int)((qlc[l] >> 4) | (((qhc[l] >> 4) & 3) << 4)) - 32);
        const float q4 = (float)((int)((qlc[l + 32] >> 4) | (((qhc[l] >> 6) & 3) << 4)) - 32);
        s0 += dsc[scb + is + 0] * q1 * (float)b[base + l];
        s1 += dsc[scb + is + 2] * q2 * (float)b[base + l + 32];
        s2 += dsc[scb + is + 4] * q3 * (float)b[base + l + 64];
        s3 += dsc[scb + is + 6] * q4 * (float)b[base + l + 96];
      }
    }
    c[row] += (s0 + s1) + (s2 + s3);
  }
  event1();
}

extern "C" {
void matvec_q6k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q6k_scalaropt<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
