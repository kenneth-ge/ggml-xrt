//===- mv_q6k.cc (aie2 / Phoenix) -------------------------*- C++ -*-===//
//
// Q6_K on-chip-dequant matrix-vector (decode gemv) core function.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// For Q4_K_M models: the Q6_K-typed tensors (attn_v, ffn_down, output) that Q4_K
// doesn't cover. Computes C[M] += dequant(A_q6K[M,K]) . B[K] for one k-tile == one
// 256-elem q6_K superblock per row. Dequant is exactly ggml `dequantize_row_q6_K`
// (ggml-quants.c): per 128-elem chunk, 4-bit low (ql) + 2-bit high (qh) combine to a
// 6-bit quant, minus 32, times int8 scale `sc[is + {0,2,4,6}]` (is=l/16) times d.
// Products accumulate in float (the gemv-bias lesson).
//
// Weight REPACKED host-side into ONE buffer (see build-q6k-gemv.sh). Per 256-elem
// superblock per output row, a **212-byte record** (nearly ggml's block_q6_K order,
// only `d` widened f16->f32):
//   [0..127]   ql      (128 raw block_q6_K.ql bytes)
//   [128..191] qh      (64  raw block_q6_K.qh bytes)
//   [192..207] scales  (16 raw int8 block_q6_K.scales)
//   [208..211] d       (f32, host-converted from ggml f16 x.d)
// A : [N][K/256][212]; b : [K] bf16; c : [N] f32; weight NOT transposed.
//
// NOTE: compiled on Linux, NOT executed here. Unvalidated scaffold — verify with the
// on-device q4_gemv_check.cpp (add Q6_K mode) before enabling dispatch.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q6k(const uint8_t *restrict a, const bfloat16 *restrict b, float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;              // 128 bytes
    const uint8_t *qh = rec + 128;        // 64 bytes
    const int8_t  *sc = (const int8_t *)(rec + 192);  // 16 int8 scales
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    float sum = 0.0f;
    for (int ch = 0; ch < 2; ch++) {   // two 128-elem chunks
      const uint8_t *qlc = ql + ch * 64;
      const uint8_t *qhc = qh + ch * 32;
      const int8_t  *scc = sc + ch * 8;
      const int base = ch * 128;
      for (int l = 0; l < 32; l++) {
        const int is = l >> 4;   // l / 16  -> 0 or 1
        const int q1 = (int)((qlc[l]      & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32;
        const int q2 = (int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32;
        const int q3 = (int)((qlc[l]      >> 4)   | (((qhc[l] >> 4) & 3) << 4)) - 32;
        const int q4 = (int)((qlc[l + 32] >> 4)   | (((qhc[l] >> 6) & 3) << 4)) - 32;
        sum += d * (float)scc[is + 0] * (float)q1 * (float)b[base + l];
        sum += d * (float)scc[is + 2] * (float)q2 * (float)b[base + l + 32];
        sum += d * (float)scc[is + 4] * (float)q3 * (float)b[base + l + 64];
        sum += d * (float)scc[is + 6] * (float)q4 * (float)b[base + l + 96];
      }
    }
    c[row] += sum;
  }
  event1();
}

extern "C" {
void matvec_q6k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q6k<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++) {
    c[i] = 0.0f;
  }
}
}
