//===- mm_q6k.cc (aie2 / Phoenix) -------------------------*- C++ -*-===//
//
// Q6_K on-chip-dequant FUSED tiled matmul core (vectorized; prefill AND padded decode).
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// Completes Q4_K_M prefill: the Q6_K-typed tensors (attn_v, ffn_down, output) that the
// Q4_K fused matmul (mm_q4k.cc) doesn't cover. Same design as mm_q4k, only the dequant +
// record size differ:
//   C[m,n] += A[m,k] . dequant(B_q6K)   for one k-tile, k-tile = DIM_K = 256 = one q6_K
//   superblock per weight column (superblock-aligned so scales/qh don't split).
//   A = activation bf16 in the A mmul sub-tile layout (memA DMA transform).
//   B = WEIGHT, packed q6_K, uploaded RAW (NO memB transform) — the core dequants and
//       scatters bf16 into the B mmul sub-tile layout (Bl1, a design-owned L1 aie.buffer),
//       then calls the validated matmul_vectorized_4x4 MAC. C = f32.
//
// deq_q6k == ggml dequantize_row_q6_K (proven on HW via mv_q6k); products promote to float.
// Scatter index map is the single_core memB transform [(k//s,s*n),(n//t,t),(s,n),(t,1)]:
//   B[k=ks*s+si, n=nt*t+ti]  ->  L1 index  ((ks*(N/t)+nt)*s + si)*t + ti      (s=8,t=4)
//
// Bl1 MUST be a design-owned L1 aie.buffer (see mm_q6k.py) — a core-local array ICEs.
// One chip only: aie2/Phoenix, bf16 MMUL r=4,s=8,t=4.
//
// UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // matmul_vectorized_4x4 (static inline) + zero_f32

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 256   // one q6_K superblock per column per k-tile
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

namespace {
// 212-byte record -> 256 dequantized floats (one q6_K superblock).
//   [0..127] ql | [128..191] qh | [192..207] scales(int8) | [208..211] d(f32)
inline void deq_q6k(const uint8_t *rec, float *out) {
  const uint8_t *ql = rec;
  const uint8_t *qh = rec + 128;
  const int8_t  *sc = (const int8_t *)(rec + 192);
  float d;
  __builtin_memcpy(&d, rec + 208, 4);
  for (int ch = 0; ch < 2; ch++) {          // two 128-elem chunks
    const uint8_t *qlc = ql + ch * 64;
    const uint8_t *qhc = qh + ch * 32;
    const int8_t  *scc = sc + ch * 8;
    const int base = ch * 128;
    for (int l = 0; l < 32; l++) {
      const int is = l >> 4;                 // l / 16 -> 0 or 1
      const int q1 = (int)((qlc[l]      & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((qlc[l]      >> 4)   | (((qhc[l] >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((qlc[l + 32] >> 4)   | (((qhc[l] >> 6) & 3) << 4)) - 32;
      out[base + l]      = d * (float)scc[is + 0] * (float)q1;
      out[base + l + 32] = d * (float)scc[is + 2] * (float)q2;
      out[base + l + 64] = d * (float)scc[is + 4] * (float)q3;
      out[base + l + 96] = d * (float)scc[is + 6] * (float)q4;
    }
  }
}
} // namespace

extern "C" {
// qB: DIM_N superblock records (212 B each) for this k-tile. A: [DIM_M,DIM_K] bf16 in
// A sub-tile layout. Bl1: design-owned L1 scratch (DIM_K*DIM_N bf16). C: [DIM_M,DIM_N] f32.
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  constexpr int s = 8, t = 4;
  float col[DIM_K];
  for (int nc = 0; nc < DIM_N; nc++) {
    deq_q6k(qB + nc * 212, col);
    const int nt = nc / t, ti = nc % t;
    for (int kk = 0; kk < DIM_K; kk++) {
      const int ks = kk / s, si = kk % s;
      Bl1[((ks * (DIM_N / t) + nt) * s + si) * t + ti] = (bfloat16)col[kk];
    }
  }
  matmul_vectorized_4x4<bfloat16, float, (DIM_M / 4), (DIM_K / 8), (DIM_N / 4), 4, 8, 4,
                        true, true>(A, Bl1, C);
}
}
