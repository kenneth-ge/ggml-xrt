//===- mm_q4k.cc (aie2 / Phoenix) -------------------------*- C++ -*-===//
//
// Q4_K on-chip-dequant TILED matmul core (vectorized; serves prefill AND decode).
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// C[m,n] += A[m,k] . dequant(B_q4K)   for one k-tile, k-tile = DIM_K = 256 = one
// q4_K superblock per weight column (superblock-aligned so scales don't split).
//   A = activation, bf16, already in the A mmul sub-tile layout (memA DMA transform).
//   B = WEIGHT, packed q4_K, uploaded raw (NO memB transform) — the core dequants it
//       and scatters bf16 into the B mmul sub-tile layout, then calls the *validated*
//       matmul_vectorized_4x4 MAC. C = f32.
//
// Only new code vs the validated bf16 path is deq (= ggml dequantize_row_q4_K, proven
// on HW via mv_q4k) + the scatter, whose index map is taken verbatim from the single_core
// memB `dimensionsToStream [(k//s,s*n),(n//t,t),(s,n),(t,1)]` transform (s=8,t=4):
//   B[k=ks*s+si, n=nt*t+ti]  ->  L1 index  ((ks*(N/t)+nt)*s + si)*t + ti
//
// Bl1 MUST be a design-owned L1 aie.buffer (see mm_q4k.py) — a core-LOCAL Bl1 array ICEs
// llvm-aie (16 KB exceeds core capacity). Passing it in as a pointer arg fixes that; the
// buffer is allocated by the IRON design (aie.buffer on the compute tile).
// One chip only: aie2/Phoenix, bf16 MMUL r=4,s=8,t=4.
//
// UNVALIDATED scaffold — compiled on Linux (no NPU). Built via build-q4k-mm.sh for the
// Qwen3-1.7B Q4_K shapes; verify on-device (q4_gemv_check, mm mode) before dispatch.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // aie_kernels/aie2/mm.cc: matmul_vectorized_4x4 (static inline), zero_*

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 256   // one q4_K superblock per column per k-tile
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

namespace {
inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
  else { *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
         *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
// 148-byte record -> 256 dequantized floats (one q4_K superblock)
inline void deq_q4k(const uint8_t *rec, float *out) {
  const uint8_t *qs = rec;
  const uint8_t *sca = rec + 128;
  float d, dmin;
  __builtin_memcpy(&d, rec + 140, 4);
  __builtin_memcpy(&dmin, rec + 144, 4);
  for (int ck = 0; ck < 4; ck++) {
    uint8_t sc, mn;
    get_scale_min_k4(2 * ck + 0, sca, &sc, &mn);
    const float d1 = d * (float)sc, m1 = dmin * (float)mn;
    get_scale_min_k4(2 * ck + 1, sca, &sc, &mn);
    const float d2 = d * (float)sc, m2 = dmin * (float)mn;
    const uint8_t *q = qs + ck * 32;
    const int base = ck * 64;
    for (int l = 0; l < 32; l++) {
      out[base + l]      = d1 * (float)(q[l] & 0x0F) - m1;
      out[base + 32 + l] = d2 * (float)(q[l] >> 4)   - m2;
    }
  }
}
} // namespace

extern "C" {
// qB: DIM_N superblock records (148 B each) for this k-tile. A: [DIM_M,DIM_K] bf16 in
// A sub-tile layout. Bl1: design-owned L1 scratch (DIM_K*DIM_N bf16) — NOT a core-local
// array (that ICEs). C: [DIM_M,DIM_N] f32 accumulator.
void matmul_q4k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  constexpr int s = 8, t = 4;
  float col[DIM_K];
  for (int nc = 0; nc < DIM_N; nc++) {
    deq_q4k(qB + nc * 148, col);
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
