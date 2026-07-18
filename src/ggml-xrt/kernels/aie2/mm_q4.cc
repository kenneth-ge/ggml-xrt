//===- mm_q4.cc (aie2 / Phoenix) --------------------------*- C++ -*-===//
//
// Q4_0 on-chip-dequant FUSED tiled matmul core (vectorized; prefill AND padded decode).
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
//   C[m,n] += A[m,k] . dequant(B_q4_0)   for one k-tile, k-tile = DIM_K = 256 = EIGHT
//   32-elem q4_0 blocks per weight column (k must be a multiple of 32; 256 keeps parity
//   with the Q4_K/Q6_K designs and amortizes DMA).
//   A = activation bf16 in the A mmul sub-tile layout (memA DMA transform).
//   B = WEIGHT, packed q4_0, uploaded RAW (NO memB transform) — the core dequants and
//       scatters bf16 into the B mmul sub-tile layout (Bl1, a design-owned L1 aie.buffer),
//       then calls the validated matmul_vectorized_4x4 MAC. C = f32.
//
// Record per column per k-tile = 8 * 20 = 160 bytes (8 blocks). Per 20-byte block (== the
// validated mv_q4 repack): 16 nibble bytes (ggml block_q4_0.qs) then f32 d.
//   lo = (qs[j] & 0x0F) - 8 -> out[j] ; hi = (qs[j] >> 4) - 8 -> out[j+16]   (j in 0..15)
// Scatter index map == single_core memB transform [(k//s,s*n),(n//t,t),(s,n),(t,1)]:
//   B[k=ks*s+si, n=nt*t+ti]  ->  L1 index  ((ks*(N/t)+nt)*s + si)*t + ti      (s=8,t=4)
//
// Bl1 MUST be a design-owned L1 aie.buffer (see mm_q4.py) — a core-local array ICEs.
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
#define DIM_K 256   // 8 q4_0 blocks per column per k-tile
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

#define Q4_BLK 32
#define Q4_REC 20   // 16 nibble bytes + f32 d

namespace {
// 160-byte record (8 * 20) -> 256 dequantized floats (8 q4_0 blocks for one column).
inline void deq_q4_0(const uint8_t *rec, float *out) {
  for (int blk = 0; blk < DIM_K / Q4_BLK; blk++) {
    const uint8_t *qs = rec + blk * Q4_REC;
    float d;
    __builtin_memcpy(&d, qs + 16, 4);
    float *o = out + blk * Q4_BLK;
    for (int j = 0; j < 16; j++) {
      o[j]      = (float)((int)(qs[j] & 0x0F) - 8) * d;
      o[j + 16] = (float)((int)(qs[j] >> 4)   - 8) * d;
    }
  }
}
} // namespace

extern "C" {
// qB: DIM_N records (160 B each) for this k-tile. A: [DIM_M,DIM_K] bf16 in A sub-tile
// layout. Bl1: design-owned L1 scratch (DIM_K*DIM_N bf16). C: [DIM_M,DIM_N] f32.
void matmul_q4_0_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  constexpr int s = 8, t = 4;
  constexpr int REC = (DIM_K / Q4_BLK) * Q4_REC;  // 160
  float col[DIM_K];
  for (int nc = 0; nc < DIM_N; nc++) {
    deq_q4_0(qB + nc * REC, col);
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
