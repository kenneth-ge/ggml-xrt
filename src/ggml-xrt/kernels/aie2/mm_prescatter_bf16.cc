//===- mm_prescatter_bf16.cc (aie2 / Phoenix) -----------------------*- C++ -*-===//
//
// PRE-SCATTERED bf16 tiled matmul core — spec-decode M=N VERIFY, pure-mmul CEILING.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// The "host pre-scatters the weight" experiment. Same matmul_vectorized_4x4 (4x8x4 bf16
// systolic MAC) as mm_q6k_vecdq.cc, but with BOTH on-core steps removed:
//   - NO on-core dequant (weight arrives already bf16), and
//   - NO on-core scatter (weight already arrives in the mmul B systolic-tile order, "Bl1"),
// so the core does NOTHING but the raw MAC. This measures the pure-mmul ceiling for the
// M=N verify, i.e. how fast the kernel is once dequant + the 8192 scalar scatter stores per
// k-tile are entirely off-core (paid by the host / pre-processing).
//
//   C[m,n] += A[m,k] . B_prescattered[k,n]   for one k-tile (k-tile = DIM_K = 256).
//   A = activation bf16 in the mmul A sub-tile layout (memA DMA transform, unchanged).
//   B = WEIGHT, already bf16 AND already in the mmul B systolic-tile ("Bl1") order — read
//       DIRECTLY as the mmul B operand. NO Bl1 L1 scratch, NO scatter.
//   C = f32.
//
// HOST PRE-SCATTER SPEC (the layout the host must produce for B) — see gemv_mm_prescatter.py
// header for the full DRAM offset math. Per k-tile, per output column nc and K position kk,
// with s=8, t=4:  nt=nc/t, ti=nc%t, ks=kk/s, si=kk%s
//     B_tile[((ks*(DIM_N/t)+nt)*s+si)*t+ti] = dequant(weight)[nc][k-tile*256 + kk]
// This is EXACTLY the Bl1 scatter target from mm_q6k_vecdq.cc, so on-device C is bit-identical
// to the fused q6k path (modulo the host doing the dequant in f64/f32 vs on-core bf16).
//
// matmul_vectorized_4x4 steps rowA(=DIM_M/4) by 4 and colB(=DIM_N/4) by 4, so DIM_M and DIM_N
// must each be a multiple of 16. Smallest M-tile is 16. One chip only: aie2/Phoenix bf16 MMUL
// r=4,s=8,t=4. zero_f32 comes from mm.cc's combos macro (-Dbf16_f32_ONLY).
//
// UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // matmul_vectorized_4x4 (static inline) + zero_f32 + <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 16    // token sub-tile (mmul needs %16); host pads draft tokens to 16
#endif
#ifndef DIM_K
#define DIM_K 256   // one k-tile (== one q6_K superblock worth of K), for parity with q6k
#endif
#ifndef DIM_N
#define DIM_N 32    // output-channel sub-tile
#endif

extern "C" {
// A: [DIM_M,DIM_K] bf16 activation in the mmul A sub-tile layout.
// B: [DIM_K,DIM_N] bf16 WEIGHT, ALREADY in the mmul B systolic-tile ("Bl1") order.
// C: [DIM_M,DIM_N] f32.
void matmul_prescatter_f32(bfloat16 *A, bfloat16 *B, float *C) {
  matmul_vectorized_4x4<bfloat16, float, (DIM_M / 4), (DIM_K / 8), (DIM_N / 4), 4, 8, 4,
                        true, true>(A, B, C);
}
}
