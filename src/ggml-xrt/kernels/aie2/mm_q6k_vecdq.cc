//===- mm_q6k_vecdq.cc (aie2 / Phoenix) ------------------*- C++ -*-===//
//
// Q6_K on-chip-dequant FUSED tiled matmul core, VECTORIZED dequant — spec-decode M=N VERIFY.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// Identical design to mm_q6k.cc (mmul primitive matmul_vectorized_4x4, the 4x8x4 bf16 systolic
// MAC — the right primitive for M>1: dense MACs, no per-token reduce). The ONLY change vs
// mm_q6k.cc: the per-column dequant is now VECTORIZED (the noscratch group-scale-in-registers
// path from mv_q6k_noscratch.cc) instead of scalar deq_q6k (which made mm_q6k.cc 4262 ms).
//
//   C[m,n] += A[m,k] . dequant(B_q6K)  for one k-tile (k-tile = DIM_K = 256 = one q6_K
//   superblock per weight column). A = activation bf16 in the A mmul sub-tile layout (memA DMA
//   transform). B = WEIGHT, packed q6_K, uploaded RAW; the core dequants (VECTORIZED) and
//   scatters bf16 into the B mmul sub-tile layout (Bl1, design-owned L1 aie.buffer), then calls
//   the validated matmul_vectorized_4x4 MAC. C = f32.
//
// Vectorized dequant == the 8-chunk QW/gsb assembly proven on HW via mv_q6k. The chunk->col[]
// store offsets below reproduce deq_q6k's natural col[] order EXACTLY (verified by a Linux
// scalar col[]-match check; see build-mm-verify.sh notes). The EXISTING scatter (col->Bl1) and
// matmul_vectorized_4x4 call are unchanged from mm_q6k.cc.
//
// Bl1 MUST be a design-owned L1 aie.buffer (see mm_q6k.py) — a core-local array ICEs.
// matmul_vectorized_4x4 steps rowA(=DIM_M/4) by 4 and colB(=DIM_N/4) by 4, so DIM_M and DIM_N
// must each be a multiple of 16 (DIM_M=4 would read out of bounds). Smallest M-tile is 16.
// One chip only: aie2/Phoenix, bf16 MMUL r=4,s=8,t=4.
//
// UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // matmul_vectorized_4x4 (static inline) + zero_f32 + <aie_api/aie.hpp>

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

// per-chunk 32-lane scale, built inline from the group-scale vector gsb (groups in lanes 12..27):
// chunk ci spans groups 2ci (low 16 lanes) and 2ci+1 (high 16 lanes). No sbuf store/load.
#define SCV(ci)                                                                            \
  aie::concat(aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci))),                         \
              aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci) + 1)))

// bit-assemble a 32-lane 6-bit chunk (low nibble from QLv, high 2 bits from QHv>>hi_shift),
// -32, then scale by group scales SCV(ci). Yields aie::vector<bfloat16,32>.
#define QW(QLv, QHv, hi_shift, ci)                                                          \
  aie::mul(aie::sub(aie::to_float<bfloat16>(aie::unpack(                                    \
                        aie::bit_or((hi_shift) < 4 ? aie::bit_and((uint8_t)0x0F, (QLv))     \
                                                   : aie::logical_downshift((QLv), 4),      \
                                    aie::upshift(aie::bit_and((uint8_t)0x03,                 \
                                        aie::logical_downshift((QHv), (hi_shift))), 4)))),   \
                    c32v),                                                                  \
           SCV(ci)).template to_vector<bfloat16>()

// 212-byte record -> 256 dequantized bf16 (one q6_K superblock), VECTORIZED.
//   [0..127] ql | [128..191] qh | [192..207] scales(int8) | [208..211] d(f32)
// Store order MUST equal scalar deq_q6k's out[] (see mm_q6k.cc). The 8 chunk->offset
// pairs below match the b-offsets each chunk MACs against in mv_q6k_noscratch.cc.
inline void deq_q6k_vec(const uint8_t *rec, bfloat16 *col) {
  const uint8_t *ql = rec;
  const uint8_t *qh = rec + 128;
  const int8_t  *sc = (const int8_t *)(rec + 192);
  float d;
  __builtin_memcpy(&d, rec + 208, 4);

  const aie::vector<bfloat16, 32> c32v = aie::broadcast<bfloat16, 32>((bfloat16)32.0f);

  // group scales in registers (lanes 12..27): gsb.get(12+g) == sc[g]*d.
  aie::vector<bfloat16, 32> gsb =
      aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(sc - 12))),
               aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();

  aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
  aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
  aie::vector<uint8_t, 32> Ha  = aie::load_unaligned_v<32>(qh);
  aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
  aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
  aie::vector<uint8_t, 32> Hb  = aie::load_unaligned_v<32>(qh + 32);

  // Chunk -> col[] offset (natural deq_q6k order):
  aie::store_v(col +   0, QW(L0a, Ha, 0, 0));   // ch0 q1  -> out[0:32]
  aie::store_v(col +  32, QW(L1a, Ha, 2, 1));   // ch0 q2  -> out[32:64]
  aie::store_v(col +  64, QW(L0a, Ha, 4, 2));   // ch0 q3  -> out[64:96]
  aie::store_v(col +  96, QW(L1a, Ha, 6, 3));   // ch0 q4  -> out[96:128]
  aie::store_v(col + 128, QW(L0b, Hb, 0, 4));   // ch1 q1  -> out[128:160]
  aie::store_v(col + 160, QW(L1b, Hb, 2, 5));   // ch1 q2  -> out[160:192]
  aie::store_v(col + 192, QW(L0b, Hb, 4, 6));   // ch1 q3  -> out[192:224]
  aie::store_v(col + 224, QW(L1b, Hb, 6, 7));   // ch1 q4  -> out[224:256]
}
} // namespace

extern "C" {
// qB: DIM_N superblock records (212 B each) for this k-tile. A: [DIM_M,DIM_K] bf16 in
// A sub-tile layout. Bl1: design-owned L1 scratch (DIM_K*DIM_N bf16). C: [DIM_M,DIM_N] f32.
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  constexpr int s = 8, t = 4;
  bfloat16 col[DIM_K];
  for (int nc = 0; nc < DIM_N; nc++) {
    deq_q6k_vec(qB + nc * 212, col);
    const int nt = nc / t, ti = nc % t;
    for (int kk = 0; kk < DIM_K; kk++) {
      const int ks = kk / s, si = kk % s;
      Bl1[((ks * (DIM_N / t) + nt) * s + si) * t + ti] = col[kk];
    }
  }
  matmul_vectorized_4x4<bfloat16, float, (DIM_M / 4), (DIM_K / 8), (DIM_N / 4), 4, 8, 4,
                        true, true>(A, Bl1, C);
}
}
