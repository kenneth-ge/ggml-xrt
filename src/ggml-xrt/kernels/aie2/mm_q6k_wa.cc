//===- mm_q6k_wa.cc (aie2 / Phoenix) --------------------------*- C++ -*-===//
//
// Q6_K on-chip-dequant FUSED tiled matmul core, VECTORIZED dequant AND VECTORIZED
// B-scatter — 16-CORE whole_array-style spec-decode M=N VERIFY.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// Same math + mmul primitive as mm_q6k_vecdq.cc (matmul_vectorized_4x4, the 4x8x4 bf16
// systolic MAC — DIM_M=16 DIM_N=32 DIM_K=256, 16 accumulators). The ONLY change vs
// mm_q6k_vecdq.cc: the col[]->Bl1 SCALAR scatter (8192-iter scalar store loop, the single
// biggest single-core cost) is REPLACED by a fully VECTORIZED scatter built from the same
// vectorized deq_q6k_vec, an aie::transpose, and 32-lane vector stores.
//
//   C[m,n] += A[m,k] . dequant(B_q6K)   for one k-tile (= DIM_K = 256 = one q6_K superblock
//   per weight column). A = activation bf16 in the A mmul sub-tile layout (memA DMA transform).
//   B = WEIGHT, packed q6_K, uploaded RAW; the core dequants (VECTORIZED) and scatters bf16
//   into the B mmul sub-tile layout (Bl1, a design-owned L1 aie.buffer, one PER CORE), then
//   calls matmul_vectorized_4x4. C = f32.
//
// --- Vectorized scatter (the key fix) ------------------------------------------------------
// mmul B sub-tile layout: Bl1[((ks*(DIM_N/t)+nt)*s+si)*t+ti] = col_nc[kk], with
//   ks = kk/s, si = kk%s (s=8), nt = nc/t, ti = nc%t (t=4).   The tile for (ks,nt) is 32
//   contiguous bf16 in [si=8][ti=4] order (matmul_vectorized_4x4 loads size_B=s*t=32 per B tile).
// Process one nt-group (4 channels ti=0..3) at a time: dequant the 4 channels (vectorized) into
// col0..col3. For each ks (32), gather the 8 k-lanes of each of the 4 channels
// (col_ti[ks*8 .. ks*8+8]) into a 32-lane [ti=4][si=8] vector via aie::concat, then
// aie::transpose(_,4,8) -> [si=8][ti=4] (== the (ks,nt) tile), and a single 32-lane store.
// No scalar inner loop at all: 8 nt-groups x 32 ks = 256 transpose+store, each 32 lanes.
//
// Bl1 MUST be a design-owned L1 aie.buffer (see gemv_mm16.py) — a core-local array ICEs.
// One chip only: aie2/Phoenix, bf16 MMUL r=4,s=8,t=4. DIM_M,DIM_N must be multiples of 16.
//
// UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // matmul_vectorized_4x4 (static inline) + zero_f32 + <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 16
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

// 212-byte record -> 256 dequantized bf16 (one q6_K superblock), VECTORIZED. (Verbatim from
// mm_q6k_vecdq.cc — proven vectorized dequant; the col[] order is the natural deq_q6k order.)
//   [0..127] ql | [128..191] qh | [192..207] scales(int8) | [208..211] d(f32)
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
// qB: DIM_N superblock records (212 B each) for this k-tile. Aplain: [DIM_M,DIM_K] bf16 RAW
// row-major activation (broadcast shim->core, NO DMA transform — keeps the memtile at the proven
// 5+5 channel budget). Al1: design-owned L1 scratch (DIM_M*DIM_K bf16) for the re-tiled mmul A
// sub-tile layout. Bl1: design-owned L1 scratch (DIM_K*DIM_N bf16). C: [DIM_M,DIM_N] f32.
void matmul_q6k_f32(uint8_t *qB, bfloat16 *Aplain, bfloat16 *Al1, bfloat16 *Bl1, float *C) {
  constexpr int r = 4, s = 8, t = 4;
  constexpr int NT = DIM_N / t;      // n-tile count (channels/4)
  constexpr int KS = DIM_K / s;      // k-tile count (k/8)
  constexpr int RT = DIM_M / r;      // m rowtile count

  // --- A re-tile (VECTORIZED): plain row-major [m,k] -> mmul A sub-tile [z][i][rr][ss] ---
  // matmul_vectorized_4x4 reads A as size_A=r*s tiles: tile(z,i) at Al1[(z*KS+i)*r*s], row-major
  // [rr=r][ss=s]. Source: Aplain[(z*r+rr)*DIM_K + i*s + ss]. The 4 rr-rows concat directly into
  // [rr][ss] order (no transpose needed). No scalar loop.
  for (int z = 0; z < RT; z++) {
    for (int i = 0; i < KS; i++) {
      aie::vector<bfloat16, 8> u0 = aie::load_v<8>(Aplain + (z * r + 0) * DIM_K + i * s);
      aie::vector<bfloat16, 8> u1 = aie::load_v<8>(Aplain + (z * r + 1) * DIM_K + i * s);
      aie::vector<bfloat16, 8> u2 = aie::load_v<8>(Aplain + (z * r + 2) * DIM_K + i * s);
      aie::vector<bfloat16, 8> u3 = aie::load_v<8>(Aplain + (z * r + 3) * DIM_K + i * s);
      aie::store_v(Al1 + (z * KS + i) * (r * s), aie::concat(u0, u1, u2, u3));  // [rr][ss]
    }
  }

  bfloat16 col0[DIM_K], col1[DIM_K], col2[DIM_K], col3[DIM_K];

  for (int nt = 0; nt < NT; nt++) {
    // Dequant the 4 channels of this n-tile (ti = 0..3), each into its own col[] (VECTORIZED).
    deq_q6k_vec(qB + (nt * t + 0) * 212, col0);
    deq_q6k_vec(qB + (nt * t + 1) * 212, col1);
    deq_q6k_vec(qB + (nt * t + 2) * 212, col2);
    deq_q6k_vec(qB + (nt * t + 3) * 212, col3);

    // VECTORIZED scatter: build each (ks,nt) [si=8][ti=4] tile by transposing the 4 channels'
    // 8 k-lanes, then a single 32-lane store. No scalar remap loop.
    for (int ks = 0; ks < KS; ks++) {
      aie::vector<bfloat16, 8> a0 = aie::load_v<8>(col0 + ks * s);
      aie::vector<bfloat16, 8> a1 = aie::load_v<8>(col1 + ks * s);
      aie::vector<bfloat16, 8> a2 = aie::load_v<8>(col2 + ks * s);
      aie::vector<bfloat16, 8> a3 = aie::load_v<8>(col3 + ks * s);
      aie::vector<bfloat16, 32> cat = aie::concat(a0, a1, a2, a3);  // [ti=4][si=8]
      aie::vector<bfloat16, 32> tile = aie::transpose(cat, t, s);   // -> [si=8][ti=4]
      aie::store_v(Bl1 + (ks * NT + nt) * (s * t), tile);
    }
  }

  matmul_vectorized_4x4<bfloat16, float, (DIM_M / 4), (DIM_K / 8), (DIM_N / 4), 4, 8, 4,
                        true, true>(Al1, Bl1, C);
}
}
