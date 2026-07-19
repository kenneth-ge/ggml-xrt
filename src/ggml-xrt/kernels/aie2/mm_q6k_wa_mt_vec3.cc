//===- mm_q6k_wa_mt_vec3.cc (aie2 / Phoenix) ------------------*- C++ -*-===//
//
// RE-VECTORIZED scatter attempt #3 on the PROVEN scalar baseline (memtile-A, DIM_M=32). NATIVE
// 32-lane throughout — no sub-register ops. Attempt #1/#2 (concat of four aie::load_v<8> col
// slices + transpose) zeroed Bl1 on HW because the concat lowered to @llvm.aie2.set.I512.I128
// sub-register INSERTS (building a 512-bit reg from 4x 128-bit pieces), which peano miscompiles.
//
// This version keeps vectors 32-wide: load each channel's 32-value chunk with a native 512-bit
// load, then a 4-way lane interleave (aie::interleave_zip, native vshuffle/vsel) whose four
// 32-lane halves ARE the four mmul-B tiles of the chunk.
//
//   v_ti = load_v<32>(col_ti + c*32)               // [ksl=4][si=8], native 512-bit load
//   (x0,x1) = interleave_zip(v0, v1, 1)            // [v0[i],v1[i],...]
//   (y0,y1) = interleave_zip(v2, v3, 1)            // [v2[i],v3[i],...]
//   (z0,z1) = interleave_zip(x0, y0, 2)            // [v0[i],v1[i],v2[i],v3[i],...] for i=0..15
//   (z2,z3) = interleave_zip(x1, y1, 2)            // ... for i=16..31
//   => z_ksl[si*4+ti] = v_ti[ksl*8+si] == Bl1 tile(ks=c*4+ksl, nt)[si][ti].   Store 32-lane.
//
// Bl1 target (mmul-B sub-tile, matches scalar): Bl1[(ks*8+nt)*32 + si*4 + ti] = col_{nt*4+ti}[ks*8+si].
// VALIDATE on HW with mm_verify_wa_vec3_bl1echo — Bl1 MUST byte-match the scalar scatter before
// trusting mm_verify. Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <utility>
#include "mm.cc"  // matmul_vectorized_4x4 + zero_f32 + <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 256
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

namespace {

#define SCV(ci)                                                                            \
  aie::concat(aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci))),                         \
              aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci) + 1)))

#define QW(QLv, QHv, hi_shift, ci)                                                          \
  aie::mul(aie::sub(aie::to_float<bfloat16>(aie::unpack(                                    \
                        aie::bit_or((hi_shift) < 4 ? aie::bit_and((uint8_t)0x0F, (QLv))     \
                                                   : aie::logical_downshift((QLv), 4),      \
                                    aie::upshift(aie::bit_and((uint8_t)0x03,                 \
                                        aie::logical_downshift((QHv), (hi_shift))), 4)))),   \
                    c32v),                                                                  \
           SCV(ci)).template to_vector<bfloat16>()

inline void deq_q6k_vec(const uint8_t *rec, bfloat16 *col) {
  const uint8_t *ql = rec;
  const uint8_t *qh = rec + 128;
  const int8_t  *sc = (const int8_t *)(rec + 192);
  float d;
  __builtin_memcpy(&d, rec + 208, 4);
  const aie::vector<bfloat16, 32> c32v = aie::broadcast<bfloat16, 32>((bfloat16)32.0f);
  aie::vector<bfloat16, 32> gsb =
      aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(sc - 12))),
               aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
  aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
  aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
  aie::vector<uint8_t, 32> Ha  = aie::load_unaligned_v<32>(qh);
  aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
  aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
  aie::vector<uint8_t, 32> Hb  = aie::load_unaligned_v<32>(qh + 32);
  aie::store_v(col +   0, QW(L0a, Ha, 0, 0));
  aie::store_v(col +  32, QW(L1a, Ha, 2, 1));
  aie::store_v(col +  64, QW(L0a, Ha, 4, 2));
  aie::store_v(col +  96, QW(L1a, Ha, 6, 3));
  aie::store_v(col + 128, QW(L0b, Hb, 0, 4));
  aie::store_v(col + 160, QW(L1b, Hb, 2, 5));
  aie::store_v(col + 192, QW(L0b, Hb, 4, 6));
  aie::store_v(col + 224, QW(L1b, Hb, 6, 7));
}

__attribute__((noinline)) void scatter_bl1(const uint8_t *qB, bfloat16 *Bl1) {
  constexpr int s = 8, t = 4;
  constexpr int NT = DIM_N / t;      // 8
  constexpr int NCH = DIM_K / 32;    // 8 chunks of 32 k-values (== KS/4)
  bfloat16 col0[DIM_K], col1[DIM_K], col2[DIM_K], col3[DIM_K];
  for (int nt = 0; nt < NT; nt++) {
    deq_q6k_vec(qB + (nt * t + 0) * 212, col0);
    deq_q6k_vec(qB + (nt * t + 1) * 212, col1);
    deq_q6k_vec(qB + (nt * t + 2) * 212, col2);
    deq_q6k_vec(qB + (nt * t + 3) * 212, col3);
    for (int c = 0; c < NCH; c++) {
      aie::vector<bfloat16, 32> v0 = aie::load_v<32>(col0 + c * 32);  // [ksl=4][si=8]
      aie::vector<bfloat16, 32> v1 = aie::load_v<32>(col1 + c * 32);
      aie::vector<bfloat16, 32> v2 = aie::load_v<32>(col2 + c * 32);
      aie::vector<bfloat16, 32> v3 = aie::load_v<32>(col3 + c * 32);
      auto x = aie::interleave_zip(v0, v1, 1);   // (x.first, x.second)
      auto y = aie::interleave_zip(v2, v3, 1);
      auto z01 = aie::interleave_zip(x.first, y.first, 2);   // tiles ksl=0,1
      auto z23 = aie::interleave_zip(x.second, y.second, 2); // tiles ksl=2,3
      const int ks0 = c * 4;
      aie::store_v(Bl1 + ((ks0 + 0) * NT + nt) * (s * t), z01.first);
      aie::store_v(Bl1 + ((ks0 + 1) * NT + nt) * (s * t), z01.second);
      aie::store_v(Bl1 + ((ks0 + 2) * NT + nt) * (s * t), z23.first);
      aie::store_v(Bl1 + ((ks0 + 3) * NT + nt) * (s * t), z23.second);
    }
  }
}
} // namespace

extern "C" {
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  scatter_bl1(qB, Bl1);
  matmul_vectorized_4x4<bfloat16, float, (DIM_M / 4), (DIM_K / 8), (DIM_N / 4), 4, 8, 4,
                        true, true>(A, Bl1, C);
}
}
