//===- mm_q6k_wa_mt_scalar.cc (aie2 / Phoenix) ----------------*- C++ -*-===//
//
// CORRECTNESS-FALLBACK core for the 16-core whole_array Q6_K verify (memtile-A, DIM_M=32). Uses
// the PROVEN SCALAR col->Bl1 scatter from mm_q6k_vecdq.cc (Bl1[((ks*(DIM_N/4)+nt)*8+si)*4+ti] =
// col[kk]) instead of the vectorized concat+aie::transpose+store scatter (which produced a ZERO
// Bl1 on HW). Slow (the 8192-iter scalar remap the whole_array design set out to remove), but
// correctness-first: if this verifies, the whole M>1 pipeline (A delivery, dequant, C gather,
// mmul) is proven end-to-end and the vectorized scatter becomes a separable optimization on a
// known-good base. Everything else identical to mm_q6k_wa_mt.cc.
//
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // matmul_vectorized_4x4 (static inline) + zero_f32 + <aie_api/aie.hpp>

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
} // namespace

extern "C" {
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  constexpr int s = 8, t = 4;
  bfloat16 col[DIM_K];
  // PROVEN SCALAR scatter (mm_q6k_vecdq.cc): dequant each channel, remap col[]->Bl1 scalar.
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
