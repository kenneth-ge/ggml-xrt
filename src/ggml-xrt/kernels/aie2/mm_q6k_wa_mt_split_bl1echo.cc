//===- mm_q6k_wa_mt_split_bl1echo.cc (aie2 / Phoenix) ---------*- C++ -*-===//
//
// SPLIT-STRUCTURE Bl1-ECHO diagnostic. Uses the EXACT noinline scatter_bl1 from
// mm_q6k_wa_mt_split.cc (concat + aie::transpose(t,s) + store), calls it, then ECHOES Bl1 -> C via
// the proven cpass C-path, SKIPPING the mmul. Isolates the noinline vectorized scatter itself:
//   Bl1 == 0 here          -> scatter_bl1 LOGIC produces/targets zeros (not reorder, not a
//                             buffer-instance mismatch with the mmul; the transpose is the suspect).
//   Bl1 nonzero here        -> scatter is fine but the real _split mmul reads a different Bl1
//                             instance (unlikely - same .py design buffer) or the mmul itself.
//
// ECHO LAYOUT (same as mm_q6k_wa_bl1echo.cc): host C[tok,ch] == (float) Bl1[tok*DIM_N + ch]
//   (first 1024 Bl1 elements, mmul-B order). Decode: ks=tok/8, nt=tok%8, si=ch/4, ti=ch%4 ->
//   dequant of channel (nt*4+ti), k-index (ks*8+si) of this core's n-tile (last k-tile).
//
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"

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
  constexpr int NT = DIM_N / t;
  constexpr int KS = DIM_K / s;
  bfloat16 col0[DIM_K], col1[DIM_K], col2[DIM_K], col3[DIM_K];
  for (int nt = 0; nt < NT; nt++) {
    deq_q6k_vec(qB + (nt * t + 0) * 212, col0);
    deq_q6k_vec(qB + (nt * t + 1) * 212, col1);
    deq_q6k_vec(qB + (nt * t + 2) * 212, col2);
    deq_q6k_vec(qB + (nt * t + 3) * 212, col3);
    for (int ks = 0; ks < KS; ks++) {
      aie::vector<bfloat16, 8> a0 = aie::load_v<8>(col0 + ks * s);
      aie::vector<bfloat16, 8> a1 = aie::load_v<8>(col1 + ks * s);
      aie::vector<bfloat16, 8> a2 = aie::load_v<8>(col2 + ks * s);
      aie::vector<bfloat16, 8> a3 = aie::load_v<8>(col3 + ks * s);
      aie::vector<bfloat16, 32> cat = aie::concat(a0, a1, a2, a3);
      aie::vector<bfloat16, 32> tile = aie::transpose(cat, t, s);
      aie::store_v(Bl1 + (ks * NT + nt) * (s * t), tile);
    }
  }
}
} // namespace

extern "C" {
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  (void)A;
  scatter_bl1(qB, Bl1);
  constexpr int r = 4, t = 4;
  for (int tok = 0; tok < DIM_M; tok++) {
    for (int ch = 0; ch < DIM_N; ch++) {
      int cb = (tok / r) * (r * DIM_N) + (tok % r) * t + (ch / t) * (r * t) + (ch % t);
      C[cb] = (float)Bl1[tok * DIM_N + ch];
    }
  }
}
}
