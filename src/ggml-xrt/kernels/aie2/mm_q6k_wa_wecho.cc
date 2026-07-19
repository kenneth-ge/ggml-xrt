//===- mm_q6k_wa_wecho.cc (aie2 / Phoenix) --------------------*- C++ -*-===//
//
// W-ECHO PROBE for the 16-core whole_array Q6_K verify (memtile-A path, DIM_M=32). Same ABI as
// mm_q6k_wa_mt.cc: matmul_q6k_f32(qB, A, Bl1, C). SKIPS mmul; DEQUANTS the FIRST weight record of
// the core's current qB tile (deq_q6k_vec, the proven vectorized dequant) and writes those
// dequant'd bf16 values into C (as f32) through the PROVEN cpass C write-back path. Isolates
// W-delivery + dequant:
//   host C reflects sane dequant magnitudes -> W arrives + dequant works (bug is A/mmul);
//   host C == 0 / garbage -> W DMA or dequant broken.
//
// ECHO LAYOUT: host C[tok, ch] == col[ch] (the ch-th dequant'd bf16 of the FIRST weight superblock
// of this core's n-tile), identical for every token tok. corebuf write uses the cpass-verified
// de-tile inverse cb = (tok/r)*(r*DIM_N)+(tok%r)*t+(ch/t)*(r*t)+(ch%t)  (r=t=4). Called per k-tile
// and overwritten, so C reflects the LAST k-tile (t=23) superblock of the core's first channel.
//   -> "correct" = C[tok, 0..31] is a plausible q6_K dequant row (nonzero, O(weight*scale*d)),
//      the same across all 32 token rows.
//
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // <aie_api/aie.hpp>

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
  (void)A; (void)Bl1;
  constexpr int r = 4, t = 4;
  bfloat16 col[DIM_K];
  deq_q6k_vec(qB, col);  // dequant the first weight record of this core's n-tile
  for (int tok = 0; tok < DIM_M; tok++) {
    for (int ch = 0; ch < DIM_N; ch++) {
      int cb = (tok / r) * (r * DIM_N) + (tok % r) * t + (ch / t) * (r * t) + (ch % t);
      C[cb] = (float)col[ch];
    }
  }
}
}
