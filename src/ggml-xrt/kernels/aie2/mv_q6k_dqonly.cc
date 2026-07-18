//===- mv_q6k_dqonly.cc (aie2 / Phoenix) — ISOLATION EXPERIMENT ----*- C++ -*-===//
//
// DEQUANT-ONLY q6_K decode gemv: does the FULL 6-bit dequant (identical QW macro / sbuf setup
// to mv_q6k.cc) but replaces the b-weighted MAC with a cheap accumulate (aie::add) so the
// dequantized vectors are consumed (not dead-code-eliminated) without the multiply-by-b cost.
// Isolates dequant (bit-ops + unpack + to_float + sub + per-group scale) from the MAC.
//
//   time(this, dequant-only)  ~=  dequant cost (+ trivial adds, - the b DMA)
// Cross-check with mv_q6k_maconly.cc: full ~= maconly + dqonly - overlap.
// PRODUCES GARBAGE OUTPUT BY DESIGN — for timing only, never ship.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

// identical to mv_q6k.cc: (ql_nibble | (qh_2bit<<4)) -> bf16, minus 32, times per-group scale.
#define QW(QLv, QHv, hi_shift, ci)                                                          \
  aie::mul(aie::sub(aie::to_float<bfloat16>(aie::unpack(                                    \
                        aie::bit_or((hi_shift) < 4 ? aie::bit_and((uint8_t)0x0F, (QLv))     \
                                                   : aie::logical_downshift((QLv), 4),      \
                                    aie::upshift(aie::bit_and((uint8_t)0x03,                 \
                                        aie::logical_downshift((QHv), (hi_shift))), 4)))),   \
                    c32v),                                                                  \
           aie::load_v<32>(sbuf + (ci) * 32)).template to_vector<bfloat16>()

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  const aie::vector<bfloat16, 32> c32v = aie::broadcast<bfloat16, 32>((bfloat16)32.0f);
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    alignas(64) bfloat16 sbuf[256];
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16, aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)sc[g])));

    aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
    aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
    aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
    aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
    aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
    aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

    // full dequant on all 8 chunks; consume via a cheap add (no multiply-by-b, no b DMA).
    aie::vector<bfloat16, 32> s = QW(L0a, Ha, 0, 0);
    s = aie::add(s, QW(L1a, Ha, 2, 1));
    s = aie::add(s, QW(L0a, Ha, 4, 2));
    s = aie::add(s, QW(L1a, Ha, 6, 3));
    s = aie::add(s, QW(L0b, Hb, 0, 4));
    s = aie::add(s, QW(L1b, Hb, 2, 5));
    s = aie::add(s, QW(L0b, Hb, 4, 6));
    s = aie::add(s, QW(L1b, Hb, 6, 7));

    c[row] += aie::reduce_add(s);
  }
  event1();
}

extern "C" {
void matvec_q6k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q6k_vec<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
