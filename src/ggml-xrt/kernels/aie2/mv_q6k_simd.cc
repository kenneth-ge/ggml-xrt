//===- mv_q6k_simd.cc (aie2 / Phoenix) --------------------*- C++ -*-===//
//
// Q6_K decode matvec — FULL SIMD (variant A): the quant bit-unpack is vectorized too, not just
// the MAC/scale. The four sub-quants q1..q4 of each 128-elem chunk land at contiguous 32-lane
// blocks (l, l+32, l+64, l+96), so each is computed as a 32-lane vector and fed straight to the
// MAC - no per-element scalar loop at all. Removes the last scalar-on-vector-engine penalty.
//
//   q = (ql_nibble) | ((qh_2bit) << 4)  in [0,63], then -32; weight = q * d*sc; acc += w*b.
// Scale still hoisted per 16-group into sbuf via broadcast. bit ops via aie::bit_and/bit_or/
// upshift/logical_downshift on uint8 vectors; -32 done in bf16.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  const bfloat16 c32 = (bfloat16)32.0f;
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    alignas(64) bfloat16 sbuf[256];
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16,
                   aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)sc[g])));

    aie::accum<accfloat, 32> acc;
    int ci = 0;
    for (int ch = 0; ch < 2; ch++) {
      // UNALIGNED: the record stride is 212 B, so ql/qh for rows>=1 are not 64B-aligned.
      // An aligned load_v of a misaligned address reads shifted bytes (row 0 works, rows>=1
      // scramble) - that was the bug, not the dtype conversion.
      aie::vector<uint8_t, 32> QL0 = aie::load_unaligned_v<32>(ql + ch * 64);
      aie::vector<uint8_t, 32> QL1 = aie::load_unaligned_v<32>(ql + ch * 64 + 32);
      aie::vector<uint8_t, 32> QH = aie::load_unaligned_v<32>(qh + ch * 32);

      aie::vector<uint8_t, 32> qs[4];
      qs[0] = aie::bit_or(aie::bit_and((uint8_t)0x0F, QL0),
                          aie::upshift(aie::bit_and((uint8_t)0x03, QH), 4));
      qs[1] = aie::bit_or(aie::bit_and((uint8_t)0x0F, QL1),
                          aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(QH, 2)), 4));
      qs[2] = aie::bit_or(aie::logical_downshift(QL0, 4),
                          aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(QH, 4)), 4));
      qs[3] = aie::bit_or(aie::logical_downshift(QL1, 4),
                          aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(QH, 6)), 4));

      for (int s = 0; s < 4; s++, ci++) {
        // unpack uint8->int16 (lane-preserving widen) BEFORE the float convert; a direct
        // uint8->bf16 conversion permutes lanes (the width change reorders), which was the bug.
        aie::vector<bfloat16, 32> qv =
            aie::sub(aie::to_float<bfloat16>(aie::unpack(qs[s])),
                     aie::broadcast<bfloat16, 32>(c32));
        aie::vector<bfloat16, 32> w =
            aie::mul(qv, aie::load_v<32>(sbuf + ci * 32)).template to_vector<bfloat16>();
        if (ci == 0)
          acc = aie::mul(w, aie::load_v<32>(b + ci * 32));
        else
          acc = aie::mac(acc, w, aie::load_v<32>(b + ci * 32));
      }
    }
    c[row] += aie::reduce_add(acc.template to_vector<float>());
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
