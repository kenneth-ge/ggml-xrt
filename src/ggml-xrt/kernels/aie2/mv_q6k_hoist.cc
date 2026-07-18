//===- mv_q6k_hoist.cc (aie2 / Phoenix) -------------------*- C++ -*-===//
//
// Q6_K decode matvec — vectorized + per-group scale HOIST. Same result as mv_q6k.cc (bit-
// identical sbuf), but the scale table is filled with 16 vector broadcast-stores instead of
// 256 scalar stores: sbuf[p] = d*sc[p/16] is constant across each 16-element group, so the
// 256-iteration scalar loop now only unpacks the quants (integer), and all scale/index
// handling leaves the inner loop. Everything else (MAC reduction) is unchanged.
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
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    alignas(64) int16_t qbuf[256];
    alignas(64) bfloat16 sbuf[256];
    // hoist: sbuf[p] = d*sc[p/16], one broadcast per 16-group (16 vector stores).
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16,
                   aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)sc[g])));

    for (int ch = 0; ch < 2; ch++) {
      const uint8_t *qlc = ql + ch * 64;
      const uint8_t *qhc = qh + ch * 32;
      const int base = ch * 128;
      for (int l = 0; l < 32; l++) {  // quant unpack only (integer)
        qbuf[base + l]      = (int16_t)((int)((qlc[l] & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32);
        qbuf[base + l + 32] = (int16_t)((int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32);
        qbuf[base + l + 64] = (int16_t)((int)((qlc[l] >> 4) | (((qhc[l] >> 4) & 3) << 4)) - 32);
        qbuf[base + l + 96] = (int16_t)((int)((qlc[l + 32] >> 4) | (((qhc[l] >> 6) & 3) << 4)) - 32);
      }
    }

    aie::accum<accfloat, 32> acc;
    for (int i = 0; i < 256; i += 32) {
      aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::load_v<32>(qbuf + i));
      aie::vector<bfloat16, 32> w =
          aie::mul(qv, aie::load_v<32>(sbuf + i)).template to_vector<bfloat16>();
      if (i == 0)
        acc = aie::mul(w, aie::load_v<32>(b + i));
      else
        acc = aie::mac(acc, w, aie::load_v<32>(b + i));
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
