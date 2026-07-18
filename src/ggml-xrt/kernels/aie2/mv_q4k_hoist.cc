//===- mv_q4k_hoist.cc (aie2 / Phoenix) -------------------*- C++ -*-===//
//
// Q4_K decode matvec — vectorized + per-group scale/min HOIST. Bit-identical to mv_q4k.cc,
// but sbuf (d*sc) and mbuf (dmin*mn) are filled with 8 vector broadcast-stores each (one per
// 32-element group) instead of 512 scalar stores. Q4_K is affine (y = d*sc*q - dmin*mn); both
// scale and min are constant per 32-group, so the 256-iteration loop now only unpacks quants.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

template <int M>
void matvec_q4k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 148;
    const uint8_t *qs = rec;
    const uint8_t *sca = rec + 128;
    float d, dmin;
    __builtin_memcpy(&d, rec + 140, 4);
    __builtin_memcpy(&dmin, rec + 144, 4);

    alignas(64) int16_t qbuf[256];
    alignas(64) bfloat16 sbuf[256];
    alignas(64) bfloat16 mbuf[256];
    // hoist: 8 groups of 32; group gi uses get_scale_min_k4(gi). One broadcast each.
    for (int gi = 0; gi < 8; gi++) {
      uint8_t sc, mn;
      get_scale_min_k4(gi, sca, &sc, &mn);
      aie::store_v(sbuf + gi * 32, aie::broadcast<bfloat16, 32>((bfloat16)(d * (float)sc)));
      aie::store_v(mbuf + gi * 32, aie::broadcast<bfloat16, 32>((bfloat16)(dmin * (float)mn)));
    }

    for (int ck = 0; ck < 4; ck++) {  // quant unpack only (integer)
      const uint8_t *q = qs + ck * 32;
      const int base = ck * 64;
      for (int l = 0; l < 32; l++) {
        qbuf[base + l] = (int16_t)(q[l] & 0x0F);
        qbuf[base + 32 + l] = (int16_t)(q[l] >> 4);
      }
    }

    aie::accum<accfloat, 32> acc;
    for (int i = 0; i < 256; i += 32) {
      aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::load_v<32>(qbuf + i));
      aie::vector<bfloat16, 32> w =
          aie::sub(aie::mul(qv, aie::load_v<32>(sbuf + i)).template to_vector<bfloat16>(),
                   aie::load_v<32>(mbuf + i));
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
void matvec_q4k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q4k_vec<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
