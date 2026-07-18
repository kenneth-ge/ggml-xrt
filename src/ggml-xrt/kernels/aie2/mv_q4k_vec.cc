//===- mv_q4k_vec.cc (aie2 / Phoenix) ---------------------*- C++ -*-===//
//
// Q4_K decode matvec — VECTORIZED. Same ABI/repack as mv_q4k.cc; drop-in matvec_q4k_f32.
// Q4_K dequant is AFFINE: y = d*sc*q - dmin*mn  (per 32-elem group, sc/mn from
// get_scale_min_k4). So vs q6_K we carry a per-element min term too.
//
// Per superblock per row: scalar integer unpack of the 256 nibbles into qbuf (int16, 0..15)
// and fill sbuf (bf16 = d*sc) + mbuf (bf16 = dmin*mn) per 32-group; integer/bf16 stores only.
// Then 8 x 32-lane vector passes: qv = to_float<bf16>(qbuf); w = qv*sbuf - mbuf (bf16);
// acc += w * b (aie::mac); reduce_add. Needs a >= 0x2000 core stack (qbuf+sbuf+mbuf = 1536 B;
// the default 0x400 overflows -> silent-wrong, which is what broke the first q6k vec attempt).
// bf16 weight rounding -> NRMSE ~1e-3 band vs f32 scalar, not bit-exact.
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
    for (int ck = 0; ck < 4; ck++) {
      uint8_t sc, mn;
      get_scale_min_k4(2 * ck + 0, sca, &sc, &mn);
      const bfloat16 d1 = (bfloat16)(d * (float)sc), m1 = (bfloat16)(dmin * (float)mn);
      get_scale_min_k4(2 * ck + 1, sca, &sc, &mn);
      const bfloat16 d2 = (bfloat16)(d * (float)sc), m2 = (bfloat16)(dmin * (float)mn);
      const uint8_t *q = qs + ck * 32;
      const int base = ck * 64;
      for (int l = 0; l < 32; l++) {
        qbuf[base + l] = (int16_t)(q[l] & 0x0F);
        qbuf[base + 32 + l] = (int16_t)(q[l] >> 4);
        sbuf[base + l] = d1;
        mbuf[base + l] = m1;
        sbuf[base + 32 + l] = d2;
        mbuf[base + 32 + l] = m2;
      }
    }

    aie::accum<accfloat, 32> acc;
    for (int i = 0; i < 256; i += 32) {
      aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::load_v<32>(qbuf + i));
      aie::vector<bfloat16, 32> sv = aie::load_v<32>(sbuf + i);
      aie::vector<bfloat16, 32> mv = aie::load_v<32>(mbuf + i);
      aie::vector<bfloat16, 32> w =
          aie::sub(aie::mul(qv, sv).template to_vector<bfloat16>(), mv);
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
