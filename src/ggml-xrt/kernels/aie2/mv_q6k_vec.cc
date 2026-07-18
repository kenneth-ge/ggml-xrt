//===- mv_q6k_vec.cc (aie2 / Phoenix) ---------------------*- C++ -*-===//
//
// Q6_K decode matvec — VECTORIZED. Same ABI/repack as mv_q6k.cc; drop-in matvec_q6k_f32.
// Kills the scalar-float pathology: the ONLY scalar work is integer bit-unpacking of the
// 6-bit quants and hoisting d*scale (16 float muls/row). All the hot float work — the
// dequant multiply and the K-reduction — runs on the vector unit:
//   * scalar loop unpacks 256 quants into qbuf (int16, already minus 32) and fills sbuf
//     (bf16 = d*scale, from a hoisted dsc[16]); integer + bf16 stores only, no float mul.
//   * vector loop (8 x 32-lane): qv = to_float<bf16>(qbuf); w = qv*sbuf (bf16);
//     acc += w * b   (aie::mac);  then reduce_add over the 32-lane accumulator.
// bf16 weight rounding matches the fused-mm numeric band (NRMSE ~1e-3), not bit-exact vs
// the scalar f32 path — verify it stays in-band on HW.
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

    bfloat16 dsc[16];
    for (int j = 0; j < 16; j++)
      dsc[j] = (bfloat16)(d * (float)sc[j]);

    alignas(32) int16_t qbuf[256];
    alignas(32) bfloat16 sbuf[256];
    for (int ch = 0; ch < 2; ch++) {
      const uint8_t *qlc = ql + ch * 64;
      const uint8_t *qhc = qh + ch * 32;
      const int scb = ch * 8;
      const int base = ch * 128;
      for (int l = 0; l < 32; l++) {
        const int is = l >> 4;
        qbuf[base + l]      = (int16_t)((int)((qlc[l] & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32);
        qbuf[base + l + 32] = (int16_t)((int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32);
        qbuf[base + l + 64] = (int16_t)((int)((qlc[l] >> 4) | (((qhc[l] >> 4) & 3) << 4)) - 32);
        qbuf[base + l + 96] = (int16_t)((int)((qlc[l + 32] >> 4) | (((qhc[l] >> 6) & 3) << 4)) - 32);
        sbuf[base + l]      = dsc[scb + is + 0];
        sbuf[base + l + 32] = dsc[scb + is + 2];
        sbuf[base + l + 64] = dsc[scb + is + 4];
        sbuf[base + l + 96] = dsc[scb + is + 6];
      }
    }

    aie::accum<accfloat, 32> acc;
    acc = aie::mul(aie::mul(aie::to_float<bfloat16>(aie::load_v<32>(qbuf)),
                            aie::load_v<32>(sbuf)).template to_vector<bfloat16>(),
                   aie::load_v<32>(b));
    for (int i = 32; i < 256; i += 32) {
      aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::load_v<32>(qbuf + i));
      aie::vector<bfloat16, 32> w =
          aie::mul(qv, aie::load_v<32>(sbuf + i)).template to_vector<bfloat16>();
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
