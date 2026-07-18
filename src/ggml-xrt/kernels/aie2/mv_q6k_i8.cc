//===- mv_q6k_i8.cc (aie2 / Phoenix) ----------------------*- C++ -*-===//
//
// Q6_K decode matvec — INTEGER MAC (the "fewer converts" lever). The bf16 SIMD path spends
// ~all its time in per-element unpack + to_float; here we keep the quants as int8 and do an
// int8xint8 MAC (the AIE's fast path), removing to_float entirely. The activation is
// dynamically quantized to int8 once per superblock (shared across all output rows).
//
//   y[k] = d*sc_g*(q[k]-32);  result = sum_k y[k]*b[k]
//        = bscale * sum_g d*sc_g * ( sum_{k in g} q[k]*bi8[k]  -  32 * sum_{k in g} bi8[k] )
// with b[k] ~= bscale*bi8[k], groups of 16 (q6_K scale granularity). int8 MAC per 16-group +
// reduce; scale/offset applied per group in float. PRECISION: int8 activation -> NRMSE will
// rise above the bf16 band (~0.005); verify on-device that it stays acceptable for generation.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q6k_i8(const uint8_t *restrict a, const bfloat16 *restrict b,
                   float *restrict c) {
  event0();

  // --- dynamic-quantize the activation b[256] -> int8, once per call (shared across rows) ---
  bfloat16 bmax = aie::reduce_max(aie::abs(aie::load_v<32>(b)));
  for (int i = 32; i < 256; i += 32) {
    bfloat16 m2 = aie::reduce_max(aie::abs(aie::load_v<32>(b + i)));
    if (m2 > bmax) bmax = m2;
  }
  const float bmaxf = (float)bmax;
  const float inv = bmaxf > 0.f ? 127.0f / bmaxf : 0.f;
  const float bscale = bmaxf / 127.0f;
  const aie::vector<bfloat16, 32> invv = aie::broadcast<bfloat16, 32>((bfloat16)inv);

  alignas(64) int8_t bi8[256];
  for (int i = 0; i < 256; i += 32) {
    aie::vector<bfloat16, 32> bs = aie::mul(aie::load_v<32>(b + i), invv).to_vector<bfloat16>();
    aie::store_v(bi8 + i, aie::to_fixed<int8_t>(bs, 0));
  }
  // per 16-group sum of bi8 (for the -32 offset), once per call.
  int sumb[16];
  for (int g = 0; g < 16; g++)
    sumb[g] = aie::reduce_add(aie::unpack(aie::load_v<16>(bi8 + g * 16)));

  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    // assemble q6_K quants (0..63) as int8 into qi8[256] (natural order), no to_float.
    alignas(64) int8_t qi8[256];
    for (int ch = 0; ch < 2; ch++) {
      aie::vector<uint8_t, 32> L0 = aie::load_unaligned_v<32>(ql + ch * 64);
      aie::vector<uint8_t, 32> L1 = aie::load_unaligned_v<32>(ql + ch * 64 + 32);
      aie::vector<uint8_t, 32> H = aie::load_unaligned_v<32>(qh + ch * 32);
      const int base = ch * 128;
      aie::store_v((uint8_t *)qi8 + base + 0,
                   aie::bit_or(aie::bit_and((uint8_t)0x0F, L0),
                               aie::upshift(aie::bit_and((uint8_t)0x03, H), 4)));
      aie::store_v((uint8_t *)qi8 + base + 32,
                   aie::bit_or(aie::bit_and((uint8_t)0x0F, L1),
                               aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(H, 2)), 4)));
      aie::store_v((uint8_t *)qi8 + base + 64,
                   aie::bit_or(aie::logical_downshift(L0, 4),
                               aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(H, 4)), 4)));
      aie::store_v((uint8_t *)qi8 + base + 96,
                   aie::bit_or(aie::logical_downshift(L1, 4),
                               aie::upshift(aie::bit_and((uint8_t)0x03, aie::logical_downshift(H, 6)), 4)));
    }

    float acc = 0.0f;
    for (int g = 0; g < 16; g++) {
      aie::vector<int8_t, 16> qg = aie::load_v<16>(qi8 + g * 16);
      aie::vector<int8_t, 16> bg = aie::load_v<16>(bi8 + g * 16);
      int idot = aie::reduce_add(aie::mul(qg, bg).to_vector<int32_t>());
      acc += (d * (float)sc[g]) * bscale * (float)(idot - 32 * sumb[g]);
    }
    c[row] += acc;
  }
  event1();
}

extern "C" {
void matvec_q6k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q6k_i8<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
