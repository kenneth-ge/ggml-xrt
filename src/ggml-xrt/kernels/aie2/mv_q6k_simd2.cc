//===- mv_q6k_simd2.cc (aie2 / Phoenix) -------------------*- C++ -*-===//
//
// Q6_K decode matvec — SIMD, UNROLLED (no vector array, 2 accumulators). The prior SIMD core
// held the 4 sub-quants in `aie::vector qs[4]` indexed by a loop var; vector-typed arrays on
// AIE spill to L1 (every access a load/store), which showed up as ~520 cyc per 32-lane chunk
// (15-50x off peak) in the compute decomposition. This version fully unrolls both 128-chunks
// into 8 named vector regs, and splits the 8-chunk MAC into TWO independent accumulators
// (even/odd chunks) so the dependent MAC chain pipelines instead of serializing.
// Identical math to mv_q6k.cc (bf16 band); pure throughput change.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

// assemble a 32-lane chunk of q6_K quants: (ql_nibble | (qh_2bit<<4)) -> bf16, minus 32,
// times the per-group scale (sbuf), as bf16 ready for the MAC.
// nibble: LOW for qh-shift {0,2}, HIGH for {4,6}. qh 2-bit field at bit `hi_shift`.
#define QW(QLv, QHv, hi_shift, ci)                                                         \
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

    // chunk 0..3 (first 128), chunk 4..7 (second 128)
    aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
    aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
    aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
    aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
    aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
    aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

    aie::accum<accfloat, 32> acc0 = aie::mul(QW(L0a, Ha, 0, 0), aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, QW(L0a, Ha, 4, 2), aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 0, 4), aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 4, 6), aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(QW(L1a, Ha, 2, 1), aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, QW(L1a, Ha, 6, 3), aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 2, 5), aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 6, 7), aie::load_v<32>(b + 224));

    c[row] += aie::reduce_add(acc0.template to_vector<float>()) +
              aie::reduce_add(acc1.template to_vector<float>());
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
