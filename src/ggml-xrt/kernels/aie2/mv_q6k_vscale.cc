//===- mv_q6k_vscale.cc (aie2 / Phoenix) -----------------*- C++ -*-===//
//
// Q6_K decode matvec, VECTORIZED SCALE SETUP. Same 212 B record + math as mv_q6k.cc (NO repack
// change), one fix: the per-group scale table was built with SCALAR fp32 — `(bfloat16)(d *
// (float)sc[g])` per g compiles to software-emulated __floatsisf + __mulsf3 (the AIE scalar unit
// has no hw fp32), 16x/row, and the 3-way isolation's "89% dequant" was almost entirely THIS, not
// the 6-bit unpack. Here the 16 int8 group scales are widened + multiplied by d with VECTOR ops
// (one hw vconv + one vmul), then broadcast per group. The 6-bit assemble/MAC path is unchanged.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

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

    // VECTORIZED per-group scale: widen the 16 int8 scales -> bf16 and multiply by d with vector
    // ops (no scalar __floatsisf/__mulsf3). (bfloat16)d is ONE scalar convert/row (was 16).
    // Use a FULL 32-lane load+unpack (the same width as the proven-correct QW weight path): a
    // half-width vunpack on a v16int8 does not map lane i->i, which sign-mangled scales per group.
    // 32-lane vunpack.s16.s8 IS lane-preserving. Load the window [sc-12 .. sc+20) which is fully
    // inside the 212 B record (last 12 B of qh, then the 16 scales, then d) -> no OOB; the 16
    // scales land in lanes 12..27, so sc[g] = gsb.get(12 + g).
    alignas(64) bfloat16 sbuf[256];
    aie::vector<int16, 32> sc16 = aie::unpack(aie::load_unaligned_v<32>((const int8_t *)sc - 12));
    aie::vector<bfloat16, 32> scb = aie::to_float<bfloat16>(sc16);
    aie::vector<bfloat16, 32> gsb =
        aie::mul(scb, aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16, aie::broadcast<bfloat16, 16>(gsb.get(12 + g)));

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
