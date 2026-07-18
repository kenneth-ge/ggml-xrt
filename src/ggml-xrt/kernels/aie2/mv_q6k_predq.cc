//===- mv_q6k_predq.cc (aie2 / Phoenix) ------------------*- C++ -*-===//
//
// Q6_K decode matvec, PRE-DEQUANTIZED record. The 3-way isolation showed the on-core 6-bit
// assembly (ql|qh -> 6-bit -> -32) is 89% of decode time (8.4 of 9.44 ms; MAC+DMA+reduce floor
// is 1.05 ms). This kernel moves that assembly OFF the core into the host repack (done once per
// weight, cached): the record already holds the quants as NATURAL-ORDER signed int8 (value-32),
// so the core does only int8->bf16 convert + per-group scale + MAC. Because the host de-interleaves
// during repack, b is read sequentially (no more +64/+128 interleave).
//
// NEW RECORD (276 B / 256-elem superblock), replaces the 212 B q6_K record for this kernel:
//   [0  :256) int8  q[256]   natural order, = (assembled 6-bit 0..63) - 32, range -32..31
//   [256:272) int8  sc[16]   per-16-group scales (element k uses sc[k/16])
//   [272:276) f32   d        superblock scale
// weight = d * sc[k/16] * q[k].  Memory: 276 B vs bf16 512 B (1.85x win); vs old 212 (compute-bound,
// so trading ~30% more weight DMA for -89% core dequant is the right trade).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#define REC_PD 276

// int8 chunk -> bf16, scaled by the per-group scale already in sbuf (natural order).
#define WPD(qptr, ci)                                                                      \
  aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>((qptr) + (ci) * 32))),      \
           aie::load_v<32>(sbuf + (ci) * 32)).template to_vector<bfloat16>()

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * REC_PD;
    const int8_t *q = (const int8_t *)rec;
    const int8_t *sc = (const int8_t *)(rec + 256);
    float d;
    __builtin_memcpy(&d, rec + 272, 4);

    // per-group scale d*sc[g] broadcast to the group's 16 lanes (natural order).
    alignas(64) bfloat16 sbuf[256];
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16, aie::broadcast<bfloat16, 16>((bfloat16)(d * (float)sc[g])));

    // 8 sequential chunks of 32, split across 2 independent accumulators to pipeline the MAC.
    aie::accum<accfloat, 32> acc0 = aie::mul(WPD(q, 0), aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, WPD(q, 2), aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, WPD(q, 4), aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, WPD(q, 6), aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(WPD(q, 1), aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, WPD(q, 3), aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, WPD(q, 5), aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, WPD(q, 7), aie::load_v<32>(b + 224));

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
