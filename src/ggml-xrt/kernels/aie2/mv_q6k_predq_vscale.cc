//===- mv_q6k_predq_vscale.cc (aie2 / Phoenix) -----------*- C++ -*-===//
//
// Q6_K decode matvec, PRE-DEQUANT + VSCALE. The next core-throughput lever after vscale: with the
// overlay fix in place matmul is compute-bound with ~6-10x DMA headroom, so trading a fatter
// weight record for less on-core work is now a pure win (earlier this was gated on going DMA-bound).
// vscale killed the scalar-fp32 scale (2.6 ms). The remaining on-core cost vs the 1.05 ms maconly
// floor is the 6-bit ASSEMBLY (ql|qh -> 6-bit -> -32). Move THAT to the host repack: the record
// carries natural-order signed int8 quants (already assembled, already -32); the core does only
// int8->bf16 convert + vectorized per-group scale + MAC. Expected ~1.05-1.5 ms (toward the floor).
//
// NEW RECORD (276 B / 256-elem superblock), for this kernel only:
//   [0  :256) int8  q[256]   natural order, = (assembled 6-bit 0..63) - 32   (range -32..31)
//   [256:272) int8  sc[16]   per-16-group scales (element k uses sc[k/16])
//   [272:276) f32   d        superblock scale.  weight = d * sc[k/16] * q[k].
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#define REC_PD 276

// int8 chunk (natural, pre-assembled, pre-(-32)) -> bf16, scaled by the per-group scale in sbuf.
#define WPD(qptr, ci)                                                                      \
  aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>((qptr) + (ci) * 32))), \
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

    // VECTORIZED per-group scale (vscale): 32-lane signed unpack from sc-12 (window fully inside
    // the record; scales land in lanes 12..27), * d, broadcast each of the 16 group scales.
    alignas(64) bfloat16 sbuf[256];
    aie::vector<bfloat16, 32> gsb =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(sc - 12))),
                 aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
    for (int g = 0; g < 16; g++)
      aie::store_v(sbuf + g * 16, aie::broadcast<bfloat16, 16>(gsb.get(12 + g)));

    // 8 sequential chunks of 32 (natural order), 2 independent accumulators. No 6-bit assembly.
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
