//===- mv_q4k_4acc.cc (aie2 / Phoenix) --------------*- C++ -*-===//
//
// Q4_K decode matvec, 4-INDEPENDENT-ACCUMULATOR variant. Base = mv_q4k_noscratch.cc (2 accumulators,
// inline-register scale, no sbuf scratch, 0xF0 hi-nibble trick that already avoids logical_downshift).
// This variant runs FOUR independent 2-MAC chains so four 6-cycle VMAC chains overlap, mirroring the
// measured q6k 4acc win. The 0xF0 hi-nibble (no logical_downshift) is retained, so the multi-accum +
// logical_downshift peano ICE is dodged just as in the 2-acc noscratch core.
//   acc0={chunks 0,4}, acc1={1,5}, acc2={2,6}, acc3={3,7} -- each a 2-MAC chain, 4 in flight.
// Combine = 3 VECTOR adds + 1 reduce_add (reduce count independent of accumulator count).
// Same 148 B record + math + result as mv_q4k_noscratch.cc (lo ci=2ck, hi ci=2ck+1, b+ci*32).
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

// LO nibble: bit_and(0x0F) in [0,15], scale gsc.get(ci), minus gmn.get(ci).
#define LO(Q, ci)                                                                          \
  aie::sub(aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::bit_and((uint8_t)0x0F, (Q)))), \
                    aie::broadcast<bfloat16, 32>(gsc.get(ci))).template to_vector<bfloat16>(), \
           aie::broadcast<bfloat16, 32>(gmn.get(ci)))
// HI nibble WITHOUT logical_downshift: bit_and(0xF0) = hi*16, scaled by gschi=gsc/16 (so
// gschi*(hi*16) = gsc*hi), minus gmn.get(ci) (min is not scaled by 1/16). Dodges the ICE.
#define HI(Q, ci)                                                                          \
  aie::sub(aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::bit_and((uint8_t)0xF0, (Q)))), \
                    aie::broadcast<bfloat16, 32>(gschi.get(ci))).template to_vector<bfloat16>(), \
           aie::broadcast<bfloat16, 32>(gmn.get(ci)))

template <int M>
void matvec_q4k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  _Pragma("clang loop unroll_count(2)")
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 148;
    const uint8_t *qs = rec;
    const uint8_t *sca = rec + 128;
    float d, dmin;
    __builtin_memcpy(&d, rec + 140, 4);
    __builtin_memcpy(&dmin, rec + 144, 4);

    alignas(64) uint8_t scarr[32], mnarr[32];
    for (int gi = 0; gi < 8; gi++)
      get_scale_min_k4(gi, sca, &scarr[gi], &mnarr[gi]);
    // inline-register scales (vscale, no scratch). gschi = gsc/16 for the 0xF0 hi nibble (one vec mul).
    aie::vector<bfloat16, 32> gsc =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(scarr))),
                 aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
    aie::vector<bfloat16, 32> gschi =
        aie::mul(gsc, aie::broadcast<bfloat16, 32>((bfloat16)(1.0f / 16.0f))).to_vector<bfloat16>();
    aie::vector<bfloat16, 32> gmn =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(mnarr))),
                 aie::broadcast<bfloat16, 32>((bfloat16)dmin)).to_vector<bfloat16>();

    aie::vector<uint8_t, 32> Q0 = aie::load_unaligned_v<32>(qs + 0);
    aie::vector<uint8_t, 32> Q1 = aie::load_unaligned_v<32>(qs + 32);
    aie::vector<uint8_t, 32> Q2 = aie::load_unaligned_v<32>(qs + 64);
    aie::vector<uint8_t, 32> Q3 = aie::load_unaligned_v<32>(qs + 96);

    // 4 INDEPENDENT accumulators, each a 2-MAC chain (mul then mac). ILP = 4 chains in flight.
    // chunk order (b-offset): 0:LO(Q0,0) 1:HI(Q0,1) 2:LO(Q1,2) 3:HI(Q1,3)
    //                         4:LO(Q2,4) 5:HI(Q2,5) 6:LO(Q3,6) 7:HI(Q3,7)
    aie::accum<accfloat, 32> acc0 = aie::mul(LO(Q0, 0), aie::load_v<32>(b + 0));
    aie::accum<accfloat, 32> acc1 = aie::mul(HI(Q0, 1), aie::load_v<32>(b + 32));
    aie::accum<accfloat, 32> acc2 = aie::mul(LO(Q1, 2), aie::load_v<32>(b + 64));
    aie::accum<accfloat, 32> acc3 = aie::mul(HI(Q1, 3), aie::load_v<32>(b + 96));
    acc0 = aie::mac(acc0, LO(Q2, 4), aie::load_v<32>(b + 128));
    acc1 = aie::mac(acc1, HI(Q2, 5), aie::load_v<32>(b + 160));
    acc2 = aie::mac(acc2, LO(Q3, 6), aie::load_v<32>(b + 192));
    acc3 = aie::mac(acc3, HI(Q3, 7), aie::load_v<32>(b + 224));

    // combine: 3 VECTOR adds then ONE reduce (reduce count independent of accumulator count).
    aie::vector<float, 32> s =
        aie::add(aie::add(acc0.template to_vector<float>(), acc1.template to_vector<float>()),
                 aie::add(acc2.template to_vector<float>(), acc3.template to_vector<float>()));
    c[row] += aie::reduce_add(s);
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
