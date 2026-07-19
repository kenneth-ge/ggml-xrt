//===- mv_mm_q4k.cc (aie2 / Phoenix) ---------------------*- C++ -*-===//
//
// Q4_K M=N "mat-MAT" for the speculative-decode VERIFY kernel. C[MT, m] += dequant(W[m,K]) . A[MT,K]
// one 256-k-block at a time. Same amortization physics as mv_mm_q6k.cc: dequant W[n]'s 8 chunks
// ONCE per output row (the q4k noscratch path: bit_and(0xF0) hi nibble + gschi=gsc/16, NO
// logical_downshift), then fill the idle MACs with all MT draft tokens. Reuses mv_q4k_noscratch.cc's
// dequant. mt-loop SEQUENTIAL with one reused accumulator. 148 B q4k records.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_MT
#define DIM_MT 8
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
// HI nibble via bit_and(0xF0)=hi*16 scaled by gschi=gsc/16 (avoids logical_downshift), minus gmn.
#define HI(Q, ci)                                                                          \
  aie::sub(aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::bit_and((uint8_t)0xF0, (Q)))), \
                    aie::broadcast<bfloat16, 32>(gschi.get(ci))).template to_vector<bfloat16>(), \
           aie::broadcast<bfloat16, 32>(gmn.get(ci)))

template <int M, int MT>
void matvec_mm_q4k(const uint8_t *restrict a, const bfloat16 *restrict b,
                   float *restrict c) {
  event0();
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

    // Dequant the 8 chunks ONCE (hoisted, amortized over MT). Chunk ci -> activation slice b+ci*32.
    aie::vector<bfloat16, 32> w0 = LO(Q0, 0);
    aie::vector<bfloat16, 32> w1 = HI(Q0, 1);
    aie::vector<bfloat16, 32> w2 = LO(Q1, 2);
    aie::vector<bfloat16, 32> w3 = HI(Q1, 3);
    aie::vector<bfloat16, 32> w4 = LO(Q2, 4);
    aie::vector<bfloat16, 32> w5 = HI(Q2, 5);
    aie::vector<bfloat16, 32> w6 = LO(Q3, 6);
    aie::vector<bfloat16, 32> w7 = HI(Q3, 7);

    for (int mt = 0; mt < MT; mt++) {
      const bfloat16 *bm = b + mt * 256;
      aie::accum<accfloat, 32> acc = aie::mul(w0, aie::load_v<32>(bm + 0));
      acc = aie::mac(acc, w1, aie::load_v<32>(bm + 32));
      acc = aie::mac(acc, w2, aie::load_v<32>(bm + 64));
      acc = aie::mac(acc, w3, aie::load_v<32>(bm + 96));
      acc = aie::mac(acc, w4, aie::load_v<32>(bm + 128));
      acc = aie::mac(acc, w5, aie::load_v<32>(bm + 160));
      acc = aie::mac(acc, w6, aie::load_v<32>(bm + 192));
      acc = aie::mac(acc, w7, aie::load_v<32>(bm + 224));
      c[mt * M + row] += aie::reduce_add(acc.template to_vector<float>());
    }
  }
  event1();
}

extern "C" {
void matvec_mm_q4k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_mm_q4k<DIM_M, DIM_MT>(a, b, c);
}
void zero_mm_f32(float *c) {
  for (int i = 0; i < DIM_MT * DIM_M; i++)
    c[i] = 0.0f;
}
}
