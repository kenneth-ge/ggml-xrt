//===- mv_q4k_noscratch.cc (aie2 / Phoenix) --------------*- C++ -*-===//
//
// Q4_K decode matvec, vscale with the sbuf/mbuf[256] round-trip removed. The isolation proved the
// vscale->maconly gap is the scale scratch L1 round-trip (q6k noscratch = 2.3x). q4k can't be
// fully unrolled (clang ICEs), so it keeps the ck/s loop; the fix is to NOT expand the scale to
// 256 lanes. vscale stored 8x store_v<32> into sbuf + 8x into mbuf (16 vec stores) and read them
// back with 16 vec loads per row. Here the 8 group scales/mins are stored ONCE (one 32-lane store
// each) into tiny arrays and broadcast per chunk from a scalar load -> 2 vec stores + cheap
// scalar loads instead of a 512-bf16 round-trip. Same 148 B record + math + result as vscale.
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

    // 8 group scales/mins: scalar bit-extract, vector widen+scale (vscale), then ONE 32-lane store
    // each into tiny arrays (not the 256-lane sbuf/mbuf) -> no per-chunk L1 round-trip.
    alignas(64) uint8_t scarr[32], mnarr[32];
    for (int gi = 0; gi < 8; gi++)
      get_scale_min_k4(gi, sca, &scarr[gi], &mnarr[gi]);
    alignas(64) bfloat16 gsc[32], gmn[32];
    aie::store_v(gsc, aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(scarr))),
                               aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>());
    aie::store_v(gmn, aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(mnarr))),
                               aie::broadcast<bfloat16, 32>((bfloat16)dmin)).to_vector<bfloat16>());

    aie::accum<accfloat, 32> acc;
    int ci = 0;
    for (int ck = 0; ck < 4; ck++) {
      aie::vector<uint8_t, 32> QS = aie::load_unaligned_v<32>(qs + ck * 32);
      aie::vector<uint8_t, 32> sub[2];
      sub[0] = aie::bit_and((uint8_t)0x0F, QS);
      sub[1] = aie::logical_downshift(QS, 4);
      for (int s = 0; s < 2; s++, ci++) {
        aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::unpack(sub[s]));
        // per-group scale/min broadcast from the tiny arrays (scalar load + splat), no 32-lane load
        aie::vector<bfloat16, 32> w =
            aie::sub(aie::mul(qv, aie::broadcast<bfloat16, 32>(gsc[ci])).template to_vector<bfloat16>(),
                     aie::broadcast<bfloat16, 32>(gmn[ci]));
        if (ci == 0)
          acc = aie::mul(w, aie::load_v<32>(b + ci * 32));
        else
          acc = aie::mac(acc, w, aie::load_v<32>(b + ci * 32));
      }
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
