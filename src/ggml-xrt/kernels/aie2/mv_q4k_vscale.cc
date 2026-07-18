//===- mv_q4k_vscale.cc (aie2 / Phoenix) -----------------*- C++ -*-===//
//
// Q4_K decode matvec, VECTORIZED SCALE SETUP. Same 148 B record + math as mv_q4k.cc (NO repack
// change). Fix: the two per-group scale tables (sbuf=d*sc, mbuf=dmin*mn) were built with SCALAR
// fp32 — `(bfloat16)(d*(float)sc)` / `(bfloat16)(dmin*(float)mn)` per group compile to software
// __floatsisf + __mulsf3 (no hw fp32 on the AIE scalar unit), 8 groups x2 = 32 calls/row, which
// the q6k disassembly proved is the real decode cost (not the nibble unpack). Here the 6-bit
// scales/mins (get_scale_min_k4, cheap scalar bit ops, unsigned 0..63) are widened + multiplied
// by d/dmin with VECTOR ops (32-lane unpack, lane-preserving; zero-extend is correct for the
// unsigned 6-bit scales). The nibble-unpack + affine MAC path is unchanged.
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

    // 6-bit scales/mins via scalar bit ops (cheap, no fp); into aligned 32-byte lanes 0..7.
    alignas(64) uint8_t scarr[32], mnarr[32];
    for (int gi = 0; gi < 8; gi++)
      get_scale_min_k4(gi, sca, &scarr[gi], &mnarr[gi]);

    // VECTORIZED: widen (uint8->bf16, 32-lane lane-preserving unpack) and multiply by d/dmin with
    // vector ops (no scalar __floatsisf/__mulsf3). (bfloat16)d/dmin = 2 scalar converts/row (was 32).
    alignas(64) bfloat16 sbuf[256];
    alignas(64) bfloat16 mbuf[256];
    aie::vector<bfloat16, 32> gsc =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(scarr))),
                 aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
    aie::vector<bfloat16, 32> gmn =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(mnarr))),
                 aie::broadcast<bfloat16, 32>((bfloat16)dmin)).to_vector<bfloat16>();
    for (int gi = 0; gi < 8; gi++) {
      aie::store_v(sbuf + gi * 32, aie::broadcast<bfloat16, 32>(gsc.get(gi)));
      aie::store_v(mbuf + gi * 32, aie::broadcast<bfloat16, 32>(gmn.get(gi)));
    }

    aie::accum<accfloat, 32> acc;
    int ci = 0;
    for (int ck = 0; ck < 4; ck++) {
      aie::vector<uint8_t, 32> QS = aie::load_unaligned_v<32>(qs + ck * 32);
      aie::vector<uint8_t, 32> sub[2];
      sub[0] = aie::bit_and((uint8_t)0x0F, QS);        // lo nibbles -> base+0..31
      sub[1] = aie::logical_downshift(QS, 4);          // hi nibbles -> base+32..63
      for (int s = 0; s < 2; s++, ci++) {
        aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::unpack(sub[s]));
        aie::vector<bfloat16, 32> w =
            aie::sub(aie::mul(qv, aie::load_v<32>(sbuf + ci * 32)).template to_vector<bfloat16>(),
                     aie::load_v<32>(mbuf + ci * 32));
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
