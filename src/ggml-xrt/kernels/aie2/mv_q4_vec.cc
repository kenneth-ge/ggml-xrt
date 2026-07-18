//===- mv_q4_vec.cc (aie2 / Phoenix) ----------------------*- C++ -*-===//
//
// Q4_0 decode matvec — VECTORIZED. Same ABI/repack as mv_q4.cc; drop-in matvec_q4_0_f32.
// Simplest quant layout (one f32 scale per 32-elem block, lo/hi nibbles at j / j+16, natural
// output order) — used to validate the vector dequant+MAC skeleton before the q6_K interleave.
//
// Per 32-elem block per row: scalar integer unpack of 16 nibble bytes into qbuf (int16, -8),
// then one 32-lane vector pass: qv = to_float<bf16>(qbuf); w = qv * d; acc = w * b;
// c[row] += reduce_add(acc). qbuf is alignas(64) so the 512-bit vector load is aligned (a
// mis-aligned load was what scrambled the first q6k vec attempt).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q4_0_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                     float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 20;
    float d;
    __builtin_memcpy(&d, rec + 16, 4);

    alignas(64) int16_t qbuf[32];
    for (int j = 0; j < 16; j++) {
      qbuf[j] = (int16_t)((int)(rec[j] & 0x0F) - 8);
      qbuf[j + 16] = (int16_t)((int)(rec[j] >> 4) - 8);
    }

    aie::vector<bfloat16, 32> qv = aie::to_float<bfloat16>(aie::load_v<32>(qbuf));
    aie::vector<bfloat16, 32> wv = aie::mul(qv, (bfloat16)d).template to_vector<bfloat16>();
    aie::accum<accfloat, 32> acc = aie::mul(wv, aie::load_v<32>(b));
    c[row] += aie::reduce_add(acc.template to_vector<float>());
  }
  event1();
}

extern "C" {
void matvec_q4_0_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q4_0_vec<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
