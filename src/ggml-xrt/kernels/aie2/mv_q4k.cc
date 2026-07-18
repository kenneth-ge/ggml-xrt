//===- mv_q4k.cc (aie2 / Phoenix) -------------------------*- C++ -*-===//
//
// Q4_K on-chip-dequant matrix-vector (decode gemv) core function.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// Computes C[M] += dequant(A_q4K[M,K]) . B[K] for one k-tile == one 256-elem q4_K
// superblock per row. Dequant math is exactly ggml `dequantize_row_q4_K` +
// `get_scale_min_k4` (ggml-quants.c): per superblock d = f32(x.d), min = f32(x.dmin);
// for each of 4 64-elem chunks, two 6-bit (scale,min) pairs give (d1,m1)/(d2,m2) and
//   y_lo[l] = d1*(qs[l]&0xF) - m1 ;  y_hi[l] = d2*(qs[l]>>4) - m2   (l in 0..31).
// Products accumulate in float (the gemv-bias lesson).
//
// Weight REPACKED host-side into ONE buffer (see handoff / build-q4k-gemv.sh). Per
// 256-elem superblock per output row, a **148-byte record**:
//   [0..127]   qs      (128 raw block_q4_K.qs bytes, unchanged)
//   [128..139] scales  (12 raw 6-bit-packed scale/min bytes, unchanged)
//   [140..143] d       (f32, host-converted from ggml f16 x.d)
//   [144..147] dmin    (f32, host-converted from ggml f16 x.dmin)
// i.e. a FIELD REORDER of ggml's block_q4_K (which is {d,dmin,scales,qs}) + the two
// f16 scales widened to f32. A : [N][K/256][148]; b : [K] bf16; c : [N] f32.
//
// NOTE: compiled on Linux, NOT executed here. Unvalidated scaffold — verify with the
// on-device q4 gemv check harness (q4_gemv_check.cpp) before enabling dispatch.
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
void matvec_q4k(const uint8_t *restrict a, const bfloat16 *restrict b, float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 148;
    const uint8_t *qs = rec;         // 128 bytes
    const uint8_t *sca = rec + 128;  // 12 bytes
    float d, dmin;
    __builtin_memcpy(&d, rec + 140, 4);
    __builtin_memcpy(&dmin, rec + 144, 4);

    float sum = 0.0f;
    for (int ck = 0; ck < 4; ck++) {   // 4 chunks of 64 elements
      uint8_t sc, mn;
      get_scale_min_k4(2 * ck + 0, sca, &sc, &mn);
      const float d1 = d * (float)sc, m1 = dmin * (float)mn;
      get_scale_min_k4(2 * ck + 1, sca, &sc, &mn);
      const float d2 = d * (float)sc, m2 = dmin * (float)mn;
      const uint8_t *q = qs + ck * 32;
      const int base = ck * 64;
      for (int l = 0; l < 32; l++) {
        const float vlo = d1 * (float)(q[l] & 0x0F) - m1;
        const float vhi = d2 * (float)(q[l] >> 4) - m2;
        sum += vlo * (float)b[base + l];
        sum += vhi * (float)b[base + 32 + l];
      }
    }
    c[row] += sum;
  }
  event1();
}

extern "C" {
void matvec_q4k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q4k<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++) {
    c[i] = 0.0f;
  }
}
}
