//===- mv_q4.cc (aie2 / Phoenix) --------------------------*- C++ -*-===//
//
// Q4_0 on-chip-dequant matrix-vector (decode gemv) core function.
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//
// Computes C[M] += dequant(A_q4[M,K]) . B[K] for one (M=DIM_M, K=DIM_K) tile,
// where DIM_K is one k-tile (== 32 == a single ggml q4_0 block per row).
//
// Weight is REPACKED host-side (see handoff) into ONE buffer of 20-byte block
// records — raw ggml block_q4_0's 18-byte stride is DMA-hostile, and 3 separate
// DDR input streams exceed the shim's 2 read-DMA channels, so weight is a single
// stream. Per 32-elem block, per row: **20 bytes = 16 nibble bytes (ggml qs) then
// one f32 scale** (host-converted from ggml f16 `d`, f32 for precision). Layout:
//   A : [M][K/32][20] uint8   (row-major; 16 nibbles + 4 scale bytes per block)
//   b : [K] bf16 activation ;  c : [M] f32 output (accumulated)
//
// Dequant math is exactly ggml `dequantize_row_q4_0` (ggml-quants.c):
//   for each 32-elem block with scale d:
//     lo = (qs[j] & 0x0F) - 8 ; hi = (qs[j] >> 4) - 8   (j in 0..15)
//     val[j] = lo*d ; val[j+16] = hi*d
// Products accumulate in float (the gemv-bias lesson: promote before multiply).
//
// NOTE: compiled on Linux, NOT executed here (no NPU). Unvalidated scaffold —
// verify with the on-device mulmat/gemv check harness before enabling dispatch.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 32
#endif

// One k-tile == one 32-elem q4_0 block per row; A tile is [M][20] bytes.
template <int M>
void matvec_q4_0(const uint8_t *restrict a, const bfloat16 *restrict b, float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 20;   // 16 nibbles + f32 scale
    float d;
    __builtin_memcpy(&d, rec + 16, 4);   // unaligned-safe (row stride is 20)
    float sum = 0.0f;
    for (int j = 0; j < 16; j++) {
      const int lo = (int)(rec[j] & 0x0F) - 8;
      const int hi = (int)(rec[j] >> 4) - 8;
      sum += (float)lo * d * (float)b[j];
      sum += (float)hi * d * (float)b[j + 16];
    }
    c[row] += sum;
  }
  event1();
}

extern "C" {
void matvec_q4_0_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_q4_0<DIM_M>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++) {
    c[i] = 0.0f;
  }
}
}
