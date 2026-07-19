//===- mm_q6k_wa_cpass.cc (aie2 / Phoenix) --------------------*- C++ -*-===//
//
// C-PASSTHROUGH PROBE for the 16-core whole_array Q6_K verify (gemv_mm16.py). Same ABI/signature
// as mm_q6k_wa.cc, but matmul_q6k_f32 SKIPS dequant+mmul entirely and writes a KNOWN pattern
// C[i] = (float)(i + 1) into the acquired C tile. The core loop, C objectfifo (core->mem gather,
// mem->shim c_dims 4-dim transform), release path, and runtime C DMA are byte-identical to the
// real kernel — only the compute is replaced. Decisive isolation:
//   host reads nonzero (values 1..512 permuted per tile) -> C write-back path is CORRECT, the
//     all-zeros bug is in the dequant+mmul compute (into eo);
//   host reads 0 -> the C DMA / gather / transform / group-id path is broken.
// The (i+1) ramp (all strictly positive, no 0) also reveals whether c_dims permutes the layout.
//
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // zero_f32 (referenced by gemv_mm16.py) + <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 256
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

extern "C" {
// Same signature as the real matmul_q6k_f32 (qB, Aplain, Al1, Bl1, C) so gemv_mm16.py is
// unchanged. A/W buffers are still acquired/released by the core loop; here we ignore them and
// just stamp the C tile. Called once per k-tile (24x); we OVERWRITE (idempotent), so after the
// k-loop C[i] == (float)(i+1).
void matmul_q6k_f32(uint8_t *qB, bfloat16 *Aplain, bfloat16 *Al1, bfloat16 *Bl1, float *C) {
  (void)qB; (void)Aplain; (void)Al1; (void)Bl1;
  for (int i = 0; i < DIM_M * DIM_N; i++)
    C[i] = (float)(i + 1);
}
}
