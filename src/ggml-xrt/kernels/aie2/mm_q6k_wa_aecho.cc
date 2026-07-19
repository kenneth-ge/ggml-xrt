//===- mm_q6k_wa_aecho.cc (aie2 / Phoenix) --------------------*- C++ -*-===//
//
// A-ECHO PROBE for the 16-core whole_array Q6_K verify (memtile-A path, DIM_M=32). Same ABI as
// mm_q6k_wa_mt.cc: matmul_q6k_f32(qB, A, Bl1, C). SKIPS dequant + mmul; instead ECHOES the mmul-A
// tile the core actually received (A, the a_dims-transformed memtile-A buffer) straight into C,
// through the PROVEN cpass C write-back path. Isolates A-delivery:
//   host C reflects the activation -> A reaches the cores (bug is mmul/W);
//   host C == 0 -> A is NOT arriving = the memtile-A DMA / a_dims transform / broadcast is broken.
//
// ECHO LAYOUT (verified against the cpass HW result C[tok,ch]=corebuf[(tok/4)*128+(tok%4)*4+
// (ch/4)*16+(ch%4)]+... ): we write corebuf so that host C[tok,ch] == A[tok, ch] for tok=0..31,
// ch=0..31. A[tok,kk] is read out of the mmul-A sub-tile layout at
//   A_mmul[((z*KS+i)*r+rr)*s+ss], z=tok/r, rr=tok%r, i=kk/s, ss=kk%s  (r=4,s=8,KS=DIM_K/s).
// The core loop calls this once per k-tile (24x) and overwrites, so at C-release C reflects the
// LAST k-tile (t=23): host C[tok,ch] == A[tok, 23*256 + ch]. A is broadcast, so every core /
// n-tile shows the same A[tok, 5888..5919] block replicated across all N channels.
//   -> "correct" = C[tok, ch] equals the input activation A[tok, 5888 + (ch mod 32)] (nonzero if
//      the activation is nonzero anywhere in that k-band; the whole point is zero vs nonzero).
//
// Licensed Apache-2.0 WITH LLVM-exception. (c) 2026 AMD Inc.
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include "mm.cc"  // zero_f32 + <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 32
#endif
#ifndef DIM_K
#define DIM_K 256
#endif
#ifndef DIM_N
#define DIM_N 32
#endif

extern "C" {
void matmul_q6k_f32(uint8_t *qB, bfloat16 *A, bfloat16 *Bl1, float *C) {
  (void)qB; (void)Bl1;
  constexpr int r = 4, s = 8, t = 4;
  constexpr int KS = DIM_K / s;
  for (int tok = 0; tok < DIM_M; tok++) {
    for (int ch = 0; ch < DIM_N; ch++) {
      // A[tok, kk=ch] out of the mmul-A sub-tile layout:
      int z = tok / r, rr = tok % r, i = ch / s, ss = ch % s;
      bfloat16 v = A[((z * KS + i) * r + rr) * s + ss];
      // corebuf index that de-tiles (c_dims) to host C[tok, ch]:
      int cb = (tok / r) * (r * DIM_N) + (tok % r) * t + (ch / t) * (r * t) + (ch % t);
      C[cb] = (float)v;
    }
  }
}
}
