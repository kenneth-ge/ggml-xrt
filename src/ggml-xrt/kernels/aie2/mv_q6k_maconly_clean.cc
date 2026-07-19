//===- mv_q6k_maconly_clean.cc (aie2 / Phoenix) — TIMING PROBE H#2a --*- C++ -*-===//
//
// CLEAN MAC-ONLY q6_K decode gemv: isolates the TRUE per-weight compute floor of the shared
// AIE2 vector datapath by keeping the EXACT structure/loop/unroll(2)/2-accumulator/2-reduce
// schedule of the SHIPPING kernel (mv_q6k_noscratch.cc) but SKIPPING ALL DEQUANT.
//
// vs. the older mv_q6k_maconly.cc (which still did unpack+to_float PER CHUNK = 2 vector-ops/mac),
// this converts the raw bytes to a bf16 operand exactly ONCE per row and REUSES that single
// vector for all 8 MACs. So the per-chunk vector-op count drops to just:  the aie::mac + the
// aie::load_v<32>(b) of the activation.  Nothing else touches the vector unit inside the chunk.
//
//   time(noscratch, full) - time(this, mac-only) = dequant's TRUE cost on the shared vector unit
//   time(this, mac-only)                         = DMA + MAC + reduce floor  (the M=1 primitive)
//
// The 212 B record is still streamed (DMA identical, set by the MLIR / REC=212); only the
// per-record COMPUTE differs. PRODUCES GARBAGE OUTPUT BY DESIGN — for timing only, never ship.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  // SAME loop + unroll(2) + 2-accumulator + 2-reduce structure as mv_q6k_noscratch.cc.
  _Pragma("clang loop unroll_count(2)")
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;

    // ONE load + ONE convert per row: NO 6-bit assembly, NO per-group scale, NO sub-32.
    // aie::to_float the raw bytes once, reuse the single bf16 vector for all 8 MACs so the
    // per-chunk vector-op count is exactly {mac + activation-load}. (Garbage math, timing only.)
    const aie::vector<bfloat16, 32> w =
        aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(ql)));

    aie::accum<accfloat, 32> acc0 = aie::mul(w, aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, w, aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, w, aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, w, aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(w, aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, w, aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, w, aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, w, aie::load_v<32>(b + 224));

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
