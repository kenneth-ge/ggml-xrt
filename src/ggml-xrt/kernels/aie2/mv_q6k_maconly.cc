//===- mv_q6k_maconly.cc (aie2 / Phoenix) — ISOLATION EXPERIMENT ---*- C++ -*-===//
//
// MAC-ONLY q6_K decode gemv: strips the 6-bit dequant (nibble|hi-bit assembly, per-group
// scale, sub-32) and keeps ONLY the memory read + a minimal uint8->bf16 convert + the exact
// same 8-MAC / 2-accumulator / 2-reduce structure as mv_q6k.cc. This isolates the cost of
// DMA + MAC + reduce from the cost of dequant.
//
//   time(mv_q6k.cc, full)  -  time(this, mac-only)  =  dequant cost
//   time(this, mac-only)                            =  DMA + MAC + reduce floor
//
// If mac-only ~= full  -> dequant is ~free; we are MAC/M=1-structural-bound (the primitive).
// If mac-only <<  full  -> dequant dominates; the lever is dequant (fused/offload/fewer converts).
// PRODUCES GARBAGE OUTPUT BY DESIGN — for timing only, never ship.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

// minimal "dequant": one unpack (uint8->int16, lane-preserving) + to_float<bf16>. No bit-ops,
// no scale, no sub. This is the cheapest way to get a bf16x32 operand out of the record.
#define QMIN(Lv) aie::to_float<bfloat16>(aie::unpack(Lv))

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;

    // identical record reads to the full kernel (same DMA/L1 traffic)
    aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
    aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
    aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
    aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
    aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
    aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

    // exact same 8-MAC / 2-independent-accumulator schedule as mv_q6k.cc, but the operands
    // skip the 6-bit assembly + scale (QMIN vs QW).
    aie::accum<accfloat, 32> acc0 = aie::mul(QMIN(L0a), aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, QMIN(L1a), aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, QMIN(L0b), aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, QMIN(L1b), aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(QMIN(Ha), aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, QMIN(Hb), aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, QMIN(L0a), aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, QMIN(L1a), aie::load_v<32>(b + 224));

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
