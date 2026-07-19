//===- mv_q6k_noreduce_timing.cc (aie2 / Phoenix) --------------*- C++ -*-===//
//
// STEP-1 TIMING PROBE (produces GARBAGE output on purpose). Identical to
// mv_q6k_noscratch.cc EXCEPT the two per-output horizontal aie::reduce_add()
// at the tail are removed. This times the MAC+dequant path WITHOUT the
// horizontal reduce, with the SAME DMA, SAME dequant, SAME 8 macs/row.
//   - if this build times ~0.45 ms (near the 23 GB/s DMA floor) the per-output
//     reduce_add IS the compute lever -> the no-reduce rewrite is worth it.
//   - if it still times ~0.9 ms the reduce is NOT the bottleneck -> stop.
//
// CRITICAL (why not just grab lane 0): reading a single scalar lane lets LLVM
// dead-code-eliminate lanes 1..31 of every dequant+mac (verified: vmac 24->12,
// vmul 44->25), which UNDER-measures and would falsely blame the reduce. To keep
// all 32 lanes live WITHOUT a horizontal reduce, each row's 32-lane accumulator
// is folded into a running 32-lane `tally` via a plain 32-lane VECTOR add (NOT a
// horizontal reduce_add), and the full 32-lane tally is stored to c at the end.
// This mirrors the real no-reduce kernel's tail (vector store, no reduce) and
// forces the compiler to compute all 32 lanes of every mac -- so the delta vs
// mv_q6k_noscratch is EXACTLY the removed reduce_add work. Output is garbage.
// Same 212 B q6_K record, same repack contract, same ABI as the gemv.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

// per-chunk 32-lane scale, built inline from the group-scale vector gsb (groups in lanes 12..27):
// chunk ci spans groups 2ci (low 16 lanes) and 2ci+1 (high 16 lanes). No sbuf store/load.
#define SCV(ci)                                                                            \
  aie::concat(aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci))),                         \
              aie::broadcast<bfloat16, 16>(gsb.get(12 + 2 * (ci) + 1)))

#define QW(QLv, QHv, hi_shift, ci)                                                          \
  aie::mul(aie::sub(aie::to_float<bfloat16>(aie::unpack(                                    \
                        aie::bit_or((hi_shift) < 4 ? aie::bit_and((uint8_t)0x0F, (QLv))     \
                                                   : aie::logical_downshift((QLv), 4),      \
                                    aie::upshift(aie::bit_and((uint8_t)0x03,                 \
                                        aie::logical_downshift((QHv), (hi_shift))), 4)))),   \
                    c32v),                                                                  \
           SCV(ci)).template to_vector<bfloat16>()

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();
  const aie::vector<bfloat16, 32> c32v = aie::broadcast<bfloat16, 32>((bfloat16)32.0f);
  // running 32-lane tally: folds every row's accumulator with a VECTOR add (no
  // horizontal reduce). Keeps all 32 lanes of every mac live (stored at the end).
  aie::accum<accfloat, 32> tally;
  tally.from_vector(aie::broadcast<float, 32>(0.0f));
  _Pragma("clang loop unroll_count(2)")
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    // group scales in registers (lanes 12..27), no sbuf scratch.
    aie::vector<bfloat16, 32> gsb =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(sc - 12))),
                 aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();

    aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
    aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
    aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
    aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
    aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
    aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

    aie::accum<accfloat, 32> acc0 = aie::mul(QW(L0a, Ha, 0, 0), aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, QW(L0a, Ha, 4, 2), aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 0, 4), aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 4, 6), aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(QW(L1a, Ha, 2, 1), aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, QW(L1a, Ha, 6, 3), aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 2, 5), aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 6, 7), aie::load_v<32>(b + 224));

    // STEP-1 PROBE: NO horizontal reduce_add. Fold both accumulators into the
    // 32-lane tally with plain vector adds (elementwise, NOT a 32->1 reduce).
    tally = aie::add(tally, aie::add(acc0, acc1));
  }
  // store the full 32-lane tally (all lanes live -> no dequant/mac lane pruned).
  // GARBAGE values, timing only. c_ty is exactly 32 floats.
  aie::store_v(c, tally.template to_vector<float>());
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
