//===- mv_q6k_8acc.cc (aie2 / Phoenix) --------------*- C++ -*-===//
//
// TIMING PROBE H#1 (accumulator ILP, maximal). Base = mv_q6k_noscratch.cc (2 accumulators,
// inline-register scale, no sbuf scratch). This variant uses EIGHT INDEPENDENT accumulators --
// one per K-chunk, all a single mul with NO dependency chain between them -- so all eight VMACs
// can be issued back-to-back and the 6-cycle VMAC latency is fully exposed to the scheduler.
// If cyc/MAC drops toward ~6 (issue-bound) rather than ~45 (stall-bound), the latency was never
// being hidden and ILP is the lever; if 8-acc register pressure causes spills and it is NOT faster
// than 4-acc, the 4-acc point is the sweet spot. Combine: add-tree (7 vector adds) + ONE reduce.
// Same 212 B record + math + result as mv_q6k_noscratch.cc -> mm_verify must still pass.
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

    // 8 INDEPENDENT accumulators, one per chunk (single mul each, no dependency chain).
    aie::accum<accfloat, 32> acc0 = aie::mul(QW(L0a, Ha, 0, 0), aie::load_v<32>(b + 0));
    aie::accum<accfloat, 32> acc1 = aie::mul(QW(L1a, Ha, 2, 1), aie::load_v<32>(b + 32));
    aie::accum<accfloat, 32> acc2 = aie::mul(QW(L0a, Ha, 4, 2), aie::load_v<32>(b + 64));
    aie::accum<accfloat, 32> acc3 = aie::mul(QW(L1a, Ha, 6, 3), aie::load_v<32>(b + 96));
    aie::accum<accfloat, 32> acc4 = aie::mul(QW(L0b, Hb, 0, 4), aie::load_v<32>(b + 128));
    aie::accum<accfloat, 32> acc5 = aie::mul(QW(L1b, Hb, 2, 5), aie::load_v<32>(b + 160));
    aie::accum<accfloat, 32> acc6 = aie::mul(QW(L0b, Hb, 4, 6), aie::load_v<32>(b + 192));
    aie::accum<accfloat, 32> acc7 = aie::mul(QW(L1b, Hb, 6, 7), aie::load_v<32>(b + 224));

    // combine: balanced add-tree (7 VECTOR adds) then ONE reduce.
    aie::vector<float, 32> s01 = aie::add(acc0.template to_vector<float>(), acc1.template to_vector<float>());
    aie::vector<float, 32> s23 = aie::add(acc2.template to_vector<float>(), acc3.template to_vector<float>());
    aie::vector<float, 32> s45 = aie::add(acc4.template to_vector<float>(), acc5.template to_vector<float>());
    aie::vector<float, 32> s67 = aie::add(acc6.template to_vector<float>(), acc7.template to_vector<float>());
    aie::vector<float, 32> s = aie::add(aie::add(s01, s23), aie::add(s45, s67));
    c[row] += aie::reduce_add(s);
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
