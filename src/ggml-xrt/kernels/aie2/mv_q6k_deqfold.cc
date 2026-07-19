//===- mv_q6k_deqfold.cc (aie2 / Phoenix) — TIMING PROBE H#2b -------*- C++ -*-===//
//
// BIAS-FOLDED q6_K decode gemv: CORRECT output (mm_verify-able), but MINIMIZES the per-chunk
// dequant vector-ops vs. the shipping mv_q6k_noscratch.cc so less of the SHARED AIE2 vector
// datapath is spent on dequant instead of the aie::mac.
//
// Math:  dq_e = (q_e - 32) * s_g ;  out += Σ_e dq_e * b_e
//              = Σ_e q_e*s_g*b_e  -  32 * Σ_e s_g*b_e
// The FIRST term is the ordinary MAC of (q*s) against b — so we DROP the per-chunk aie::sub(-32)
// entirely (8 vector subs/row removed). The SECOND term is a per-row scalar BIAS that folds the
// -32 offset out of the inner loop:
//      Σ_e s_g(e)*b_e = Σ_g s_g * (Σ_{e in group g} b_e) = dot(scale[16], Bg[16])
// where Bg[g] = per-group sum of the activation. b is CONSTANT across all M rows of a matvec
// call, so Bg[16] is computed ONCE per call (hoisted above the row loop) and reused. The per-row
// cost of the fold is then just one vector mul + one reduce (dot(scale,Bg)) instead of 8 subs.
//
// Also: the per-group combined scale is built ONCE/row as a clean 16-lane vector (gscale16, groups
// in lanes 0..15) — replacing noscratch's 32-lane gsb (groups in lanes 12..27) — and both the
// per-chunk SCV() and the bias dot() reuse it (no redundant scale rebuild).
//
// Per-chunk vector-ops vs noscratch: SAME bit-assembly + unpack + to_float + scale-mul + mac,
// but the aie::sub(c32v) is GONE. Net/row: -8 vsub, +1 vmul +1 vreduce (bias dot). Correct math.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

// per-chunk 32-lane scale, built inline from the clean group-scale vector gscale16 (groups in
// lanes 0..15): chunk ci spans groups 2ci (low 16 lanes) and 2ci+1 (high 16 lanes).
#define SCV(ci)                                                                            \
  aie::concat(aie::broadcast<bfloat16, 16>(gscale16.get(2 * (ci))),                         \
              aie::broadcast<bfloat16, 16>(gscale16.get(2 * (ci) + 1)))

// dequant WITHOUT the -32 sub: (ql_nibble | (qh_2bit<<4)) -> bf16, times per-group scale.
// The -32 offset is folded into the per-row bias term (see below), NOT applied per element.
#define QW(QLv, QHv, hi_shift, ci)                                                          \
  aie::mul(aie::to_float<bfloat16>(aie::unpack(                                             \
               aie::bit_or((hi_shift) < 4 ? aie::bit_and((uint8_t)0x0F, (QLv))             \
                                          : aie::logical_downshift((QLv), 4),               \
                           aie::upshift(aie::bit_and((uint8_t)0x03,                         \
                               aie::logical_downshift((QHv), (hi_shift))), 4)))),           \
           SCV(ci)).template to_vector<bfloat16>()

template <int M>
void matvec_q6k_vec(const uint8_t *restrict a, const bfloat16 *restrict b,
                    float *restrict c) {
  event0();

  // Per-group sums of the activation, Bg[g] = Σ_{i=0..15} b[16*g + i], computed ONCE per matvec
  // call (b is constant across the M rows) and reused by every row's bias dot. Each 32-lane load
  // covers two groups (low 16 / high 16 lanes).
  alignas(32) bfloat16 bg[16];
  for (int k = 0; k < 8; k++) {
    aie::vector<bfloat16, 32> vb = aie::load_v<32>(b + 32 * k);
    bg[2 * k]     = aie::reduce_add(vb.template extract<16>(0));
    bg[2 * k + 1] = aie::reduce_add(vb.template extract<16>(1));
  }
  const aie::vector<bfloat16, 16> Bg16 = aie::load_v<16>(bg);

  _Pragma("clang loop unroll_count(2)")
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = a + row * 212;
    const uint8_t *ql = rec;
    const uint8_t *qh = rec + 128;
    const int8_t *sc = (const int8_t *)(rec + 192);
    float d;
    __builtin_memcpy(&d, rec + 208, 4);

    // clean group scales in lanes 0..15 (no sbuf, no lane-12 offset): reused by SCV() and bias.
    const aie::vector<bfloat16, 16> gscale16 =
        aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<16>(sc))),
                 aie::broadcast<bfloat16, 16>((bfloat16)d)).to_vector<bfloat16>();

    aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
    aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
    aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
    aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
    aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
    aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

    // MAC of (q*scale) against b — NO per-chunk -32 sub. Same 8-MAC / 2-accumulator schedule.
    aie::accum<accfloat, 32> acc0 = aie::mul(QW(L0a, Ha, 0, 0), aie::load_v<32>(b + 0));
    acc0 = aie::mac(acc0, QW(L0a, Ha, 4, 2), aie::load_v<32>(b + 64));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 0, 4), aie::load_v<32>(b + 128));
    acc0 = aie::mac(acc0, QW(L0b, Hb, 4, 6), aie::load_v<32>(b + 192));

    aie::accum<accfloat, 32> acc1 = aie::mul(QW(L1a, Ha, 2, 1), aie::load_v<32>(b + 32));
    acc1 = aie::mac(acc1, QW(L1a, Ha, 6, 3), aie::load_v<32>(b + 96));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 2, 5), aie::load_v<32>(b + 160));
    acc1 = aie::mac(acc1, QW(L1b, Hb, 6, 7), aie::load_v<32>(b + 224));

    // folded -32 offset: bias = 32 * dot(scale, Bg). One vmul + one vreduce, replaces 8 vsubs.
    const float bias =
        32.0f * aie::reduce_add(aie::mul(gscale16, Bg16).template to_vector<float>());

    c[row] += aie::reduce_add(acc0.template to_vector<float>()) +
              aie::reduce_add(acc1.template to_vector<float>()) - bias;
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
