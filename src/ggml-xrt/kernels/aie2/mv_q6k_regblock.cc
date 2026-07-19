//===- mv_q6k_regblock.cc (aie2 / Phoenix) --------------*- C++ -*-===//
//
// H#3 TIMING PROBE (register-block activation reuse). Fork of mv_q6k_noscratch.cc.
// Same 212 B record, inline group-scale (no sbuf), 2-acc per-output-dot structure, same result.
//
// KEY CHANGE vs noscratch: noscratch re-loads the WHOLE activation vector b[256] (8x load_v<32>)
// once PER OUTPUT ROW -> M*8 act loads per matvec call (=24576/core for 2048x6144). But b is
// IDENTICAL across all output rows in a k-tile. Here we register-block B=4 rows: load the 8
// activation chunks ONCE into registers, then dequant+MAC 4 rows' weights against those SAME
// live b vectors before advancing. Act loads drop B-fold: 8 per 4 rows = 2/row effective
// (=6144/core). This is NOT the outputs-in-lanes rank-1 that failed — each output is still its
// own dot with its own 2 accumulators; we only amortize the activation loads.
//
// Alignment: b is the activation object-fifo L1 buffer (bf16[256], 64B-aligned) so its loads are
// already aie::load_v<32> (aligned) in both noscratch and here. The WEIGHT record is 212 B / row
// (not a multiple of 64) so per-row ql/qh/sc loads are inherently unaligned-strided and MUST stay
// aie::load_unaligned_v. No weight load can be safely promoted to aie::load_v without a 64B stride.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef REG_B
#define REG_B 4
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
  constexpr int B = REG_B;
  static_assert(M % B == 0, "DIM_M must be divisible by REG_B");

  for (int rb = 0; rb < M; rb += B) {
    // ---- load the activation chunks ONCE per B-row block (aligned, shared across B rows) ----
    const aie::vector<bfloat16, 32> b0 = aie::load_v<32>(b + 0);
    const aie::vector<bfloat16, 32> b1 = aie::load_v<32>(b + 32);
    const aie::vector<bfloat16, 32> b2 = aie::load_v<32>(b + 64);
    const aie::vector<bfloat16, 32> b3 = aie::load_v<32>(b + 96);
    const aie::vector<bfloat16, 32> b4 = aie::load_v<32>(b + 128);
    const aie::vector<bfloat16, 32> b5 = aie::load_v<32>(b + 160);
    const aie::vector<bfloat16, 32> b6 = aie::load_v<32>(b + 192);
    const aie::vector<bfloat16, 32> b7 = aie::load_v<32>(b + 224);

    // DO NOT unroll(full) the inner row loop: fully unrolling forces all B rows' dequant temps
    // (unpack/to_float/sub/mul/concat) live at once and overflows AIE2's vector RF -> heavy spill
    // (B=4 full-unroll measured stack 0xaa0 / vlda 252). Left rolled, the 8 loop-invariant b
    // vectors stay in registers (LICM-hoisted) and are reused across the B rows: stack 0x760
    // (< shipping 0x7a0), vlda 89 (< shipping 119). That is the real -B-fold activation-load win.
    _Pragma("clang loop unroll(disable)")
    for (int j = 0; j < B; j++) {
      const int row = rb + j;
      const uint8_t *rec = a + row * 212;
      const uint8_t *ql = rec;
      const uint8_t *qh = rec + 128;
      const int8_t *sc = (const int8_t *)(rec + 192);
      float d;
      __builtin_memcpy(&d, rec + 208, 4);

      // group scales in registers (lanes 12..27), no sbuf scratch. Record is 212B-strided -> unaligned.
      aie::vector<bfloat16, 32> gsb =
          aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_unaligned_v<32>(sc - 12))),
                   aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();

      aie::vector<uint8_t, 32> L0a = aie::load_unaligned_v<32>(ql);
      aie::vector<uint8_t, 32> L1a = aie::load_unaligned_v<32>(ql + 32);
      aie::vector<uint8_t, 32> Ha = aie::load_unaligned_v<32>(qh);
      aie::vector<uint8_t, 32> L0b = aie::load_unaligned_v<32>(ql + 64);
      aie::vector<uint8_t, 32> L1b = aie::load_unaligned_v<32>(ql + 96);
      aie::vector<uint8_t, 32> Hb = aie::load_unaligned_v<32>(qh + 32);

      // reuse the block-shared b vectors (no per-row activation reload)
      aie::accum<accfloat, 32> acc0 = aie::mul(QW(L0a, Ha, 0, 0), b0);
      acc0 = aie::mac(acc0, QW(L0a, Ha, 4, 2), b2);
      acc0 = aie::mac(acc0, QW(L0b, Hb, 0, 4), b4);
      acc0 = aie::mac(acc0, QW(L0b, Hb, 4, 6), b6);

      aie::accum<accfloat, 32> acc1 = aie::mul(QW(L1a, Ha, 2, 1), b1);
      acc1 = aie::mac(acc1, QW(L1a, Ha, 6, 3), b3);
      acc1 = aie::mac(acc1, QW(L1b, Hb, 2, 5), b5);
      acc1 = aie::mac(acc1, QW(L1b, Hb, 6, 7), b7);

      c[row] += aie::reduce_add(acc0.template to_vector<float>()) +
                aie::reduce_add(acc1.template to_vector<float>());
    }
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
