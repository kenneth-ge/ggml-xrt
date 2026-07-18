//===- mv_mm_q6k.cc (aie2 / Phoenix) ---------------------*- C++ -*-===//
//
// Q6_K M=N "mat-MAT" for the speculative-decode VERIFY kernel. Computes, per core's
// output rows, C[MT, m] += dequant(W[m, K]) . A[MT, K] one 256-k-block at a time.
//
// THE PHYSICS: the M=1 gemv is weight-DMA-bound (the MACs sit mostly idle while the
// weight streams). Here we dequant W[n]'s 8 chunks ONCE per output row (the exact
// noscratch QW/gsb path), then fill those idle MACs with ALL MT draft tokens'
// activations. MT tokens cost ~one weight stream instead of MT gemvs.
//
// Reuses mv_q6k_noscratch.cc's dequant VERBATIM (QW macro + gsb group-scale-in-regs).
// The mt-loop is SEQUENTIAL with ONE accumulator reused per token -> avoids the
// q6k logical_downshift + multi-accum ICE (validated shape).
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32   // output rows (weight records) this core-call processes
#endif
#ifndef DIM_MT
#define DIM_MT 8   // draft tokens M (activation rows)
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

// a: [M, 212] q6k weight records for this core's M output rows, one 256-k-block.
// b: [MT, 256] bf16 activations for MT draft tokens, same 256-k-block, broadcast.
// c: [MT, M] f32 output tile, k-accumulated across blocks (zeroed once per output tile).
template <int M, int MT>
void matvec_mm_q6k(const uint8_t *restrict a, const bfloat16 *restrict b,
                   float *restrict c) {
  event0();
  const aie::vector<bfloat16, 32> c32v = aie::broadcast<bfloat16, 32>((bfloat16)32.0f);
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

    // Dequant this output row's 8 chunks ONCE (hoisted out of the mt-loop = amortized
    // over the MT draft tokens). Chunk ci maps to activation slice b + ci*32.
    aie::vector<bfloat16, 32> w0 = QW(L0a, Ha, 0, 0);
    aie::vector<bfloat16, 32> w1 = QW(L1a, Ha, 2, 1);
    aie::vector<bfloat16, 32> w2 = QW(L0a, Ha, 4, 2);
    aie::vector<bfloat16, 32> w3 = QW(L1a, Ha, 6, 3);
    aie::vector<bfloat16, 32> w4 = QW(L0b, Hb, 0, 4);
    aie::vector<bfloat16, 32> w5 = QW(L1b, Hb, 2, 5);
    aie::vector<bfloat16, 32> w6 = QW(L0b, Hb, 4, 6);
    aie::vector<bfloat16, 32> w7 = QW(L1b, Hb, 6, 7);

    // One accumulator reused per token (sequential mt-loop = no multi-accum ICE).
    for (int mt = 0; mt < MT; mt++) {
      const bfloat16 *bm = b + mt * 256;
      aie::accum<accfloat, 32> acc = aie::mul(w0, aie::load_v<32>(bm + 0));
      acc = aie::mac(acc, w1, aie::load_v<32>(bm + 32));
      acc = aie::mac(acc, w2, aie::load_v<32>(bm + 64));
      acc = aie::mac(acc, w3, aie::load_v<32>(bm + 96));
      acc = aie::mac(acc, w4, aie::load_v<32>(bm + 128));
      acc = aie::mac(acc, w5, aie::load_v<32>(bm + 160));
      acc = aie::mac(acc, w6, aie::load_v<32>(bm + 192));
      acc = aie::mac(acc, w7, aie::load_v<32>(bm + 224));
      c[mt * M + row] += aie::reduce_add(acc.template to_vector<float>());
    }
  }
  event1();
}

extern "C" {
void matvec_mm_q6k_f32(uint8_t *a, bfloat16 *b, float *c) {
  matvec_mm_q6k<DIM_M, DIM_MT>(a, b, c);
}
void zero_mm_f32(float *c) {
  for (int i = 0; i < DIM_MT * DIM_M; i++)
    c[i] = 0.0f;
}
}
