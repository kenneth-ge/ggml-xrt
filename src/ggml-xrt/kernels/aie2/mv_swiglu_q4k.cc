//===- mv_swiglu_q4k.cc (aie2 / Phoenix) -----------------*- C++ -*-===//
//
// FUSED FFN SwiGLU core: gate + up (both q4k decode matvec) + silu(gate)*up, in ONE dispatch.
// Removes the NPU->GPU->NPU round-trip that today runs silu*mul on the iGPU between the up matmul
// and the down matmul -> mechanism-robust dispatch/handoff reduction (the real per-token cost).
//
// Weight layout (per output row, per 256-k-block): an INTERLEAVED 296 B record = gate_q4k(148) ++
// up_q4k(148). This keeps the weight DMA 4-D identical to gemv_mc16 (REC=296) and fits Phoenix's
// 2-MM2S/column budget (1 weight + 1 activation + 1 output S2MM); the core splits gate/up by
// pointer offset. Scales are vectorized (vscale, no scalar-fp32). silu(x)=x*0.5*(tanh(x/2)+1) via
// getTanhBf16 LUT. Host: A = per-record [gateq4k ++ upq4k]; B = shared x[K]; C = fused y[N_ff].
//===----------------------------------------------------------------------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#define RECq 148   // one q4k record
#define RECi 296   // interleaved gate++up record

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// dot of ONE q4k record (148 B) with activation b[256] -> float (vscale: vectorized scale setup).
static inline float q4k_dot(const uint8_t *restrict rec, const bfloat16 *restrict b) {
  const uint8_t *qs = rec;
  const uint8_t *sca = rec + 128;
  float d, dmin;
  __builtin_memcpy(&d, rec + 140, 4);
  __builtin_memcpy(&dmin, rec + 144, 4);

  alignas(64) uint8_t scarr[32], mnarr[32];
  for (int gi = 0; gi < 8; gi++)
    get_scale_min_k4(gi, sca, &scarr[gi], &mnarr[gi]);

  // NOSCRATCH + 2-ACCUMULATOR (matches mv_q4k_noscratch.cc, the 1a fast core). Inline-register
  // scale; hi nibble via bit_and(0xF0)=hi*16 with gschi=gsc/16 (NO logical_downshift, which ICEs
  // peano with 2 accumulators). 2 accs pipeline the MAC (the pre-1a single-acc dot was 2x slower).
  aie::vector<bfloat16, 32> gsc =
      aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(scarr))),
               aie::broadcast<bfloat16, 32>((bfloat16)d)).to_vector<bfloat16>();
  aie::vector<bfloat16, 32> gschi =
      aie::mul(gsc, aie::broadcast<bfloat16, 32>((bfloat16)(1.0f / 16.0f))).to_vector<bfloat16>();
  aie::vector<bfloat16, 32> gmn =
      aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::load_v<32>(mnarr))),
               aie::broadcast<bfloat16, 32>((bfloat16)dmin)).to_vector<bfloat16>();
#define SG_LO(Q, ci)                                                                       \
  aie::sub(aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::bit_and((uint8_t)0x0F, (Q)))), \
                    aie::broadcast<bfloat16, 32>(gsc.get(ci))).template to_vector<bfloat16>(), \
           aie::broadcast<bfloat16, 32>(gmn.get(ci)))
#define SG_HI(Q, ci)                                                                       \
  aie::sub(aie::mul(aie::to_float<bfloat16>(aie::unpack(aie::bit_and((uint8_t)0xF0, (Q)))), \
                    aie::broadcast<bfloat16, 32>(gschi.get(ci))).template to_vector<bfloat16>(), \
           aie::broadcast<bfloat16, 32>(gmn.get(ci)))
  aie::vector<uint8_t, 32> Q0 = aie::load_unaligned_v<32>(qs + 0);
  aie::vector<uint8_t, 32> Q1 = aie::load_unaligned_v<32>(qs + 32);
  aie::vector<uint8_t, 32> Q2 = aie::load_unaligned_v<32>(qs + 64);
  aie::vector<uint8_t, 32> Q3 = aie::load_unaligned_v<32>(qs + 96);
  aie::accum<accfloat, 32> acc0 = aie::mul(SG_LO(Q0, 0), aie::load_v<32>(b + 0));
  acc0 = aie::mac(acc0, SG_LO(Q1, 2), aie::load_v<32>(b + 64));
  acc0 = aie::mac(acc0, SG_LO(Q2, 4), aie::load_v<32>(b + 128));
  acc0 = aie::mac(acc0, SG_LO(Q3, 6), aie::load_v<32>(b + 192));
  aie::accum<accfloat, 32> acc1 = aie::mul(SG_HI(Q0, 1), aie::load_v<32>(b + 32));
  acc1 = aie::mac(acc1, SG_HI(Q1, 3), aie::load_v<32>(b + 96));
  acc1 = aie::mac(acc1, SG_HI(Q2, 5), aie::load_v<32>(b + 160));
  acc1 = aie::mac(acc1, SG_HI(Q3, 7), aie::load_v<32>(b + 224));
  return aie::reduce_add(acc0.template to_vector<float>()) +
         aie::reduce_add(acc1.template to_vector<float>());
}

// one 256-k-block: accumulate gate and up for all DIM_M rows from the interleaved weight buffer.
template <int M>
static void fused_gate_up_impl(const uint8_t *restrict w, const bfloat16 *restrict b,
                               float *restrict gate, float *restrict up) {
  event0();
  for (int row = 0; row < M; row++) {
    const uint8_t *rec = w + row * RECi;
    gate[row] += q4k_dot(rec, b);          // gate_q4k at +0
    up[row] += q4k_dot(rec + RECq, b);     // up_q4k   at +148
  }
  event1();
}

// silu(gate)*up over DIM_M f32 elements -> out (f32). silu(x)=x*0.5*(tanh(x/2)+1).
template <int M>
static void silu_mul_impl(const float *restrict gate, const float *restrict up,
                          float *restrict out) {
  const aie::vector<bfloat16, 16> h = aie::broadcast<bfloat16, 16>((bfloat16)0.5f);
  const aie::vector<bfloat16, 16> one = aie::broadcast<bfloat16, 16>((bfloat16)1.0f);
  for (int i = 0; i < M; i += 16) {
    aie::vector<bfloat16, 16> g =
        aie::accum<accfloat, 16>(aie::load_v<16>(gate + i)).to_vector<bfloat16>();
    aie::vector<bfloat16, 16> u =
        aie::accum<accfloat, 16>(aie::load_v<16>(up + i)).to_vector<bfloat16>();
    // getTanhBf16's LUT covers input [-4,4) (32 elems, step 0.25) and does NOT clamp; raw gate
    // dots over K reach +/-76 so hx=gate/2 indexes out of the LUT -> garbage. tanh saturates to
    // +/-1 outside +/-4, so clamping hx into range is exact and fixes the large-|gate| outputs.
    aie::vector<bfloat16, 16> hx = aie::mul(g, h).to_vector<bfloat16>();
    hx = aie::min(hx, aie::broadcast<bfloat16, 16>((bfloat16)3.9f));
    hx = aie::max(hx, aie::broadcast<bfloat16, 16>((bfloat16)-3.9f));
    aie::vector<bfloat16, 16> th = getTanhBf16(hx);
    aie::vector<bfloat16, 16> sig = aie::mul(aie::add(th, one), h).to_vector<bfloat16>();
    aie::vector<bfloat16, 16> silu = aie::mul(g, sig).to_vector<bfloat16>();
    aie::accum<accfloat, 16> y = aie::mul(silu, u);
    aie::store_v(out + i, y.to_vector<float>());
  }
}

extern "C" {
// per-k-block: w=(DIM_M, 296) interleaved gate++up, b=x[256], accumulate gate[] and up[].
void fused_gate_up_q4k(uint8_t *w, bfloat16 *b, float *gate, float *up) {
  fused_gate_up_impl<DIM_M>(w, b, gate, up);
}
void silu_mul_f32(float *gate, float *up, float *out) { silu_mul_impl<DIM_M>(gate, up, out); }
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
