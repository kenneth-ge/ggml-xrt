//===- mv_bf16_noreduce.cc (aie2 / Phoenix) --------------------*- C++ -*-===//
//
// STEP-3a PROOF: dense bf16 M=1 gemv with NO per-output horizontal reduce_add.
//
// Outputs-in-lanes / rank-1 dataflow (mirrors mlir-aie mv.cc matvec_vectorized):
// the M=32 outputs of one output-tile live PERMANENTLY in the 32 accumulator
// lanes. K is consumed by broadcast-scalar x weight-column MACs
// (aie::accumulate, 8 columns/call): acc[n] += b[k] * W[n,k]. After the k-tile
// each lane IS a finished partial output -- ZERO horizontal reduce.
//
// Weight layout REQUIRED (K-major, host pre-repacked, contiguous):
//   for one output-tile of M outputs and one k-tile of K columns, W is [K][M]
//   bf16: element (col k, output n) at W[k*M + n]. So for a given column the M
//   outputs' weights are contiguous -> a single load_v<M> feeds one rank-1
//   update, no 4-byte-granularity DMA transpose / filter_even-odd needed
//   (that dance in mv.cc only exists because it transposes bf16 in the DMA;
//   here the host repack already lands the weight K-major).
//
// c[] is read-modify-written each k-tile call (accumulate across K/k k-tiles),
// mirroring the q6k gemv's per-call `c[row] +=`. The RMW is a single 32-lane
// vector load + add + store -- NOT a reduce.
//
// TRADEOFF vs q6k: bf16 weight is 2 B/wt vs q6k's ~0.83 B/wt (212 B / 256 wts),
// so ~2.4x more weight DMA. Since the q6k gemv is DMA-hidden/compute-bound this
// isolates the reduce-elimination win; on the NPU the bf16 DMA cost must be
// weighed separately (see report).
//
// ABI mirrors the q6k gemv: matvec(weight@a, act@b, out@c); zero_scalar_f32.
// Compile-only validation on Linux/WSL; NOT executed on NPU.
//===----------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32      // outputs per tile == accumulator lanes
#endif
#ifndef DIM_K
#define DIM_K 256     // columns per k-tile (matches q6k QK for DMA comparability)
#endif

// Load 8 K-major weight columns (each M outputs contiguous) starting at wp,
// and rank-1-accumulate them into `a` with the 8 broadcast coeffs from `bv`:
//   a[i] += bv[0]*W(col0,i) + ... + bv[7]*W(col7,i)   (NO horizontal reduce)
#define ACC8(a, wp, bv)                                                        \
  do {                                                                         \
    aie::vector<bfloat16, M> w0 = aie::load_v<M>((wp) + 0 * M);                 \
    aie::vector<bfloat16, M> w1 = aie::load_v<M>((wp) + 1 * M);                 \
    aie::vector<bfloat16, M> w2 = aie::load_v<M>((wp) + 2 * M);                 \
    aie::vector<bfloat16, M> w3 = aie::load_v<M>((wp) + 3 * M);                 \
    aie::vector<bfloat16, M> w4 = aie::load_v<M>((wp) + 4 * M);                 \
    aie::vector<bfloat16, M> w5 = aie::load_v<M>((wp) + 5 * M);                 \
    aie::vector<bfloat16, M> w6 = aie::load_v<M>((wp) + 6 * M);                 \
    aie::vector<bfloat16, M> w7 = aie::load_v<M>((wp) + 7 * M);                 \
    (a) = aie::accumulate<M>((a), (bv), 0, w0, w1, w2, w3, w4, w5, w6, w7);     \
  } while (0)

template <int M, int K>
void matvec_bf16_kmajor(const bfloat16 *restrict W, const bfloat16 *restrict b,
                        float *restrict c) {
  event0();
  // outputs-in-lanes: the SAME M outputs held in all four M-lane accumulators;
  // each covers a DISJOINT set of columns so the four rank-1 chains are
  // independent -> 4-way ILP hides the fma latency (a single-accumulator chain
  // would serialize and stall). NO horizontal reduce anywhere.
  aie::accum<accfloat, M> acc0, acc1, acc2, acc3;
  acc0.from_vector(aie::load_v<M>(c));            // seed w/ current c (RMW across k-tiles)
  const aie::vector<float, M> z = aie::broadcast<float, M>(0.0f);
  acc1.from_vector(z);
  acc2.from_vector(z);
  acc3.from_vector(z);

  const bfloat16 *restrict wp = W;                // W[k*M + n], K-major
  // 32 columns per iteration: 4 independent 8-column rank-1 updates.
  for (int col = 0; col < K; col += 32) {
    aie::vector<bfloat16, 8> b0 = aie::load_v<8>(b + col + 0);
    aie::vector<bfloat16, 8> b1 = aie::load_v<8>(b + col + 8);
    aie::vector<bfloat16, 8> b2 = aie::load_v<8>(b + col + 16);
    aie::vector<bfloat16, 8> b3 = aie::load_v<8>(b + col + 24);
    ACC8(acc0, wp + 0 * M, b0);
    ACC8(acc1, wp + 8 * M, b1);
    ACC8(acc2, wp + 16 * M, b2);
    ACC8(acc3, wp + 24 * M, b3);
    wp += 32 * M;
  }
  // combine the 4 partial-K accumulators: elementwise 32-lane adds, NOT a reduce.
  auto acc = aie::add(aie::add(acc0, acc1), aie::add(acc2, acc3));
  aie::store_v(c, acc.template to_vector<float>());
  event1();
}

extern "C" {
void matvec_bf16_f32(bfloat16 *a, bfloat16 *b, float *c) {
  matvec_bf16_kmajor<DIM_M, DIM_K>(a, b, c);
}
void zero_scalar_f32(float *c) {
  for (int i = 0; i < DIM_M; i++)
    c[i] = 0.0f;
}
}
