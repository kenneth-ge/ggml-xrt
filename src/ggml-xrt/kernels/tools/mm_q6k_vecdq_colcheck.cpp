// mm_q6k_vecdq_colcheck.cpp — Linux (host g++) scalar check, NO aie.
//
// Validates that the vectorized-dequant chunk->col[] store mapping in aie2/mm_q6k_vecdq.cc
// reproduces the SAME col[] element order as the scalar deq_q6k in aie2/mm_q6k.cc.
// Both run on the same random 212-byte q6_K record; float-exact compare (the mapping is a pure
// reordering of identical arithmetic, so a correct mapping is bit-identical in float — bf16
// rounding on the NPU is orthogonal to the position/scale-group mapping this check guards).
//
//   g++ -O2 -std=c++17 mm_q6k_vecdq_colcheck.cpp -o /tmp/colcheck && /tmp/colcheck

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

// --- reference: scalar deq_q6k, copied verbatim from aie2/mm_q6k.cc ---
static void deq_q6k(const uint8_t *rec, float *out) {
  const uint8_t *ql = rec;
  const uint8_t *qh = rec + 128;
  const int8_t  *sc = (const int8_t *)(rec + 192);
  float d;
  memcpy(&d, rec + 208, 4);
  for (int ch = 0; ch < 2; ch++) {
    const uint8_t *qlc = ql + ch * 64;
    const uint8_t *qhc = qh + ch * 32;
    const int8_t  *scc = sc + ch * 8;
    const int base = ch * 128;
    for (int l = 0; l < 32; l++) {
      const int is = l >> 4;
      const int q1 = (int)((qlc[l]      & 0x0F) | (((qhc[l] >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((qlc[l + 32] & 0x0F) | (((qhc[l] >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((qlc[l]      >> 4)   | (((qhc[l] >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((qlc[l + 32] >> 4)   | (((qhc[l] >> 6) & 3) << 4)) - 32;
      out[base + l]      = d * (float)scc[is + 0] * (float)q1;
      out[base + l + 32] = d * (float)scc[is + 2] * (float)q2;
      out[base + l + 64] = d * (float)scc[is + 4] * (float)q3;
      out[base + l + 96] = d * (float)scc[is + 6] * (float)q4;
    }
  }
}

// --- scalar MODEL of the vectorized chunk->col[] store mapping (mm_q6k_vecdq.cc) ---
// Mirrors the QW(QLv,QHv,hi_shift,ci) 6-bit assembly + SCV(ci) group scales + store offset.
//   gsb.get(12+g) == sc[g]*d ; SCV(ci): lanes[0..15]=group 2ci, lanes[16..31]=group 2ci+1.
struct Chunk { int ql_off, qh_off, hi_shift, ci, store_off; };

static void deq_q6k_vec_model(const uint8_t *rec, float *col) {
  const uint8_t *ql = rec;
  const uint8_t *qh = rec + 128;
  const int8_t  *sc = (const int8_t *)(rec + 192);
  float d;
  memcpy(&d, rec + 208, 4);

  // exactly the 8 aie::store_v(col+off, QW(...)) lines in mm_q6k_vecdq.cc:
  //   L0a=ql+0, L1a=ql+32, Ha=qh+0 ; L0b=ql+64, L1b=ql+96, Hb=qh+32
  const Chunk chunks[8] = {
    {  0,  0, 0, 0,   0},  // QW(L0a, Ha, 0, 0) -> col[0:32]
    { 32,  0, 2, 1,  32},  // QW(L1a, Ha, 2, 1) -> col[32:64]
    {  0,  0, 4, 2,  64},  // QW(L0a, Ha, 4, 2) -> col[64:96]
    { 32,  0, 6, 3,  96},  // QW(L1a, Ha, 6, 3) -> col[96:128]
    { 64, 32, 0, 4, 128},  // QW(L0b, Hb, 0, 4) -> col[128:160]
    { 96, 32, 2, 5, 160},  // QW(L1b, Hb, 2, 5) -> col[160:192]
    { 64, 32, 4, 6, 192},  // QW(L0b, Hb, 4, 6) -> col[192:224]
    { 96, 32, 6, 7, 224},  // QW(L1b, Hb, 6, 7) -> col[224:256]
  };
  for (const Chunk &c : chunks) {
    const uint8_t *QL = ql + c.ql_off;
    const uint8_t *QH = qh + c.qh_off;
    for (int l = 0; l < 32; l++) {
      const int low  = (c.hi_shift < 4) ? (QL[l] & 0x0F) : (QL[l] >> 4);
      const int high = ((QH[l] >> c.hi_shift) & 0x03) << 4;
      const int q    = (low | high) - 32;
      const int g    = (l < 16) ? (2 * c.ci) : (2 * c.ci + 1);  // SCV group -> sc index
      col[c.store_off + l] = d * (float)sc[g] * (float)q;
    }
  }
}

int main() {
  srand(12345);
  int fails = 0, trials = 2000;
  for (int t = 0; t < trials; t++) {
    uint8_t rec[212];
    for (int i = 0; i < 208; i++) rec[i] = (uint8_t)(rand() & 0xFF);
    float d = ((rand() % 2000) - 1000) / 137.0f;  // signed non-trivial scale
    memcpy(rec + 208, &d, 4);

    float ref[256], got[256];
    deq_q6k(rec, ref);
    deq_q6k_vec_model(rec, got);

    for (int i = 0; i < 256; i++) {
      if (ref[i] != got[i]) {
        if (fails < 8)
          printf("MISMATCH trial=%d col[%d]: scalar=%.6f model=%.6f\n", t, i, ref[i], got[i]);
        fails++;
      }
    }
  }
  if (fails == 0) {
    printf("PASS: vectorized chunk->col[] mapping matches scalar deq_q6k order "
           "(all 256 elems, %d random records, float-exact).\n", trials);
    return 0;
  }
  printf("FAIL: %d mismatched elements across %d records.\n", fails, trials);
  return 1;
}
