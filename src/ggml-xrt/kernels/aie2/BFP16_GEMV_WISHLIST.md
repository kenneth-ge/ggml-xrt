# bfp16 (block floating point) for Q6_K decode gemv — feasibility verdict

**Verdict: NOT FEASIBLE on aie2 / Phoenix (XDNA1). No prototype built.**

Two independent blockers, either of which is fatal for our target:

## 1. bfp16 does not exist on aie2/Phoenix at all — it is arch-gated to aie2p/Strix

In the mlir-aie aie_api headers, the block-floating-point element types
(`bfp16ebs8`, `bfp16ebs16`), the `aie::block_vector<>` container, and the
`mmul_bfp16_bfp16` matmul intrinsic are all compiled out unless the target arch
is AIE2P (Strix). The gates are:

- `#if __AIE_ARCH__ == 21` — in
  `detail/aie2/vector_native_types.hpp` (bfp types + `is_block_floating_point`)
- `#if AIE_API_ML_VERSION >= 210` — in `block_vector.hpp` (the `block_vector` class)
- File layout: `detail/aie2p/block_vector*.hpp` and
  `detail/aie2p/mmul_bfp16_bfp16.hpp` exist; there is **no** counterpart under
  `detail/aie2/` (Phoenix).

Arch numbering (`detail/config.hpp`): AIE1 = 10, **aie2/Phoenix = 20**,
aie2p/Strix = 21. We build `--target=aie2-none-unknown-elf` → `__AIE_ARCH__ == 20`
→ every bfp16 symbol is absent. A bfp16 kernel would not even compile here.

The bfp16 matmul programming examples confirm this: they live under
`programming_examples/ml/block_datatypes/` and their lit tests are named
`run_strix_*` (Strix-only).

## 2. Even on aie2p, bfp16 is MMUL-ONLY — there is no vector-MAC / dot path

This is the crux the task flagged as a likely STOP condition, and it holds. On
aie2p, `block_vector` is referenced only in:

- `detail/aie2p/block_vector*.hpp` (the type),
- `detail/aie2p/accum.hpp` + `array_helpers.hpp` (pack/unpack to/from accum),
- `detail/aie2p/mmul.hpp` + `detail/aie2p/mmul_bfp16_bfp16.hpp` (the **matmul** intrinsic).

The `mul`/`mac` that accept bfp16 are **methods of the `mmul_bfp16_bfp16`
struct** — they take r×s×t matrix sub-tiles (8×8×8 and 8×8×16) with
`a_sign`/`b_sign` flags, i.e. a matmul MAC, not a vector·vector dot. The
elementwise operators (`detail/aie2p/mul.hpp`, `mul_acc32_fp.hpp`) and
`aie::reduce_*` have **no** `block_vector` overload anywhere.

Our decode gemv (`mv_q6k.cc` / `mv_q6k_simd2.cc`) is M=1: it uses
`aie::mul` / `aie::mac` on `aie::vector<bfloat16,32>` accumulating into
`aie::accum<accfloat,32>`, then `aie::reduce_add`. bfp16 offers no drop-in for
any of those ops. To exploit bfp16 you must reformulate as an `aie::mmul`
matmul (M≥16), which defeats the point of a decode gemv.

## Implication for the M=16 verify / prefill matmul path

`mm_q6k.cc` already uses `aie::mmul<4,8,4,bfloat16,...>` — the bf16 matmul
intrinsic, which is exactly the shape family bfp16 could substitute
(`mmul_bfp16_bfp16<8,8,8|16>`). So bfp16 is *architecturally* relevant only to
that M≥16 matmul path, **not** the gemv. But it is still gated to aie2p/Strix,
so it cannot be enabled on Phoenix. If/when a Strix (aie2p) target is added,
swapping the verify matmul's B operand (and optionally A) to `bfp16ebs8` is the
place to try it: the shared-exponent-per-8 or per-16 block maps naturally onto
Q6_K's per-16 sub-block scales, and bfp16 mmul on Strix is ~2x the bf16 mmul
MAC throughput per the AMD block-datatype examples. That is a Strix-only future
item, with its own NRMSE validation needed (bfp16's 8-bit shared exponent +
8-bit mantissa has a distinct precision profile from per-element bf16).

## Bottom line

- No `mv_q6k_bfp16.cc` and no `gemv_q6k_*_bfp16.xclbin` were produced — they
  cannot compile for aie2/Phoenix.
- The decode gemv stays on the bf16 SIMD mac+reduce core (`mv_q6k_simd2.cc`,
  ~36 ms for q6k 6144x2048 4-col).
- bfp16 is a Strix (aie2p) + matmul-only lever; revisit only if an aie2p backend
  target is introduced, and only for the M≥16 matmul/verify path.
