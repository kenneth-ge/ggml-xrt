# ggml-xrt — kernel wishlist for the Linux/WSL build side

Kernels are compiled **only on Linux/WSL** (mlir-aie / IRON / peano / Triton-XDNA; ELF wheels,
no Windows build). The Windows host consumes `.xclbin` + instruction blobs as data. This doc is
the running wishlist of kernels/shapes/optimizations to produce on the Linux side, ordered by
value. Target silicon: **aie2 / Phoenix / NPU1** (`aie.device(npu1_1col)` for single-core,
4-column `whole_array` for the wide designs). Everything the NPU consumes is **BF16** (weights are
host-dequantized from Q4_K etc. before dispatch).

Build scripts live in `src/ggml-xrt/kernels/` (`build-mm-xclbin.sh`, `build-qwen3-matmuls.sh`,
`build-ops.sh`, `headless_shim.py`). The validated host ABI every kernel must keep: instance
name `MLIR_AIE`, `kernel(opcode=3, instr@grp1, ninstr_words, bo0@grp3, bo1@grp4, bo2@grp5)`,
BF16 buffers, seq/M tiling done host-side.

Legend: **[P0]** required for target models · **[P1]** broadens model/op coverage · **[P2]** nice-to-have.

---

## 1. Matmul: two structural items

- **[P0] Real M=1 gemv kernels** (decode path). Today decode (M=1) runs the M=32 `single_core`
  tile with the other 31 rows zero-padded. Build a proper `matrix_vector`/gemv design per `(K,N)`
  (mlir-aie `programming_examples/basic/matrix_vector`). Naming suggestion:
  `mul_mat_aie2_bf16_f32_1x{K}x{N}_gemv.xclbin` so the host's largest-tile-≤-M selector picks it
  for M=1. **The prebuilt gemv is SCALAR (the vectorized `mv.cc` path is
  erroneous) → ~5% AIE util, slower than the vectorized tiled kernel padding M=1; it is host
  opt-in (`GGML_XRT_USE_GEMV`) and OFF by default until a VECTORIZED gemv is built. Vectorizing
  the matvec is the P0 part of this item.**
- **[P1] Runtime-M matmul** (one xclbin per `(K,N)` covering all M). The current designs bake
  M/K/N into static `aiex.npu.dma_memcpy_nd` descriptors, so we ship one kernel per (M-tile,K,N)
  and the host tiles M (looping `ceil(M/tile)` launches). A design with runtime-parameterized DMA
  sizes would collapse the kernel set into one kernel per `(K,N)`. This is IRON surgery but
  eliminates a whole class of shape gaps.

## 2. Matmul shape/tiling — real constraint + remaining gaps

**`N % 128 == 0` is NOT the constraint** (that's only the default `n_tile=32 × 4 cols`). The real
buildability rule (verified empirically — see `docs/ggml-xrt-plan.md` §7a for the authoritative
statement and current per-model shape table):
1. `N % (n_tile × n_aie_cols) == 0` (tileability);
2. `n_tile % 16 == 0` for bf16 (smallest n-tile is 16);
3. output DMA stride `N × n_tile ≤ 1048576` (the BD `[1:1048576]` stride range).

A 2-column (`_2c`) design with a smaller n-tile makes many non-128-multiple N buildable, and host
**N-padding** covers the rest. Most shapes that were previously thought blocked are now **built**
(e.g. Gemma4 `2816×2112` `_2c`, Qwen3.5-35B-A3B dense FFN `2048×4304` via an N-padded `4352`
kernel, a `2176` N-block kernel, and several Gemma4 variant tiers). Defer to the plan-doc table for
the live set — don't duplicate it here.

Remaining Linux-side asks:
- **[P1] Keep emitting ≤stride-limit N-block kernels** (e.g. `2176 = 128×17`) for the K's needed
  by very-large-N FFNs (`N=17408` on Qwen3-14B/Qwen3.5-27B). `N=17408` can't be one dispatch even
  at the smallest n-tile (constraint 3), so the host **N-tiles** over these block kernels
  (Windows-side dispatch — handoff step 9); the kernel ask is just to provide the block shapes.
- **[P2] lm_head (`N=151936`)** — CPU today; would use the same host N-tiling if ever moved.

## 3. MoE (Gemma4-26B-A4B, Qwen3.5-A3B, general MoE)

The expert FFNs are ordinary GEMMs — most shapes are (or can be) built as in §2. What's missing is
the batched dispatch and the routing:

- **[P1] Native grouped/indexed expert GEMM (`MUL_MAT_ID`).** The host can already implement
  MUL_MAT_ID by grouping tokens per expert and calling the single-expert kernel per group, so this
  is not strictly blocking — but a kernel that takes the stacked expert weights `[K,N,n_expert]` +
  an id/offset list and does the batched expert matmul in a single dispatch (instead of a host
  loop over experts) would be cleaner.
- **[P2] Router/gating on NPU** (top-k / `ARGSORT`, router softmax, weighted combine). Cheap ops;
  Vulkan already has `topk_moe`. Only worth an NPU kernel if we pursue full-layer residency. Low
  priority — leave on GPU.
- Expert weight shapes to ensure exist (both tiers + gemv): see §2 per-model lists.

## 4. RoPE (currently: works via a NORMAL kernel + host permutation)

The prebuilt `rope_{128,256}` are mlir-aie `ml/rope` = **NORMAL/GPT-J** (adjacent-pair) kernels
that take a full **cos/sin LUT** (no positions, no baked freqs). We run NEOX by host-permuting
in/out and computing ggml's cos/sin cache on the host. Wishlist:

- **[P1] Native NEOX kernel** — pairs `(x[j], x[j+half])` directly, so the host doesn't permute
  in/out every call. (Qwen3/Llama/GLM are all NEOX.)
- **[P1] On-device frequency generation** — a kernel that takes **positions (+ freq_base,
  freq_scale, and optional freq_factors)** and generates the rotation internally, instead of the
  host building and uploading a full per-element cos/sin table each call.
- **[P2] Partial rotation** (`n_dims < head_dim`) and **larger seq tiles** (currently 32).
- **[P2] MROPE / vision-rope** variants if we target multimodal.
- Head dims present: 128, 256. Add others if new models need them.

## 5. Norm / activation kernels

- **[P0] Parameterized epsilon for RMS_NORM.** The kernel bakes **eps=1e-5**; Qwen3 uses 1e-6,
  Gemma uses its own. Either parameterize eps as a scalar kernel arg, or build per-eps variants.
  (Currently negligible vs bf16 error, but it should be correct.)
- **[P1] Fused RMSNorm × weight (gamma).** ggml keeps the norm and the weight-multiply as separate
  ops (`RMS_NORM` then `MUL`); the kernel currently does gamma=1 (norm only). A fused norm+scale
  kernel saves an op/round-trip per norm. Gemma also does `(1+weight)` scaling — support a bias/one
  offset.
- **[P1] More RMS_NORM row sizes.** Have 128/256/2048/2816/5120. Add per new hidden/head dims
  (e.g. 3072, 4096, 5120 variants as models require); keep the 32-row seq tile.
- **[P1] Fused SwiGLU / GeGLU.** Qwen uses SwiGLU = `silu(gate) * up`; Gemma uses GeGLU =
  `gelu(gate) * up`. Today that's SILU/GELU + a separate elementwise MUL (MUL not yet a kernel). A
  fused gate-activation-times-up kernel removes an op and a round-trip.
- **[P2] SCALE / ADD / MUL elementwise kernels.** Trivial (build like `silu_16384`); needed if we
  want attention/residuals fully NPU-resident (see §6). Also enables non-fused GeGLU/SwiGLU.
- **[P2] Elementwise tile sizes** — silu/gelu are single `*_16384` tiles the host loops over; fine,
  but a couple of sizes could reduce tail waste.

## 6. Attention primitives (needed for full-layer NPU residency)

To keep a contiguous range of layers fully on the NPU (fewer backend crossings), the NPU needs
*every* op in a layer. Missing:

- **[P1] SOFT_MAX** with causal/sliding-window mask and scale (Gemma4 has sliding-window
  attention). Dynamic over seq len.
- **[P1] Dynamic-seq attention-score matmuls** — `Q·Kᵀ` and `scores·V` with `K=head_dim=128` and
  **N = seq len (dynamic)**. Needs runtime-N (§1) or per-seq-len kernels; this is the main blocker
  for on-NPU attention.
- **[P2] KV-cache-friendly layouts** — if attention runs on NPU, the K/V cache copy/view ops and
  their strides need to be NPU-consumable.

Note: this section is for the full-layer-residency end-state (keeping a contiguous range of layers
entirely on the NPU to avoid backend crossings), not near-term.

## 7. Quantization — native quantized matmul (the memory fix)

- **[P0] Native quantized matmul: NPU consumes quantized weights directly (dequant on-chip).**
  Motivation (concrete): the NPU only takes BF16 today, so the host dequantizes each quantized
  weight to BF16 (**~4× the Q4_K size**) and holds it resident (cached per weight). For a full
  model that BF16 set dominates RAM (e.g. ~2.8 GB for Qwen3-1.7B's NPU matmuls, on top of the
  ~1 GB Q4_K). A kernel that reads the **quantized blocks from DDR and dequantizes inside the AIE
  tile** removes the 4× expansion entirely (host uploads the raw quantized weight; no BF16 copy,
  no host dequant). This is the real fix for the memory pressure.

  **Build guidance for the mlir-aie side:**
  - **Start with Q4_0** (simplest block: 32 elems, one f16 scale + 32×4-bit) to prove the on-chip
    dequant→matmul path, then do **Q4_K** (the shipped model: `block_q4_K`, 256-elem superblock =
    8×32 sub-blocks, 6-bit scales+mins packed, one f16 `d` + f16 `dmin`; see ggml `ggml-common.h`
    `block_q4_K` and the reference `dequantize_row_q4_K` for the exact unpack math). Then Q6_K,
    Q8_0 if useful. Scope: Q4-and-below is the priority.
  - **Kernel shape:** reuse the existing `whole_array` matmul tiling (4-col preferred, per §1/§2),
    but the A/weight operand is the packed quant format: the core reads a quant block, unpacks it
    to bf16 in local memory (aie2 vector intrinsics), and feeds the existing bf16 MAC. Weight
    stays quantized in DDR the whole time.
  - **ABI:** same `MLIR_AIE(opcode=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5)` convention.
    Only the **A (weight) operand dtype changes** to the packed quant layout; B (activation) stays
    bf16, C stays f32. **The host must upload the weight in the kernel's expected quant block
    layout** — document exactly what layout/tiling the DDR-side weight buffer must have (e.g.
    per-tile block ordering) so the host can arrange it (likely the raw ggml row-major quant bytes,
    but confirm any tile reordering the DMA expects). Keep it **untransposed vs transposed**
    explicit per design (the bf16 path needs N×K→K×N; state what the quant path needs).
  - **Naming:** `mul_mat_aie2_<qtype>_f32_<M>x<K>x<N>_<cols>c.xclbin` with `<qtype>` ∈
    `{q4_0, q4k, q6k, q8_0}` (matching the host dtype token we'll add). Also emit a M=1 gemv
    variant `..._1x{K}x{N}_gemv.xclbin` for decode (same as the bf16 gemv, quant weight).
  - **Shapes:** the Qwen3-1.7B set first (`2048×2048, 2048×1024, 2048×6144, 6144×2048`, both tiers
    + gemv), then the other target models' shapes (defer to plan §7a table).

  Host counterpart (our side, once the kernels exist): add the qtype dtype tokens to
  `ggml_xrt_dtype_token`, have `supports_op`/`find` prefer a native-quant kernel when the weight
  is that quant type, and in the dispatch upload the **raw quantized weight bytes** (no
  dequant/BF16 cache) — which also removes the `GGML_XRT_LOW_MEM` tradeoff. Until then, weights go
  through host BF16 dequant (default cached/fast; `GGML_XRT_LOW_MEM=1` for the low-RAM path).

### STATUS — Q4_0 decode gemv **HARDWARE-VALIDATED** (2026-07-18, NPU Phoenix)

All three built shapes run on the NPU and match the CPU reference **essentially exactly**
(NRMSE 0.00000, max_abs_err ≤ 2.4e-4 on sums of magnitude ~20–70; mean(npu−cpu) ≈ 0,
mean(npu/cpu) = 1.000000 ± 2e-6):

| Shape (K×N) | xclbin | NRMSE | max_abs_err |
|---|---|---|---|
| 2048×2048 (Q/O) | `…_1x2048x2048_gemv` | 0.00000 | 7e-5 |
| 2048×1024 (K/V) | `…_1x2048x1024_gemv` | 0.00000 | 6e-5 |
| 6144×2048 (down) | `…_1x6144x2048_gemv` | 0.00000 | 2.1e-4 |

Verified across seeds {1,2,3,7,9} and an all-ones activation (bit-exact, max_abs_err 0.0).
**No K-proportional bias**: K=6144 (192 blocks/row) shows the same ~0 mean error as K=2048
(64 blocks/row), so the float accumulation in `mv_q4.cc` is sound. The repack contract below
is confirmed correct as written — the host implementation needed no deviation from it.

Harness: `C:\dev\xrt-sdk\work\q4_gemv_check.cpp` (+ `cc_q4_gemv.bat`, `run_q4_gemv.bat`),
raw XRT, bypasses the ggml backend since the q4 dispatch branch is not wired yet. It
quantizes a random f32 weight with `ggml_quantize_chunk(GGML_TYPE_Q4_0, …)`, repacks per the
contract, and references against a host dequant of the *same* blocks (so q4 quantization
error is excluded and the kernel is measured on its own).

Original note (pre-validation):

Kernel authored: `kernels/aie2/mv_q4.cc` (dequant math ported exactly from ggml
`dequantize_row_q4_0`, cross-checked vs ggml-hexagon/Vulkan; products promoted to float per
the gemv-bias lesson) + `kernels/gemv_q4.py` design + `kernels/build-q4-gemv.sh`. Built for
Qwen3-1.7B decode shapes: `mul_mat_aie2_q4_0_f32_1x{2048x2048,2048x1024,6144x2048}_gemv.xclbin`
(gate/up N=6144 > 2048 → no gemv, same as bf16). Compiles clean; **not run on hardware**.

**Host REPACK contract (the host must produce exactly this):** ONE weight buffer, row-major
`[N][K/32][20]` bytes — per 32-elem q4_0 block per output row: **16 nibble bytes** (ggml
`block_q4_0.qs`, unchanged) then a **4-byte f32 scale** (host converts ggml f16 `d`→f32).
(Chosen because raw 18-byte ggml blocks are DMA-hostile and 3 separate DDR streams exceed the
shim's 2 read-DMA channels — so nibbles+scale are one interleaved stream; this mirrors
ggml-hexagon's "repacked" quant weights.) Activation B stays bf16 `[K]`; C is f32 `[N]`.
**Weight is NOT transposed** (native `[N,K]`, unlike the bf16 tiled matmul). ABI unchanged:
`kernel(op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5)`.

**Host counterpart still TODO** (now unblocked — kernels are validated): add a `q4_0` dtype
token; in `supports_op`/`find`, when the
weight is Q4_0 and it's a decode (M=1) op, prefer the `..._q4_0_f32_..._gemv.xclbin` and
upload the **repacked weight (no BF16 dequant/cache)** — this is the memory win. A separate
q4-gemv dispatch branch is needed (different A operand: repacked quant, no transpose).
**Q4_0 result: hardware-validated (NRMSE 0.0, bit-exact vs host q4_0 dequant, no K-bias).**

### STATUS — Q4_K decode gemv **HARDWARE-VALIDATED** (2026-07-18, NPU Phoenix)

All three built shapes run on the NPU and match the CPU reference **essentially exactly**
(NRMSE 0.00000; max_abs_err ≤ 3.1e-4 on sums of magnitude ~20–70; mean(npu−cpu) ≈ 0,
mean(npu/cpu) = 1.000000 ± 1e-6):

| Shape (K×N) | xclbin | NRMSE | max_abs_err |
|---|---|---|---|
| 2048×2048 (Q/O) | `…_q4k_…_1x2048x2048_gemv` | 0.00000 | 8e-5 |
| 2048×1024 (K/V) | `…_q4k_…_1x2048x1024_gemv` | 0.00000 | 7e-5 |
| 6144×2048 (down) | `…_q4k_…_1x6144x2048_gemv` | 0.00000 | 3.1e-4 |

Verified across seeds {1,2,3,7,9,11} and all-ones activations. **No K-proportional bias**:
K=6144 (24 superblocks/row) shows the same ~0 mean error as K=2048 (8/row). The 6-bit packed
scale/min unpacking (`get_scale_min_k4`, including the `j>=4` split-nibble branch, which every
superblock exercises via chunks 2–3) and the affine `y = d·q − min` dequant are both correct.
The 148-byte repack contract below is confirmed correct as written.

Harness: the same `C:\dev\xrt-sdk\work\q4_gemv_check.cpp` extended with a Q4_K mode
(auto-detected from `q4k` in the xclbin filename; `run_q4k_gemv.bat`). Q4_0 re-verified after
the change — still exact. Note `block_q4_K` is `{d, dmin, scales[12], qs[128]}` = 144 B, so the
repack is a field reorder plus f16→f32 widening of `d`/`dmin`, exactly as specified.

Original note (pre-validation):

Kernel authored: `kernels/aie2/mv_q4k.cc` (dequant ported exactly from ggml
`dequantize_row_q4_K` + `get_scale_min_k4`: 4 chunks × 64, two 6-bit scale/min pairs per
chunk, affine `y = d·q − min`; float accumulation) + `kernels/gemv_q4k.py` +
`kernels/build-q4k-gemv.sh`. Built for Qwen3-1.7B decode shapes
`mul_mat_aie2_q4k_f32_1x{2048x2048,2048x1024,6144x2048}_gemv.xclbin`. Compiles clean; **not
run on hardware** — verify with `q4_gemv_check.cpp` (Q4_K mode); its constant-offset/scale/
row-permute/truncated-prefix diagnostics should apply directly.

**Host REPACK contract (Q4_K):** ONE buffer, row-major `[N][K/256][148]`. Per 256-elem
superblock per output row, **148 bytes** = `qs[128]` (raw `block_q4_K.qs`) + `scales[12]`
(raw 6-bit-packed) + **f32 d** + **f32 dmin** (host converts ggml f16 `x.d`/`x.dmin`→f32).
This is a **field reorder** of ggml `block_q4_K` (`{d,dmin,scales,qs}`) with the two f16
scales widened to f32 — same principle as Q4_0. K must be a multiple of 256. B bf16, C f32,
weight NOT transposed, ABI unchanged. Host counterpart: add a `q4k` dtype token; same
native-quant dispatch branch as Q4_0 (upload repacked weight, no BF16 dequant/cache).

**Q4_K result: hardware-validated (NRMSE 0.0, bit-exact, no K-bias; split-nibble
`get_scale_min_k4` branch exercised millions of times).**

### STATUS — Q6_K decode gemv **HARDWARE-VALIDATED** → Q4_K_M decode fully covered (2026-07-18)

All three built shapes run on the NPU and match the CPU reference **essentially exactly**
(NRMSE 0.00000; max_abs_err ≤ 2.3e-4; mean(npu/cpu) = 1.000000 ± 1e-6):

| Shape (K×N) | xclbin | NRMSE | max_abs_err |
|---|---|---|---|
| 2048×2048 | `…_q6k_…_1x2048x2048_gemv` | 0.00000 | 7e-5 |
| 2048×1024 | `…_q6k_…_1x2048x1024_gemv` | 0.00000 | 9e-5 |
| 6144×2048 | `…_q6k_…_1x6144x2048_gemv` | 0.00000 | 2.3e-4 |

Verified across seeds {1,2,3,7,9} and all-ones activations; no K-proportional bias. The
6-bit reassembly (4-bit `ql` + 2-bit `qh` at all four shift positions 0/2/4/6, −32 offset) and
the strided int8 scale indexing `scales[is + {0,2,4,6}]` are both correct. The 212-byte repack
contract is confirmed as written.

**Q4_K_M decode is now fully validated native-quant**: Q4_K (attn_q/k/o, ffn_gate/up) +
Q6_K (attn_v, ffn_down) both pass on all three Qwen3-1.7B decode shapes. The only Q4_K_M tensor
still off the native-quant path is `output` (Q6_K, N=151936) — unchanged, stays CPU/GPU since no
gemv exists at that N. All nine kernels (Q4_0/Q4_K/Q6_K × 3 shapes) re-verified together after
the harness gained Q6_K mode.

Original note (pre-validation):

Q4_K_M = Q4_K (attn_q/k/o, ffn_gate/up) + **Q6_K** (attn_v, ffn_down, output). Built the Q6_K
gemv to complete it: `kernels/aie2/mv_q6k.cc` (ported exactly from ggml `dequantize_row_q6_K`:
4-bit `ql` + 2-bit `qh` → 6-bit quant −32, int8 `scales[is+{0,2,4,6}]`, ×d; float accum) +
`kernels/gemv_q6k.py` + `kernels/build-q6k-gemv.sh`. Built for Qwen3-1.7B
`mul_mat_aie2_q6k_f32_1x{2048x2048,2048x1024,6144x2048}_gemv.xclbin`. Compiles clean; **not
run on hardware** — verify with `q4_gemv_check.cpp` (add a Q6_K mode; the harness already
auto-detects by filename).

**Q6_K repack contract:** ONE buffer `[N][K/256][212]`. Per 256-elem superblock per row, 212
bytes = `ql[128]` + `qh[64]` + `scales[16]` (raw int8) + **f32 d** (host converts ggml f16
`x.d`→f32). This is ggml `block_q6_K`'s native field order with only `d` widened — a near-copy
(unlike Q4_K's reorder). K%256==0; B bf16; C f32; weight NOT transposed. Same native-quant
host dispatch branch (q6k dtype token; upload repacked weight, no BF16 cache). With this,
Qwen3-1.7B-Q4_K_M decode is fully native-quant on the NPU except the `output` tensor (Q6_K,
N=151936 — stays CPU/GPU, or would need N-tiling).

### STATUS — fused tiled-prefill quant matmul **BUILT (UNVALIDATED)** (2026-07-18)

The tiled-prefill quant matmul is done for all three types. Design: `mm_q4k.py` / `mm_q6k.py`
/ `mm_q4.py` + cores `aie2/mm_q4k.cc` / `mm_q6k.cc` / `mm_q4.cc`; recipes `build-q4k-mm.sh`,
`build-q6k-mm.sh`, `build-q4-mm.sh`. `C[M,N] += A_act[M,K] . dequant(W)`, C f32, A bf16.

- **The ICE fix (option a).** A core-local `Bl1[DIM_K*DIM_N]` (16 KB bf16) dequant scratch
  ICEs llvm-aie — an AIE core can't hold a local array that big. The fix: the core takes
  `Bl1` as a **pointer arg to a design-owned L1 `aie.buffer`** declared on the compute tile
  (`Bl1 = buffer(compute_tile, bl1_ty, ...)`). Core dequants each superblock record and
  scatters bf16 into `Bl1` in the mmul B sub-tile layout, then runs the validated
  `matmul_vectorized_4x4` MAC. **B is delivered RAW (no memB transform)**; A/C keep the
  standard mmul transforms.
- **k-tile forced to 256** (one Q4_K/Q6_K superblock, or 8 Q4_0 blocks) so scales/qh never
  split across tiles. Same host repack contracts as the gemvs (`[N][K/256][148|212]`, or
  `[N][K/32][20]` for Q4_0).
- **L1 budget.** `Bl1` (k*n*2) + double-buffered A/B/C must fit the 64 KB bank. `m=32,n=32`
  overflows (`Bl1` alone is 16 KB); the recipes fall back `32×32 → 32×16 → 16×32`, so all
  Qwen3-1.7B shapes build at `m=16,n=32` or `m=32,n=16`. Host chunks tokens to `m` and pads
  the decode M=1 → m (this kernel can also serve decode, though the gemvs are faster there).
- Built for Qwen3-1.7B shapes (2048×2048, 2048×1024, 6144×2048) for Q4_0/Q4_K/Q6_K.

**HW validation (2026-07-18, Windows agent):** 8 of 9 pass in the expected bf16 band
(NRMSE ~0.0034–0.0040, no K-proportional drift). One shape was broken and is now worked
around:

- **q6k 6144×2048 M=32 was WRONG on rows 16–31** (the second m-tile): rows 0–15 correct
  (per-row NRMSE ~0.003), rows 16–31 garbage (per-row NRMSE ~0.60, max abs err ~113), a
  clean split at the m-tile boundary, reproduced across seeds. Q4_0/Q4_K at the same shape
  and Q6_K at K=2048 all pass.
- **Root cause: the multi-m-tile B re-stream.** `single_core`'s runtime issues the FULL
  packed-weight DMA once per m-tile (weight is identical across token rows, so it's
  re-streamed). The FIRST stream is always correct; only the SECOND consecutive stream
  corrupts, and only for the largest transfer — q6k 6144×2048 is the sole shape whose
  per-m-tile B stream exceeds ~8 MB (10.4 MB; q4k=7.3, q4_0=7.9, q6k@2048=3.5). Every static
  DMA limit (shim step≤2²⁰, size0/1≤1023, iter≤64) is satisfied and the two B DMAs are
  byte-identical, so it's a runtime/resource edge on back-to-back >8 MB streams, not a
  descriptor error.
- **Fix (shipped): serialize m-tiles, keep M=32.** A `dma_wait(outC)` between m-tiles (reusing
  bd_ids 0/1/2) makes each m-tile's B stream fully drain before the next starts — so the second
  stream behaves like the always-correct single stream, at M=32. Enabled via
  `mm_q*.py --serialize-mtiles`; the recipes auto-set it when `N*(K/256)*REC > 8 MB`. Smaller
  shapes keep the faster ping-pong path (byte-identical to the validated ones). `q6k 6144x2048`
  is now `32x6144x2048_mm` again (M=32), the earlier `16x...` single-m-tile workaround dropped.
  Trade-off: serialize loses the DMA/compute ping-pong overlap for the affected shape.
  **UNVALIDATED — verify on-device.** If serialize still fails, fall back to the M=16 single-
  m-tile build (recoverable from git commit `7eac7e55`) or the host BF16-tiled path.
- **Deeper fix (TODO, real perf win): weight-stationary B.** Hold each n-tile's packed weight
  resident in the mem-tile (L2, 512 KB — a 159 KB n-tile fits) and replay it to the core across
  m-tiles, so the full weight streams from DDR **once** total (not `M_div_m×`). Halves prefill B
  traffic. No reference design does this (stock `single_core` also re-streams B per m-tile), so
  it's novel L2-orchestration best done with HW in the loop — deferred.

**Still UNVALIDATED:** the serialize-m-tiles q6k 6144 fix. Verify (`q4_gemv_check.cpp`, mm mode),
then regenerate the overlay/ELF set (§8) for these shapes.

Decode (gemv) + prefill (fused matmul) for Q4_0/Q4_K/Q6_K are now both complete on the Linux
build side (8/9 HW-validated; the 9th has a shipped single-m-tile workaround pending validation).

## 8. Shared hw_context across kernels (fixes the 5-context limit)

- **[P0] One `hw_context` for many kernels.** The Phoenix NPU allows only ~5 concurrent
  `hw_context`s (the 6th `register_xclbin`/`hw_context` create fails `0xc01e0009`). Each of our
  kernels is a separate `.xclbin` = a separate context, so a full model (>5 distinct kernels)
  overflows; the host currently works around it with an LRU context cap (`GGML_XRT_MAX_CONTEXTS`,
  default 4) that evicts + reloads, which **thrashes** when many kernels are used. The real fix is
  to stop needing a context per kernel:
  - **Preferred (FastFlowLM-proven): a base overlay xclbin + per-op instruction *modules*.** Build
    all ops against one common AIE overlay/config, ship each op as an instruction sequence, and on
    the host create **one** `hw_context` from the overlay and load per-op work via
    `xrt::ext::kernel` + `xrt::module` (aiebu ELF from the transaction blob) into that single
    context. No per-kernel context → no 5-limit, no LRU thrashing. (This is the `kernel(3,0,0,…)`
    ext-kernel path, vs our current `register_xclbin`→`hw_context`→`kernel`.) Host change:
    adopt the ext-kernel/module dispatch for a shared context.
  - **Alternative: one xclbin containing multiple kernels** (matmul + rms + silu + rope packed into
    a single NPU config, addressed by kernel name) → one `register_xclbin` + one `hw_context`,
    `xrt::kernel(ctx, "<name>")` per op. Simpler host change but the packed kernels must co-fit the
    array/columns.
  Either removes the biggest blocker to running the full op set on the NPU concurrently.

### FEASIBILITY VERDICT (Linux-side study)

**Approach 1 (overlay + per-op ELF modules) is feasible and is the right fix — toolchain
supports it.** `aiecc --aie-generate-xclbin --aie-generate-elf` emits the overlay xclbin +
an instruction **ELF module** (verified: built xclbin+ELF for two shapes here). Host pattern
is the mlir-aie `vector_scalar_add` example verbatim: `register_xclbin` + `hw_context(uuid)`
**once**, then `xrt::elf`→`xrt::module`→`xrt::ext::kernel(ctx, mod, name)` per op, call
`kernel(opcode, 0, 0, bo…)` (instrs come from the module). `runlist` batches ops on one ctx.

**Key structural finding (measured, not assumed):** the overlay = array config **+ the core
program**, and the core bakes its loop bounds — so there is NO single globally-shape-independent
overlay for free. BUT the split is favorable for decode:
- **gemv (decode) core loops `range_(0xFFFFFFFF)` with only a `K_div_k` inner count → the
  overlay depends on K ONLY, not output N.** Proven: same-K gemv shapes (e.g. K=2048, N=2048
  vs 1024) generate a **byte-identical device/overlay MLIR**; only the runtime instruction
  sequence differs. So **decode needs one hw_context per distinct K**, serving every projection
  of that K via cheap per-shape ELF modules. Qwen3-1.7B decode: K∈{2048,6144} → **2 contexts**
  for all matmuls — under the ~5 limit, no LRU thrash. This directly fixes the hot path.
- **tiled prefill matmul** bakes M,N in the core loop → overlay per (M,N,tile) → far less
  shareable via this route. Prefill is less context-pressured at steady state (runs once), so
  accept the LRU cap there for now.

**Approach 2 (spatially packing many kernels in one xclbin) is limited:** one `whole_array`
matmul already uses all 4×4 tiles, so matmul can't spatially co-locate with other ops; only a
few single-core ops (gemv/rms/silu/rope) could share tiles. Useful to bundle those, secondary
to Approach 1.

**Endgame (one overlay for everything):** make the cores use **runtime loop counts** (RTP /
dynamic `K_div_k`, `M_div_m`) — the §1 runtime-M/N item — so a single overlay serves any K/N
and the whole model is ONE context. That's the FastFlowLM structure; it's real IRON work but
the definitive fix.

**Recommended sequencing:** (1) restructure the **gemv/decode** build to emit one overlay per
distinct K + per-shape instruction ELFs, host adopts the module dispatch → fixes the decode
5-context problem now (biggest pain). (2) runtime-loop cores for a universal overlay (endgame).
(3) multi-kernel xclbin to bundle small ops if still needed.

### BUILT — decode overlay + per-shape ELF module set (step 1 above) — UNVALIDATED

Compiled on Linux/WSL, **NOT executed on an NPU** — correctness is validated later on Windows.
This delivers step 1 of the sequencing: one overlay per `(dtype,K)` group + a lightweight
instruction **ELF module** per output-`N` shape.

> **REGENERATED 2026-07-18 — quant overlays are now 4-COLUMN SIMD (must-do after the gemv perf pass).**
> The quant decode kernels were promoted to 4-column SIMD (vectorized MAC + `aie::unpack` cores
> `mv_q4*.cc`/`mv_q6k.cc`, built from the new `gemv_mc.py --cols 4`; ~137× on q6k down_proj). The
> old frozen single-column quant overlays (commit `25ff872f`) made the host run the OLD SCALAR core
> whenever `GGML_XRT_OVERLAY` was on (≈24× slower), so overlays were force-OFF as a stopgap.
> `build-overlay-elf.sh` now generates **quant** (`q4_0`/`q4k`/`q6k`) MLIR via
> `gemv_mc.py --qtype <dt> --cols 4` (bf16 stays single-column via `gemv.py`, `mv.cc` unchanged), and
> the whole set is regenerated: **17 overlays, 31 shape ELFs.** Quant overlays are ~26–38 KB (4-core
> program) vs bf16 ~13 KB; quant ELFs are 2432 B (4-col instr stream) vs bf16 1120 B. `manifest.json`
> now tags every overlay/shape with `cols` (4 quant / 1 bf16) and `core` (`simd_4col` / `scalar_bf16`)
> so the host can confirm the overlay path carries the SIMD core and **re-enable `GGML_XRT_OVERLAY`**
> once validated on HW. Quant shapes built: `q4_0` K∈{2048,6144}; `q4k` K=2048 N∈{1024,2048,**6144**}
> + K=6144 N=2048; `q6k` K∈{2048,6144} — matching the promoted standalone
> `mul_mat_aie2_<dt>_f32_1x{K}x{N}_gemv.xclbin`.
>
> **VERIFY — the overlay/ELF split still holds at 4 columns (the key question): YES.** With
> `gemv_mc.py --mode full` the per-column core loops only `range_(K_div_k)` (depends on **K**, not N);
> all N-dependence lives in the `runtime_sequence` DMA descriptors → the per-N ELF. Measured: for a
> fixed `(dtype,K)` at cols=4, two different N emit **byte-identical device/core MLIR** (checked q6k
> K=2048 N=1024 vs N=2048), and every additional N builds ELF-only with
> `--xclbin-input <that overlay>` and loads by construction. So the group key is unchanged —
> `(dtype,K)` at a fixed `cols=4` — and the per-model `hw_context` count (`max=3` across the lineup)
> is unchanged; only the core inside each overlay/ELF is now the SIMD 4-col version.

**Regenerate:** `src/ggml-xrt/kernels/build-overlay-elf.sh` (no args). It enumerates the
ground-truth shape set by globbing the existing prebuilt gemv xclbins
(`prebuilt/**/mul_mat_aie2_<dt>_f32_1x{K}x{N}_gemv.xclbin`, `<dt>∈bf16/q4_0/q4k/q6k`), excludes
the out-of-scope `qwen3.5-122b-a10b` / `qwen3.5-397b-a17b` dirs, groups by `(dtype,K)`, then:
first shape of a group → `aiecc.py --aie-generate-xclbin --aie-generate-elf … --xclbin-name=…
--elf-name=…`; every other `N` of that group → `aiecc.py --xclbin-input=<overlay> --aie-generate-elf
… --elf-name=…` (ELF only, against the one overlay). Built set: **17 overlays, 31 shape ELFs**
(quant via `gemv_mc.py --cols 4`; see the 2026-07-18 regeneration note above).

**Directory layout — `src/ggml-xrt/kernels/prebuilt/overlays/`:**
```
<dtype>_k<K>_overlay.xclbin     one per (dtype,K) group — register once per group
<dtype>_1x{K}x{N}_gemv.elf      one per shape — the instruction module loaded into that context
manifest.json                   shape → {overlay, elf, kernel_name} + per-model overlay counts
```
The overlay for a group is byte-identical across its N's except for the auto-generated xclbin
UUID (two 16-byte UUID fields); the host reads the UUID from whichever overlay file it registers,
so this is a non-issue. All ELFs of a group are built with `--xclbin-input <that overlay>`, so
they are load-compatible with it **by construction**.

**`manifest.json`** has:
- `overlays[]`: `{dtype, K, overlay, elf_count}` — the 17 groups.
- `shapes[]`: `{dtype, K, N, overlay, elf, kernel_name:"MLIR_AIE"}` — host maps a requested
  shape → (overlay to register, ELF module to load).
- `models{}`: per model in the lineup, `overlays_by_dtype` + `max_overlays_any_dtype`. A live
  deployment uses ONE weight dtype, so its `hw_context` count = distinct K for that dtype.
  **Max across the whole lineup = 3** (Qwen3.5-0.8B, Qwen3.5-35B-A3B, Gemma4-E2B), well under
  the ~5 limit — the point of the restructuring. (Gemma4-31B has no gemv shapes; it is
  prefill-only in the prebuilt set.)

**Host load protocol** (mlir-aie `vector_scalar_add` pattern; per shape look it up in the
manifest):
```
# once per (dtype,K) group actually used by the model:
dev.register_xclbin( xrt::xclbin(<overlay>) )
ctx = xrt::hw_context(dev, overlay.get_uuid())
# per shape in that group:
mod = xrt::module( xrt::elf(<elf>) )
k   = xrt::ext::kernel(ctx, mod, "MLIR_AIE")     # kernel_name from manifest
# dispatch (instrs come from the module, not a bo):
k(3, 0, 0, A_bo, B_bo, C_bo)                      # opcode=3; A=weight, B=activation, C=output
# xrt::runlist(ctx) can batch several shape kernels on the one context.
```
This replaces the current per-shape `register_xclbin → hw_context → kernel` (one context per
shape) on the decode path, removing the 5-context overflow and the LRU thrash.

### BUILT — activation (silu/gelu) overlay + per-length ELF set — UNVALIDATED (2026-07-18)

Extends step 1 to the single-core **activation** ops. Compiled on Linux/WSL, **NOT executed on
an NPU** — correctness is validated later on Windows.

**Per-op collapsibility (measured on this tree, not assumed):** each op's stock mlir-aie `ml/`
design was generated at two sizes and the device/core MLIR diffed (everything outside
`aie.runtime_sequence` = the overlay). Verdict:

| op | collapsible? | why |
|---|---|---|
| **silu** | **YES** | `ml/silu` uses a FIXED L1 line buffer (`line_size=1024`); the transfer `length` only rewrites the `runtime_sequence` DMA taps. Device/core MLIR is byte-identical across lengths; the two xclbins differ in **66 bytes** (the 2×16-byte UUID fields + the xclbin JSON metadata). |
| **gelu** | **YES** | identical structure to silu (shares the `ml/gelu` design + fixed 1024 line). |
| **rms_norm** | **NO** with stock `ml/rmsnorm`; **YES** with the RTP redesign | Stock `ml/rmsnorm` sizes the ObjectFifo L1 buffer as `memref<embedding_dim×bf16>` **and** bakes `cols` as an `arith.constant` (both in the overlay). The RTP redesign (`rms_norm_rtp.py`) fixes both — see the RTP section below — so ONE overlay serves ALL `(seq, cols)`. |
| **rope** | **NO** with stock `ml/rope`; **YES** with the RTP redesign | Same stock issue (L1 buffer + baked `dims`). `rope_rtp.py` applies the identical RTP + fixed-buffer fix → ONE overlay serves ALL `(seq, head_dim)`. |

**Regenerate:** `src/ggml-xrt/kernels/build-op-overlay-elf.sh` (no args). Builds silu, gelu, and the
packed rms_norm+rope set. For silu/gelu: first length → `aiecc.py --aie-generate-xclbin
--aie-generate-elf …`; every other length → `aiecc.py --xclbin-input=<overlay> --aie-generate-elf …`
(ELF only). For rms/rope see the packed section below. Built set: **3 overlays, 40 shape ELFs.**

### BUILT — rms_norm & rope: RTP collapse + spatial packing into ONE hw_context — UNVALIDATED

The stock `ml/rmsnorm`/`ml/rope` don't collapse (row length baked into both the L1 buffer and an
`arith.constant cols`). Both are fixed the same way, in `rms_norm_rtp.py` / `rope_rtp.py`:
- **FIXED max L1 buffer** (`LMAX`: rms 6144 ≥ max row 5376; rope 256 ≥ max head_dim) → L1
  allocation is shape-independent.
- **`cols`/`dims` as a RUNTIME PARAMETER** — an L1 `aie.buffer` written by the runtime sequence
  (`aiex.npu.rtp_write`, lowered from `rtp[0]=n`), read by the core (`rtp[0]`), instead of a baked
  constant. A lock (`set_lock` in the sequence / `use_lock(Acquire)` in the core) orders the
  RTP-write before the read.
- **core loops `range_(0xFFFFFFFF)`** (gemv pattern) so the row count (`seq`) isn't baked either.

**Measured:** the device/core MLIR (everything outside `aie.runtime_sequence`) is now **byte-identical
across every `(seq, cols)` / `(seq, dims)`** — `cols`/`dims` appears only as `aiex.npu.rtp_write(...)`
inside the runtime sequence (→ the ELF). So ONE overlay serves ALL shapes of each op, exactly like
silu. Verified by building each shape's ELF `--xclbin-input` the one overlay (all succeed).

**Spatially PACKED (Approach 2) — rms_norm + rope share ONE hw_context.** `rms_rope_packed.py`
(low-level/placed style, because the high-level IRON API rejects an ObjectFifo whose producer
endpoint is never driven) puts **both** cores in ONE `aie.device(npu1)`: rms on `tile(0,2)` (col 0),
rope on `tile(1,2)` (col 1), each with its own shim DMA path, fixed L1 buffers, RTP scalar and lock.
Both cores + all fifos + both RTP buffers/locks are **always declared**, so the device/core config
(= the overlay) is **identical** regardless of which op a dispatch drives; `--op {rms,rope}` controls
only the `runtime_sequence` (which RTP is written, which lock is set, which DMAs run) → that lives in
the ELF. The idle op's core just blocks on its unset lock — harmless. Verified: overlay built from the
rms variant; every rms ELF (per `cols`) **and** every rope ELF (per `dims`) load `--xclbin-input
rms_rope_overlay.xclbin` and build clean.

**Directory layout — `src/ggml-xrt/kernels/prebuilt/op-overlays/` (3 overlays, 40 ELFs):**
```
silu_overlay.xclbin        + silu_<L>.elf            (6 lengths)  — 1 hw_context
gelu_overlay.xclbin        + gelu_<L>.elf            (6 lengths)  — 1 hw_context
rms_rope_overlay.xclbin    + rms_norm_s<seq>_c<cols>.elf (24)     \_ 1 SHARED hw_context
                           + rope_s<seq>_d<dims>.elf     (4)      /   (both ops)
manifest.json              shape → {op, overlay, elf, kernel_name:"MLIR_AIE", buffers, lmax}
```
`rms_norm` ELFs cover `seq∈{1,32} × cols∈{128,256,1024,1536,2048,2560,2816,3072,3840,4096,5120,5376}`;
`rope` covers `seq∈{1,32} × head_dim∈{128,256}`. (`rms_norm_rtp.py`/`rope_rtp.py` remain as the
single-op, single-column variants if a 2-column footprint is ever undesirable; the shipped prebuilt
uses the packed overlay.)

**Host load / dispatch protocol** (per shape, from the manifest):
```
# once — establishes the ONE shared hw_context for BOTH rms_norm and rope:
dev.register_xclbin( xrt::xclbin("rms_rope_overlay.xclbin") )
ctx = xrt::hw_context(dev, overlay.get_uuid())
# rms_norm(cols) dispatch:
k = xrt::ext::kernel(ctx, xrt::module(xrt::elf("rms_norm_s1_c<cols>.elf")), "MLIR_AIE")
k(3, 0, 0, IN_bo, OUT_bo)                 # 2 buffers; rows uploaded PADDED to lmax=6144
# rope(head_dim) dispatch (same ctx):
k = xrt::ext::kernel(ctx, xrt::module(xrt::elf("rope_s1_d<dim>.elf")), "MLIR_AIE")
k(3, 0, 0, IN_bo, LUT_bo, OUT_bo)         # 3 buffers; rows padded to lmax=256
```
**HOST ABI note:** rms/rope rows are fixed `LMAX` L1 objects, so the host uploads each row **padded to
LMAX** (real data in `[0:cols|dims]`); the kernel reduces/rotates only the valid prefix, so the rms
divisor stays `cols` and the padding is ignored. (Trade-off: padding a `head_dim=128` q/k-norm row to
6144 wastes DMA on a tiny op; if that shows up in profiling, add a small-`LMAX` bucket overlay — but it
does not change the shared-context count.) NEOX rope stays host-permuted (rope.cc is GPT-J, §4).

**Result / decode working-set impact.** All four single-core activation/norm ops now cost **THREE
`hw_context`s total** — silu (1) + gelu (1) + rms_norm&rope packed (1) — regardless of how many shapes
each is used at. For the Qwen3-1.7B decode set this turns the old `rms@2048 + rms@128(q/k) + rope@128 +
silu` (≈4 contexts) into `rms_rope(1) + silu(1)` = **2**, so the full decode set (≈2 gemv overlays +
1 tiled matmul + these) lands **at/under the ~5 limit** without LRU thrash. silu/gelu can't join the
pack (each uses all 4 columns); rms+rope fit in 2 columns.

**Feasibility note — kernel-side N=6144 decode gemv (secondary, unbuilt).** The gemv design
(`gemv.py`) emits the output in `m`-sized sub-tiles (default `m=32`), so the inB broadcast BD outer
count is `N/m`; `N=6144` → 192 tiles, over the broadcast BD `[1:64]` limit → no N=6144 gemv exists.
A larger output sub-tile **`m=96`** gives `6144/96 = 64` tiles (≤64), which is buildable **iff** a
`mv_96x32.o` core variant is compiled (`-DDIM_M=96`) and the `m×k` A tile + double-buffered C still
fit L1. This is a durable kernel-side alternative to host N-tiling of gate/up; feasible but
**not built or validated** here (host N-tiling remains the Windows-side path).

## 9. Build-pipeline / packaging asks

- **[P1] Emit the instruction blob with an honest extension.** The `_insts.txt` files are actually
  raw uint32 binary; the Windows reader content-sniffs to cope. Please emit `_insts.bin` (raw) so
  the format matches the name.
- **[P1] Per-kernel scalar params** where noted (eps, freq_base) so we don't need a combinatorial
  set of baked variants.
- **[P2] A manifest** (JSON) of built artifacts: `{op, arch, dtin, dtout, K, N, M_tile, cols,
  extra}` so the host can validate coverage and the packaged `GGML_XRT_KERNEL_DIR` is stable/versioned.
- **[P2] Keep `.xclbin`** (don't strip to `.pdi`) — the Windows XRT path registers the xclbin.

## 10. Notes for whoever builds these

- New matmul `(K,N)`: buildable when it satisfies the §2 rule (`N % (n_tile × n_aie_cols) == 0`,
  `n_tile % 16 == 0`, `N × n_tile ≤ 1048576`) — pick `n_tile`/`n_aie_cols` to fit, or host-N-pad;
  very large N is host-N-tiled over block kernels. Otherwise it stays on GPU.
- The host already handles: M-tiling (largest tile ≤ token count, else smallest), Q4_K→BF16 dequant
  (cached per weight, transposed N×K→K×N), F32→BF16 activation conversion, and per-op AOT gating
  (an op is only claimed if a matching artifact exists, else it routes to Vulkan/CPU).
- Hybrid policy: Gated DeltaNet / `SSM_CONV` / `SSM_SCAN` (Qwen3.5) and anything non-conformant stay
  on the GPU (Vulkan ships those shaders). The NPU takes conformant weight matmuls + the enabled ops.
