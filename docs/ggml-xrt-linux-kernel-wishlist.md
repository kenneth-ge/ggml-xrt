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
  for M=1.
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

**Next kernels:** the tiled-prefill quant matmul (harder: unpack into `mm.cc`'s mmul-tiled
layout) — the remaining item for full native-quant coverage. Q4_K_M's few Q6_K tensors keep
the host-dequant fallback (6-bit out of scope).

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
