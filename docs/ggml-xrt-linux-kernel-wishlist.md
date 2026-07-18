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

## 2. Matmul shape/tiling gaps (needed for broader model coverage)

The stock designs impose: **whole_array (4-col) needs N % 128 == 0**, and **large N overflows the
DMA stride range** (fails around N=17408); **single_core** needs a tiled dim ≤ 64 tiles (N ≤ 2048).
These block several models:

- **[P0] Non-conformant N (`N % 128 != 0`).** Blocks Gemma4-26B-A4B `2816×2112` and Qwen3.5-35B-A3B
  dense FFN `2048×4304`. Need a design that pads/handles N not a multiple of 128, or a documented
  host N-padding path + a kernel that tolerates a padded N.
- **[P0] Large N (DMA stride overflow).** Blocks Qwen3-14B & Qwen3.5-27B FFN `5120×17408` and
  `17408×5120`. Either an in-kernel N-tiling design, or emit kernels for ≤2048 128-aligned column
  blocks so the host can N-tile and concatenate (analogous to the existing M-tiling).
- **[P1] lm_head `(2048,151936)` etc.** Currently CPU. Huge N — needs the large-N/N-tiling work
  above; low priority (one matmul per token, and CPU handles it fine).

Per-model shape status (from the prebuilt table; "both tiers" = prefill M=256 `_4c` + decode):
- **Qwen3-1.7B**: complete. **Add M=1 gemv** variants for all four `(K,N)`.
- **Qwen3-14B**: have 5120×5120, 5120×1024, 17408×5120; **need 5120×17408** (gate/up, large-N).
- **Gemma4-26B-A4B**: have 2816×4096, 2816×2048, 4096×2816(prefill), 2816×704(decode),
  704×2816(prefill); **need 2816×2112 (N%128), plus decode tiers for 4096×2816 / 704×2816 /
  2816×704**.
- **Qwen3.5-27B (hybrid)**: have attention 5120×6144, 5120×1024, 6144×5120; **FFN 5120×17408 /
  17408×5120 need large-N** (else GPU); DeltaNet layers → GPU (by design).
- **Qwen3.5-35B-A3B (hybrid MoE)**: have attention 2048×4096, 2048×512, 4096×2048; expert FFN
  2048×512, 512×2048; **dense FFN 2048×4304 needs N%128**; DeltaNet + router/gating → GPU.

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

## 7. Quantization

- **[P1] Native quantized matmul (skip host dequant).** Today weights are host-dequantized Q4_K→BF16
  before the NPU sees them (cached per weight). A kernel that consumes Q4_K/Q4_0 blocks directly
  would let the NPU read the quantized weights as-is, removing the host dequant-to-BF16 step and
  its BF16 weight copy (which is 2× the quantized size in memory). Q4-and-below is the scope.

## 8. Build-pipeline / packaging asks

- **[P1] Emit the instruction blob with an honest extension.** The `_insts.txt` files are actually
  raw uint32 binary; the Windows reader content-sniffs to cope. Please emit `_insts.bin` (raw) so
  the format matches the name.
- **[P1] Per-kernel scalar params** where noted (eps, freq_base) so we don't need a combinatorial
  set of baked variants.
- **[P2] A manifest** (JSON) of built artifacts: `{op, arch, dtin, dtout, K, N, M_tile, cols,
  extra}` so the host can validate coverage and the packaged `GGML_XRT_KERNEL_DIR` is stable/versioned.
- **[P2] Keep `.xclbin`** (don't strip to `.pdi`) — the Windows XRT path registers the xclbin.

## 9. Notes for whoever builds these

- New matmul `(K,N)`: N must be `%128` for the 4-col `whole_array`, or `≤2048` for `single_core`;
  otherwise it needs the §2 tiling work or stays on GPU.
- The host already handles: M-tiling (largest tile ≤ token count, else smallest), Q4_K→BF16 dequant
  (cached per weight, transposed N×K→K×N), F32→BF16 activation conversion, and per-op AOT gating
  (an op is only claimed if a matching artifact exists, else it routes to Vulkan/CPU).
- Hybrid policy: Gated DeltaNet / `SSM_CONV` / `SSM_SCAN` (Qwen3.5) and anything non-conformant stay
  on the GPU (Vulkan ships those shaders). The NPU takes conformant weight matmuls + the enabled ops.
