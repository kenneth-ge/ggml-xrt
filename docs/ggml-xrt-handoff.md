# ggml-xrt — Windows handoff

Branch: **`ggml-xrt-backend`** on `github.com/kenneth-ge/ggml-hsa`. Full design +
coverage detail is in **`docs/ggml-xrt-plan.md`** — read that first. This file is the
short "what to do on Windows" companion.

## What this project is

A new ggml backend (`ggml-xrt`) that runs AMD **Phoenix (aie2/NPU1)** AIE kernels via the
**XRT** runtime, for **hybrid NPU + GPU (Vulkan) inference** on Windows. It exists because
`ggml-hsa` is HSA/ROCr and Linux-only; XRT is the Windows NPU path. Kernels are compiled
**ahead-of-time on Linux/WSL** (the mlir-aie toolchain is Linux-only) and loaded as data on
Windows.

Bring-up target: **Qwen3-1.7B dense**. Also has matmul shapes for Qwen3-14B,
Gemma4-26B-A4B, Qwen3.5-27B, Qwen3.5-35B-A3B. Scope: dense/standard ops on the NPU; MoE
routing, Gated DeltaNet/SSM, non-conformant tiling → GPU.

## What is DONE (Linux/WSL side, all pushed, compiles clean)

- `include/ggml-xrt.h`, `src/ggml-xrt/ggml-xrt.cpp` — full ggml-backend vtable against the
  XRT C++ API. Registers safely on a machine with no NPU (reports 0 devices).
- **Dispatch** (AOT-gated `supports_op` → op runs on NPU only if a matching xclbin exists,
  else GPU): `MUL_MAT` (host M-tiling), `RMS_NORM` (row-tiled, `ROW_TILE=32`), `SILU`/`GELU`
  (flat elementwise tiling). `ROPE` present but disabled.
- **Unified-memory buffers**: XRT host-visible `bo`, `is_host=true` (copy-free CPU↔NPU).
- **Prebuilt kernels** in `src/ggml-xrt/kernels/prebuilt/` (aie2, bf16→f32, **compiled, never
  executed**): per-model MUL_MAT shape sets + `ops/{rms_norm_*,rope_*,silu_16384,gelu_16384}`.
  Reproducible via `kernels/build-*.sh` (+ `headless_shim.py`).
- **Hybrid build verified compiling**: `cmake -DGGML_XRT=ON -DGGML_VULKAN=ON` → 3 backends,
  both libs link. ggml-vulkan already ships GPU shaders for the offloaded ops (DeltaNet, SSM,
  topk_moe, geglu, flash_attn, dequant-all-quants).

## Hardware-validation status (updated 2026-07-18, on NPU Phoenix / Ryzen 7 7840U)

**MUL_MAT is now hardware-validated.** The first NPU matmul ran and matched the CPU
reference bit-exactly (NRMSE 0.0) for M=1/100/256 on the 256³ toy kernel. Getting there
fixed three `TODO(hw)` bugs in `ggml-xrt.cpp`:
1. Use `xrt::device::register_xclbin()` + `hw_context(dev, uuid)`, **not** `load_xclbin()`
   (the XDNA/NPU shim rejects the legacy `load_axlf`: "not supported").
2. `_insts.txt` files are actually raw binary blobs — the instruction reader now
   content-sniffs instead of assuming hex text.
3. The weight (`bo_b`) must be **transposed N×K→K×N**: the stock mlir-aie matmul expects B
   row-major K×N, but ggml stores the weight transposed. (The earlier "b-col-major, feed
   directly" assumption was wrong.)

The kernel ABI (`opcode=3, instr@grp1, ninstr-words, A@grp3, B@grp4, C@grp5`, kernel name
`MLIR_AIE`) was confirmed from the xclbin metadata and needed no change.

**Still unvalidated:** RMS_NORM / SILU / GELU / ROPE dispatch (still `TODO(hw)`), the real
Qwen3 weight shapes, host BF16 dequant, and end-to-end hybrid `[xrt,vulkan,cpu]` inference.
Treat those paths as scaffold until run on-device.

## Windows to-do (in order)

1. ~~**Environment**~~ **DONE.** XRT dev SDK staged at `C:\dev\xrt-sdk` (headers from the XRT
   source tree + import lib regenerated from the **driver-store** `xrt_coreutil.dll`; the
   RyzenAI SDK copy is the wrong version). Configure with `-DXILINX_XRT=C:/dev/xrt-sdk`. Vulkan
   SDK 1.4.350.0 installed. `xrt-smi examine` lists `NPU Phoenix`. At runtime the driver-store
   dir must be on PATH. MSVC needs `/Zc:__cplusplus` on the ggml-xrt target (XRT headers pick
   `std::any` only when `__cplusplus>=201703L`).
2. ~~**Graft into llama.cpp**~~ **DONE.** Fresh clone at `C:\Users\kennyge2\projects\llama.cpp`
   (HEAD vendors ggml **0.17.0**, backend interface byte-identical to this fork). Copied
   `src/ggml-xrt/{ggml-xrt.cpp,CMakeLists.txt}` + `include/ggml-xrt.h`; wired `option(GGML_XRT)`,
   `ggml_add_backend(XRT)`, and the 3 `ggml-backend-reg.cpp` points. Builds under MSVC;
   `llama-cli --list-devices` shows `XRT0: NPU Phoenix`. Prebuilt kernels stay in the fork,
   referenced via `GGML_XRT_KERNEL_DIR`.
3. ~~**Validate the XRT ABI**~~ **DONE & hardware-validated** — see the validation-status
   section above. ABI arg order/opcode/groups were correct; fixed register_xclbin,
   binary-insts reader, and the weight transpose. One MUL_MAT matches CPU bit-exactly.
4. ~~**Point at kernels**~~ **DONE & hardware-validated.** With `GGML_XRT_KERNEL_DIR` =
   `src/ggml-xrt/kernels/prebuilt` (recursive; Qwen3-1.7B matmuls live in the root, ops in
   `ops/`), all four Qwen3-1.7B weight shapes run on the NPU and match CPU (NRMSE 0.0):
   2048×2048 (Q/O), 2048×1024 (K/V), 2048×6144 (gate/up, whole_array 4-col), 6144×2048 (down).
   `find_mul_mat_xclbin` resolves the smallest-M kernel and the host tiles M over it; both the
   single_core (1c) and whole_array (4c) designs are proven. Original step-4 text:
   set `GGML_XRT_KERNEL_DIR` to the model's `prebuilt/<model>/` dir
   (+ `prebuilt/ops/`). Naming: `mul_mat_aie2_<dti>_<dto>_<M>x<K>x<N>_<cols|1c>.xclbin`,
   `<op>_<size>_aie2.xclbin`. Keep `ROW_TILE`/tile-length in the code in sync with the
   built artifacts.
5. ~~**Host BF16 dequant**~~ **DONE & hardware-validated.** Done *in-backend* (not via a
   llama.cpp loader hook, which only uploads raw bytes): `supports_op` now accepts any
   BF16-convertible weight/activation type (F16, BF16, F32, and every quant with a `to_float`
   trait), and the MUL_MAT dispatch host-dequantizes weight→BF16 (`to_float` then
   `ggml_fp32_to_bf16_row`, transposed and **cached per weight data ptr** since weights are
   constant) and converts the activation→BF16 per M-tile. Output stays F32. Verified: F16 and
   **Q4_K** weights × F32 activation match CPU within bf16 precision (NRMSE ~0.004) on the toy
   256³ and the Qwen3-1.7B 2048×2048 / 2048×6144 shapes.
6. **Scheduler — DONE & validated (MUL_MAT-only).** NPU output now matches CPU end-to-end
   (coherent generation) with MUL_MAT-only on the NPU. Ran `Qwen3-1.7B-Q4_K_M.gguf` through
   `llama-cli` (`-ngl 0`, weights on CPU) and the NPU automatically took the conformant ops via
   the scheduler — our device is `GGML_BACKEND_DEVICE_TYPE_ACCEL`, which llama.cpp does NOT use
   for `-ngl` layer offload (it's skipped like CPU), but the scheduler still routes matching ops
   to it (like BLAS). Confirmed loaded+run on-device: all four projection matmuls
   (2048×2048, 2048×1024, 2048×6144, 6144×2048) **and** `rms_norm_2048`; the model produced a
   token. Weights dequant Q4_K→BF16 in-backend; no Vulkan yet (`GGML_VULKAN=OFF`).
   Since then: NPU output matches CPU end-to-end (confirmed by re-running); llama.cpp was rebuilt
   with `-DGGML_VULKAN=ON` and the `[xrt, vulkan, cpu]` 3-way hybrid runs (Vulkan takes offloaded
   layers via `-ngl`, the NPU takes CPU-resident conformant ops). NOTE: do NOT run llama-cli
   inside Claude Code (crashes it); llama.cpp also has a benign teardown hang on exit (present in
   stock llama.cpp). Run inference in a normal terminal.
7. **Ops on the NPU — MUL_MAT, RMS_NORM, SILU, GELU, and NEOX RoPE all validated and default-on.**
   All op kernels are BF16-in/BF16-out; the dispatch converts F32↔BF16 per tile (the earlier
   dispatch fed them raw F32, which is why RMS_NORM produced garbage until fixed). Unit harnesses
   vs CPU all pass: `mulmat_check` (bf16 + Q4_K, prefill + decode), `rmsnorm_check`, `silu_check`
   (silu + gelu), `rope_check`. RoPE detail: the prebuilt `rope_{128,256}` are NORMAL/GPT-J kernels
   that take a cos/sin LUT; NEOX is realized by host permutation of the in/out pairs + computing
   ggml's cos/sin cache on the host, restricted to full-width NEOX (`n_dims == head_dim`; other
   modes fall through to GPU). RMS_NORM eps note: the kernel bakes 1e-5 vs Qwen3's 1e-6
   (functionally negligible). Op-split visibility: XRT `graph_compute` logs a per-graph op summary
   (`GGML_XRT_ENABLE_LOG=1`); pair with `GGML_SCHED_DEBUG=2` for the full cross-backend split.
   Validate any new op with a `*_check.cpp` harness (NPU vs CPU) before enabling it.
8. **Zero-copy NPU↔Vulkan — feasibility proven, enabler landed.** Validated end-to-end: a
   standalone harness imported an XRT `host_only` bo into ggml-vulkan (via
   `VK_EXT_external_memory_host` / `ggml_backend_dev_buffer_from_host_ptr`) and a Vulkan op read
   the shared pages with zero copy, matching CPU (0 mismatches / 1,048,576 elems; late-write
   aliasing confirmed — the VkBuffer aliases the same physical pages, no copy at import). Both open
   questions answered yes: XRT `bo.map()` is always ≥4096-aligned, and the `host_only` pages are
   ordinary importable host memory (HOST_VISIBLE|HOST_COHERENT).
   - **Enabler landed** (`ggml-xrt.cpp`): buffers are 4096-aligned and their size is rounded up to
     a 4096 multiple, so the whole allocation satisfies Vulkan's `minImportedHostPointerAlignment`.
   - **ggml-vulkan needs no change** — the import primitive already exists in both the fork and the
     llama.cpp trees (`ggml_vk_buffer_from_host_ptr`). Do NOT globally flip
     `caps.buffer_from_host_ptr` (it breaks llama's mmap loader on unaligned gguf pointers);
     instead the coordinator calls `ggml_backend_dev_buffer_from_host_ptr(vk_dev, base, import_size)`
     directly.
   - **`hsa_buffer` implemented + scheduler-level zero-copy PROVEN.** A shared buffer type
     `XRT_HSA`: allocates an XRT `host_only` bo, imports it into Vulkan (Vulkan-native context +
     iface), keeps a registry `{bo, host_base P}`, and `ggml_xrt_tensor_host_ptr(t)` =
     `P + (t->data − get_base)` translates the vk-sentinel address to the real host ptr for the XRT
     dispatch (all MUL_MAT/RMS_NORM/SILU/GELU/ROPE reads route through it; non-hsa → `t->data`
     unchanged). Vulkan side is a 7-line `supports_buft` accepting `XRT_HSA`. Harness
     `tests/hsa_buffer_check.cpp` drives a real `ggml_backend_sched` graph where one hsa tensor is
     read by the NPU (mul_mat) **and** the iGPU (scale): scheduler inserts **0 copies** for it,
     both outputs match CPU (NRMSE 0), late-write aliasing confirmed. Non-hsa buffers unaffected
     (mulmat/rms_norm/silu/rope still pass).
   - **`is_host = true` — full UMA zero-copy (CPU + NPU + iGPU), done.** `get_base` returns the
     real XRT host ptr `P` (pages wrapped via `ggml_backend_cpu_buffer_from_ptr`), so CPU and NPU
     use `P` directly and a CPU op on a shared tensor no longer forces a copy. Vulkan resolves `P`
     via ggml-vulkan's pinned-host path: hsa alloc imports+registers `P` through
     `register/unregister_host_ptr` hooks (exposed via `get_proc_address`, so ggml-xrt stays
     decoupled). ggml-vulkan gained a `ggml_vk_tensor_mem` resolver (host_get first, else
     `dev_buffer + vk_tensor_offset`), routed through the graph-analysis passes
     (`overlaps_unsynced`, `ggml_vk_tensors_overlap`) **and** the `mul_mat`/`mul_mat_id` dst sites,
     so a shared hsa tensor works as a Vulkan op **input and output**. Applied to both the fork's
     `src/ggml-vulkan/` and the llama.cpp build tree. Validated on hardware: CPU+NPU+iGPU read AND
     write shared hsa tensors with **0 cross-backend copies** (NRMSE ~0), non-hsa ops unaffected.
     Caveat: `im2col`/`im2col_3d` dst (conv path, non-transformer) not converted — a follow-up only
     if a policy ever makes an im2col output an hsa tensor.
   - **Remaining for full use**: a placement policy choosing *which* tensors become `hsa_buffer`s
     (only NPU↔iGPU↔CPU handoff tensors). The mechanism is done; the "which tensors" wiring is not.

9. **Benchmark NPU vs GPU/CPU and decide placement — do this FIRST; it gates step 10.**
   All of this runs on the shapes that **already work** (no new code), and the results decide
   whether the remaining matmul-on-NPU work is worth building. Flagged in
   `docs/ggml-xrt-plan.md` §7a (column-count note, op default-on note). Measure, don't assume:
   - **Per-op NPU vs Vulkan vs CPU** for MUL_MAT (across sizes and M), RMS_NORM, SiLU, GELU.
     Phoenix per-op NPU speed is unproven/contested. **If the GPU wins for large matmuls, the
     FFN shapes (4304/17408) should just stay on the GPU and step 10 is not worth doing.** Set
     op placement (and whether `GGML_XRT_ENABLE_OPS` defaults on) from the data.
   - **Column count A/B** — `variants/` cols=2 vs cols=4 for the same shape (padding-waste win
     is concrete; the two-independent-matmuls-on-disjoint-2-col-partitions idea is speculative
     and needs a concurrent-dispatch prototype).
   - **M-tile sweep** for prefill.
   Chicken-and-egg note: 4304/17408 can't be measured on the NPU until step 10 builds their
   dispatch — so use the *runnable* shapes' NPU-vs-GPU verdict (it generalizes) to decide
   whether to build step 10 at all.

10. **N-padding & N-tiling dispatch — ONLY if step 9 shows the NPU is competitive for large
    matmuls** (otherwise leave 4304/17408 on the GPU; it already works). Kernels are already
    built; this is host dispatch in `ggml_backend_xrt_mul_mat` (see plan §7a), analogous to the
    existing M-tiling:
    - **N-padding** (for `N=4304`, Qwen3.5-35B dense FFN — `16×269`, un-tileable exactly): when
      the exact `(K,N)` kernel is absent but a padded `(K,N_pad)` kernel exists
      (`…_256x2048x4352_4c`), allocate `bo_b`/`bo_c` at `N_pad`, zero-pad the weight columns,
      run, copy back only the first `N` output columns. Gate `supports_op` on an exact **or** a
      `≥N` padded kernel.
    - **N-tiling** (for `N=17408`, Qwen3-14B/27B FFN — output-stride overflow, can't be one
      dispatch): loop the output columns over the `N_block` kernel (`…_5120x2176_4c`, `8×2176`),
      slicing weight `[K, n0:n0+N_block]` and output `[M, n0:n0+N_block]` per launch.
    Then sweep the N-block size. Both are needed only for those two shapes; every other
    target-model matmul already runs one-shot.

11. **True M=1 gemv decode path — DONE (wired & validated).** The M==1 branch in
    `ggml_backend_xrt_mul_mat` now uses the gemv kernel when a `1x{K}x{N}_gemv` artifact exists
    (weight A untransposed @grp3 with its own cache, activation B @grp4, output C @grp5, one
    launch); shapes without a gemv (e.g. gate/up 2048×6144) fall back to the tiled path, and M>1
    is unchanged. Validated vs CPU (NRMSE 0) for Q/O 2048×2048, K/V 2048×1024 (bf16 + Q4_K), with
    the 2048×6144 fallback and M=256 tiled path still passing. Original notes below. Prebuilt gemv
    xclbins `mul_mat_aie2_bf16_f32_1x{K}x{N}_gemv.xclbin` exist for Qwen3-1.7B Q/O
    (2048×2048), K/V (2048×1024), down (6144×2048) — a proper matrix-vector kernel for
    decode (M=1), avoiding the ~16× padded-MAC waste of the M=16 fallback. **They have a
    DIFFERENT ABI from the tiled matmul — do not reuse the matmul dispatch:**
    - `C[N] = A[N,K] · B[K]`. **A = the WEIGHT in ggml's native `[N,K]` layout → NO transpose**
      (the tiled matmul needs the N×K→K×N transpose; the gemv does not). B = the activation
      vector `[K]`. C = output `[N]`. Kernel call: `kernel(opcode=3, instr@grp1, ninstr, A_bo,
      B_bo, C_bo)` with `A=weight, B=activation, C=output`.
    - Wire a gemv branch in `ggml_backend_xrt_mul_mat`: when the token count is 1 and a
      `…_1x{K}x{N}_gemv.xclbin` exists, use it (upload weight untransposed as A, activation as
      B, one launch, no M loop). The filename's leading `M=1` already makes the largest-tile
      selector prefer it only for decode.
    - **gate/up (N=6144) has no gemv** (output dim 6144 > the single-core broadcast BD limit of
      64 tiles) → keep using its M=64 whole_array fallback, or host-N-tile the gemv output.
    - The kernel uses the **scalar** matvec (upstream's vectorized path is marked erroneous);
      switch to vectorized later for throughput. Built from repo `kernels/gemv.py` +
      `kernels/aie2/mv.cc` (bf16 combo enabled; stock mv.cc comments it out). Validate with the
      existing gemv/mulmat check harness before enabling.

12. **Native-quant decode gemv (Q4_0, Q4_K, Q6_K) — kernels HARDWARE-VALIDATED, host dispatch
    still TODO.** The on-chip-dequant gemv kernels
    `mul_mat_aie2_{q4_0,q4k,q6k}_f32_1x{2048x2048,2048x1024,6144x2048}_gemv.xclbin` (from
    `kernels/aie2/{mv_q4.cc,mv_q4k.cc,mv_q6k.cc}` + the matching `gemv_q*.py`) all ran on the NPU
    and match CPU **exactly** (NRMSE 0.00000 on all **nine** kernels, multiple seeds; no
    K-proportional bias). **This covers Q4_K_M decode end to end**: Q4_K_M is not a tensor dtype
    but a mixture — Q4_K (attn_q/k/o, ffn_gate/up) + Q6_K (attn_v, ffn_down) — and both halves
    now pass on all three Qwen3-1.7B decode shapes. Only `output` (Q6_K, N=151936) stays off the
    NPU, as it already did. Full detail + results tables are in
    `docs/ggml-xrt-linux-kernel-wishlist.md` §7 STATUS.
    - **Both repack contracts are confirmed as documented.** ONE weight buffer, row-major, weight
      **NOT transposed** (native `[N,K]`); activation B bf16 `[K]`, output C f32 `[N]`; ABI
      unchanged (`op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5`, kernel `MLIR_AIE`).
      - **Q4_0** — `[N][K/32][20]`: 16 nibble bytes (`block_q4_0.qs`) + f32 scale (f16 `d`→f32).
      - **Q4_K** — `[N][K/256][148]`: `qs[128]` + `scales[12]` (raw 6-bit packed) + f32 `d` +
        f32 `dmin`. A field reorder of `block_q4_K` (`{d,dmin,scales,qs}` = 144 B) with the f16
        scales widened to f32.
      - **Q6_K** — `[N][K/256][212]`: `ql[128]` + `qh[64]` + `scales[16]` (raw int8) + f32 `d`.
        Near-copy: `block_q6_K` (210 B) already ends with `d`, so only the widening differs.
    - **WIRED & validated end-to-end through the backend** (`GGML_XRT_NATIVE_QUANT=1`).
      `ggml_backend_xrt_mul_mat` gained a native-quant branch ahead of the bf16 gemv branch: at
      M==1, if `src0->type` is Q4_0/Q4_K/Q6_K and a matching `…_<qtok>_f32_1x{K}x{N}_gemv.xclbin`
      exists, it repacks the raw quantized weight per the contract above and uploads **that** —
      no BF16 dequant, no BF16 cache. Repacked weights are cached per weight host ptr in a
      separate `quant_weight_bos` map (`GGML_XRT_LOW_MEM=1` re-packs into a pooled per-shape bo
      instead). Helpers: `ggml_xrt_quant_token`, `ggml_xrt_quant_rec_bytes`,
      `ggml_xrt_repack_quant_weight`, `ggml_xrt_find_quant_gemv_xclbin`.
      - **Measured resident weight memory** (per weight, NPU-side), M=1:

        | Weight | native-quant | BF16 cache | ratio |
        |---|---|---|---|
        | Q4_K 2048×2048 | 2368 KiB | 8192 KiB | 3.5× |
        | Q4_0 6144×2048 | 7680 KiB | 24576 KiB | 3.2× |
        | Q6_K 6144×2048 | 10176 KiB | 24576 KiB | 2.4× |

      - **Accuracy end-to-end**: NRMSE 0.0040–0.0043 vs CPU for all three types × all three
        shapes — i.e. exactly the bf16-activation rounding floor the bf16 matmul already sits at
        (0.0044), so on-chip dequant costs nothing in accuracy.
      - **`supports_op` deliberately unchanged**: the op is still claimed via the existing
        BF16-convertible test, so the native-quant branch is a pure fast/lean path *inside*
        dispatch. Anything it can't handle falls through to the tiled path rather than failing.
        Verified fallbacks: M>1 prefill, a shape with no quant gemv (2048×6144), a non-quant
        (F16) weight, and `GGML_XRT_NATIVE_QUANT` unset — all take the tiled path and still pass.
      - **OFF by default**, same reason as `GGML_XRT_USE_GEMV`: these are SCALAR matvec designs
        (~single-lane AIE util), so they are slower than padding M=1 through the vectorized tiled
        kernel. It is a **RAM-vs-latency** switch today. Flip the default once a vectorized gemv
        exists — at that point native-quant becomes a straight win (less DRAM traffic *and* less
        RAM) and this should become the standard decode path.
    - Validation harness: `C:\dev\xrt-sdk\work\q4_gemv_check.cpp` (+ `cc_q4_gemv.bat`,
      `run_q4_gemv.bat`, `run_q4k_gemv.bat`, `run_q6k_gemv.bat`) — raw XRT, no backend dependency;
      handles all three formats, auto-detected from the xclbin filename. Adding Q8_0 or another
      quant is ~20 lines (block struct + repack + reference). Re-run all nine after any change.
    - gate/up (N=6144) still has no gemv (broadcast BD limit), same as bf16 → M=64 fallback.

## Environment variables (host backend)

| Variable | Default | Effect |
|---|---|---|
| `GGML_XRT_KERNEL_DIR` | — | Root of the prebuilt xclbin tree, searched **recursively**. Required for the NPU to claim any op (dispatch is AOT-gated). |
| `GGML_XRT_ENABLE_OPS` | on | Claim RMS_NORM / SiLU / GELU / NEOX-RoPE in addition to MUL_MAT. |
| `GGML_XRT_NATIVE_QUANT` | **off** | M=1 decode: feed Q4_0/Q4_K/Q6_K weights to the NPU **raw** (on-chip dequant) instead of host-dequanting to BF16. ~2.4–3.5× less resident weight RAM; slower today (scalar gemv). See step 12. |
| `GGML_XRT_USE_GEMV` | **off** | M=1 decode: use the bf16 gemv kernels. Off because the prebuilt gemv is scalar and loses to the vectorized tiled kernel. |
| `GGML_XRT_LOW_MEM` | off | Don't cache per-weight device buffers; refill a pooled per-shape bo each call. Much slower; for memory-constrained one-off runs. Composes with `GGML_XRT_NATIVE_QUANT`. |
| `GGML_XRT_MAX_CONTEXTS` | 4 | LRU cap on concurrent `hw_context`s (Phoenix allows ~5; the 6th fails `0xc01e0009`). |
| `GGML_XRT_ENABLE_LOG` | off | Per-graph op summary + kernel load/evict + native-quant repack sizes. Pair with `GGML_SCHED_DEBUG=2` for the full cross-backend split. |

To run decode with native-quant weights on the NPU (from a normal terminal — **not** inside
Claude Code, it crashes the session):

```
set GGML_XRT_KERNEL_DIR=C:\Users\kennyge2\projects\ggml-xrt\src\ggml-xrt\kernels\prebuilt
set GGML_XRT_NATIVE_QUANT=1
set GGML_XRT_ENABLE_LOG=1
llama-cli -m Qwen3-1.7B-Q4_K_M.gguf -ngl 0 -p "hello" -n 32
```

The `[ggml-xrt] native-quant …` lines confirm which weights took the path and how much RAM
each saved.

## Rebuilding kernels (must stay on Linux/WSL)

`src/ggml-xrt/kernels/{build-mm-xclbin.sh,build-qwen3-matmuls.sh,build-ops.sh,build-gemv.sh}`
+ `headless_shim.py`, `gemv.py`, `aie2/{rms_norm.cc,mv.cc}`. Requires an IRON env (mlir-aie +
llvm-aie/peano); no NPU needed to compile. **Shape rule (see plan §7a — NOT "%128"):** for
the 4-col whole_array matmul, `N % (n_tile × cols) == 0` with `n_tile % 16 == 0` and
`N/(n_tile×cols) ≤ 64` tiles and output stride `N×n_tile ≤ 1048576`; pick `(n_tile, cols)` to
fit (e.g. cols=2 for N=2112). N with a large prime factor (4304) needs host N-pad; very large N
(17408) needs host N-tile. single_core min M-tile = 16; whole_array min M-tile = 64.
