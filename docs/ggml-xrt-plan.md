# ggml-xrt: AMD XDNA NPU backend via XRT (Windows-first)

Status: **scaffold** (this branch `ggml-xrt-backend`). Host backend skeleton builds and
registers; NPU op dispatch is stubbed with `TODO(xrt)`. This document scopes the project
and gives a prioritized checklist to reach the milestone.

## 1. Goal & milestone

Run a **very basic dense transformer** end-to-end on the AMD **Phoenix (PHX / XDNA1 /
`aie2` / NPU1)** NPU **on Windows**, through a new `ggml-xrt` backend that dispatches
precompiled AIE kernels via the **XRT** runtime.

Milestone = first token(s) produced by a small **dense** model (e.g. a Qwen3-dense or
Llama-class model, or GLM-4 dense) with **at least `MUL_MAT` running on the NPU** and the
rest of the graph correct (CPU fallback allowed for non-NPU ops). This is a correctness /
bring-up milestone, not a performance target.

### Explicit scope boundaries (agreed)

- **Dense transformers only.** No MoE (`MUL_MAT_ID`, expert routing) in the milestone.
- **No SSM / linear-attention** (rules out Qwen3.5-style Gated DeltaNet). Target GLM-style
  and the **older Qwen 3** series.
- **Q4 quantization and below** (Q4_K_M / Q4_0 and smaller). Weights are dequantized to
  **BF16** on the host before the NPU sees them — no native quantized matmul.
- **Windows runtime.** Kernels are compiled in WSL/Linux; the host binary is built and run
  on Windows.

## 1a. Bring-up target model

**Qwen3-1.7B (dense).** Phase-2 target: **Gemma 4 26B-A4B** (MoE — deferred; needs
`MUL_MAT_ID`, expert routing/`ARGSORT`, `GELU`/GeGLU, and sliding-window attention).

Qwen3-1.7B dimensions (from HF `config.json`):

| Field | Value |
|---|---|
| hidden_size (H) | 2048 |
| intermediate_size (I) | 6144 |
| num_hidden_layers | 28 |
| num_attention_heads | 16 (Q dim = 16·128 = 2048) |
| num_key_value_heads | 8 (KV dim = 8·128 = 1024) |
| head_dim | 128 |
| vocab_size (V) | 151936 |
| activation | SiLU (SwiGLU) |
| rms_norm_eps | 1e-6 |
| rope_theta | 1e6 (NEOX) |
| tie_word_embeddings | true |
| sliding window | none |

Qwen3-4B (same family, if we scale up): H=2560, I=9728, 36 layers, 32 Q / 8 KV
heads, head_dim 128, V=151936.

### Weight matmul shapes (K = in, N = out; M = token count, dynamic)

| Projection | K | N |
|---|---|---|
| Q proj | 2048 | 2048 |
| K / V proj | 2048 | 1024 |
| O proj | 2048 | 2048 |
| gate / up proj | 2048 | 6144 |
| down proj | 6144 | 2048 |
| lm_head (tied) | 2048 | 151936 |

Distinct `(K,N)`: `(2048,2048)`, `(2048,1024)`, `(2048,6144)`, `(6144,2048)`,
`(2048,151936)`. Plus per-head attention-score matmuls (`Q·Kᵀ`, `scores·V`, K=128,
dynamic in seq len).

### Consequences for the matmul kernel

- Shapes are **large** (K/N up to 6144, lm_head N=151936) — the toy `single_core`
  256³ kernel does not apply; real projections need the multi-core `whole_array`
  design (4 columns on aie2/Phoenix) with proper tiling.
- **M (tokens) is dynamic**: 1 for decode, prompt-length for prefill. Options:
  - **Decode (M=1)** → gemv; use the mlir-aie `matrix_vector` design.
  - **Prefill** → fixed chunk M (e.g. 128) via `whole_array`, loop over chunks.
  - (Longer term: a runtime-M matmul so one xclbin covers all token counts.)
- **lm_head** (N=151936) is huge; keep on **CPU** for the milestone.
- **QK-norm caveat**: Qwen3 applies RMSNorm to per-head Q and K (a Qwen3 feature not
  flagged in `config.json`). Small op; run on CPU for bring-up.

## 2. Why a new backend (not ggml-hsa)

`ggml-hsa` talks to the NPU through ROCr's **HSA** API (141 `hsa_*` calls, 0 XRT calls) and
the Linux `amdxdna` driver. There is **no HSA/ROCr on Windows**; the Windows NPU stack is
**XRT** (`xrt::device` / `xrt::hw_context` / `xrt::kernel` / `xrt::bo`). So Windows NPU
execution requires a distinct backend with an XRT host/dispatch layer. The **kernel
artifacts and the Python kernel-generation pipeline are reusable**; only the host runtime is
rewritten.

## 3. Architecture

### Build split (this is the key structural fact)

| Domain | Where | Why |
|---|---|---|
| **AIE kernels → `.xclbin` + `_insts.bin`** | **WSL / Linux** | The mlir-aie / IRON / peano / Triton-XDNA toolchain is Linux-only (ELF wheels). No Windows build exists. |
| **`ggml-xrt` host backend + llama.cpp** | **Windows** | MSVC/clang against the Windows XRT SDK; loads the xclbins as data. |

The `.xclbin` + instruction-sequence artifacts are **OS-agnostic AIE device code** for the
same `aie2` silicon, so kernels built in WSL load unchanged under Windows XRT. Only the
loader/dispatch differs (XRT on Windows vs HSA on Linux).

### Runtime dispatch (target design)

Per op, the host backend will:
1. Build a kernel key from op + tensor shapes/dtypes (reuse `src/ggml-hsa/kernel-discovery`
   naming, e.g. `MUL_MAT_aie2_bf16_bf16_bf16_<shapes>`).
2. `device.load_xclbin("<GGML_XRT_KERNEL_DIR>/<key>.xclbin")` → `xrt::hw_context`.
3. Look up `xrt::kernel` by name; read `<key>_insts.bin` into an `xrt::bo`.
4. Bind src/dst tensor buffers as `xrt::bo`, sync inputs.
5. `run = kernel(opcode, insts_bo, insts_len, in_bo…, out_bo); run.wait();`
6. Sync output bo back to host.

## 4. What exists in this branch now

- `include/ggml-xrt.h` — public API (mirrors `ggml-hsa.h`).
- `src/ggml-xrt/ggml-xrt.cpp` — full ggml-backend vtable: reg / device / buffer-type /
  buffer / backend. Compiles against XRT and links `xrt_coreutil`.
  - Device open is **lazy and exception-guarded**: on a machine with no NPU (WSL), the
    backend registers and reports **0 devices** instead of crashing (verified).
  - Buffers use **host memory** for now (so set/get/clear work and it is testable without a
    device). To be migrated to `xrt::bo`.
  - `supports_op` currently claims **only no-op tensors**; real ops are enabled as their
    dispatch lands. `ggml_backend_xrt_compute_node()` is the single `TODO(xrt)` seam.
- `src/ggml-xrt/CMakeLists.txt` — XRT discovery (CMake package or `-DXILINX_XRT=<path>`),
  `GGML_XRT` option, backend registration in `ggml-backend-reg.cpp`, public header wired.

Verified: `cmake -DGGML_XRT=ON` configures, `libggml-xrt.so` builds warning-free, links
`libxrt_coreutil.so.2`, and `ggml_backend_xrt_reg()` + `get_device_count()==0` run safely
without an NPU.

## 5. Prioritized checklist

Priorities: **P0** = required for the milestone; **P1** = needed for a real/correct run;
**P2** = hardening / perf / later.

### Group A — Kernel artifacts (build in WSL)

- **A1 (P0) Fix the IRON env for offline compilation.** `import aie.iron` eagerly opens
  `pyxrt.device(0)` (`.../aie/utils/__init__.py:157`, `CachedXRTRuntime()`), which fails with
  no NPU. Provide a fake-device shim (patch `pyxrt.device` to a stub whose `get_info` returns
  a name containing `"Phoenix"`) so MLIR generation works headless. Kernel *arch* comes from
  the design, not the live device, so generated MLIR is unaffected.
- **A2 (P0) Compile a `MUL_MAT` (bf16) xclbin for `aie2`.** Reuse
  `mlir-aie/programming_examples/basic/matrix_multiplication` or `src/ggml-hsa/kernels/
  iron_kernels/gemm.py`; emit `.xclbin` (keep it — do not strip to PDI). Target `--target
  AIE2`, `--n-aie-cols 4` (PHX 4×4).
- **A3 (P1) Emit xclbins from the ggml-hsa pipeline.** `build_iron.py` currently emits
  `.pdi`+`_insts.bin`; add an option to keep the `.xclbin` (Triton path already produces
  `aie.xclbin`). Lets the full op set be generated with existing naming.
- **A4 (P1) Compile the minimal dense-transformer op set** to xclbins: `MUL_MAT` (bf16),
  `SILU`, `RMS_NORM`, `ROPE` (NEOX), plus reuse existing `SOFT_MAX`, `SCALE`, `ADD`, `MUL`.
- **A5 (P2) Package artifacts** into a versioned `GGML_XRT_KERNEL_DIR` layout + manifest so
  the Windows build consumes a stable set.

### Group B — ggml-xrt host backend (Windows)

- **B1 (P0) Kernel loader / cache.** Implement xclbin discovery + `load_xclbin` +
  `hw_context` + `kernel` lookup + insts `bo`, keyed by the kernel name. Port the naming from
  `src/ggml-hsa/kernel-discovery.cpp`.
- **B2 (P0) `MUL_MAT` dispatch** in `ggml_backend_xrt_compute_node()`; enable it in
  `supports_op` (bf16, contiguous, shape constraints matching the compiled tiling).
- **B3 (P1) Migrate buffers to `xrt::bo`.** Replace host-malloc buffers with device-visible
  `xrt::bo`; `get_base` returns `bo.map()`; sync on set/get.
- **B4 (P1) Add remaining dense ops** (`SILU`, `RMS_NORM`, `ROPE`, `SOFT_MAX`) to dispatch +
  `supports_op`, gated on artifact availability.
- **B5 (P2) `synchronize` / batching / kernel-arg caching** for throughput.

### Group C — Model bring-up

- **C1 (P0) Host dequant-to-BF16.** Quantized weights (Q4_K_M etc.) → BF16 on load via
  ggml's `ggml_get_type_traits(t)->to_float` (type-agnostic; covers Q4-and-below for free).
  So the NPU only ever sees BF16.
- **C2 (P1) Attention from primitives.** Build attention as `MUL_MAT`+`SCALE`+`SOFT_MAX`
  (masked)+`MUL_MAT` — no fused FlashAttention in the milestone.
- **C3 (P1) End-to-end logit compare** vs CPU backend on a tiny dense model.

### Group D — Build / integration (Windows)

- **D1 (P0) Confirm the Windows XRT dev SDK** (headers + `xrt_coreutil` import lib) is
  installed, not just the runtime driver. `-DXILINX_XRT=%XILINX_XRT%`.
- **D2 (P0) Graft ggml-xrt into llama.cpp.** llama.cpp vendors its own ggml; point it at this
  fork (`-DLLAMA_USE_SYSTEM_GGML`) or copy `src/ggml-xrt` + header + CMake/reg wiring into
  llama.cpp's `ggml/`. Match a llama.cpp commit vendoring ggml ≈ v0.17.0.
- **D3 (P1) MSVC build pass.** Verify the backend compiles under MSVC (the scaffold is
  compile-checked with GCC/Linux XRT; watch for `ggml_aligned_malloc`, `snprintf`, and XRT
  header quirks on MSVC).

### Group E — Test / validation

- **E1 (P1) `test-backend-ops -b XRT`** per enabled op on `aie2`.
- **E2 (P1) BF16-dequant accuracy** (perplexity / logit drift vs CPU Q4_K_M reference).
- **E3 (P2) A small XRT smoke test** under `tests/` (load xclbin, run `MUL_MAT`, compare).

## 6. Risks / open questions

- **Windows AIE kernel compilation is not available** — kernels must be built in WSL. If AMD
  later ships a Windows mlir-aie/peano, revisit.
- **XRT NPU dispatch ABI**: exact `kernel(opcode, insts, …)` argument order and the
  `insts.bin` format must match what the mlir-aie toolchain emits for `aie2`; validate against
  an mlir-aie XRT host example before wiring B2.
- **Model memory**: even small dense models + KV cache must fit PHX constraints; may need
  weight streaming / partial offload.
- **MSVC vs GCC**: scaffold verified on GCC; MSVC pass (D3) is unproven.

## 7. Reuse map (ggml-hsa → ggml-xrt)

| Reused | From |
|---|---|
| Kernel Python generation + tiling | `src/ggml-hsa/kernels/**` (esp. `iron_kernels/gemm.py`) |
| Kernel naming scheme | `src/ggml-hsa/kernel-discovery.cpp` |
| Op-support / dequant patterns | `src/ggml-hsa/ggml-hsa.cpp`, `type-traits.hpp` |
| Backend vtable shape | `src/ggml-hsa/ggml-hsa.cpp` (mirrored in `ggml-xrt.cpp`) |

## 7a. Kernel build status (prebuilt artifacts)

All artifacts are aie2 (Phoenix), bf16→f32, **compiled but NOT executed** (no NPU in the
dev environment). Layout: `kernels/prebuilt/<model>/` for matmuls, `kernels/prebuilt/ops/`
for elementwise/norm.

**MUL_MAT** (prefill = whole_array M=256 `_4c`; decode = single_core M=32 `_1c`, or
whole_array M=128 `_4c` for wide N):

| Model | Built (K×N) | Not yet built — reason |
|---|---|---|
| Qwen3-1.7B | 2048×2048, 2048×1024, 2048×6144, 6144×2048 (both tiers) | — (complete) |
| Qwen3-14B | 5120×5120, 5120×1024, 17408×5120 (both tiers) | **5120×17408** (gate/up): DMA stride out of range at N=17408 |
| Gemma4-26B-A4B | 2816×4096, 2816×2048, 4096×2816(prefill), 2816×704(decode), 704×2816(prefill) | **2816×2112** (N%128≠0), plus decode variants for 4096×2816 / 704×2816 / 2816×704(prefill) |
| Qwen3.5-27B (hybrid) | attention: 5120×6144, 5120×1024, 6144×5120 (both tiers) | FFN 5120×17408 / 17408×5120 → **GPU** (N stride); DeltaNet layers → GPU |
| Qwen3.5-35B-A3B (hybrid MoE) | attention: 2048×4096, 2048×512, 4096×2048; expert FFN: 2048×512, 512×2048 (both tiers) | dense FFN 2048×4304 (N%128≠0) → **GPU**; DeltaNet + router/gating → GPU |

**Hybrid-model policy (per user):** shapes needing complex tiling (N%128≠0 or N too large
for the DMA range) and the "difficult" new ops (Gated DeltaNet / `SSM_CONV`/`SSM_SCAN`, MoE
routing/`ARGSORT`) are **left on the GPU**. Because `supports_op` is AOT-gated, the NPU
simply doesn't claim them and the scheduler routes them to Vulkan automatically — no code
change needed. The NPU takes the conformant attention/FFN/expert weight matmuls only.

**Ops** (`prebuilt/ops/`, aie2, size-encoded `<tag>_<size>_aie2.xclbin`):
- **RMS_NORM** — `rms_norm_{128,256,2048,2816,5120}` (covers Qwen3/14B/Gemma4/Qwen3.5
  hidden sizes + head_dims 128/256). Authored in `src/ggml-xrt/kernels/aie2/rms_norm.cc`
  (the `ml/rmsnorm` example was aie2p-only; scalar `aie::invsqrt` pulls in `sqrtf`, absent
  in the aie2 peano runtime, so it uses the vector reciprocal-sqrt intrinsic). Built with a
  fixed row tile `seq=32`; the host tiles the token dimension over it.
- **RoPE** — `rope_{128,256}` (head_dims). Dispatch disabled (see below).
- **SiLU / GELU** — one tileable elementwise kernel each (`silu_16384`, `gelu_16384`); the
  host tiles the flat element count over the 16384 tile.
- lm_head → CPU.

**Op dispatch wiring** (`ggml-xrt.cpp`): `supports_op` + `graph_compute` handle `MUL_MAT`,
`RMS_NORM` (row-wise, `ROW_TILE=32`, keyed on `ne[0]`), and unary `SILU`/`GELU` (flat
elementwise, host-tiled over the kernel's tile length). All AOT-gated: an op is claimed only
if a shape-matching artifact exists, else it runs on the GPU. `ROPE` dispatch is present but
disabled (position/frequency arg binding unvalidated) → GPU. All unvalidated on hardware
(TODO(hw): arg layouts, and keep `ROW_TILE`/tile length in sync with the built artifacts).

**Stock-matmul shape constraints found (aie2, tile 32):**
- whole_array (4 cols): **N % 128 == 0** required; large N (e.g. 17408) overflows the DMA
  stride range.
- single_core: a tiled dim must be **≤ 64 tiles** (so N ≤ 2048).
- **Fix for uncovered shapes:** host-side **N-tiling** — split N into ≤2048, 128-aligned
  column blocks and concatenate outputs (the backend MUL_MAT dispatch already tiles M; N-
  tiling is the analogous extension). Alternatively tune (m,k,n)/`n_aie_cols`.

## 8. Dynamic-M matmul strategy

**Finding:** the mlir-aie matmul/gemv designs bake M/K/N into *static* DMA descriptors
(`aiex.npu.dma_memcpy_nd [..][..]` with compile-time constant sizes/strides). There is
no runtime-M xclbin out of the box; a single "any-M" kernel would require IRON surgery to
parameterize the descriptor sizes.

**Approach (AOT-friendly): host-side M-tiling over a fixed small-M-tile kernel.** The
kernel is compiled for a fixed M tile; the host makes M dynamic by looping over
`ceil(tokens / M_tile)` chunks and zero-padding the final chunk. K,N stay fixed per weight.

- **Prefill** (M = prompt length): chunk into `M_tile` blocks.
- **Decode** (M = 1): pad up to the smallest tile (weight-bandwidth-bound, so the wasted
  M-compute is largely hidden), or later a dedicated gemv (`matrix_vector`) for efficiency.

This keeps the kernel set **finite** (one small-M kernel per `(K,N)`, plus the M=256
prefill kernels already built) → fully AOT, no JIT. Host tiling loop lives in the
`ggml-xrt` `MUL_MAT` dispatch (checklist B2).

**Tile/design constraints found empirically (aie2, tile 32):**

- `whole_array` (4 cols) requires **M ≥ 128** (M split across 4 rows × 32); M=64 fails.
- `single_core` works at **M=32** but a tiled dimension may not exceed **64 tiles** (the
  `aiex.npu.dma_memcpy_nd` BD range is `[1:64]`). So single-core handles N ≤ 2048
  (≤64 N-tiles) but **not N=6144** (192 tiles) — the gate/up decode kernel therefore uses
  `whole_array` M=128 (N split across 4 cols → 48 tiles/col).

**Built artifact set (Qwen3-1.7B, `kernels/prebuilt/`):** for each weight `(K,N)`, a prefill
kernel (`whole_array` M=256, `*_4c`) and a decode kernel (`single_core` M=32 `*_1c`, except
gate/up which is `whole_array` M=128 `*_4c`). lm_head omitted (CPU).

**Future optimization:** true runtime-M via parameterized DMA descriptors (one xclbin per
`(K,N)` for all M), and a separate efficient M=1 gemv for decode.

## 9. Hybrid NPU + GPU execution

Goal: run some compute on the NPU (e.g. the weight GEMMs, or a contiguous range of inner
layers) and the rest on the GPU (Vulkan on the 780M iGPU), with CPU fallback.

**Mechanism — the ggml scheduler already does this.** Create a `ggml_backend_sched` over
`[xrt(NPU), vulkan(GPU), cpu]`. It partitions the graph across backends by `supports_op` +
tensor/buffer placement and inserts copies at backend boundaries. No custom partitioner
needed.

**How to avoid the JIT limitation — isolate the AOT constraint to the NPU:**

1. **`supports_op` = "is there a precompiled xclbin for this exact op+shape+dtype?"** On
   Windows we cannot JIT, so the NPU backend must claim an op *only* when a matching AOT
   artifact exists in `GGML_XRT_KERNEL_DIR` (this replaces ggml-hsa's JIT-trial probe). Any
   op without an artifact returns false and the scheduler routes it to the GPU/CPU
   automatically.
2. **The GPU has no AOT limitation.** Vulkan compiles shaders at runtime and handles
   arbitrary/dynamic shapes, so it absorbs everything the NPU can't take (attention with
   dynamic seq-len, norms, RoPE, activations, lm_head, embeddings). The finite, fixed-`(K,N)`
   weight matmuls are exactly the set that *is* AOT-precompilable — so put those on the NPU.
3. **Placement drives assignment.** ggml prefers to run an op on the backend where its
   inputs/weights live. Put a chosen layer's weights in an NPU buffer → its matmuls run on
   the NPU; leave the rest in GPU buffers. This is how "some inner layers on NPU" is realized.

**Split granularity — minimize boundary crossings.** Each NPU↔GPU boundary inserts a tensor
copy. Trade-off:

- **Fine-grained** (only `MUL_MAT` on NPU, everything else GPU): minimal kernel authoring,
  but a copy around every matmul. Fine for the bring-up milestone.
- **Coarse-grained** (a contiguous block of whole layers entirely on NPU): fewest copies,
  but requires NPU kernels for *all* ops in those layers (MUL_MAT + RMS_NORM + RoPE +
  SOFT_MAX + SiLU) so the layer stays NPU-resident. This is the efficient end state.

**APU advantage:** on Phoenix the NPU and iGPU share system memory, so cross-backend handoff
is a host-memory copy (cheap), and can eventually be made **zero-copy** via shared
host-visible buffers (XRT `bo` host-only mapped + Vulkan external-memory import) — an
optimization worth taking once the split works.

**Recommended path:** (1) bring-up with fine-grained split — `MUL_MAT` on NPU (via the
finite AOT xclbin set), everything else on Vulkan; (2) grow NPU op coverage (RMS_NORM, RoPE,
SiLU) so a contiguous inner-layer range can run fully NPU-resident, minimizing copies; (3)
add zero-copy NPU↔GPU sharing. GPU backend on Windows = **Vulkan** (native AMD 780M support;
HIP-on-Windows is limited).

### Zero-copy build status

- **Mechanism**: host-pointer import via `VK_EXT_external_memory_host` (unified memory), NOT
  dma-buf/handle export. ggml-vulkan already implements the import (`ImportMemoryHostPointerInfoEXT`,
  `minImportedHostPointerAlignment`, its host buffer type). The two runtimes share one host
  allocation.
- **ggml-xrt side — done**: tensor buffers are XRT host-visible `bo`s (`get_base` = `bo.map()`),
  and the buffer type reports `is_host = true`, so the allocation is directly CPU/NPU-visible
  and exposable to a Vulkan importer. Compiled.
- **Hybrid build — verified compiling**: with `libvulkan-dev glslang-tools spirv-tools glslc
  spirv-headers` installed, `cmake -DGGML_XRT=ON -DGGML_VULKAN=ON` configures with 3 backends
  (XRT + Vulkan + CPU) and both `libggml-xrt.so` and `libggml-vulkan.so` build cleanly.
  (`spirv-headers` is required by ggml-vulkan's `find_package(SPIRV-Headers CONFIG)`.)
- **GPU already covers the offloaded ops**: ggml-vulkan ships compute shaders for exactly the
  work the NPU declines — `gated_delta_net`, `ssm_conv`, `ssm_scan` (Qwen3.5 DeltaNet),
  `topk_moe` (MoE routing), `geglu`/`swiglu`, `flash_attn`, `rope_neox`, and dequant for all
  quant types. So the hybrid split is well-supported: the NPU takes conformant weight matmuls
  (+ RMS_NORM/SiLU/GELU where enabled), Vulkan takes everything else.
- **On-hardware unknowns (TODO(hw))**: (1) alignment — the XRT `bo` base must meet Vulkan's
  `minImportedHostPointerAlignment` (typically 4 KB); (2) whether XRT `host_only` `bo` memory
  is importable at all (must be ordinary host pages, not a special carveout). Only testable on
  the NPU.
