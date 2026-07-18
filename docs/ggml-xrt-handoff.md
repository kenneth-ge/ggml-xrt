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

## CRITICAL: nothing is hardware-validated

No NPU exists in the dev/WSL environment, so every NPU code path **compiles but has never
run**. All spots needing on-device validation are marked `TODO(hw)` in `ggml-xrt.cpp`. Treat
the dispatch as a *starting scaffold*, not known-correct.

## Windows to-do (in order)

1. **Environment**: install the **XRT dev SDK** (headers + `xrt_coreutil` import lib; set
   `-DXILINX_XRT=%XILINX_XRT%`) and the **Vulkan SDK**. Confirm `xrt-smi examine` lists the NPU.
2. **Graft into llama.cpp**: llama.cpp vendors its own ggml. Point it at this fork
   (`-DLLAMA_USE_SYSTEM_GGML`) or copy `src/ggml-xrt/` + `include/ggml-xrt.h` + the
   `GGML_XRT` CMake wiring + `ggml-backend-reg.cpp` lines into llama.cpp's `ggml/`. Match a
   llama.cpp commit vendoring ggml ≈ v0.17.0.
3. **Validate the XRT ABI** (`TODO(hw)` in `ggml_backend_xrt_mul_mat`): confirm
   `kernel(opcode=3, bo_instr@grp1, count, A@grp3, B@grp4, C@grp5)` arg order, the
   **b-col-major** layout (ggml weight is row-major NxK), bo memory **groups**, and the
   instruction-file format, against an mlir-aie XRT host example. Fix, then test one MUL_MAT
   vs CPU reference.
4. **Point at kernels**: set `GGML_XRT_KERNEL_DIR` to the model's `prebuilt/<model>/` dir
   (+ `prebuilt/ops/`). Naming: `mul_mat_aie2_<dti>_<dto>_<M>x<K>x<N>_<cols|1c>.xclbin`,
   `<op>_<size>_aie2.xclbin`. Keep `ROW_TILE`/tile-length in the code in sync with the
   built artifacts.
5. **Host BF16 dequant**: quantized GGUF weights → BF16 on load (ggml
   `ggml_get_type_traits(t)->to_float`) so the NPU only sees BF16.
6. **Scheduler**: run llama.cpp with `[xrt, vulkan, cpu]` so the NPU takes conformant
   matmuls and Vulkan takes the rest. Verify end-to-end logits vs CPU.
7. **Enable ops incrementally**: turn on RMS_NORM/SiLU/GELU on the NPU once validated;
   author RoPE dispatch; then consider coarse per-layer NPU residency.
8. **Zero-copy (optimization, later)**: import the XRT `bo` host pointer into Vulkan via
   `VK_EXT_external_memory_host` (ggml-vulkan already supports host-pointer import).
   Unknowns: bo base must meet `minImportedHostPointerAlignment` (~4 KB); XRT `host_only` bo
   must be ordinary importable host pages.

## Rebuilding kernels (must stay on Linux/WSL)

`src/ggml-xrt/kernels/{build-mm-xclbin.sh,build-qwen3-matmuls.sh,build-ops.sh}` +
`headless_shim.py`. Requires an IRON env (mlir-aie + llvm-aie/peano). No NPU needed to
compile. New shapes: N must be %128 (4-col) or ≤2048 (single-core); otherwise leave on GPU.
