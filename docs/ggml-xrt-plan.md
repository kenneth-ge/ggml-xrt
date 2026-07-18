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
