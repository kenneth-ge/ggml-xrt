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
6. **Scheduler — FIRST TOKEN ACHIEVED on NPU (partial).** Ran `Qwen3-1.7B-Q4_K_M.gguf` through
   `llama-cli` (`-ngl 0`, weights on CPU) and the NPU automatically took the conformant ops via
   the scheduler — our device is `GGML_BACKEND_DEVICE_TYPE_ACCEL`, which llama.cpp does NOT use
   for `-ngl` layer offload (it's skipped like CPU), but the scheduler still routes matching ops
   to it (like BLAS). Confirmed loaded+run on-device: all four projection matmuls
   (2048×2048, 2048×1024, 2048×6144, 6144×2048) **and** `rms_norm_2048`; the model produced a
   token. Weights dequant Q4_K→BF16 in-backend; no Vulkan yet (`GGML_VULKAN=OFF`).
   REMAINING: (a) full logit / multi-token correctness vs a CPU-only run (compare — force CPU by
   pointing `GGML_XRT_KERNEL_DIR` at an empty dir so `supports_op` returns false everywhere);
   (b) rebuild llama with `-DGGML_VULKAN=ON` for the true `[xrt,vulkan,cpu]` hybrid.
   NOTE: do NOT run llama-cli inside Claude Code (crashes it); llama.cpp also has a benign
   teardown hang on exit (present in stock llama.cpp). Run inference in a normal terminal.
7. **Enable ops incrementally**: RMS_NORM/SiLU/GELU are now **gated OFF by default** behind
   `GGML_XRT_ENABLE_OPS=1` — they are not numerically validated, and an unvalidated on-device
   RMS_NORM corrupts every layer's activations (observed: degenerate "GGGG…" output). Default is
   MUL_MAT-only (step 6 scope). To validate an op: build a small unit harness like
   `mulmat_check.cpp` (NPU vs CPU for that op), fix its ABI/layout, THEN enable via the env var.
   RoPE dispatch still unwritten. Then consider coarse per-layer NPU residency.
8. **Zero-copy (optimization, later)**: import the XRT `bo` host pointer into Vulkan via
   `VK_EXT_external_memory_host` (ggml-vulkan already supports host-pointer import).
   Unknowns: bo base must meet `minImportedHostPointerAlignment` (~4 KB); XRT `host_only` bo
   must be ordinary importable host pages.

## Rebuilding kernels (must stay on Linux/WSL)

`src/ggml-xrt/kernels/{build-mm-xclbin.sh,build-qwen3-matmuls.sh,build-ops.sh}` +
`headless_shim.py`. Requires an IRON env (mlir-aie + llvm-aie/peano). No NPU needed to
compile. New shapes: N must be %128 (4-col) or ≤2048 (single-core); otherwise leave on GPU.
