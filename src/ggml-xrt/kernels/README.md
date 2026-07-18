# ggml-xrt kernels (WSL / Linux build)

The AIE (XDNA NPU) kernels consumed by the `ggml-xrt` backend are compiled to
`.xclbin` + instruction-sequence artifacts on **Linux/WSL** — the mlir-aie / IRON /
peano toolchain is Linux-only. The Windows `ggml-xrt` host binary loads these
artifacts unchanged via XRT (they are OS-agnostic AIE device code for the same
`aie2` silicon).

See `../../../docs/ggml-xrt-plan.md` for the full plan.

## Prerequisites (WSL)

- An IRON environment (mlir-aie + `llvm-aie`/peano). This repo was validated against
  the setup in `~/ironenv` + `~/mlir-aie`.
- No NPU is required to *compile* kernels. Because `import aie.iron` eagerly opens a
  device, use `headless_shim.py` (installed as `sitecustomize.py`) to stub it out.

## Build a MUL_MAT xclbin (Phoenix / aie2)

```bash
# 1. Headless shim so aie.iron imports without an NPU
mkdir -p /tmp/xrtshim && cp headless_shim.py /tmp/xrtshim/sitecustomize.py
export PYTHONPATH=/tmp/xrtshim:$PYTHONPATH

# 2. IRON / mlir-aie env
source ~/ironenv/bin/activate
export MLIR_AIE_INSTALL=$(python3 -c "import mlir_aie; print(mlir_aie.__path__[0])")
source ~/mlir-aie/utils/env_setup.sh "$MLIR_AIE_INSTALL"
export PEANO_INSTALL_DIR=~/ironenv/lib/python3.12/site-packages/llvm-aie

# 3. Compile (dev=npu == Phoenix/aie2/npu1; peano, no chess)
cd ~/mlir-aie/programming_examples/basic/matrix_multiplication/single_core
make M=256 K=256 N=256 dtype_in=bf16 dtype_out=f32 \
     build/final_256x256x256_32x32x32.xclbin \
     build/insts_256x256x256_32x32x32.txt
```

Outputs: `build/final_*.xclbin` (device image) and `build/insts_*.txt|.bin` (control
instruction sequence). Copy both into the backend's `GGML_XRT_KERNEL_DIR`, named per
the kernel-key scheme in `src/ggml-hsa/kernel-discovery.cpp`.

`prebuilt/` holds artifacts produced by this recipe for the Windows side to consume
directly. Regenerate them with `build-mm-xclbin.sh`.

> Note: `--dev npu` targets Phoenix/npu1/aie2; `--dev npu2` targets Strix/aie2p.
