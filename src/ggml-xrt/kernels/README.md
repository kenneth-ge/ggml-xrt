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

`prebuilt/` holds artifacts for the Windows side to consume directly:

- `build-mm-xclbin.sh` — the minimal 256³ demo matmul.
- `build-qwen3-matmuls.sh` — the Qwen3-1.7B weight-matmul set in two tiers:
  - **prefill**: `whole_array`, 4 columns, `M=256` (`*_4c`) — host chunks tokens into
    M-sized blocks (`whole_array` requires M ≥ 128).
  - **decode/small-batch**: `single_core`, `M=32` (`*_1c`) — host pads 1 token up to 32.

Together with host-side M-tiling this covers any token count with a finite, AOT set (no
JIT). See `docs/ggml-xrt-plan.md` §8 (dynamic-M) and §9 (hybrid NPU+GPU split).

## Decode gemv kernels (`gemv.py` + `build-gemv.sh`)

For the decode path (one token, M=1), a true matrix-vector kernel avoids the padded-MAC
waste of the small-M matmul tiles. `gemv.py` is a **parameterized** IRON design; build with:

```bash
# ./build-gemv.sh <out_subdir_under_prebuilt> <K> <N> [<K> <N> ...]
./build-gemv.sh qwen3-14b       5120 1024              # one shape
./build-gemv.sh gemma4-26b-a4b  2816 2048  2816 704    # several at once
./build-gemv.sh .               2048 2048              # into prebuilt/ root
./build-gemv.sh                                        # no args -> Qwen3-1.7B set
```

Each `K N` is a ggml `MUL_MAT` (weight `[N,K]`, activation `[K]`); output is
`mul_mat_aie2_bf16_f32_1x{K}x{N}_gemv.xclbin` (leading `M=1` so the backend picks it only
for decode). To add a model: pass its `(K,N)` projection shapes.

**Rule of thumb — which shapes get a gemv:** only output `N ≤ 2048` (and `N%32==0`,
`K%32==0`); the script SKIPs the rest and prints why. That's the single-core broadcast BD
limit (`N/32 ≤ 64`). In practice this is the K/V projections and small MoE-expert FFNs;
wide outputs (Q/O at 4096–6144, gate/up, down at ≥2560) can't gemv and keep the small-M
`whole_array` matmul fallback (or would need host N-tiling of the gemv output).

gemv ABI differs from the tiled matmul — **weight is fed untransposed** (native `[N,K]`);
`B` = activation vector, `C` = output. Wire a separate dispatch branch (handoff step 11).
Kernel source: `gemv.py` + `aie2/mv.cc` (the bf16→f32 combo is enabled here; stock
`mlir-aie` `mv.cc` comments it out, and its scalar matvec must promote operands to float
before multiplying — else it accumulates a K-proportional bias).

> Note: `--dev npu` targets Phoenix/npu1/aie2; `--dev npu2` targets Strix/aie2p.
