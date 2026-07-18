#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build true M=1 gemv (matrix-vector) xclbins for the decode path, aie2/Phoenix.
# Uses the parameterized gemv.py (this dir) + a repo copy of mv.cc with the
# bf16->f32 combo enabled (the stock aie_kernels/aie2/mv.cc comments it out).
#
# gemv ABI differs from the tiled matmul (see docs/ggml-xrt-handoff.md step 11):
#   C[N] = A[N,K] . B[K];  A = WEIGHT in ggml-native [N,K] layout (NO transpose),
#   B = activation vector [K], C = output [N].  kernel(op=3, instr, ninstr, A, B, C).
#
# Naming: mul_mat_aie2_bf16_f32_1x{K}x{N}_gemv.xclbin  (leading M=1 so the backend's
# largest-tile-<=token selector picks it only for decode, M=1).
#
# Shapes: only N (output) with N/m <= 64 build (single-core gemv broadcast BD limit),
# i.e. N <= 2048 at m=32. Wide-N (e.g. gate/up 6144) falls back to the M=64 whole_array
# matmul. Compiled on Linux/WSL, NOT executed here.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
CLANG="${PEANO_INSTALL_DIR}/bin/clang++"
AK="${MLIR_AIE_SRC}/aie_kernels/aie2"   # for mv.cc's relative includes (../, zero.cc)

W="$(mktemp -d)"; cd "$W"
"${CLANG}" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw \
  -DNDEBUG -DDIM_M=32 -DDIM_K=32 -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv.cc" -o mv_32x32.o

gemv () { # K N   (ggml K, N; gemv -M is the output N)
  local K=$1 N=$2
  python "${here}/gemv.py" --dev npu -M "$N" -K "$K" -m 32 -k 32 --dtype_in bf16 --dtype_out f32 > aie.mlir
  aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
    --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=g_insts.bin aie.mlir
  cp g.xclbin   "${DST}/mul_mat_aie2_bf16_f32_1x${K}x${N}_gemv.xclbin"
  cp g_insts.bin "${DST}/mul_mat_aie2_bf16_f32_1x${K}x${N}_gemv_insts.bin"
  echo "gemv OK ${K}x${N}"
}

# Qwen3-1.7B: Q/O, K/V, down. gate/up (N=6144) exceeds the gemv BD limit -> M=64 fallback.
gemv 2048 2048
gemv 2048 1024
gemv 6144 2048
echo "done -> ${DST}"
