#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build true M=1 gemv (matrix-vector) xclbins for the decode path, aie2/Phoenix.
# One gemv kernel per weight matmul with output dim N <= 2048 (see LIMITS below).
#
#   Usage:  ./build-gemv.sh [<out_subdir> <K> <N> [<K> <N> ...]]
#     e.g.  ./build-gemv.sh qwen3-14b 5120 1024
#           ./build-gemv.sh .          2048 2048  2048 1024  6144 2048
#     no args -> rebuild the Qwen3-1.7B set into prebuilt/ (Q/O, K/V, down).
#   <out_subdir> is under prebuilt/ (use "." for the prebuilt root).
#   Each K N is a ggml MUL_MAT (weight [N,K], activation [K]) -> gemv output dim N.
#
# What a gemv is: C[N] = A[N,K].B[K].  A = WEIGHT in ggml-native [N,K] layout (NO
# transpose, unlike the tiled matmul); B = activation vector [K]; C = output [N].
# Kernel ABI: kernel(opcode=3, instr@grp1, ninstr, A_bo, B_bo, C_bo). Backend picks
# it only for decode (filename leading M=1). See docs/ggml-xrt-handoff.md step 11.
#
# LIMITS (aie2, m=32): output N must be a multiple of 32 AND N/32 <= 64  =>  N <= 2048
# (the single-core broadcast DMA BD range is [1:64]); K must be a multiple of 32.
# Wider-N projections (e.g. gate/up 6144, Q 5120) can't gemv -> they use the small-M
# whole_array matmul fallback (M=64), or need host N-tiling of the gemv output.
#
# Uses gemv.py (this dir) + aie2/mv.cc (bf16->f32 combo enabled; stock mv.cc comments
# it out). Compiled on Linux/WSL, NOT executed here.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then set -- . 2048 2048 2048 1024 6144 2048; fi   # default: Qwen3-1.7B
subdir="$1"; shift
DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
CLANG="${PEANO_INSTALL_DIR}/bin/clang++"
AK="${MLIR_AIE_SRC}/aie_kernels/aie2"

W="$(mktemp -d)"; cd "$W"
"${CLANG}" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw \
  -DNDEBUG -DDIM_M=32 -DDIM_K=32 -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv.cc" -o mv_32x32.o

while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % 32)) -ne 0 ] || [ $((N / 32)) -gt 64 ] || [ $((K % 32)) -ne 0 ]; then
    echo "SKIP ${K}x${N} (needs N%32==0, N<=2048, K%32==0)"; continue
  fi
  python "${here}/gemv.py" --dev npu -M "$N" -K "$K" -m 32 -k 32 --dtype_in bf16 --dtype_out f32 > aie.mlir
  aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
    --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=g_insts.bin aie.mlir >/dev/null 2>&1
  cp g.xclbin    "${DST}/mul_mat_aie2_bf16_f32_1x${K}x${N}_gemv.xclbin"
  cp g_insts.bin "${DST}/mul_mat_aie2_bf16_f32_1x${K}x${N}_gemv_insts.bin"
  echo "OK gemv ${K}x${N} -> ${subdir}/"
done
echo "done -> ${DST}"
