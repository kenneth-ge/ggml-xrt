#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the PRE-SCATTERED bf16 M=N VERIFY xclbin: the "host pre-scatters the weight"
# ceiling experiment. The weight arrives ALREADY bf16 AND already in the mmul B systolic-tile
# ("Bl1") order, so the core (aie2/mm_prescatter_bf16.cc) does NO on-core dequant and NO
# scatter — just the raw matmul_vectorized_4x4 MAC. Measures the pure-mmul ceiling vs the
# fused-q6k path (build-mm-verify.sh / mm_q6k_vecdq.cc). Design generator: gemv_mm_prescatter.py.
#
#   Usage: ./build-mm-prescatter.sh [<out_subdir> <M> <K> <N>]
#     default -> bench 16 6144 2048  (Qwen3 down q6_K K=6144 N=2048, one mmul M-tile).
#
# M-tile note: matmul_vectorized_4x4 steps rowA(=M/4) by 4 and colB(=N/4) by 4, so M and N must
# each be a multiple of 16. Smallest accepted M is 16 (rowA=4, one M-tile). With M=m=16 there
# is a single m-tile => single B stream => no serialize needed.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

subdir="${1:-bench}"
M="${2:-16}"; K="${3:-6144}"; N="${4:-2048}"
mm=16   # token sub-tile (== M, single m-tile); must be %16 for the 4x4 mmul
nn=32   # output-channel sub-tile; N must be divisible by nn and %16
DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

W="$(mktemp -d)"; cd "$W"

# Core: pre-scattered bf16 matmul (no dequant, no scatter). Object MUST be named
# mm_prescatter.o (gemv_mm_prescatter.py @core name).
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
  -DDIM_M="$mm" -DDIM_K=256 -DDIM_N="$nn" -Dbf16_f32_ONLY \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mm_prescatter_bf16.cc" -o mm_prescatter.o

python "${here}/gemv_mm_prescatter.py" --dev npu -M "$M" -K "$K" -N "$N" -m "$mm" -n "$nn" > aie.mlir

aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
  --peano "${PEANO_INSTALL_DIR}" --xclbin-name=mmprescatter.xclbin \
  --aie-generate-npu-insts --npu-insts-name=mmprescatter_insts.bin aie.mlir

out="mm_verify_prescatter_bf16_${K}x${N}_m${M}"
cp mmprescatter.xclbin    "${DST}/${out}.xclbin"
cp mmprescatter_insts.bin "${DST}/${out}_insts.bin"
echo "OK mm-prescatter (pre-scattered bf16, no dequant/scatter) M=${M} K=${K} N=${N} (m=${mm} n=${nn}) -> ${subdir}/${out}.xclbin"
