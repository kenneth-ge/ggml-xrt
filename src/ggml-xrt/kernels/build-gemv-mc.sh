#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the REAL multi-column decode gemvs (aie2/Phoenix) for Qwen3-1.7B, all three dtypes,
# via gemv_mc.py (--cols 4 default). HW-validated ~4x over single-column vec, arithmetically
# exact (NRMSE unchanged). Drop-in: same _1x{K}x{N}_gemv.xclbin names/ABI as the single-column
# kernels (M=1 decode; N is split across the 4 columns internally). Overwrites the
# single-column reals in prebuilt/.
#
#   Usage: ./build-gemv-mc.sh [<out_subdir>]     (default: prebuilt/)
# N must be divisible by cols*m (=128 at cols=4,m=32); every Qwen3-1.7B decode shape qualifies,
# including gate/up N=6144 (48 tiles/col, so no m=96 hack needed).
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLS="${COLS:-4}"; MM="${MM:-32}"
subdir="${1:-.}"; DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

# qtype:coresrc:coreobj:outtok  and the Qwen3-1.7B decode shapes per dtype (K N ...)
# Q4_K_M = q4k (q/k/o, gate/up) + q6k (v, down). q4_0 built too for Q4_0 models.
declare -A CORE=( [q4_0]="mv_q4.cc:mv_q4_32x32.o" [q4k]="mv_q4k.cc:mv_q4k.o" [q6k]="mv_q6k.cc:mv_q6k.o" )
declare -A NAME=( [q4_0]="q4_0" [q4k]="q4k" [q6k]="q6k" )
declare -A SHAPES=( [q4_0]="2048 2048 2048 1024 6144 2048" [q4k]="2048 2048 2048 1024 2048 6144" [q6k]="2048 1024 6144 2048" )

W="$(mktemp -d)"; cd "$W"
for qt in q4_0 q4k q6k; do
  IFS=: read -r src obj <<< "${CORE[$qt]}"
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/${src}" -o "${obj}"
  set -- ${SHAPES[$qt]}
  while [ "$#" -ge 2 ]; do
    K="$1"; N="$2"; shift 2
    if [ $((N % (COLS * MM))) -ne 0 ]; then echo "SKIP ${qt} ${K}x${N} (N%(cols*m)!=0)"; continue; fi
    python "${here}/gemv_mc.py" --dev npu --qtype "$qt" -M "$N" -K "$K" -m "$MM" --cols "$COLS" > aie.mlir 2>/dev/null
    if ! aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
         --peano "${PEANO_INSTALL_DIR}" --xclbin-name=g.xclbin \
         --aie-generate-npu-insts --npu-insts-name=g_insts.bin aie.mlir >/dev/null 2>&1; then
      echo "FAIL ${qt} ${K}x${N}"; continue
    fi
    cp g.xclbin    "${DST}/mul_mat_aie2_${NAME[$qt]}_f32_1x${K}x${N}_gemv.xclbin"
    cp g_insts.bin "${DST}/mul_mat_aie2_${NAME[$qt]}_f32_1x${K}x${N}_gemv_insts.bin"
    echo "OK ${qt} ${K}x${N} (${COLS}col)"
  done
done
echo "done -> ${DST}"
