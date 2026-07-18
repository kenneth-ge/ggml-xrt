#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the multi-column gemv-stream ceiling benchmark (aie2/Phoenix). Gutted known-good gemv
# (dequant/MAC removed) replicated across 1/2/4 columns; each column streams a K x N q6_K
# weight from DDR. Host times a dispatch: aggregate GB/s = (cols * N*(K/256)*212) / wall.
#   C=1 must reproduce ~3.59 GB/s (the working single-col floor); C=4 answers whether parallel
#   columns multiply DDR bandwidth (go/no-go for CPU-parity decode).
#
#   Usage: ./build-stream-gemv.sh          # K=6144 N=2048 per col, cols 1/2/4 -> prebuilt/stream/
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/stream"; mkdir -p "$DST"
K="${K:-6144}"; N="${N:-2048}"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M=32 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k.cc" -o mv_q6k.o

for cols in 1 2 4; do
  python "${here}/stream_gemv.py" --dev npu -K "$K" -N "$N" -m 32 --cols "$cols" > aie.mlir 2>/dev/null
  if ! aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
       --peano "${PEANO_INSTALL_DIR}" --xclbin-name=sg.xclbin \
       --aie-generate-npu-insts --npu-insts-name=sg_insts.bin aie.mlir >/dev/null 2>&1; then
    echo "FAIL stream-gemv ${cols}col"; continue
  fi
  cp sg.xclbin    "${DST}/stream_gemv_aie2_${cols}col_${K}x${N}.xclbin"
  cp sg_insts.bin "${DST}/stream_gemv_aie2_${cols}col_${K}x${N}_insts.bin"
  perb=$(( N * (K / 256) * 212 ))
  echo "OK stream-gemv ${cols}col (${cols} x ${perb} B = $(( cols * perb )) B total)"
done
echo "done -> ${DST}"
