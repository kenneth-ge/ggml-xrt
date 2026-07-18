#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the 8-weight-channel 16-core decode gemv bench xclbins (aie2/Phoenix) for q6k 6144x2048
# via gemv_mc16_8ch.py:
#   gemv_q6k_6144x2048_16core_8ch.xclbin         - full drop-in kernel (7 weight ch + b broadcast)
#   gemv_q6k_6144x2048_16core_8ch_stream.xclbin  - weight-only ceiling (all 8 MM2S = weight, gutted)
# UNVALIDATED scaffold; compile-only (no NPU here). -> prebuilt/bench/
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
QT="${QT:-q6k}"; K="${K:-6144}"; N="${N:-2048}"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
declare -A CORE=( [q4_0]="mv_q4.cc:mv_q4_32x32.o" [q4k]="mv_q4k.cc:mv_q4k.o" [q6k]="mv_q6k.cc:mv_q6k.o" )
IFS=: read -r src obj <<< "${CORE[$QT]}"
W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" -c "${here}/aie2/${src}" -o "${obj}"

build() { # <extra-args> <name-suffix>
  local args="$1" suf="$2"
  python "${here}/gemv_mc16_8ch.py" --dev npu --qtype "$QT" -M "$N" -K "$K" -m "$MM" --rows 4 $args > aie.mlir 2>/dev/null
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
    cp g.xclbin "${DST}/gemv_${QT}_${K}x${N}_16core_8ch${suf}.xclbin"
    cp gi.bin   "${DST}/gemv_${QT}_${K}x${N}_16core_8ch${suf}_insts.bin"
    echo "OK  gemv_${QT}_${K}x${N}_16core_8ch${suf}"
  else
    echo "FAIL gemv_${QT}_${K}x${N}_16core_8ch${suf} (see below)"; tail -30 aiecc.log; return 1
  fi
}

build "--stream" "_stream"
build "" ""
echo "done -> ${DST}"
