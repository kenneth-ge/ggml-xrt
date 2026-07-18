#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the q6k 6144x2048 decode-gemv bottleneck DECOMPOSITION xclbins (full / stream /
# compute). Same shape + DMA structure; only the core work differs. Host times each dispatch:
#   b(compute) ~= c(full)          -> compute-bound (vectorize the dequant+MAC)
#   a(stream)  ~= c(full) << b     -> streaming-bound
#   c(full) >> a(stream)+b(compute)-> per-dispatch / per-descriptor overhead dominates
#
#   Usage: ./build-gemv-bench.sh          # q6k 6144x2048 (down_proj) into prebuilt/bench/
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
K="${K:-6144}"; N="${N:-2048}"; MM="${MM:-32}"

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
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k.cc" -o mv_q6k.o

for mode in full stream compute; do
  python "${here}/gemv_q6k_bench.py" --dev npu -M "$N" -K "$K" -m "$MM" --mode "$mode" > aie.mlir 2>/dev/null
  if ! aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
       --peano "${PEANO_INSTALL_DIR}" --xclbin-name=b.xclbin \
       --aie-generate-npu-insts --npu-insts-name=b_insts.bin aie.mlir >/dev/null 2>&1; then
    echo "FAIL ${mode} ${K}x${N}"; continue
  fi
  cp b.xclbin    "${DST}/gemv_q6k_${K}x${N}_${mode}.xclbin"
  cp b_insts.bin "${DST}/gemv_q6k_${K}x${N}_${mode}_insts.bin"
  echo "OK bench ${mode} ${K}x${N} -> bench/"
done
echo "done -> ${DST}"
