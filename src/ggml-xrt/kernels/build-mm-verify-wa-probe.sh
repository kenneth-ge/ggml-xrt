#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build a PROBE xclbin for the 16-core whole_array Q6_K verify: same memtile-A path + DIM_M=32 +
# cpass C write-back as mm_verify_wa_q6k_..._m32, but with a probe core that isolates one stage.
#
#   Usage: ./build-mm-verify-wa-probe.sh <core_cc_basename> <out_name> [<K> <N>]
#     e.g.  ./build-mm-verify-wa-probe.sh mm_q6k_wa_aecho mm_verify_wa_aecho
#           ./build-mm-verify-wa-probe.sh mm_q6k_wa_wecho mm_verify_wa_wecho
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

core_cc="${1:?core .cc basename required (e.g. mm_q6k_wa_aecho)}"
outname="${2:?out name required (e.g. mm_verify_wa_aecho)}"
K="${3:-6144}"; N="${4:-2048}"
M=32; mm=32; nn=32
subdir="bench"; DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

W="$(mktemp -d)"; cd "$W"

# Probe core (memtile-A signature: qB, A, Bl1, C). Object MUST be named mm_q6k.o.
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
  -DDIM_M="$mm" -DDIM_K=256 -DDIM_N="$nn" -Dbf16_f32_ONLY \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/${core_cc}.cc" -o mm_q6k.o

python "${here}/gemv_mm16.py" --dev npu -M "$M" -K "$K" -N "$N" -m "$mm" -n "$nn" --a-memtile > aie.mlir

aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
  --peano "${PEANO_INSTALL_DIR}" --xclbin-name=mmverify.xclbin \
  --aie-generate-npu-insts --npu-insts-name=mmverify_insts.bin aie.mlir

cp mmverify.xclbin    "${DST}/${outname}.xclbin"
cp mmverify_insts.bin "${DST}/${outname}_insts.bin"
echo "OK probe ${core_cc} -> ${subdir}/${outname}.xclbin"
