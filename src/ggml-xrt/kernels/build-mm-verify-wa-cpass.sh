#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the C-PASSTHROUGH PROBE xclbin for the 16-core whole_array Q6_K verify: identical design
# (gemv_mm16.py) but the core (aie2/mm_q6k_wa_cpass.cc) SKIPS dequant+mmul and stamps
# C[i]=(float)(i+1). Isolates the C write-back/gather/transform path from the compute path.
#
#   Usage: ./build-mm-verify-wa-cpass.sh [<out_subdir> <M> <K> <N>]
#     default -> bench 16 6144 2048.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

subdir="${1:-bench}"
M="${2:-16}"; K="${3:-6144}"; N="${4:-2048}"
mm=16
nn=32
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

# Core: C-passthrough probe. Object MUST be named mm_q6k.o (gemv_mm16.py @core name).
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
  -DDIM_M="$mm" -DDIM_K=256 -DDIM_N="$nn" -Dbf16_f32_ONLY \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mm_q6k_wa_cpass.cc" -o mm_q6k.o

python "${here}/gemv_mm16.py" --dev npu -M "$M" -K "$K" -N "$N" -m "$mm" -n "$nn" > aie.mlir

aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
  --peano "${PEANO_INSTALL_DIR}" --xclbin-name=mmverify.xclbin \
  --aie-generate-npu-insts --npu-insts-name=mmverify_insts.bin aie.mlir

out="mm_verify_wa_q6k_${K}x${N}_m${M}_cpass"
cp mmverify.xclbin    "${DST}/${out}.xclbin"
cp mmverify_insts.bin "${DST}/${out}_insts.bin"
echo "OK mm-verify-wa CPASS probe (C[i]=(float)(i+1)) M=${M} K=${K} N=${N} -> ${subdir}/${out}.xclbin"
