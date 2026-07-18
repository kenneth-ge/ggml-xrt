#!/bin/bash
# VECTORIZED-SCALE fix (aie2/Phoenix): rebuild the 16-core decode gemv with mv_${QT}_vscale.cc,
# which replaces the scalar-fp32 per-group scale setup (__floatsisf/__mulsf3, the real ~89% of
# decode time) with vector ops. SAME repack contract + ABI as the shipping gemv (drop-in, no host
# change). QT=q6k (212 B) or q4k (148 B); q4_0 already has no software-fp scale so it's untouched.
#   QT=q6k|q4k   MODE=bench|real   e.g.: QT=q4k MODE=real ./build-gemv-vscale.sh 2048 2048 2048 1024 2048 6144
# Default: bench build at 6144x2048 (q6k). Pass shapes ("K N K N ...") for the real _1x{K}x{N}_gemv.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
QT="${QT:-q6k}"       # q6k | q4k
MODE="${MODE:-bench}"   # bench -> prebuilt/bench/gemv_..._vscale.xclbin ; real -> prebuilt/mul_mat_..._gemv.xclbin
DST="${here}/prebuilt"; [ "$MODE" = bench ] && DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
declare -A OBJ=( [q6k]="mv_q6k.o" [q4k]="mv_q4k.o" )
if [ "$#" -eq 0 ]; then set -- 6144 2048; fi
W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_${QT}_vscale.cc" -o "${OBJ[$QT]}"
while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % (MM * 16))) -ne 0 ]; then echo "SKIP ${K}x${N} (N%(m*16)!=0)"; continue; fi
  python "${here}/gemv_mc16.py" --dev npu --qtype "$QT" -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>/dev/null
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
    if [ "$MODE" = bench ]; then
      cp g.xclbin "${DST}/gemv_${QT}_${K}x${N}_16core_vscale.xclbin"; cp gi.bin "${DST}/gemv_${QT}_${K}x${N}_16core_vscale_insts.bin"
      echo "OK vscale ${QT} ${K}x${N} -> bench/gemv_${QT}_${K}x${N}_16core_vscale.xclbin"
    else
      cp g.xclbin "${DST}/mul_mat_aie2_${QT}_f32_1x${K}x${N}_gemv.xclbin"; cp gi.bin "${DST}/mul_mat_aie2_${QT}_f32_1x${K}x${N}_gemv_insts.bin"
      echo "OK vscale ${QT} ${K}x${N} -> mul_mat_aie2_${QT}_f32_1x${K}x${N}_gemv.xclbin (REAL, drop-in)"
    fi
  else echo "FAIL vscale ${QT} ${K}x${N}"; fi
done
echo "done -> ${DST}"
