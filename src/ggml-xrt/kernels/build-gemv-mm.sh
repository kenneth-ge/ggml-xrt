#!/bin/bash
# Build the 16-core M=N quantized VERIFY kernel (aie2/Phoenix): C[MT,N] = A[MT,K] . dequant(W[N,K]).
# The weight dequant+DMA is amortized over MT draft tokens (aie2/mv_mm_q6k.cc + gemv_mm.py), so MT
# tokens cost ~one gemv (the weight stream) instead of MT gemvs. Default: q6k, MT=8, K=6144 N=2048
# (Qwen3-1.7B down proj), -> prebuilt/bench/mm_verify_q6k_6144x2048_m8.xclbin.
#   QT=q6k  MT=8   e.g.:  ./build-gemv-mm.sh 6144 2048   (or pass "K N K N ..." for more shapes)
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
QT="${QT:-q6k}"       # q6k
MT="${MT:-8}"         # draft tokens M
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
declare -A OBJ=( [q6k]="mv_mm_q6k.o" [q4k]="mv_mm_q4k.o" )
declare -A SRC=( [q6k]="mv_mm_q6k.cc" [q4k]="mv_mm_q4k.cc" )
if [ "$#" -eq 0 ]; then set -- 6144 2048; fi
W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_MT="$MT" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/${SRC[$QT]}" -o "${OBJ[$QT]}"
while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % (MM * 16))) -ne 0 ]; then echo "SKIP ${K}x${N} (N%(m*16)!=0)"; continue; fi
  python "${here}/gemv_mm.py" --dev npu --qtype "$QT" -M "$N" -K "$K" -m "$MM" --rows 4 --mt "$MT" > aie.mlir 2>err.log || { echo "GEN FAIL ${QT} ${K}x${N}"; cat err.log; continue; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
    cp g.xclbin "${DST}/mm_verify_${QT}_${K}x${N}_m${MT}.xclbin"
    cp gi.bin "${DST}/mm_verify_${QT}_${K}x${N}_m${MT}_insts.bin"
    echo "OK mm_verify ${QT} ${K}x${N} MT=${MT} -> bench/mm_verify_${QT}_${K}x${N}_m${MT}.xclbin"
  else echo "FAIL mm_verify ${QT} ${K}x${N} MT=${MT}"; tail -40 aiecc.log; fi
done
echo "done -> ${DST}"
