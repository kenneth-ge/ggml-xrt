#!/bin/bash
# ISOLATION EXPERIMENT (aie2/Phoenix): build the q6k 6144x2048 16-core decode gemv in THREE
# variants that share identical DMA/core/ABI structure and differ ONLY in the per-record work:
#   full     = mv_q6k.cc          (real: dequant + MAC)
#   maconly  = mv_q6k_maconly.cc  (MAC + reduce + DMA, dequant stripped) -> DMA+MAC floor
#   dqonly   = mv_q6k_dqonly.cc   (full dequant, MAC replaced by cheap add) -> dequant cost
#
# Measuring all three on-device answers where the ~9.5 ms goes:
#   dequant cost ~= full - maconly ;  MAC/M=1 floor ~= maconly.
# maconly/dqonly produce GARBAGE output (timing only). Distinct xclbin names, drops in prebuilt/bench/.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
K="${K:-6144}"; N="${N:-2048}"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
# variant tag -> source file
declare -A SRC=( [full]="mv_q6k.cc" [maconly]="mv_q6k_maconly.cc" [dqonly]="mv_q6k_dqonly.cc" )
W="$(mktemp -d)"; cd "$W"
for v in full maconly dqonly; do
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/${SRC[$v]}" -o mv_q6k.o
  python "${here}/gemv_mc16.py" --dev npu --qtype q6k -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>/dev/null
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
    cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_${v}.xclbin"
    cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_${v}_insts.bin"
    echo "OK ${v} q6k ${K}x${N} -> bench/gemv_q6k_${K}x${N}_16core_${v}.xclbin"
  else echo "FAIL ${v} q6k ${K}x${N}"; fi
done
echo "done -> ${DST}"
