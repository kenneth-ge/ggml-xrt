#!/bin/bash
# TIMING PROBES (aie2/Phoenix): build the q6k 6144x2048 16-core decode gemv in two variants that
# share identical DMA/core/ABI structure with the SHIPPING kernel (mv_q6k_noscratch.cc) and differ
# ONLY in the per-record compute, to test whether q6k dequant is a large SERIAL fraction on the one
# AIE2 vector datapath (competing with aie::mac):
#   maconly_clean = mv_q6k_maconly_clean.cc  (H#2a: one convert reused, per-chunk = mac+load only)
#   deqfold       = mv_q6k_deqfold.cc         (H#2b: correct output, -32 bias-folded, 8 vsubs gone)
# Compare on-device vs. the shipping gemv_q6k_6144x2048_16core_noscratch.xclbin (~1.05 ms):
#   dequant TRUE cost  ~=  noscratch - maconly_clean ;  bias-fold win  ~=  noscratch - deqfold.
# maconly_clean produces GARBAGE output (timing only); deqfold is mm_verify-able. DIM_M=32.
# Distinct xclbin names, drop in prebuilt/bench/. MIRRORS build-gemv-isolation.sh (only src+tag swapped).
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
declare -A SRC=( [maconly_clean]="mv_q6k_maconly_clean.cc" [deqfold]="mv_q6k_deqfold.cc" )
W="$(mktemp -d)"; cd "$W"
for v in maconly_clean deqfold; do
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
