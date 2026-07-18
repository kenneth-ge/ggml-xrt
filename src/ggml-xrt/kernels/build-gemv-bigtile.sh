#!/bin/bash
# LARGER-TILE experiment (aie2/Phoenix): build the q6k 6144x2048 16-core decode gemv with
# batched object_fifo acquire (KK k-blocks per acquire) via gemv_mc16_bigtile.py. Same real
# kernel (mv_q6k.cc) + ABI as the shipping 16-core gemv; only the acquire granularity changes.
# Builds kk=1 (== baseline control) and kk=2, kk=4 so the Windows agent can measure whether
# cutting acquire/loop overhead helps (i.e. whether decode is overhead- vs compute-bound).
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
W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k.cc" -o mv_q6k.o
for kk in 1 2 4; do
  python "${here}/gemv_mc16_bigtile.py" --dev npu --qtype q6k -M "$N" -K "$K" -m "$MM" --rows 4 --kk "$kk" > aie.mlir 2>err.txt || { echo "FAIL gen kk=${kk}"; sed -n '1,5p' err.txt; continue; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
    cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_kk${kk}.xclbin"
    cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_kk${kk}_insts.bin"
    echo "OK kk=${kk} q6k ${K}x${N} -> bench/gemv_q6k_${K}x${N}_16core_kk${kk}.xclbin"
  else echo "FAIL build kk=${kk}"; fi
done
echo "done -> ${DST}"
