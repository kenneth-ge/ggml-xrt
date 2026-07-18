#!/bin/bash
# PRE-DEQUANT experiment (aie2/Phoenix): build the q6k 6144x2048 16-core decode gemv reading the
# NEW 276-B pre-assembled-int8 record (mv_q6k_predq.cc) via gemv_mc16.py --qtype q6k_pd. Kills the
# 89% on-core 6-bit dequant by moving assembly into the host repack. Drops in prebuilt/bench/.
# Requires the host to produce the new record (see relay spec). Correct output (validatable).
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
  -c "${here}/aie2/mv_q6k_predq_vscale.cc" -o mv_q6k.o
python "${here}/gemv_mc16.py" --dev npu --qtype q6k_pd -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>err.txt || { echo "FAIL gen"; sed -n '1,8p' err.txt; exit 1; }
if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
     --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
  cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_predq.xclbin"
  cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_predq_insts.bin"
  echo "OK predq q6k ${K}x${N} -> bench/gemv_q6k_${K}x${N}_16core_predq.xclbin"
else echo "FAIL build predq"; fi
echo "done -> ${DST}"
