#!/bin/bash
# STEP-3 CONTIGUOUS WEIGHT LAYOUT gemv (aie2/Phoenix): the M=1 q6k weight-DMA lever. Builds the
# 16-core decode gemv with a FULLY-CONTIGUOUS weight DMA (gemv_mc16_contig.py) + the winning
# 4-accumulator compute kernel (mv_q6k_4acc.cc). Requires the weight to be pre-repacked into the
# contiguous A[c][i][t][jj][REC] layout (tools/repack_q6k_contig.py -> host: add
# ggml_xrt_repack_quant_weight_contig, see report). Same ABI (matvec_q6k_f32), same C order,
# byte-identical dequant -> mm_verify must still pass. Compile-only (WSL); bench-only, NOT shipped.
#
# Gate: only meaningful if the Step-2 contigstream probe times ~0.45 ms (~23 GB/s). If it does,
# this kernel should land near that floor (compute is in the DMA shadow) vs the shipping ~0.95 ms.
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
  -c "${here}/aie2/mv_q6k_4acc.cc" -o mv_q6k.o 2>core.log || { echo "FAIL core"; sed -n '1,40p' core.log; exit 1; }

python "${here}/gemv_mc16_contig.py" --dev npu --qtype q6k -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>py.log \
  || { echo "FAIL pygen"; sed -n '1,40p' py.log; exit 1; }
if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
     --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
  cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_contiglayout.xclbin"
  cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_contiglayout_insts.bin"
  echo "OK contiglayout -> bench/gemv_q6k_${K}x${N}_16core_contiglayout.xclbin"
else echo "FAIL build"; tail -40 aiecc.log; exit 1; fi
echo "contiglayout done -> ${DST}"
