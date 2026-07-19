#!/bin/bash
# TIMING PROBES for the M=1 q6k decode gemv stall analysis (aie2/Phoenix, 6144x2048, 16-core).
# Mirrors build-gemv-sweep.sh (compile .cc -> mv_q6k.o, gemv_mc16*.py -> aie.mlir -> aiecc xclbin).
# Compile-only (WSL); bench-only, NOT shipped. Output names match the sweep convention so the
# measuring agent can mm_time each against shipping gemv_q6k_6144x2048_16core.xclbin (~1.05ms).
#
#   regblock : H#3  register-block B=4 activation reuse (mv_q6k_regblock.cc, gemv_mc16.py, m=32)
#   fifo1    : H#4  weight-stream object_fifo depth 1 (mv_q6k_noscratch.cc, gemv_mc16_wfifo.py --wdepth 1)
#   fifo2    : H#4  weight-stream object_fifo depth 2 = shipping sanity (--wdepth 2)
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

QT=q6k; K=6144; N=2048; MM=32

# variant: "<tag> <src.cc> <pygen> <extra-py-args>"
VARIANTS=(
  "regblock mv_q6k_regblock.cc  gemv_mc16.py        "
  "fifo1    mv_q6k_noscratch.cc gemv_mc16_wfifo.py  --wdepth 1"
  "fifo2    mv_q6k_noscratch.cc gemv_mc16_wfifo.py  --wdepth 2"
)

build_one() {
  local tag="$1" src="$2" pygen="$3"; shift 3; local pyargs="$*"
  local w; w="$(mktemp -d)"; cd "$w"
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/${src}" -o mv_q6k.o 2>core.log || { echo "FAIL core ${tag}"; sed -n '1,40p' core.log; return; }
  python "${here}/${pygen}" --dev npu --qtype "$QT" -M "$N" -K "$K" -m "$MM" --rows 4 $pyargs > aie.mlir 2>py.log \
    || { echo "FAIL pygen ${tag}"; sed -n '1,40p' py.log; return; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
    cp g.xclbin "${DST}/gemv_${QT}_${K}x${N}_16core_${tag}.xclbin"
    cp gi.bin   "${DST}/gemv_${QT}_${K}x${N}_16core_${tag}_insts.bin"
    echo "OK ${tag} -> bench/gemv_${QT}_${K}x${N}_16core_${tag}.xclbin"
  else echo "FAIL build ${tag}"; tail -30 aiecc.log; fi
}

for v in "${VARIANTS[@]}"; do build_one $v; done
echo "timingprobe done -> ${DST}"
