#!/bin/bash
# VECTORIZED-SCALE fix (aie2/Phoenix): rebuild q6k 16-core decode gemv with mv_q6k_vscale.cc, which
# replaces the scalar-fp32 per-group scale setup (__floatsisf/__mulsf3, ~89% of decode time) with
# vector ops. SAME 212 B repack contract + ABI as the shipping q6k gemv (drop-in, no host change).
# Default: bench build at 6144x2048. Pass shapes ("K N K N ...") to build the real _1x{K}x{N}_gemv
# names into prebuilt/ for rollout once validated.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
MODE="${MODE:-bench}"   # bench -> prebuilt/bench/gemv_..._vscale.xclbin ; real -> prebuilt/mul_mat_..._gemv.xclbin
DST="${here}/prebuilt"; [ "$MODE" = bench ] && DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
if [ "$#" -eq 0 ]; then set -- 6144 2048; fi
W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k_vscale.cc" -o mv_q6k.o
while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % (MM * 16))) -ne 0 ]; then echo "SKIP ${K}x${N} (N%(m*16)!=0)"; continue; fi
  python "${here}/gemv_mc16.py" --dev npu --qtype q6k -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>/dev/null
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
    if [ "$MODE" = bench ]; then
      cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_vscale.xclbin"; cp gi.bin "${DST}/gemv_q6k_${K}x${N}_16core_vscale_insts.bin"
      echo "OK vscale ${K}x${N} -> bench/gemv_q6k_${K}x${N}_16core_vscale.xclbin"
    else
      cp g.xclbin "${DST}/mul_mat_aie2_q6k_f32_1x${K}x${N}_gemv.xclbin"; cp gi.bin "${DST}/mul_mat_aie2_q6k_f32_1x${K}x${N}_gemv_insts.bin"
      echo "OK vscale ${K}x${N} -> mul_mat_aie2_q6k_f32_1x${K}x${N}_gemv.xclbin (REAL, drop-in)"
    fi
  else echo "FAIL vscale ${K}x${N}"; fi
done
echo "done -> ${DST}"
