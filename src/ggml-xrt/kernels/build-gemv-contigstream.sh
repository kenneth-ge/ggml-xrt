#!/bin/bash
# STEP-2 CONTIGUOUS-STREAM BW PROBE (aie2/Phoenix): build pure-drain weight-DMA probes for the
# q6k down 6144x2048 16-core gemv. Same 4-column/shim topology + 6784 B object granule as the
# shipping kernel; cores only acquire/release (NO dequant, NO MAC) so wall == weight-DMA time.
# Two variants differ ONLY in the DDR access pattern of the weight npu_dma_memcpy_nd:
#   contigstream  : one fully-contiguous 2.6 MB blast per column (6784 B bursts)   <- the GATE
#   stridedstream : shipping 212 B burst / 5088 B stride pattern (control ~11 GB/s reference)
# The measuring agent times each: if contig ~0.45 ms (~23 GB/s) and strided ~0.95 ms (~11 GB/s),
# the striding IS the lever (Step 3 worth it); if contig ALSO caps ~0.95 ms, the shim/memtile is
# the wall (lever dead — STOP). Compile-only (WSL); bench-only, NOT shipped.
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
# object only needs zero_scalar_f32 (drain cores never call matvec); maconly_clean provides it.
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k_maconly_clean.cc" -o mv_q6k.o

# variant tag -> (pattern, output-name-suffix)
declare -A PAT=( [contigstream]="contig" [stridedstream]="strided" )
for v in contigstream stridedstream; do
  python "${here}/gemv_contigstream.py" --dev npu -K "$K" -N "$N" -m "$MM" --cols 4 --depth 2 \
    --pattern "${PAT[$v]}" > aie.mlir 2>py.log \
    || { echo "FAIL pygen ${v}"; sed -n '1,40p' py.log; continue; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
    cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_${v}.xclbin"
    cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_${v}_insts.bin"
    echo "OK ${v} -> bench/gemv_q6k_${K}x${N}_16core_${v}.xclbin"
  else echo "FAIL build ${v}"; tail -40 aiecc.log; fi
done
echo "contigstream probe done -> ${DST}"
