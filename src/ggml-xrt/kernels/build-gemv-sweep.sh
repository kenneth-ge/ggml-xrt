#!/bin/bash
# CHEAP SHAPE-SWEEP bench matrix (aie2/Phoenix): build the two biggest decode gemvs at m-tile
# {16,64} (baseline m=32 already shipped) + a 4-accumulator q6k variant, so the measuring agent
# can pick each shape's optimum. Diminishing-returns tuning (matmul near floor) — bench only.
# Builds run in PARALLEL (one aiecc per variant, backgrounded).
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

# variant: "<tag> <qtype> <src.cc> <K> <N> <m>"
VARIANTS=(
  "m16    q6k mv_q6k_noscratch.cc 6144 2048 16"
  "m64    q6k mv_q6k_noscratch.cc 6144 2048 64"
  "4acc   q6k mv_q6k_4acc.cc      6144 2048 32"
  "m16    q4k mv_q4k_noscratch.cc 2048 6144 16"
  "m64    q4k mv_q4k_noscratch.cc 2048 6144 64"
)
build_one() {
  local tag="$1" qt="$2" src="$3" K="$4" N="$5" m="$6"
  local w; w="$(mktemp -d)"; cd "$w"
  local obj; case "$qt" in q6k) obj=mv_q6k.o;; q4k) obj=mv_q4k.o;; esac
  if [ $((N % (m*16))) -ne 0 ]; then echo "SKIP ${qt} ${K}x${N} ${tag} (N%(m*16)!=0)"; return; fi
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$m" -DDIM_K=256 \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/${src}" -o "$obj" 2>/dev/null || { echo "FAIL core ${qt} ${tag}"; return; }
  python "${here}/gemv_mc16.py" --dev npu --qtype "$qt" -M "$N" -K "$K" -m "$m" --rows 4 > aie.mlir 2>/dev/null
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
    cp g.xclbin "${DST}/gemv_${qt}_${K}x${N}_16core_${tag}.xclbin"; cp gi.bin "${DST}/gemv_${qt}_${K}x${N}_16core_${tag}_insts.bin"
    echo "OK ${qt} ${K}x${N} ${tag} (m=${m}) -> bench/gemv_${qt}_${K}x${N}_16core_${tag}.xclbin"
  else echo "FAIL build ${qt} ${K}x${N} ${tag}"; fi
}
pids=()
for v in "${VARIANTS[@]}"; do build_one $v & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "sweep done -> ${DST}"
