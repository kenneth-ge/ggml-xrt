#!/bin/bash
# Build the REAL 16-core decode gemvs (aie2/Phoenix) for Qwen3-1.7B, all dtypes, via gemv_mc16.py
# (all 16 compute tiles; ~3.8x over 4-core). Drop-in: same _1x{K}x{N}_gemv names/ABI, natural C[N].
# N must be divisible by m*16 (=512 at m=32). Overwrites the 4-core reals in prebuilt/.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
subdir="${1:-.}"; DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
declare -A CORE=( [q4_0]="mv_q4.cc:mv_q4_32x32.o" [q4k]="mv_q4k.cc:mv_q4k.o" [q6k]="mv_q6k.cc:mv_q6k.o" )
declare -A NAME=( [q4_0]="q4_0" [q4k]="q4k" [q6k]="q6k" )
declare -A SHAPES=( [q4_0]="2048 2048 2048 1024 6144 2048" [q4k]="2048 2048 2048 1024 2048 6144" [q6k]="2048 1024 6144 2048" )
W="$(mktemp -d)"; cd "$W"
for qt in q4_0 q4k q6k; do
  IFS=: read -r src obj <<< "${CORE[$qt]}"
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" -c "${here}/aie2/${src}" -o "${obj}"
  set -- ${SHAPES[$qt]}
  while [ "$#" -ge 2 ]; do
    K="$1"; N="$2"; shift 2
    if [ $((N % (MM * 16))) -ne 0 ]; then echo "SKIP ${qt} ${K}x${N} (N%(m*16)!=0)"; continue; fi
    python "${here}/gemv_mc16.py" --dev npu --qtype "$qt" -M "$N" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>/dev/null
    if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
      cp g.xclbin "${DST}/mul_mat_aie2_${NAME[$qt]}_f32_1x${K}x${N}_gemv.xclbin"; cp gi.bin "${DST}/mul_mat_aie2_${NAME[$qt]}_f32_1x${K}x${N}_gemv_insts.bin"; echo "OK ${qt} ${K}x${N} (16-core)"
    else echo "FAIL ${qt} ${K}x${N}"; fi
  done
done
echo "done -> ${DST}"
