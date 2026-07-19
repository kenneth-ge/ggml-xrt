#!/bin/bash
# RESIDUAL-ADD kernel (aie2/Phoenix, single core). Compiles eltwise_add.cc
# (ggml residual add: c = a + b, f32, n_embd via -DDIM_N) and builds a per-N
# xclbin. Pure elementwise: no LUT, no accumulator, no partial-link needed.
#
# UNVALIDATED on NPU: compiled on Linux/WSL only. Math: c[i] = a[i] + b[i].
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NS=(${NS:-2048})
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

FAIL=0
for N in "${NS[@]}"; do
  W="$(mktemp -d)"; cd "$W"
  CFLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
-Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
-DDIM_N=${N} \
-I ${MLIR_AIE_SRC}/aie_kernels/aie2 -I ${MLIR_AIE_SRC}/aie_kernels \
-I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
  # shellcheck disable=SC2086
  "${PEANO_INSTALL_DIR}/bin/clang++" $CFLAGS -c "${here}/aie2/eltwise_add.cc" -o eltwise_add.o
  python "${here}/eltwise_add.py" --dev npu -N "$N" > aie.mlir 2>err.txt || { echo "FAIL gen N=${N}"; sed -n '1,20p' err.txt; FAIL=1; continue; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=e.xclbin --aie-generate-npu-insts --npu-insts-name=ei.bin aie.mlir >aiecc.log 2>&1; then
    cp e.xclbin "${DST}/eltwise_add_${N}.xclbin"; cp ei.bin "${DST}/eltwise_add_${N}_insts.bin"
    echo "OK  eltwise_add N=${N} -> bench/eltwise_add_${N}.xclbin"
  else echo "FAIL eltwise_add build N=${N}"; tail -25 aiecc.log; FAIL=1; fi
done
exit $FAIL
