#!/bin/bash
# DECODE-ATTENTION soft_max_ext kernel (aie2/Phoenix, single core). Compiles
# softmax_ext.cc (ggml soft_max_ext: probs = softmax(scores*scale + mask), scale
# baked via -DSCALE, padded n_kv via -DDIM_N) and builds a per-N xclbin.
#
# getExpBf16 (natural-e LUT) needs the exp LUT tables (exp_ilut_ab/cd,
# exp_flut_ab/cd, m_inv_lut) from lut_based_ops.cpp; we compile that TU and
# partial-link (ld.lld -r) with the core object so @core("softmax_ext.o")
# resolves the LUT symbols - identical scheme to build-gemv-swiglu.sh (tanh LUT).
#
# UNVALIDATED on NPU: compiled on Linux/WSL only. Math per ggml soft_max_ext.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 1/sqrt(128); override with SCALE=... for a different head_dim.
SCALE="${SCALE:-0.08838834764831843f}"
NS=(${NS:-512 1024})
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
-DDIM_N=${N} -DSCALE=${SCALE} \
-I ${MLIR_AIE_SRC}/aie_kernels/aie2 -I ${MLIR_AIE_SRC}/aie_kernels \
-I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
  # shellcheck disable=SC2086
  "${PEANO_INSTALL_DIR}/bin/clang++" $CFLAGS -c "${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2/lut_based_ops.cpp" -o lut.o
  # shellcheck disable=SC2086
  "${PEANO_INSTALL_DIR}/bin/clang++" $CFLAGS -c "${here}/aie2/softmax_ext.cc" -o softmax_ext_core.o
  "${PEANO_INSTALL_DIR}/bin/ld.lld" -r softmax_ext_core.o lut.o -o softmax_ext.o
  python "${here}/attn_softmax.py" --dev npu -N "$N" > aie.mlir 2>err.txt || { echo "FAIL gen N=${N}"; sed -n '1,20p' err.txt; FAIL=1; continue; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=s.xclbin --aie-generate-npu-insts --npu-insts-name=si.bin aie.mlir >aiecc.log 2>&1; then
    cp s.xclbin "${DST}/attn_softmax_${N}.xclbin"; cp si.bin "${DST}/attn_softmax_${N}_insts.bin"
    echo "OK  attn_softmax N=${N} -> bench/attn_softmax_${N}.xclbin"
  else echo "FAIL attn_softmax build N=${N}"; tail -25 aiecc.log; FAIL=1; fi
done
exit $FAIL
