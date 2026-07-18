#!/bin/bash
# FUSED FFN SwiGLU decode gemv (aie2/Phoenix): gate+up+silu+mul in ONE 16-core dispatch, removing
# the NPU->GPU->NPU round-trip (silu*mul currently on the iGPU). Uses mv_swiglu_q4k.cc (vscale q4k
# dot x2 + getTanhBf16 silu epilogue). INTERLEAVED weight record (296 B = gate148 ++ up148).
#   A = [N_ff][K/256][296] ; B = shared x[K] bf16 ; C = fused y[N_ff] f32.
# Qwen3-1.7B default: K=2048 (hidden), N_ff=6144. Bench artifact; needs backend wiring
# (detect gate/up matmuls + silu + mul, dispatch this) + host repack (interleave gate/up q4k packs).
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
K="${K:-2048}"; NFF="${NFF:-6144}"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
W="$(mktemp -d)"; cd "$W"
CFLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M=${MM} -DDIM_K=256 -I ${MLIR_AIE_SRC}/aie_kernels/aie2 -I ${MLIR_AIE_SRC}/aie_kernels -I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
# getTanhBf16 needs the tanh LUT tables (tanh_lut_ab/cd) from lut_based_ops.cpp; compile it and
# partial-link (ld.lld -r) with the core so @core("mv_swiglu.o") resolves the LUT symbols.
# shellcheck disable=SC2086
"${PEANO_INSTALL_DIR}/bin/clang++" $CFLAGS -c "${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2/lut_based_ops.cpp" -o lut.o
# shellcheck disable=SC2086
"${PEANO_INSTALL_DIR}/bin/clang++" $CFLAGS -c "${here}/aie2/mv_swiglu_q4k.cc" -o mv_swiglu_core.o
"${PEANO_INSTALL_DIR}/bin/ld.lld" -r mv_swiglu_core.o lut.o -o mv_swiglu.o
python "${here}/gemv_swiglu.py" --dev npu -M "$NFF" -K "$K" -m "$MM" --rows 4 > aie.mlir 2>err.txt || { echo "FAIL gen"; sed -n '1,12p' err.txt; exit 1; }
if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
     --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
  cp g.xclbin "${DST}/gemv_swiglu_q4k_${K}x${NFF}_16core.xclbin"; cp gi.bin "${DST}/gemv_swiglu_q4k_${K}x${NFF}_16core_insts.bin"
  echo "OK swiglu q4k K=${K} N_ff=${NFF} -> bench/gemv_swiglu_q4k_${K}x${NFF}_16core.xclbin"
else echo "FAIL swiglu build"; fi