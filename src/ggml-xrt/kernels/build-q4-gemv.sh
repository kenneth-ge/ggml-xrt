#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build Q4_0 on-chip-dequant decode gemv xclbins (aie2). One per weight matmul with
# output N <= 2048. C[N] = dequant(A_q4[N,K]) . B[K]  (decode, M=1 token).
#
#   Usage: ./build-q4-gemv.sh [<out_subdir> <K> <N> [<K> <N> ...]]
#     no args -> Qwen3-1.7B decode shapes (Q/O, K/V, down) into prebuilt/.
#
# Weight REPACK contract (host must produce this; see mv_q4.cc + handoff step 11q):
#   ONE buffer, row-major [N][K/32][20] bytes. Per 32-elem q4_0 block, per output row:
#   16 nibble bytes (ggml block_q4_0.qs, unchanged) then a 4-byte f32 scale
#   (host converts ggml f16 `d` -> f32). Activation B is bf16 [K]; output C is f32 [N].
#   ABI: kernel(op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5); weight NOT transposed.
#
# UNVALIDATED — compiled on Linux, no NPU here. Verify with the on-device q4 gemv
# check harness before wiring the native-quant dispatch branch.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then set -- . 2048 2048 2048 1024 6144 2048; fi
subdir="$1"; shift
DST="${here}/prebuilt/${subdir}"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

W="$(mktemp -d)"; cd "$W"
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M=32 -DDIM_K=32 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q4.cc" -o mv_q4_32x32.o

while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % 32)) -ne 0 ] || [ $((N / 32)) -gt 64 ] || [ $((K % 32)) -ne 0 ]; then
    echo "SKIP ${K}x${N} (needs N%32==0, N<=2048, K%32==0)"; continue
  fi
  python "${here}/gemv_q4.py" --dev npu -M "$N" -K "$K" -m 32 -k 32 > aie.mlir
  aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
    --xclbin-name=q4.xclbin --aie-generate-npu-insts --npu-insts-name=q4_insts.bin aie.mlir >/dev/null 2>&1
  cp q4.xclbin    "${DST}/mul_mat_aie2_q4_0_f32_1x${K}x${N}_gemv.xclbin"
  cp q4_insts.bin "${DST}/mul_mat_aie2_q4_0_f32_1x${K}x${N}_gemv_insts.bin"
  echo "OK q4_0 gemv ${K}x${N} -> ${subdir}/"
done
echo "done -> ${DST}"
