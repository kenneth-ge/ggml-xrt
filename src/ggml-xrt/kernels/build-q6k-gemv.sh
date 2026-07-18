#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build Q6_K on-chip-dequant decode gemv xclbins (aie2). One per weight matmul with
# output N <= 2048. C[N] = dequant(A_q4K[N,K]) . B[K]  (decode, M=1 token).
#
#   Usage: ./build-q6k-gemv.sh [<out_subdir> <K> <N> [<K> <N> ...]]
#     no args -> Qwen3-1.7B decode shapes (Q/O, K/V, down) into prebuilt/.
#   K must be a multiple of 256 (q6_K superblock).
#
# Host REPACK contract (see mv_q6k.cc + handoff): ONE buffer, row-major [N][K/256][148].
# Per 256-elem q6_K superblock per output row, 148 bytes = qs[128] (raw block_q6_K.qs) +
# scales[12] (raw) + f32 d + f32 dmin (host converts ggml f16 x.d/x.dmin -> f32). This is
# a FIELD REORDER of ggml block_q6_K {d,dmin,scales,qs}. B bf16 [K]; C f32 [N]; weight NOT
# transposed. ABI: kernel(op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5).
#
# UNVALIDATED — compiled on Linux, no NPU here. Verify with q4_gemv_check.cpp (Q6_K mode).
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
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M=32 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k.cc" -o mv_q6k.o

while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  if [ $((N % 32)) -ne 0 ] || [ $((N / 32)) -gt 64 ] || [ $((K % 256)) -ne 0 ]; then
    echo "SKIP ${K}x${N} (needs N%32==0, N<=2048, K%256==0)"; continue
  fi
  python "${here}/gemv_q6k.py" --dev npu -M "$N" -K "$K" -m 32 > aie.mlir
  aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
    --xclbin-name=q6k.xclbin --aie-generate-npu-insts --npu-insts-name=q6k_insts.bin aie.mlir >/dev/null 2>&1
  cp q6k.xclbin    "${DST}/mul_mat_aie2_q6k_f32_1x${K}x${N}_gemv.xclbin"
  cp q6k_insts.bin "${DST}/mul_mat_aie2_q6k_f32_1x${K}x${N}_gemv_insts.bin"
  echo "OK q6_K gemv ${K}x${N} -> ${subdir}/"
done
echo "done -> ${DST}"
