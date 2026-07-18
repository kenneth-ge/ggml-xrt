#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build Q4_K on-chip-dequant decode gemv xclbins (aie2). One per weight matmul with
# output N <= 2048. C[N] = dequant(A_q4K[N,K]) . B[K]  (decode, M=1 token).
#
#   Usage: ./build-q4k-gemv.sh [<out_subdir> <K> <N> [<K> <N> ...]]
#     no args -> Qwen3-1.7B decode shapes (Q/O, K/V, down) into prebuilt/.
#   K must be a multiple of 256 (q4_K superblock).
#
# Host REPACK contract (see mv_q4k.cc + handoff): ONE buffer, row-major [N][K/256][148].
# Per 256-elem q4_K superblock per output row, 148 bytes = qs[128] (raw block_q4_K.qs) +
# scales[12] (raw) + f32 d + f32 dmin (host converts ggml f16 x.d/x.dmin -> f32). This is
# a FIELD REORDER of ggml block_q4_K {d,dmin,scales,qs}. B bf16 [K]; C f32 [N]; weight NOT
# transposed. ABI: kernel(op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5).
#
# UNVALIDATED — compiled on Linux, no NPU here. Verify with q4_gemv_check.cpp (Q4_K mode).
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

# Pick the output sub-tile m for a given N: the broadcast BD count is N/m and must be
# <= 64 (the shim iteration-size [1:64] limit). N<=2048 uses m=32; larger N (e.g. gate/up
# N=6144) steps m up by 32 until N/m<=64 and N%m==0 (6144 -> m=96 -> 64 tiles). This lets
# N=6144 gate/up decode use a native gemv instead of host N-tiling over the 2048 gemv.
pick_m() {  # $1=N -> echoes m, or empty if none <=192 works
  local N="$1" m=32
  while [ "$m" -le 192 ]; do
    if [ $((N % m)) -eq 0 ] && [ $((N / m)) -le 64 ]; then echo "$m"; return; fi
    m=$((m + 32))
  done
}

while [ "$#" -ge 2 ]; do
  K="$1"; N="$2"; shift 2
  m="$(pick_m "$N")"
  if [ -z "$m" ] || [ $((K % 256)) -ne 0 ]; then
    echo "SKIP ${K}x${N} (need K%256==0 and an m<=192 with N%m==0, N/m<=64)"; continue
  fi
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$m" \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/mv_q4k.cc" -o mv_q4k.o
  python "${here}/gemv_q4k.py" --dev npu -M "$N" -K "$K" -m "$m" > aie.mlir
  if ! aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=q4k.xclbin --aie-generate-npu-insts --npu-insts-name=q4k_insts.bin aie.mlir >/dev/null 2>&1; then
    echo "FAIL q4_K gemv ${K}x${N} (m=${m}; L1/shape)"; continue
  fi
  cp q4k.xclbin    "${DST}/mul_mat_aie2_q4k_f32_1x${K}x${N}_gemv.xclbin"
  cp q4k_insts.bin "${DST}/mul_mat_aie2_q4k_f32_1x${K}x${N}_gemv_insts.bin"
  echo "OK q4_K gemv ${K}x${N} (m=${m}) -> ${subdir}/"
done
echo "done -> ${DST}"
