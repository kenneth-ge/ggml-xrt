#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the DIM_M=32 16-core whole_array Q6_K verify xclbin. DIM_M=32 => rowA=DIM_M/4=8 => the
# matmul_vectorized_4x4 outer `z += 4` loop runs 2 iterations, satisfying
# AIE_LOOP_MIN_ITERATION_COUNT(2) (DIM_M=16 -> rowA=4 -> 1 iter VIOLATED it => dropped-store UB =>
# the all-zeros bug). A is delivered pre-tiled via the MEMTILE DMA transform (mm_q6k_wa_mt.cc,
# --a-memtile) because the on-core A re-tile would need 2x16KB A buffers and overflow the 64KB
# tile at M=32. Vectorized dequant+scatter, C gather/transform unchanged.
#
#   Usage: ./build-mm-verify-wa-m32.sh [<out_subdir> <M> <K> <N>]
#     default -> bench 32 6144 2048.
#
# PADDING CONTRACT (host): pad the draft-token count (4/8/16) up to 32 rows of A[32,K]; the real
# tokens are the first N rows, the remaining rows are don't-care padding and their C outputs
# (rows >= real-token-count of C[32,N]) are ignored by the harness.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

subdir="${1:-bench}"
M="${2:-32}"; K="${3:-6144}"; N="${4:-2048}"
mm="$M"   # token sub-tile == M (single m-tile); M=32 -> rowA=8 -> 2 outer iters
nn=32     # per-core output-channel sub-tile
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

# Core: MEMTILE-A variant (A pre-tiled by DMA, no Al1). Object MUST be named mm_q6k.o.
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
  -DDIM_M="$mm" -DDIM_K=256 -DDIM_N="$nn" -Dbf16_f32_ONLY \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mm_q6k_wa_mt.cc" -o mm_q6k.o

python "${here}/gemv_mm16.py" --dev npu -M "$M" -K "$K" -N "$N" -m "$mm" -n "$nn" --a-memtile > aie.mlir

aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
  --peano "${PEANO_INSTALL_DIR}" --xclbin-name=mmverify.xclbin \
  --aie-generate-npu-insts --npu-insts-name=mmverify_insts.bin aie.mlir

out="mm_verify_wa_q6k_${K}x${N}_m${M}"
cp mmverify.xclbin    "${DST}/${out}.xclbin"
cp mmverify_insts.bin "${DST}/${out}_insts.bin"
echo "OK mm-verify-wa DIM_M=${M} (memtile-A, min-iter fix) K=${K} N=${N} -> ${subdir}/${out}.xclbin"
