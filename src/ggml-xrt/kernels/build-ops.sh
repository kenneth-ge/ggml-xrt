#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the size-encoded elementwise/norm op xclbins for Phoenix (aie2) into
# prebuilt/ops/. Uses the mlir-aie ml/ examples + the headless shim (no NPU needed).
#
#   RMS_NORM {128,256,2048,2816,5120}  (row tile seq=32; uses repo aie2/rms_norm.cc
#                                       because ml/rmsnorm ships only an aie2p core)
#   RoPE     {128,256}                 (row tile seq=32)
#   SiLU/GELU 16384                     (single tileable elementwise kernel)
#
# See README.md and ../../../docs/ggml-xrt-plan.md section 7a.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/ops"; mkdir -p "$DST"
AIE2="${here}/aie2"

shim_dir="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim_dir}/sitecustomize.py"
export PYTHONPATH="${shim_dir}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
PE="${MLIR_AIE_SRC}/programming_examples/ml"

for e in 128 256 2048 2816 5120; do
    ( cd "$PE/rmsnorm" && make clean >/dev/null 2>&1
      make devicename=npu VPATH="$AIE2" sequence_length=32 embedding_dim=$e build/final.xclbin )
    cp "$PE/rmsnorm/build/final.xclbin" "$DST/rms_norm_${e}_aie2.xclbin"
    cp "$PE/rmsnorm/build/insts.bin"    "$DST/rms_norm_${e}_aie2_insts.bin"
done
for e in 128 256; do
    ( cd "$PE/rope" && make clean >/dev/null 2>&1
      make devicename=npu sequence_length=32 embedding_dim=$e build/final.xclbin )
    cp "$PE/rope/build/final.xclbin" "$DST/rope_${e}_aie2.xclbin"
    cp "$PE/rope/build/insts.bin"    "$DST/rope_${e}_aie2_insts.bin"
done
for op in silu gelu; do
    ( cd "$PE/$op" && make clean >/dev/null 2>&1
      make devicename=npu length=16384 cols=4 chans=2 build/final.xclbin )
    cp "$PE/$op/build/final.xclbin" "$DST/${op}_16384_aie2.xclbin"
    cp "$PE/$op/build/insts.bin"    "$DST/${op}_16384_aie2_insts.bin"
done
echo "done -> $DST"
