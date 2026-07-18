#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the Qwen3-1.7B dense weight-matmul xclbins for Phoenix (aie2/npu1, 4 cols)
# using the mlir-aie multi-core `whole_array` design + the headless shim (no NPU
# required). Produces bf16->f32 MUL_MAT artifacts at a fixed prefill tile M.
# See README.md and ../../../docs/ggml-xrt-plan.md.
#
# NOTE: M is fixed (default 256). Decode (M=1) and a runtime-M kernel are TODO
# (see the plan doc's dynamic-M strategy). lm_head (2048x151936) is intentionally
# omitted — it runs on CPU for the bring-up milestone.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
M="${M:-256}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shim_dir="$(mktemp -d)"
cp "${here}/headless_shim.py" "${shim_dir}/sitecustomize.py"
export PYTHONPATH="${shim_dir}:${PYTHONPATH:-}"

# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"
export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

ex="${MLIR_AIE_SRC}/programming_examples/basic/matrix_multiplication/whole_array"
mkdir -p "${here}/prebuilt"

# Distinct Qwen3-1.7B weight matmul shapes: "K N"
# Q/O proj (2048x2048), K/V proj (2048x1024), gate/up (2048x6144), down (6144x2048)
for kn in "2048 2048" "2048 1024" "2048 6144" "6144 2048"; do
    set -- $kn; K=$1; N=$2
    tgt="build/final_${M}x${K}x${N}_32x32x32_4c.xclbin"
    echo "=== K=$K N=$N (M=$M) ==="
    ( cd "${ex}" && make M="$M" K="$K" N="$N" dtype_in=bf16 dtype_out=f32 n_aie_cols=4 "$tgt" )
    name="mul_mat_aie2_bf16_f32_${M}x${K}x${N}_32x32x32_4c"
    cp "${ex}/${tgt}"                                   "${here}/prebuilt/${name}.xclbin"
    cp "${ex}/build/insts_${M}x${K}x${N}_32x32x32_4c.txt" "${here}/prebuilt/${name}_insts.txt"
done
echo "done -> ${here}/prebuilt/"
