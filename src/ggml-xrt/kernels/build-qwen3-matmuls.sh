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

mm="${MLIR_AIE_SRC}/programming_examples/basic/matrix_multiplication"
wa="${mm}/whole_array"
sc="${mm}/single_core"
mkdir -p "${here}/prebuilt"

# Distinct Qwen3-1.7B weight matmul shapes: "K N"
# Q/O proj (2048x2048), K/V proj (2048x1024), gate/up (2048x6144), down (6144x2048)
SHAPES=("2048 2048" "2048 1024" "2048 6144" "6144 2048")

# Prefill tier: whole_array, 4 columns, M=${M} (needs M>=128). Host chunks tokens
# into M-sized blocks. See docs/ggml-xrt-plan.md section 8 (dynamic-M strategy).
for kn in "${SHAPES[@]}"; do
    set -- $kn; K=$1; N=$2
    tgt="build/final_${M}x${K}x${N}_32x32x32_4c.xclbin"
    echo "=== prefill: K=$K N=$N (M=$M, 4c) ==="
    ( cd "${wa}" && make M="$M" K="$K" N="$N" dtype_in=bf16 dtype_out=f32 n_aie_cols=4 "$tgt" )
    name="mul_mat_aie2_bf16_f32_${M}x${K}x${N}_32x32x32_4c"
    cp "${wa}/${tgt}"                                     "${here}/prebuilt/${name}.xclbin"
    cp "${wa}/build/insts_${M}x${K}x${N}_32x32x32_4c.txt" "${here}/prebuilt/${name}_insts.txt"
done

# Decode/small-batch tier: single_core, M=32 (min tile). Host pads 1 token -> 32.
# EXCEPTION: N=6144 (gate/up) exceeds the single-core DMA descriptor limit (192 N-tiles
# > the [1:64] BD range), so it uses whole_array M=128 (N split across 4 cols -> 48
# tiles/col). See docs/ggml-xrt-plan.md section 8.
for kn in "${SHAPES[@]}"; do
    set -- $kn; K=$1; N=$2
    if [ "$N" -ge 4096 ]; then
        tgt="build/final_128x${K}x${N}_32x32x32_4c.xclbin"
        echo "=== decode(wide): K=$K N=$N (M=128, 4c) ==="
        ( cd "${wa}" && make M=128 K="$K" N="$N" dtype_in=bf16 dtype_out=f32 n_aie_cols=4 "$tgt" )
        name="mul_mat_aie2_bf16_f32_128x${K}x${N}_32x32x32_4c"
        cp "${wa}/${tgt}"                                     "${here}/prebuilt/${name}.xclbin"
        cp "${wa}/build/insts_128x${K}x${N}_32x32x32_4c.txt"  "${here}/prebuilt/${name}_insts.txt"
    else
        tgt="build/final_32x${K}x${N}_32x32x32.xclbin"
        echo "=== decode: K=$K N=$N (M=32, 1c) ==="
        ( cd "${sc}" && make M=32 K="$K" N="$N" dtype_in=bf16 dtype_out=f32 "$tgt" )
        name="mul_mat_aie2_bf16_f32_32x${K}x${N}_32x32x32_1c"
        cp "${sc}/${tgt}"                                  "${here}/prebuilt/${name}.xclbin"
        cp "${sc}/build/insts_32x${K}x${N}_32x32x32.txt"   "${here}/prebuilt/${name}_insts.txt"
    fi
done
echo "done -> ${here}/prebuilt/"
