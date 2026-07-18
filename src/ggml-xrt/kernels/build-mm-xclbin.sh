#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Reproducibly build a MUL_MAT xclbin (+ instruction sequence) for Phoenix (aie2/npu1)
# on WSL/Linux, using the headless shim so no NPU is required. See README.md.
#
# Env overrides: IRONENV, MLIR_AIE_SRC, M, K, N, DTYPE_IN, DTYPE_OUT.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
M="${M:-256}"; K="${K:-256}"; N="${N:-256}"
DTYPE_IN="${DTYPE_IN:-bf16}"; DTYPE_OUT="${DTYPE_OUT:-f32}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Headless shim (stub xrt::device so aie.iron imports with no NPU)
shim_dir="$(mktemp -d)"
cp "${here}/headless_shim.py" "${shim_dir}/sitecustomize.py"
export PYTHONPATH="${shim_dir}:${PYTHONPATH:-}"

# 2. IRON / mlir-aie env
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"
export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

# 3. Compile (dev=npu == Phoenix/aie2/npu1; peano backend)
ex="${MLIR_AIE_SRC}/programming_examples/basic/matrix_multiplication/single_core"
suffix="${M}x${K}x${N}_32x32x32"
( cd "${ex}" && make M="${M}" K="${K}" N="${N}" dtype_in="${DTYPE_IN}" dtype_out="${DTYPE_OUT}" \
      "build/final_${suffix}.xclbin" "build/insts_${suffix}.txt" )

# 4. Publish into prebuilt/
name="mul_mat_aie2_${DTYPE_IN}_${DTYPE_OUT}_${suffix}"
mkdir -p "${here}/prebuilt"
cp "${ex}/build/final_${suffix}.xclbin" "${here}/prebuilt/${name}.xclbin"
cp "${ex}/build/insts_${suffix}.txt"    "${here}/prebuilt/${name}_insts.txt"
cp "${ex}/build/aie_${suffix}.mlir"     "${here}/prebuilt/${name}.mlir"
echo "wrote ${here}/prebuilt/${name}.{xclbin,_insts.txt,.mlir}"
