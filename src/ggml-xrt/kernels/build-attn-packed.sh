#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the SPATIALLY-PACKED decode-attention overlay (attn_packed.py): all THREE
# attention primitives - QK^T gemv, soft_max_ext, scores*V gemv - co-located in ONE
# aie.device so they share a SINGLE hw_context (Phoenix allows ~5 total). Mirrors the
# rms+rope packing (build-op-overlay-elf.sh) but with three cores on three columns.
#
# LEVEL-A packing: the three cores are co-located but separately dispatched. The OVERLAY
# (array + all 3 core programs) is identical for every --op; --op only selects which op's
# DMAs the runtime_sequence drives. We build the overlay from the QK^T variant and (for
# demonstration) also emit the softmax + sv ELF instruction streams against it.
#
# Kernel objects:
#   mv_32x32.o     : bf16->f32 scalar matvec (aie2/mv.cc, -DDIM_M=32 -DDIM_K=32); links
#                    into BOTH gemv cores (matvec_scalar_bf16_f32 + zero_scalar_f32).
#   softmax_ext.o  : soft_max_ext (aie2/softmax_ext.cc) partial-linked (ld.lld -r) with
#                    lut_based_ops.cpp so getExpBf16's exp LUT symbols resolve - identical
#                    LUT-link scheme to build-attn-softmax.sh / build-gemv-swiglu.sh.
#
# Compile-only validation on Linux/WSL; NOT executed on NPU. Windows validates on hardware.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- shapes (buckets; overlay is identical for all) ------------------------
M_QK="${M_QK:-512}"; K_QK="${K_QK:-128}"      # QK^T : M=n_kv(free), K=head_dim
N_SM="${N_SM:-512}"                            # softmax padded n_kv
M_SV="${M_SV:-128}"; K_SV="${K_SV:-512}"      # scores*V : M=head_dim, K=n_kv bucket
SCALE="${SCALE:-0.08838834764831843f}"        # 1/sqrt(128)

DST="${here}/prebuilt/bench"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
CLANG="${PEANO_INSTALL_DIR}/bin/clang++"
AK="${MLIR_AIE_SRC}/aie_kernels/aie2"

W="$(mktemp -d)"; cd "$W"

# --- mv_32x32.o (shared by both gemv cores) --------------------------------
"${CLANG}" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
  -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw \
  -DNDEBUG -DDIM_M=32 -DDIM_K=32 -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv.cc" -o mv_32x32.o

# --- softmax_ext.o (+ exp LUT partial-link) --------------------------------
SMFLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
-Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
-DDIM_N=${N_SM} -DSCALE=${SCALE} \
-I ${AK} -I ${MLIR_AIE_SRC}/aie_kernels -I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
# shellcheck disable=SC2086
"${CLANG}" $SMFLAGS -c "${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2/lut_based_ops.cpp" -o lut.o
# shellcheck disable=SC2086
"${CLANG}" $SMFLAGS -c "${here}/aie2/softmax_ext.cc" -o softmax_ext_core.o
"${PEANO_INSTALL_DIR}/bin/ld.lld" -r softmax_ext_core.o lut.o -o softmax_ext.o

gen() { # $1=op -> aie_$1.mlir
  python "${here}/attn_packed.py" -d npu --op "$1" \
    --M_qk "$M_QK" --K_qk "$K_QK" --N_sm "$N_SM" --M_sv "$M_SV" --K_sv "$K_SV" \
    > "aie_$1.mlir" 2>"err_$1.txt" || { echo "FAIL gen $1"; sed -n '1,20p' "err_$1.txt"; exit 1; }
}

# ---- overlay + insts from the QK^T variant (overlay holds all 3 cores) -----
gen qkt
if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
     --peano "${PEANO_INSTALL_DIR}" --xclbin-name=attn.xclbin \
     --aie-generate-npu-insts --npu-insts-name=attn_qkt_insts.bin aie_qkt.mlir \
     >aiecc_qkt.log 2>&1; then
  cp attn.xclbin "${DST}/attn_packed.xclbin"
  cp attn_qkt_insts.bin "${DST}/attn_packed_insts.bin"
  cp attn_qkt_insts.bin "${DST}/attn_packed_qkt_insts.bin"
  echo "OK  attn_packed overlay (3 cores) -> bench/attn_packed.xclbin"
else
  echo "FAIL attn_packed overlay build"; tail -30 aiecc_qkt.log; exit 1
fi

# ---- per-op instruction streams against the SAME overlay -------------------
for op in softmax sv; do
  gen "$op"
  if aiecc.py --xclbin-input="${DST}/attn_packed.xclbin" \
       --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --aie-generate-npu-insts --npu-insts-name="attn_${op}_insts.bin" "aie_${op}.mlir" \
       >"aiecc_${op}.log" 2>&1; then
    cp "attn_${op}_insts.bin" "${DST}/attn_packed_${op}_insts.bin"
    echo "OK  attn_packed ${op} insts -> bench/attn_packed_${op}_insts.bin"
  else
    echo "FAIL attn_packed ${op} insts"; tail -30 "aiecc_${op}.log"; exit 1
  fi
done

echo "DONE -> ${DST}"
ls -la "${DST}"/attn_packed* 2>/dev/null
