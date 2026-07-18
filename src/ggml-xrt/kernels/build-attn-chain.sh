#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the LEVEL-B on-chip-CHAINED decode-attention overlay (attn_chain.py):
# QK^T -> soft_max_ext -> scores*V wired core-to-core via object_fifos so the
# WHOLE attention runs in ONE dispatch with NO L3 round-trips between the sub-ops
# (removes the NPU<->CPU handoffs that Level-A's 3-separate-dispatch still paid).
#
# Three cores co-located in ONE aie.device (ONE hw_context), chained:
#   Core A (QK^T,     tile(0,2)) --scores(bf16, memtile-aggregated)-->
#   Core B (softmax,  tile(1,2)) --probs(bf16, core-to-core)--------->
#   Core C (scores*V, tile(2,2)) --> out[head_dim] (f32) to L3.
#
# Kernel objects (one per core, each linked into just its core's ELF):
#   mv_qkt.o       : bf16->bf16 float-accumulate matvec (aie2/mv_bf16out.cc,
#                    -DDIM_M=m -DDIM_K=head_dim); QK^T scores stay bf16 for softmax
#                    (matvec_scalar_bf16_bf16 + zero_scalar_bf16).
#   softmax_ext.o  : soft_max_ext (aie2/softmax_ext.cc, -DDIM_N=n_kv -DSCALE=...)
#                    partial-linked (ld.lld -r) with lut_based_ops.cpp for the exp
#                    LUT symbols (same LUT-link scheme as build-attn-packed.sh).
#   mv_sv.o        : bf16->f32 scalar matvec (aie2/mv.cc, -DDIM_M=m -DDIM_K=KC);
#                    scores*V final output (matvec_scalar_bf16_f32 + zero_scalar_f32).
#                    Core C K-CHUNKS the n_kv contraction into KC-blocks so DIM_K=KC
#                    (NOT n_kv): its V tile is (m,KC)=16KB regardless of n_kv, so the
#                    same overlay builds at 512/1024/2048 (n_kv=1024 no longer
#                    overflows Core C's L1). mv_sv.o is n_kv-INDEPENDENT (KC baked).
#
# Builds all N_KV buckets in N_KV_LIST (default "512 1024 2048"): mv_qkt.o (HEAD_DIM
# baked) and mv_sv.o (KC baked) are compiled ONCE; only softmax_ext.o (DIM_N=n_kv)
# and the generated MLIR vary per bucket.
#
# Compile-only validation on Linux/WSL; NOT executed on NPU. Windows validates on HW.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- shapes (single head, n_kv buckets) ------------------------------------
N_KV_LIST="${N_KV_LIST:-512 1024 2048}"  # KV cache length buckets to build
HEAD_DIM="${HEAD_DIM:-128}"  # attention head dim
M_TILE="${M_TILE:-32}"       # gemv M tile
KC="${KC:-256}"              # scores*V n_kv k-chunk (Core C V tile = m*KC = 16KB)
SCALE="${SCALE:-0.08838834764831843f}"  # 1/sqrt(128)

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

CFLAGS="-O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
-Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG"

W="$(mktemp -d)"; cd "$W"

# --- Core A: mv_qkt.o (bf16->bf16, DIM_M=m, DIM_K=head_dim) -- n_kv-INDEP ----
# shellcheck disable=SC2086
"${CLANG}" $CFLAGS -DDIM_M="${M_TILE}" -DDIM_K="${HEAD_DIM}" \
  -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_bf16out.cc" -o mv_qkt.o

# --- Core C: mv_sv.o (bf16->f32, DIM_M=m, DIM_K=KC) -- n_kv-INDEP (KC baked) -
# K-chunked: DIM_K=KC (NOT n_kv). Core C accumulates n_kv/KC chunks (c[row]+=).
# shellcheck disable=SC2086
"${CLANG}" $CFLAGS -DDIM_M="${M_TILE}" -DDIM_K="${KC}" \
  -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv.cc" -o mv_sv.o

# ---- per-bucket: softmax (DIM_N varies) + MLIR + overlay -------------------
for N_KV in ${N_KV_LIST}; do
  echo "==== building attn_chain n_kv=${N_KV} (KC=${KC}) ===="

  # --- Core B: softmax_ext.o (+ exp LUT partial-link), DIM_N=n_kv ----------
  SMFLAGS="${CFLAGS} -DDIM_N=${N_KV} -DSCALE=${SCALE} \
-I ${AK} -I ${MLIR_AIE_SRC}/aie_kernels -I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
  # shellcheck disable=SC2086
  "${CLANG}" $SMFLAGS -c "${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2/lut_based_ops.cpp" -o lut.o
  # shellcheck disable=SC2086
  "${CLANG}" $SMFLAGS -c "${here}/aie2/softmax_ext.cc" -o softmax_ext_core.o
  "${PEANO_INSTALL_DIR}/bin/ld.lld" -r softmax_ext_core.o lut.o -o softmax_ext.o

  # ---- generate chained MLIR --------------------------------------------
  python "${here}/attn_chain.py" -d npu --n_kv "${N_KV}" --head_dim "${HEAD_DIM}" \
    -m "${M_TILE}" --kc "${KC}" \
    > aie_chain.mlir 2>err_chain.txt || { echo "FAIL gen chain n_kv=${N_KV}"; sed -n '1,40p' err_chain.txt; exit 1; }

  # ---- overlay + insts ---------------------------------------------------
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
       --peano "${PEANO_INSTALL_DIR}" --xclbin-name=attn_chain.xclbin \
       --aie-generate-npu-insts --npu-insts-name=attn_chain_insts.bin aie_chain.mlir \
       >aiecc_chain.log 2>&1; then
    cp attn_chain.xclbin "${DST}/attn_chain_${N_KV}.xclbin"
    cp attn_chain_insts.bin "${DST}/attn_chain_${N_KV}_insts.bin"
    echo "OK  attn_chain overlay (3 chained cores, 1 dispatch) -> bench/attn_chain_${N_KV}.xclbin"
  else
    echo "FAIL attn_chain overlay build n_kv=${N_KV}"; tail -60 aiecc_chain.log; exit 1
  fi
done

echo "DONE -> ${DST}"
for N_KV in ${N_KV_LIST}; do ls -la "${DST}"/attn_chain_${N_KV}* 2>/dev/null; done
