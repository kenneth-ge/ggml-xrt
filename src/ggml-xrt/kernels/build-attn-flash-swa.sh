#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the FLASH-ATTN-EXT-TARGETING chained decode-attention overlay -- SLIDING
# WINDOW (SWA) / WITH-MASK variant (attn_flash_swa.py): QK^T -> soft_max_ext ->
# scores*V wired core-to-core in ONE dispatch, L3 DMA layouts pinned to ggml's
# fused GGML_OP_FLASH_ATTN_EXT tensor contract so the ggml-xrt backend can claim
# that ONE op directly (keeping flash-attn ON -> fast prefill).
#
# This is the WITH-MASK sibling of build-attn-flash.sh (full-causal, no mask).
# The ONLY structural difference: this overlay READS a bf16 mask[n_kv] and ADDS
# it inside softmax_ext, so the SAME chain is correct for SLIDING-WINDOW (SWA)
# models (Gemma etc.) -- the window (+ causal + padding) is baked into the mask
# by the backend. Same three cores + same .o files as attn_flash (kernels are
# layout/shape independent); NEITHER K NOR V needs a transpose -- Core C reads
# flash V in its NATIVE head_dim-contiguous layout via a column-accumulate kernel
# (mv_vt.cc), keeping V zero-copy and sidestepping the bf16 transpose-in-DMA the
# row-major matvec would have needed (a 1-element bf16 stride = 2 bytes is not
# divisible by 4 -> shim DMA rejects it). See attn_flash_swa.py.
#
# Kernel objects:
#   mv_qkt.o       : bf16->bf16 float-accumulate matvec (aie2/mv_bf16out.cc,
#                    -DDIM_M=m -DDIM_K=head_dim)   -- Core A (QK^T).
#   softmax_ext.o  : soft_max_ext (aie2/softmax_ext.cc, -DDIM_N=n_kv -DSCALE=...)
#                    partial-linked with lut_based_ops.cpp   -- Core B. READS +
#                    ADDS the bf16 mask (the SWA path; the ONLY difference vs.
#                    attn_flash, which uses the no-mask softmax_ext_nomask.o).
#   mv_vt.o        : bf16->f32 COLUMN-accumulate scores*V (aie2/mv_vt.cc,
#                    -DHD=head_dim -DKC=kc) native flash V, K-chunked (V tile =
#                    kc*head_dim = 16KB at kc=64)   -- Core C (scores*V).
#
# mask NOTE: flash mask is HARD-asserted F16 by ggml; aie2 has NO native f16, so
# this overlay reads a bf16 mask and the BACKEND must host-convert f16->bf16
# (n_kv elems) before DMA. The SWA window (+ causal + padding) is encoded in the
# mask content (out-of-window positions -> large-negative sentinel). KV (K,V) may
# be bf16 (zero-copy) if the model runs a bf16 KV cache.
#
# Compile-only validation on Linux/WSL; NOT executed on NPU. Windows validates on HW.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- shapes (n_kv buckets) -------------------------------------------------
N_KV_LIST="${N_KV_LIST:-512 1024 2048}"  # KV cache length buckets to build
HEAD_DIM="${HEAD_DIM:-128}"  # attention head dim
M_TILE="${M_TILE:-32}"       # gemv M tile
KC="${KC:-64}"               # scores*V n_kv chunk (Core C V tile = KC*HEAD_DIM = 16KB)
SCALE="${SCALE:-0.08838834764831843f}"  # 1/sqrt(128) = flash `scale` param for HD=128
# Multi-head (mh) overlay: ALL n_head Q-heads for one decode token in ONE
# dispatch, GQA (Q head p uses KV head p//(N_HEAD/N_HEAD_KV)). Qwen3-ish buckets
# for structural validation; re-shape N_HEAD/N_HEAD_KV/HEAD_DIM for Gemma (SWA).
N_HEAD="${N_HEAD:-16}"           # Q heads processed per dispatch
N_HEAD_KV="${N_HEAD_KV:-8}"      # KV heads (GQA group = N_HEAD/N_HEAD_KV)

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

# --- Core C: mv_vt.o (bf16->f32 COLUMN-accumulate, HD=head_dim, KC=kc) -------
# Native flash V (head_dim contiguous, NO transpose), K-chunked -> V tile =
# KC*HD (16KB), so the same overlay builds at 512/1024/2048. n_kv-INDEP (KC baked).
# shellcheck disable=SC2086
"${CLANG}" $CFLAGS -DHD="${HEAD_DIM}" -DKC="${KC}" \
  -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_vt.cc" -o mv_vt.o

# ---- per-bucket: softmax (DIM_N varies) + MLIR + overlay -------------------
for N_KV in ${N_KV_LIST}; do
  echo "==== building attn_flash_swa n_kv=${N_KV} (KC=${KC}) ===="

  # --- Core B: softmax_ext.o (+ exp LUT partial-link), DIM_N=n_kv ----------
  # This is the SWA path: softmax_ext reads + adds the bf16 mask.
  SMFLAGS="${CFLAGS} -DDIM_N=${N_KV} -DSCALE=${SCALE} \
-I ${AK} -I ${MLIR_AIE_SRC}/aie_kernels -I ${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2 -I ${MLIR_AIE_INSTALL}/include"
  # shellcheck disable=SC2086
  "${CLANG}" $SMFLAGS -c "${MLIR_AIE_INSTALL}/aie_runtime_lib/AIE2/lut_based_ops.cpp" -o lut.o
  # shellcheck disable=SC2086
  "${CLANG}" $SMFLAGS -c "${here}/aie2/softmax_ext.cc" -o softmax_ext_core.o
  "${PEANO_INSTALL_DIR}/bin/ld.lld" -r softmax_ext_core.o lut.o -o softmax_ext.o

  # ---- single-head overlay (--n_head 1) ----------------------------------
  python "${here}/attn_flash_swa.py" -d npu --n_kv "${N_KV}" --head_dim "${HEAD_DIM}" \
    -m "${M_TILE}" --kc "${KC}" --n_head 1 --n_head_kv 1 \
    > aie_flash_swa.mlir 2>err_flash_swa.txt || { echo "FAIL gen flash_swa n_kv=${N_KV}"; sed -n '1,40p' err_flash_swa.txt; exit 1; }

  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
       --peano "${PEANO_INSTALL_DIR}" --xclbin-name=attn_flash_swa.xclbin \
       --aie-generate-npu-insts --npu-insts-name=attn_flash_swa_insts.bin aie_flash_swa.mlir \
       >aiecc_flash_swa.log 2>&1; then
    cp attn_flash_swa.xclbin "${DST}/attn_flash_swa_${N_KV}.xclbin"
    cp attn_flash_swa_insts.bin "${DST}/attn_flash_swa_${N_KV}_insts.bin"
    echo "OK  attn_flash_swa overlay (flash_attn_ext SWA/with-mask, 1 dispatch) -> bench/attn_flash_swa_${N_KV}.xclbin"
  else
    echo "FAIL attn_flash_swa overlay build n_kv=${N_KV}"; tail -60 aiecc_flash_swa.log; exit 1
  fi

  # ---- MULTI-HEAD overlay: ALL N_HEAD heads (GQA) in ONE dispatch --------
  python "${here}/attn_flash_swa.py" -d npu --n_kv "${N_KV}" --head_dim "${HEAD_DIM}" \
    -m "${M_TILE}" --kc "${KC}" --n_head "${N_HEAD}" --n_head_kv "${N_HEAD_KV}" \
    > aie_flash_swa_mh.mlir 2>err_flash_swa_mh.txt || { echo "FAIL gen mh n_kv=${N_KV}"; sed -n '1,40p' err_flash_swa_mh.txt; exit 1; }

  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
       --peano "${PEANO_INSTALL_DIR}" --xclbin-name=attn_flash_swa_mh.xclbin \
       --aie-generate-npu-insts --npu-insts-name=attn_flash_swa_mh_insts.bin aie_flash_swa_mh.mlir \
       >aiecc_flash_swa_mh.log 2>&1; then
    cp attn_flash_swa_mh.xclbin "${DST}/attn_flash_swa_mh_${N_KV}.xclbin"
    cp attn_flash_swa_mh_insts.bin "${DST}/attn_flash_swa_mh_${N_KV}_insts.bin"
    echo "OK  attn_flash_swa MULTI-HEAD overlay (N_HEAD=${N_HEAD}, N_HEAD_KV=${N_HEAD_KV}, 1 dispatch) -> bench/attn_flash_swa_mh_${N_KV}.xclbin"
  else
    echo "FAIL attn_flash_swa multi-head overlay build n_kv=${N_KV}"; tail -60 aiecc_flash_swa_mh.log; exit 1
  fi
done

echo "DONE -> ${DST}"
for N_KV in ${N_KV_LIST}; do ls -la "${DST}"/attn_flash_swa_${N_KV}* "${DST}"/attn_flash_swa_mh_${N_KV}* 2>/dev/null; done
