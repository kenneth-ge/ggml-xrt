#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build Q4_K on-chip-dequant FUSED tiled-matmul xclbins (aie2/Phoenix). Serves prefill
# (M tokens) and padded decode (M=1 -> m). C[M,N] += A_act[M,K] . dequant(W_q4K).
#
#   Usage: ./build-q4k-mm.sh [<out_subdir> <K> <N> [<K> <N> ...]]
#     no args -> Qwen3-1.7B Q4_K weight shapes into prebuilt/.
#   K must be a multiple of 256 (q4_K superblock). N must be divisible by n (=32).
#
# Host REPACK contract (== mv_q4k.cc): ONE buffer row-major [N][K/256][148]. Per 256-elem
# superblock per output channel, 148 B = qs[128] + scales[12] + f32 d + f32 dmin (field
# reorder of ggml block_q4_K). A bf16 [M,K]; C f32 [M,N]; weight NOT transposed.
# ABI: kernel(op=3, instr@grp1, ninstr, A@grp3, B@grp4, C@grp5).
#
# L1 budget (Phoenix core, 64 KB): 2*(m*k*2)[A] + 2*(n*148)[Bq] + 2*(m*n*4)[C]
#   + (k*n*2)[Bl1 scratch] + stack. m=32,k=256,n=32 -> ~65 KB (tight); if aiecc reports
#   L1 overflow, drop to -m 32 -n 16 or -m 16 -n 32 (recipe retries n=16 automatically).
#
# UNVALIDATED — compiled on Linux, no NPU here. Verify with q4_gemv_check.cpp (mm mode).
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="${M:-32}"   # prefill token-tile height built into the xclbin; host chunks/pads tokens
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
build_core() {  # $1=m $2=n  -> mm_q4k.o compiled for these tile dims
  "${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
    -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
    -Wno-missing-template-arg-list-after-template-kw -DNDEBUG \
    -DDIM_M="$1" -DDIM_K=256 -DDIM_N="$2" -Dbf16_f32_ONLY \
    -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/mm_q4k.cc" -o mm_q4k.o
}

try_shape() {  # $1=K $2=N $3=m $4=n $5=M $6=serialize(0/1)  -> 0 on success
  local K="$1" N="$2" mm="$3" nn="$4" SM="$5" SER="$6"
  local ser_flag=""; [ "$SER" = "1" ] && ser_flag="--serialize-mtiles"
  build_core "$mm" "$nn"
  python "${here}/mm_q4k.py" --dev npu -M "$SM" -K "$K" -N "$N" -m "$mm" -n "$nn" $ser_flag > aie.mlir 2>/dev/null || return 1
  aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
    --peano "${PEANO_INSTALL_DIR}" --xclbin-name=q4kmm.xclbin \
    --aie-generate-npu-insts --npu-insts-name=q4kmm_insts.bin aie.mlir >/dev/null 2>&1 || return 1
}

CFGS=("32 32" "32 16" "16 32")   # (m n) fallbacks; first that fits L1 + divides N wins. m must be %16 (aie2 4x4 MMUL); M=16 speculative-verify uses m=16 (single m-tile, one weight sweep).
SHAPES=("$@")
REC=148  # per-m-tile B-stream bytes = N*(K/256)*REC
for ((i = 0; i + 1 < ${#SHAPES[@]}; i += 2)); do
  K="${SHAPES[i]}"; N="${SHAPES[i+1]}"
  if [ $((K % 256)) -ne 0 ]; then echo "SKIP ${K}x${N} (K%256!=0)"; continue; fi
  # Multi-m-tile re-streams the full packed weight per m-tile; on HW the 2nd 10 MB+
  # stream corrupts (observed on q6k 6144x2048). Keep M=32 but SERIALIZE m-tiles
  # (dma_wait between them) for shapes whose per-m-tile B stream exceeds ~8 MB;
  # smaller shapes keep the faster ping-pong path.
  bstream=$(( N * (K / 256) * REC )); SM="$M"; SER=0
  if [ "$bstream" -gt 8388608 ]; then SER=1; echo "  NOTE ${K}x${N}: B-stream ${bstream} B > 8 MB -> serialize m-tiles (M=${SM})"; fi
  ok=0
  for cfg in "${CFGS[@]}"; do
    mm="${cfg% *}"; nn="${cfg#* }"
    if [ $((N % nn)) -ne 0 ]; then continue; fi
    if [ $((SM % mm)) -ne 0 ]; then continue; fi   # M must be a multiple of m
    if try_shape "$K" "$N" "$mm" "$nn" "$SM" "$SER"; then
      cp q4kmm.xclbin    "${DST}/mul_mat_aie2_q4k_f32_${SM}x${K}x${N}_mm.xclbin"
      cp q4kmm_insts.bin "${DST}/mul_mat_aie2_q4k_f32_${SM}x${K}x${N}_mm_insts.bin"
      echo "OK q4_K mm ${SM}x${K}x${N} (m=${mm} n=${nn}) -> ${subdir}/"
      ok=1; break
    fi
    echo "  retry ${K}x${N}: m=${mm} n=${nn} failed (L1/shape), trying next cfg"
  done
  [ "$ok" -eq 1 ] || echo "FAIL q4_K mm ${K}x${N} (all cfgs)"
done
echo "done -> ${DST}"
