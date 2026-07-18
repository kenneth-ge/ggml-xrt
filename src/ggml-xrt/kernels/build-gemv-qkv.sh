#!/bin/bash
# FUSED q/k/v decode gemv (aie2/Phoenix): one dispatch for q_proj+k_proj+v_proj (shared activation)
# instead of three -> cuts NPU dispatch count (the real per-dispatch-penalty bottleneck). Uses the
# VSCALE cores (mv_q4k_vscale.cc / mv_q6k_vscale.cc compiled into the mv_q4k.o / mv_q6k.o names the
# generator links) so the fusion does NOT re-introduce the scalar-fp32 scale penalty we just killed.
# Frozen 3-buffer ABI: A = q_w ++ k_w ++ v_w ; B = shared activation b[K] ; C = q_out ++ k_out ++ v_out.
# Qwen3-1.7B default: K=2048, Nq=2048 (q4k), Nkv=1024 (k q4k, v q6k). Bench artifact; needs backend
# wiring (graph-level q/k/v fusion in ggml_backend_xrt_mul_mat) to be used in-model.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K="${K:-2048}"; NQ="${NQ:-2048}"; NKV="${NKV:-1024}"; MM="${MM:-32}"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
W="$(mktemp -d)"; cd "$W"
# compile the VSCALE cores into the .o names the qkv generator links
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" -c "${here}/aie2/mv_q4k_vscale.cc" -o mv_q4k.o
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" -c "${here}/aie2/mv_q6k_vscale.cc" -o mv_q6k.o
python "${here}/gemv_qkv_fused.py" --dev npu -K "$K" --Nq "$NQ" --Nkv "$NKV" -m "$MM" --qk q4k --v q6k > aie.mlir 2>err.txt || { echo "FAIL gen"; sed -n '1,8p' err.txt; exit 1; }
if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
     --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >/dev/null 2>&1; then
  cp g.xclbin "${DST}/gemv_qkv_fused_${K}_vscale.xclbin"; cp gi.bin "${DST}/gemv_qkv_fused_${K}_vscale_insts.bin"
  echo "OK qkv fused (vscale) K=${K} Nq=${NQ} Nkv=${NKV} -> bench/gemv_qkv_fused_${K}_vscale.xclbin"
else echo "FAIL qkv build"; fi