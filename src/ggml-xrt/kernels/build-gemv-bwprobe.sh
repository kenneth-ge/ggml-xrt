#!/bin/bash
# WEIGHT-DMA DATAFLOW BW PROBES (aie2/Phoenix): decide whether the M=1 q6k gemv's ~11 GB/s
# weight-DMA is a HARDWARE DRAM wall or a dataflow/channel limit with headroom. Every variant reads
# the FULL 10.4 MB q6k weight of the shipping down 6144x2048 gemv from DRAM with a FIXED
# fully-contiguous access pattern; cores only acquire/release (NO dequant, NO MAC) so wall == pure
# weight-DMA time. The access pattern is held constant across ALL variants — only the DATAFLOW moves:
#
#   contig_c1 / c2 / c4   CHANNEL SCALING: 10.4 MB total split over 1 / 2 / 4 shim columns.
#                         scales ~linearly with cols => channel-limited (headroom); flat => DRAM wall.
#   contig_nomt           MEMTILE BYPASS: weight shim->core direct (no memtile link) vs c4's
#                         shim->memtile->core. faster => memtile routing is the bottleneck.
#   contig_d4 / d8        MORE IN-FLIGHT: weight object_fifo depth 4 / 8 (vs c4's depth 2).
#                         faster => the shallow fifo was starving the DMA; flat => already saturated.
#
# Named gemv_q6k_<K>x<N>_16core_<variant>.xclbin so the measuring agent mm_times each.
# Compile-only (WSL); bench-only, NOT shipped. Do NOT commit/push.
set -euo pipefail
IRONENV="${IRONENV:-$HOME/ironenv}"; MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MM="${MM:-32}"
K="${K:-6144}"; N="${N:-2048}"
DST="${here}/prebuilt/bench"; mkdir -p "$DST"
shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"; export PYTHONPATH="${shim}:${PYTHONPATH:-}"
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

W="$(mktemp -d)"; cd "$W"
# drain cores only need zero_scalar_f32 (never call matvec); maconly_clean provides it.
"${PEANO_INSTALL_DIR}/bin/clang++" -O2 -std=c++20 --target=aie2-none-unknown-elf \
  -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body \
  -Wno-missing-template-arg-list-after-template-kw -DNDEBUG -DDIM_M="$MM" -DDIM_K=256 \
  -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/mv_q6k_maconly_clean.cc" -o mv_q6k.o

# variant tag -> generator args
build_one() {
  local tag="$1"; shift
  python "${here}/gemv_bwprobe.py" --dev npu -K "$K" -N "$N" -m "$MM" "$@" > aie.mlir 2>py.log \
    || { echo "FAIL pygen ${tag}"; sed -n '1,40p' py.log; return 1; }
  if aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
       --xclbin-name=g.xclbin --aie-generate-npu-insts --npu-insts-name=gi.bin aie.mlir >aiecc.log 2>&1; then
    cp g.xclbin "${DST}/gemv_q6k_${K}x${N}_16core_${tag}.xclbin"
    cp gi.bin   "${DST}/gemv_q6k_${K}x${N}_16core_${tag}_insts.bin"
    echo "OK ${tag} -> bench/gemv_q6k_${K}x${N}_16core_${tag}.xclbin"
  else echo "FAIL build ${tag}"; tail -40 aiecc.log; return 1; fi
}

# CHANNEL SCALING (contiguous drain at 1/2/4 shim columns, depth 2, through memtile)
build_one contig_c1 --cols 1 --depth 2 || true
build_one contig_c2 --cols 2 --depth 2 || true
build_one contig_c4 --cols 4 --depth 2 || true
# MEMTILE BYPASS (4 columns, shim->core direct)
build_one contig_nomt --cols 4 --depth 2 --nomt || true
# MORE IN-FLIGHT (4 columns, through memtile, deeper weight fifo)
build_one contig_d4 --cols 4 --depth 4 || true
build_one contig_d8 --cols 4 --depth 8 || true
echo "bwprobe done -> ${DST}"
