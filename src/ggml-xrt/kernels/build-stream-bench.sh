#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the DDR->AIE streaming bandwidth microbenchmark xclbins (aie2/Phoenix). Pure weight
# streaming, no compute — establishes the achievable DDR READ bandwidth ceiling for decode.
# Host times a dispatch and reports:  GB/s = total_bytes / wall_time.  Target >= 50 GB/s.
#
#   Usage: ./build-stream-bench.sh            # default matrix into prebuilt/stream/
#   Files: stream_aie2_<cols>col_<MiB>mib.xclbin (+ _insts.bin)
#
# Configs: {1,2,4} columns x {2.5, 10} MiB. 1col = what the current gemv path does; 4col tests
# whether parallel shim channels saturate LPDDR5 (the real ceiling question). If even 4col
# can't beat ~37 GB/s, the NPU-decode path is bounded below the CPU — learn it in one round.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="${here}/prebuilt/stream"; mkdir -p "$DST"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"

MIB=$((1024 * 1024))
# label:bytes
SIZES=("2p5:$((MIB * 5 / 2))" "10:$((MIB * 10))")
COLS=(1 2 4)
TILE="${TILE:-2048}"; DEPTH="${DEPTH:-4}"

W="$(mktemp -d)"; cd "$W"
for cols in "${COLS[@]}"; do
  for sz in "${SIZES[@]}"; do
    lbl="${sz%%:*}"; bytes="${sz##*:}"
    if [ $((bytes % (cols * TILE))) -ne 0 ]; then echo "SKIP ${cols}col ${lbl}MiB (not divisible by cols*tile)"; continue; fi
    python "${here}/stream_bench.py" --dev npu -B "$bytes" --cols "$cols" --tile "$TILE" --depth "$DEPTH" > aie.mlir 2>/dev/null
    if ! aiecc.py --aie-generate-xclbin --no-compile-host --no-xchesscc --no-xbridge \
         --peano "${PEANO_INSTALL_DIR}" --xclbin-name=sb.xclbin \
         --aie-generate-npu-insts --npu-insts-name=sb_insts.bin aie.mlir >/dev/null 2>&1; then
      echo "FAIL ${cols}col ${lbl}MiB"; continue
    fi
    cp sb.xclbin    "${DST}/stream_aie2_${cols}col_${lbl}mib.xclbin"
    cp sb_insts.bin "${DST}/stream_aie2_${cols}col_${lbl}mib_insts.bin"
    echo "OK stream ${cols}col ${lbl}MiB (${bytes} B, tile=${TILE}, depth=${DEPTH})"
  done
done
echo "done -> ${DST}"
