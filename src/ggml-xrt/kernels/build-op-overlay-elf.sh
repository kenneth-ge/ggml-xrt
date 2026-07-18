#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the SINGLE-hw_context overlay + per-shape ELF module set for the
# COLLAPSIBLE single-core elementwise activation ops (silu, gelu), aie2/Phoenix.
#
# WHY: Phoenix allows only ~5 concurrent XRT hw_contexts, and each xclbin == one
# context. This mirrors build-overlay-elf.sh (the gemv decode set) for the
# activation ops so that all N lengths of an op collapse to ONE overlay + N cheap
# instruction ELF modules, instead of one xclbin (== one context) per length.
#
# COLLAPSIBILITY (verified on this tree, see docs/ggml-xrt-linux-kernel-wishlist.md §8):
#   silu/gelu (ml/silu, ml/gelu) use a FIXED L1 line buffer (line_size=1024); the
#   transfer `length` only changes the runtime_sequence DMA descriptors (taps), NOT
#   the AIE array config or the core program. So the overlay is length-INDEPENDENT:
#   two lengths generate a device/core MLIR that is byte-identical outside
#   `aie.runtime_sequence`, and their xclbins differ only in the auto-generated UUID
#   (66 bytes across the 2×16-byte UUID fields + the xclbin JSON metadata). Every
#   length's ELF is built with `--xclbin-input <overlay>`, so it is load-compatible
#   with that overlay by construction.
#
#   rms_norm and rope are NOT collapsible with the stock ml/ designs: their ObjectFifo
#   L1 buffer is sized `memref<embedding_dim x bf16>` AND the core bakes `cols` as an
#   `arith.constant`, so both the L1 allocation and the core program change per shape
#   → the overlay changes per shape → no collapse. Fixing them needs an RTP `cols`
#   (runtime parameter, not a baked constant) + a fixed max L1 buffer. Documented in
#   the wishlist §8; those ops keep their per-shape xclbins in prebuilt/ops/ for now.
#
# OUTPUT (src/ggml-xrt/kernels/prebuilt/op-overlays/):
#   <op>_overlay.xclbin   one per op (silu, gelu) — register once
#   <op>_<L>.elf          one per length — the instruction module
#   manifest.json         length -> (overlay, elf, kernel_name) + notes
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Correctness is validated
# later on Windows. This script only proves buildability + overlay-sharing.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${here}/prebuilt/op-overlays"

# Activation lengths (bf16 elements). Must be a multiple of 1024 (line_size) and of
# cols*chans (=8); every multiple of 1024 satisfies both. Covers the FFN-intermediate
# tile sizes across the lineup; the host tiles/loops any length over the nearest tile.
LENGTHS=(2048 4096 6144 8192 12288 16384)
COLS=4
CHANS=2

rm -rf "$OUT"; mkdir -p "$OUT"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
PE="${MLIR_AIE_SRC}/programming_examples/ml"

FAIL=0
W="$(mktemp -d)"

for op in silu gelu; do
  # Build the core kernel archive (kernels.a: <op>.cc.o + lut_based_ops.o) once, via
  # the example Makefile, so we reuse the exact clang flags / VPATH (aie_kernels/aie2).
  ( cd "${PE}/${op}" && make clean >/dev/null 2>&1 \
      && make devicename=npu length=16384 cols="${COLS}" chans="${CHANS}" build/kernels.a >/dev/null 2>&1 )
  cp "${PE}/${op}/build/kernels.a" "${W}/kernels.a"

  ov_built=""
  for L in "${LENGTHS[@]}"; do
    ( cd "${W}" && python3 "${PE}/${op}/${op}.py" -d npu -l "${L}" -co "${COLS}" -ch "${CHANS}" > "shape.mlir" )
    elf_name="${op}_${L}.elf"
    ov_name="${op}_overlay.xclbin"
    if [ -z "${ov_built}" ]; then
      if ( cd "${W}" && aiecc.py --aie-generate-xclbin --aie-generate-elf --no-compile-host \
             --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
             --xclbin-name="ov.xclbin" --elf-name="mod.elf" shape.mlir >aiecc.log 2>&1 ); then
        cp "${W}/ov.xclbin"  "${OUT}/${ov_name}"
        cp "${W}/mod.elf"    "${OUT}/${elf_name}"
        ov_built="${ov_name}"
        echo "OK  overlay+elf  ${op} L=${L}  -> ${ov_name}, ${elf_name}"
      else
        echo "FAIL overlay ${op} L=${L}"; tail -20 "${W}/aiecc.log"; FAIL=1
      fi
    else
      cp "${OUT}/${ov_built}" "${W}/ov.xclbin"
      if ( cd "${W}" && aiecc.py --xclbin-input=ov.xclbin --aie-generate-elf --no-compile-host \
             --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
             --elf-name="mod.elf" shape.mlir >aiecc.log 2>&1 ); then
        cp "${W}/mod.elf" "${OUT}/${elf_name}"
        echo "OK  elf         ${op} L=${L}  -> ${elf_name}"
      else
        echo "FAIL elf ${op} L=${L}"; tail -20 "${W}/aiecc.log"; FAIL=1
      fi
    fi
  done
done

# --- manifest ---------------------------------------------------------------
OUT="$OUT" python3 - "${LENGTHS[@]}" <<'PY'
import os, sys, json, glob, re
OUT = os.environ["OUT"]
lengths = [int(x) for x in sys.argv[1:]]
ops = []
shapes = []
for op in ("silu", "gelu"):
    ov = f"{op}_overlay.xclbin"
    if not os.path.exists(os.path.join(OUT, ov)):
        continue
    elfs = sorted(glob.glob(os.path.join(OUT, f"{op}_*.elf")),
                  key=lambda p: int(re.search(r'_(\d+)\.elf$', p).group(1)))
    ops.append({"op": op, "overlay": ov, "elf_count": len(elfs),
                "cols": 4, "chans": 2, "line_size": 1024})
    for e in elfs:
        L = int(re.search(r'_(\d+)\.elf$', e).group(1))
        shapes.append({"op": op, "length": L, "overlay": ov,
                       "elf": os.path.basename(e), "kernel_name": "MLIR_AIE"})
manifest = {
    "note": ("Single-hw_context overlay + per-length ELF module set for the "
             "collapsible activation ops (silu, gelu), aie2/Phoenix. Register one "
             "overlay per op; load each length's ELF as an xrt::module into that "
             "context. length only changes the DMA instruction stream; the overlay "
             "(array config + core program) is length-independent. UNVALIDATED "
             "(compiled on Linux, not executed on NPU)."),
    "kernel_name": "MLIR_AIE",
    "opcode": 3,
    "host_call": "kernel(3, 0, 0, IN_bo, OUT_bo)  # instrs come from the module",
    "not_collapsible": {
        "rms_norm": ("ObjectFifo L1 buffer = memref<embedding_dim> and the core bakes "
                     "cols as arith.constant -> overlay differs per shape. Needs RTP "
                     "cols + fixed max L1 buffer to collapse. Keeps per-shape xclbins "
                     "in prebuilt/ops/."),
        "rope": ("Same as rms_norm: ObjectFifo L1 buffer = memref<embedding_dim> + "
                 "baked cols constant -> overlay differs per head_dim. Keeps per-shape "
                 "xclbins in prebuilt/ops/."),
    },
    "op_count": len(ops),
    "shape_count": len(shapes),
    "ops": ops,
    "shapes": shapes,
}
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"manifest: {len(ops)} ops, {len(shapes)} shapes")
PY

echo ""
echo "Artifacts in ${OUT}:"
echo "  overlays: $(ls "${OUT}"/*_overlay.xclbin 2>/dev/null | wc -l)"
echo "  elfs:     $(ls "${OUT}"/*.elf 2>/dev/null | wc -l)"
[ "$FAIL" -eq 0 ] && echo "BUILD OK" || { echo "BUILD HAD FAILURES"; exit 1; }
