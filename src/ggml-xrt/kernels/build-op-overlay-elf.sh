#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the SINGLE-hw_context overlay + per-shape ELF module sets for the single-core
# activation / norm ops, aie2/Phoenix: silu, gelu, rms_norm, rope. Mirrors
# build-overlay-elf.sh (the gemv decode set) so each op class collapses to ONE overlay
# (== one hw_context) + N cheap instruction ELF modules, instead of one xclbin (== one
# context) per shape.
#
# HOW EACH OP COLLAPSES (see docs/ggml-xrt-linux-kernel-wishlist.md §8):
#   silu / gelu : the ml/silu, ml/gelu designs use a FIXED L1 line (line_size=1024); the
#                 transfer `length` only rewrites the runtime_sequence DMA taps. Overlay
#                 is length-independent by construction.
#   rms_norm / rope : the STOCK ml/ designs bake the row length into the L1 ObjectFifo
#                 buffer AND into an arith.constant `cols` in the core -> overlay changes
#                 per shape. We fix that with rms_norm_rtp.py / rope_rtp.py:
#                   * FIXED max L1 buffer (LMAX)      -> L1 alloc shape-independent
#                   * `cols`/`dims` as an RTP (aiex.npu.rtp_write in the runtime_sequence,
#                     read from an L1 buffer by the core) -> not baked in the core
#                   * core loops range_(0xFFFFFFFF)   -> row count (seq) not baked
#                 => the overlay (array config + core program) is byte-identical for every
#                    (seq, cols/dims); shape lives entirely in the ELF (DMA counts + the
#                    rtp_write value). ONE overlay serves ALL shapes of the op.
#                 HOST ABI for rms/rope: rows are fixed LMAX objects, so the host uploads
#                 each row padded to LMAX (real data in [0:cols|dims]); the kernel reduces
#                 / rotates only the valid prefix, so padding is ignored (rms divisor stays
#                 = cols). Pick LMAX per op below.
#
# OUTPUT (src/ggml-xrt/kernels/prebuilt/op-overlays/):
#   <op>_overlay.xclbin          one per op — register once (one hw_context)
#   silu_<L>.elf / gelu_<L>.elf  per length
#   rms_norm_s<seq>_c<cols>.elf  per (seq,cols)   (all share rms_norm_overlay.xclbin)
#   rope_s<seq>_d<dims>.elf      per (seq,dims)   (all share rope_overlay.xclbin)
#   manifest.json                shape -> {overlay, elf, kernel_name, buffers} + LMAX/ABI
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Windows validates on hardware.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${here}/prebuilt/op-overlays"

# --- shape sets -------------------------------------------------------------
LENGTHS=(2048 4096 6144 8192 12288 16384)   # silu/gelu (multiple of 1024)
COLS=4; CHANS=2
RMS_SEQ=(1 32)                               # decode row (1) + prefill tile (32)
RMS_COLS=(128 256 1024 1536 2048 2560 2816 3072 3840 4096 5120 5376)
RMS_LMAX=6144                                # >= max rms row (5376)
ROPE_SEQ=(1 32)
ROPE_DIMS=(128 256)
ROPE_LMAX=256                                # >= max head_dim (256)

rm -rf "$OUT"; mkdir -p "$OUT"

shim="$(mktemp -d)"; cp "${here}/headless_shim.py" "${shim}/sitecustomize.py"
export PYTHONPATH="${shim}:${PYTHONPATH:-}"
# shellcheck disable=SC1091
source "${IRONENV}/bin/activate"
MLIR_AIE_INSTALL="$(python3 -c 'import mlir_aie; print(mlir_aie.__path__[0])')"; export MLIR_AIE_INSTALL
# shellcheck disable=SC1091
source "${MLIR_AIE_SRC}/utils/env_setup.sh" "${MLIR_AIE_INSTALL}" >/dev/null 2>&1
export PEANO_INSTALL_DIR="${IRONENV}/lib/python3.12/site-packages/llvm-aie"
CLANG="${PEANO_INSTALL_DIR}/bin/clang++"
PE="${MLIR_AIE_SRC}/programming_examples/ml"
PEANOWRAP2="-O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
-Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw -DNDEBUG"

FAIL=0
W="$(mktemp -d)"

ov_plus_elf() { # $1=xclbin-out $2=elf-out $3=mlir ; build overlay + first elf
  ( cd "${W}" && aiecc.py --aie-generate-xclbin --aie-generate-elf --no-compile-host \
      --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
      --xclbin-name="$1" --elf-name="$2" "$3" >aiecc.log 2>&1 )
}
elf_only() { # $1=overlay $2=elf-out $3=mlir
  ( cd "${W}" && aiecc.py --xclbin-input="$1" --aie-generate-elf --no-compile-host \
      --no-xchesscc --no-xbridge --peano "${PEANO_INSTALL_DIR}" \
      --elf-name="$2" "$3" >aiecc.log 2>&1 )
}

# =========================== silu / gelu ====================================
for op in silu gelu; do
  ( cd "${PE}/${op}" && make clean >/dev/null 2>&1 \
      && make devicename=npu length=16384 cols="${COLS}" chans="${CHANS}" build/kernels.a >/dev/null 2>&1 )
  cp "${PE}/${op}/build/kernels.a" "${W}/kernels.a"
  ov=""
  for L in "${LENGTHS[@]}"; do
    ( cd "${W}" && python3 "${PE}/${op}/${op}.py" -d npu -l "${L}" -co "${COLS}" -ch "${CHANS}" > shape.mlir )
    elf="${op}_${L}.elf"; ovn="${op}_overlay.xclbin"
    if [ -z "${ov}" ]; then
      if ov_plus_elf "ov.xclbin" "mod.elf" "shape.mlir"; then
        cp "${W}/ov.xclbin" "${OUT}/${ovn}"; cp "${W}/mod.elf" "${OUT}/${elf}"; ov="${ovn}"
        echo "OK  ${op} overlay+elf L=${L}"
      else echo "FAIL ${op} overlay L=${L}"; tail -15 "${W}/aiecc.log"; FAIL=1; fi
    else
      cp "${OUT}/${ov}" "${W}/ov.xclbin"
      if elf_only "ov.xclbin" "mod.elf" "shape.mlir"; then
        cp "${W}/mod.elf" "${OUT}/${elf}"; echo "OK  ${op} elf L=${L}"
      else echo "FAIL ${op} elf L=${L}"; tail -15 "${W}/aiecc.log"; FAIL=1; fi
    fi
  done
done

# ================= rms_norm + rope PACKED (RTP, one shared overlay) ==========
# Both single-core ops live in ONE overlay (rms on col0 tile(0,2), rope on col1
# tile(1,2)); per-op ELFs (rms per-cols, rope per-dims) all load against it, so both ops
# share ONE hw_context and are dispatched independently by loading the right ELF module.
# See rms_rope_packed.py. The overlay is built from the rms variant; every other shape
# (rms cols, rope dims, seq) is an ELF-only build against that one overlay.
"${CLANG}" ${PEANOWRAP2} -I "${MLIR_AIE_SRC}/aie_kernels/aie2" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${here}/aie2/rms_norm.cc" -o "${W}/rms_norm.o"
"${CLANG}" ${PEANOWRAP2} -I "${MLIR_AIE_SRC}/aie_kernels/aie2p" -I "${MLIR_AIE_INSTALL}/include" \
  -c "${MLIR_AIE_SRC}/aie_kernels/aie2p/rope.cc" -o "${W}/rope.o"
PACK_OV="rms_rope_overlay.xclbin"
ov=""
# rms shapes
for s in "${RMS_SEQ[@]}"; do for c in "${RMS_COLS[@]}"; do
  ( cd "${W}" && python3 "${here}/rms_rope_packed.py" -d npu --op rms -s "${s}" -n "${c}" > shape.mlir )
  elf="rms_norm_s${s}_c${c}.elf"
  if [ -z "${ov}" ]; then
    if ov_plus_elf "ov.xclbin" "mod.elf" "shape.mlir"; then
      cp "${W}/ov.xclbin" "${OUT}/${PACK_OV}"; cp "${W}/mod.elf" "${OUT}/${elf}"; ov="${PACK_OV}"
      echo "OK  packed overlay+elf rms s=${s} c=${c}"
    else echo "FAIL packed overlay rms s=${s} c=${c}"; tail -15 "${W}/aiecc.log"; FAIL=1; fi
  else
    cp "${OUT}/${ov}" "${W}/ov.xclbin"
    if elf_only "ov.xclbin" "mod.elf" "shape.mlir"; then
      cp "${W}/mod.elf" "${OUT}/${elf}"; echo "OK  packed elf rms s=${s} c=${c}"
    else echo "FAIL packed elf rms s=${s} c=${c}"; tail -15 "${W}/aiecc.log"; FAIL=1; fi
  fi
done; done
# rope shapes (ELF-only against the packed overlay)
for s in "${ROPE_SEQ[@]}"; do for d in "${ROPE_DIMS[@]}"; do
  ( cd "${W}" && python3 "${here}/rms_rope_packed.py" -d npu --op rope -s "${s}" -n "${d}" > shape.mlir )
  elf="rope_s${s}_d${d}.elf"
  cp "${OUT}/${PACK_OV}" "${W}/ov.xclbin"
  if elf_only "ov.xclbin" "mod.elf" "shape.mlir"; then
    cp "${W}/mod.elf" "${OUT}/${elf}"; echo "OK  packed elf rope s=${s} d=${d}"
  else echo "FAIL packed elf rope s=${s} d=${d}"; tail -15 "${W}/aiecc.log"; FAIL=1; fi
done; done

# =========================== manifest =======================================
OUT="$OUT" RMS_LMAX="$RMS_LMAX" ROPE_LMAX="$ROPE_LMAX" python3 - <<'PY'
import os, re, json, glob
OUT = os.environ["OUT"]
def elfs(pat, key):
    out = []
    for e in sorted(glob.glob(os.path.join(OUT, pat))):
        out.append((os.path.basename(e), key(os.path.basename(e))))
    return out
shapes = []
for e, L in elfs("silu_*.elf", lambda b: {"length": int(re.search(r'_(\d+)\.elf$', b).group(1))}):
    shapes.append({"op": "silu", **L, "overlay": "silu_overlay.xclbin", "elf": e,
                   "kernel_name": "MLIR_AIE", "buffers": ["in", "out"]})
for e, L in elfs("gelu_*.elf", lambda b: {"length": int(re.search(r'_(\d+)\.elf$', b).group(1))}):
    shapes.append({"op": "gelu", **L, "overlay": "gelu_overlay.xclbin", "elf": e,
                   "kernel_name": "MLIR_AIE", "buffers": ["in", "out"]})
for e, K in elfs("rms_norm_*.elf",
                 lambda b: dict(zip(("seq", "cols"),
                                    map(int, re.search(r'_s(\d+)_c(\d+)\.elf$', b).groups())))):
    shapes.append({"op": "rms_norm", **K, "overlay": "rms_rope_overlay.xclbin", "elf": e,
                   "kernel_name": "MLIR_AIE", "buffers": ["in", "out"],
                   "lmax": int(os.environ["RMS_LMAX"])})
for e, K in elfs("rope_*.elf",
                 lambda b: dict(zip(("seq", "dims"),
                                   map(int, re.search(r'_s(\d+)_d(\d+)\.elf$', b).groups())))):
    shapes.append({"op": "rope", **K, "overlay": "rms_rope_overlay.xclbin", "elf": e,
                   "kernel_name": "MLIR_AIE", "buffers": ["in", "lut", "out"],
                   "lmax": int(os.environ["ROPE_LMAX"])})
ops = {}
for s in shapes:
    ops.setdefault(s["overlay"], {"overlay": s["overlay"], "ops": set(), "elf_count": 0})
    ops[s["overlay"]]["ops"].add(s["op"])
    ops[s["overlay"]]["elf_count"] += 1
overlays = [{"overlay": k, "ops": sorted(v["ops"]), "elf_count": v["elf_count"]}
            for k, v in sorted(ops.items())]
manifest = {
    "note": ("Overlay + per-shape ELF module sets for the single-core activation/norm ops "
             "(silu, gelu, rms_norm, rope), aie2/Phoenix. Register an overlay (ONE "
             "hw_context) and load a shape's ELF as an xrt::module into that context. "
             "silu/gelu: 1 overlay each, shape = DMA taps only. rms_norm+rope are "
             "SPATIALLY PACKED into ONE overlay (rms core on col0, rope core on col1): "
             "both ops SHARE ONE hw_context, selected by which ELF module is loaded. "
             "rms/rope shape (cols/dims) is an RTP baked in the ELF; fixed L1 buffer "
             "(lmax). => all four ops = THREE hw_contexts total. UNVALIDATED (Linux)."),
    "kernel_name": "MLIR_AIE", "opcode": 3,
    "host_call": "kernel(3, 0, 0, *buffer_bos)  # buffers per shape.buffers; instrs from the module",
    "rtp_note": ("rms_norm/rope: the row length is written at dispatch by the ELF's "
                 "aiex.npu.rtp_write; the host uploads rows PADDED to lmax (real data in "
                 "[0:cols|dims]); the kernel touches only the valid prefix (rms divisor = cols)."),
    "packed_note": ("rms_rope_overlay.xclbin hosts BOTH the rms_norm and rope cores; the "
                    "rms ELFs and rope ELFs are load-compatible with it by construction "
                    "(built --xclbin-input rms_rope_overlay.xclbin). Dispatch rms by "
                    "loading a rms_norm_* ELF; dispatch rope by loading a rope_* ELF; the "
                    "idle op's core blocks harmlessly on its unset lock."),
    "hw_context_count": len(overlays),
    "shape_count": len(shapes),
    "overlays": overlays,
    "shapes": shapes,
}
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"manifest: {len(overlays)} overlays (hw_contexts), {len(shapes)} shapes")
PY

echo ""
echo "Artifacts in ${OUT}:"
echo "  overlays: $(ls "${OUT}"/*_overlay.xclbin 2>/dev/null | wc -l)"
echo "  elfs:     $(ls "${OUT}"/*.elf 2>/dev/null | wc -l)"
[ "$FAIL" -eq 0 ] && echo "BUILD OK" || { echo "BUILD HAD FAILURES"; exit 1; }
