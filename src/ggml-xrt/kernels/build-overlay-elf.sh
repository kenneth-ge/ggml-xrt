#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Build the SINGLE-hw_context overlay + per-shape ELF module set for the decode
# (gemv, M=1) NPU path, aie2/Phoenix.
#
# WHY: Phoenix allows only ~5 concurrent XRT hw_contexts, and each xclbin == one
# context. A full model has many gemv (K,N) shapes; one-xclbin-per-shape overflows
# and the host thrashes (LRU-evicts contexts). Fix: build ONE overlay xclbin per
# (dtype,K) group (the AIE array config + core program, which for these gemv designs
# depends only on kernel-type and K, NOT on output N), and ship every N of that
# group as a lightweight instruction ELF *module* loaded into that one context via
# xrt::module + xrt::ext::kernel. => distinct hw_contexts per model == distinct
# (dtype,K) groups it touches (<= 5).
#
# 2026-07-18 UPDATE (perf pass): the QUANT decode kernels (q4_0/q4k/q6k) are now 16-CORE
# (4 cols x 4 rows = all of Phoenix's compute tiles), built from gemv_mc16.py --rows 4
# (per-column memtile distributes the weight to 4 core-rows, b broadcast, C gathered;
# ~3.82x over 4-core, e.g. q6k down 36.5->9.55ms). Cores mv_q4*.cc/mv_q6k.cc (q6k=simd2).
# The overlay/ELF split STILL HOLDS: each core loops range_(0xFFFFFFFF)/range_(K_div_k)
# (K, not N; no baked Mdm), and all N-dependence (Mdm, DMA offsets) is in the
# runtime_sequence -> the per-N ELF. Verified: same-(dtype,K) N shapes emit byte-identical
# device/core MLIR at 16 cores. Overlay group is (dtype,K); N must be divisible by m*16=512.
# The previous frozen 4-core/single-col quant overlays are stale and REPLACED here so the
# overlay path runs the 16-core kernel (host had GGML_XRT_OVERLAY=0 as a stopgap). bf16
# gemv stays single-column (mv.cc unchanged) via gemv.py.
#
# Ground truth: the shape set is enumerated by globbing the existing prebuilt gemv
# xclbins (src/ggml-xrt/kernels/prebuilt/**/mul_mat_aie2_<dt>_f32_1x{K}x{N}_gemv.xclbin).
# Out-of-scope model dirs (qwen3.5-122b-a10b, qwen3.5-397b-a17b) are excluded.
#
# OUTPUT (src/ggml-xrt/kernels/prebuilt/overlays/):
#   <dtype>_k<K>_overlay.xclbin     one per (dtype,K) group (register once per group)
#   <dtype>_1x{K}x{N}_gemv.elf      one per shape (the instruction module)
#   manifest.json                   shape -> (overlay, elf, kernel_name) + per-model counts
#
# Build mechanism (proven; both steps succeed on Linux/WSL headless, NO NPU):
#   first shape of a (dtype,K) group -> overlay + its ELF:
#     aiecc.py --aie-generate-xclbin --aie-generate-elf --no-compile-host \
#       --no-xchesscc --no-xbridge --peano $PEANO --xclbin-name=overlay.xclbin \
#       --elf-name=<name>.elf <shape>.mlir
#   every additional same-(dtype,K) shape -> ELF ONLY, against that overlay:
#     aiecc.py --xclbin-input=overlay.xclbin --aie-generate-elf --no-compile-host \
#       --no-xchesscc --no-xbridge --peano $PEANO --elf-name=<name>.elf <shape>.mlir
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Correctness is validated
# later on Windows. This script only proves buildability.
set -euo pipefail

IRONENV="${IRONENV:-$HOME/ironenv}"
MLIR_AIE_SRC="${MLIR_AIE_SRC:-$HOME/mlir-aie}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREBUILT="${here}/prebuilt"
OUT="${PREBUILT}/overlays"
EXCLUDE_RE='qwen3\.5-122b-a10b|qwen3\.5-397b-a17b'
QUANT_COLS="${QUANT_COLS:-4}"   # 4 columns (fixed in gemv_mc16.py)
QUANT_ROWS="${QUANT_ROWS:-4}"   # quant decode gemv is 16-core (4 cols x 4 rows) via gemv_mc16.py

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
AK="${MLIR_AIE_SRC}/aie_kernels/aie2"

# --- per-dtype config -------------------------------------------------------
# dtype -> core .cc | .o name | clang defines | design .py | python gen args
# q4k/q6k use the VSCALE cores (vectorized scale setup; kills the software-fp32 scale that made the
# pre-vscale overlays ~3.5x slower in-model). q4_0/bf16 unchanged (q4_0 already had no software-fp).
dt_cc()   { case "$1" in bf16) echo mv.cc;; q4_0) echo mv_q4.cc;; q4k) echo mv_q4k_vscale.cc;; q6k) echo mv_q6k_vscale.cc;; esac; }
dt_o()    { case "$1" in bf16) echo mv_32x32.o;; q4_0) echo mv_q4_32x32.o;; q4k) echo mv_q4k.o;; q6k) echo mv_q6k.o;; esac; }
dt_def()  { case "$1" in bf16|q4_0) echo "-DDIM_M=32 -DDIM_K=32";; q4k|q6k) echo "-DDIM_M=32";; esac; }
dt_py()   { case "$1" in bf16) echo gemv.py;; q4_0) echo gemv_q4.py;; q4k) echo gemv_q4k.py;; q6k) echo gemv_q6k.py;; esac; }
dt_args() { case "$1" in bf16) echo "-m 32 -k 32 --dtype_in bf16 --dtype_out f32";; q4_0) echo "-m 32 -k 32";; q4k|q6k) echo "-m 32";; esac; }

compile_o() { # $1=dtype ; compiles the core .o into cwd (idempotent)
  local dt="$1" o; o="$(dt_o "$dt")"
  [ -f "$o" ] && return 0
  # shellcheck disable=SC2046
  "${CLANG}" -O2 -std=c++20 --target=aie2-none-unknown-elf -Wno-parentheses -Wno-attributes \
    -Wno-macro-redefined -Wno-empty-body -Wno-missing-template-arg-list-after-template-kw \
    -DNDEBUG $(dt_def "$dt") -I "${AK}" -I "${MLIR_AIE_INSTALL}/include" \
    -c "${here}/aie2/$(dt_cc "$dt")" -o "$o"
}

# --- enumerate ground-truth shape set --------------------------------------
mapfile -t SHAPES < <(find "$PREBUILT" -name 'mul_mat_aie2_*_f32_1x*x*_gemv.xclbin' \
  | grep -vE "$EXCLUDE_RE" \
  | sed -E 's#.*/mul_mat_aie2_(.+)_f32_1x([0-9]+)x([0-9]+)_gemv\.xclbin#\1 \2 \3#' \
  | sort -u -k1,1 -k2,2n -k3,3n)

echo "Ground-truth gemv shapes (dtype K N): ${#SHAPES[@]}"

W="$(mktemp -d)"; cd "$W"
FAIL=0
declare -A GROUP_OVERLAY   # "dtype K" -> overlay filename (built)

for line in "${SHAPES[@]}"; do
  read -r dt K N <<<"$line"
  o="$(dt_o "$dt")"
  compile_o "$dt"
  # gen MLIR for this shape.
  #  - QUANT (q4_0/q4k/q6k): the promoted decode kernels are 16-CORE (4 cols x 4 rows),
  #    built via gemv_mc16.py --rows 4. Per column a memtile distributes the weight to the
  #    4 core-rows, b is broadcast, C is gathered. The overlay stays a (dtype,K) group:
  #    the per-core loop is range_(0xFFFFFFFF)/range_(K_div_k) (K only, no baked Mdm) and
  #    all N-dependence (Mdm, offsets) lives in the runtime_sequence -> the per-N ELF.
  #    (Verified: same-(dtype,K) N shapes emit byte-identical device/core MLIR at 16 cores.)
  #    Cores: mv_q4*.cc/mv_q6k.cc (q6k = simd2 unroll). N must be divisible by m*16=512.
  #  - bf16: single-column (mv.cc unchanged), via gemv.py.
  # shellcheck disable=SC2046
  if [ "$dt" = bf16 ]; then
    python "${here}/gemv.py" --dev npu -M "$N" -K "$K" $(dt_args "$dt") > shape.mlir
  else
    python "${here}/gemv_mc16.py" --dev npu --qtype "$dt" -M "$N" -K "$K" -m 32 --rows "${QUANT_ROWS}" > shape.mlir
  fi
  elf_name="${dt}_1x${K}x${N}_gemv.elf"
  ov_name="${dt}_k${K}_overlay.xclbin"
  key="${dt} ${K}"
  if [ -z "${GROUP_OVERLAY[$key]:-}" ]; then
    # first shape of this (dtype,K) group: build overlay + ELF
    if aiecc.py --aie-generate-xclbin --aie-generate-elf --no-compile-host --no-xchesscc \
         --no-xbridge --peano "${PEANO_INSTALL_DIR}" --xclbin-name="ov.xclbin" \
         --elf-name="mod.elf" shape.mlir >aiecc.log 2>&1; then
      cp ov.xclbin "${OUT}/${ov_name}"
      cp mod.elf   "${OUT}/${elf_name}"
      GROUP_OVERLAY[$key]="${ov_name}"
      echo "OK  overlay+elf  ${dt} K=${K} N=${N}  -> ${ov_name}, ${elf_name}"
    else
      echo "FAIL overlay ${dt} K=${K} N=${N}"; tail -20 aiecc.log; FAIL=1
    fi
  else
    # additional shape: ELF only, against the group overlay
    cp "${OUT}/${GROUP_OVERLAY[$key]}" ov.xclbin
    if aiecc.py --xclbin-input=ov.xclbin --aie-generate-elf --no-compile-host --no-xchesscc \
         --no-xbridge --peano "${PEANO_INSTALL_DIR}" --elf-name="mod.elf" shape.mlir >aiecc.log 2>&1; then
      cp mod.elf "${OUT}/${elf_name}"
      echo "OK  elf         ${dt} K=${K} N=${N}  -> ${elf_name}"
    else
      echo "FAIL elf ${dt} K=${K} N=${N}"; tail -20 aiecc.log; FAIL=1
    fi
  fi
done

# --- manifest ---------------------------------------------------------------
OVERLAYS_DIR="$OUT" PREBUILT_DIR="$PREBUILT" EXCLUDE_RE="$EXCLUDE_RE" QUANT_COLS="$QUANT_COLS" python3 - <<'PY'
import os, re, json, glob
OUT = os.environ["OVERLAYS_DIR"]; PREBUILT = os.environ["PREBUILT_DIR"]
pat = re.compile(r'.*/mul_mat_aie2_(.+)_f32_1x(\d+)x(\d+)_gemv\.xclbin$')

# built shapes = the ELFs we produced
shapes = []
for elf in sorted(glob.glob(os.path.join(OUT, "*_gemv.elf"))):
    m = re.match(r'(.+)_1x(\d+)x(\d+)_gemv\.elf$', os.path.basename(elf))
    dt, K, N = m.group(1), int(m.group(2)), int(m.group(3))
    if dt == "bf16":
        ncores, coredesc = 1, "scalar_bf16"
    else:
        ncores = int(os.environ.get("QUANT_COLS", "4")) * int(os.environ.get("QUANT_ROWS", "4"))
        coredesc = {"q6k": "16core_simd2", "q4k": "16core_loopSIMD",
                    "q4_0": "16core_SIMD"}.get(dt, "16core")
    shapes.append({"dtype": dt, "K": K, "N": N,
                   "overlay": f"{dt}_k{K}_overlay.xclbin",
                   "elf": os.path.basename(elf),
                   "kernel_name": "MLIR_AIE",
                   "ncores": ncores,
                   "core": coredesc})
shapes.sort(key=lambda s: (s["dtype"], s["K"], s["N"]))

# overlays summary
groups = {}
for s in shapes:
    groups.setdefault((s["dtype"], s["K"]), 0)
    groups[(s["dtype"], s["K"])] += 1
overlays = [{"dtype": dt, "K": K, "overlay": f"{dt}_k{K}_overlay.xclbin", "elf_count": n,
             "ncores": (1 if dt == "bf16"
                        else int(os.environ.get("QUANT_COLS", "4")) * int(os.environ.get("QUANT_ROWS", "4")))}
            for (dt, K), n in sorted(groups.items())]

# per-model overlay counts (in-scope lineup). A model dir's gemv xclbins define its
# shapes; a live deployment uses ONE dtype, so #hw_contexts = distinct K per dtype.
MODELS = {
    "Qwen3-1.7B":       ".",              # prebuilt root (non-recursive)
    "Qwen3.5-0.8B":     "qwen3.5-0.8b",
    "Qwen3.5-2B":       "qwen3.5-2b",
    "Qwen3.5-4B":       "qwen3.5-4b",
    "Qwen3.5-9B":       "qwen3.5-9b",
    "Qwen3.5-27B":      "qwen3.5-27b",
    "Qwen3.5-35B-A3B":  "qwen3.5-35b-a3b",
    "Gemma4-E2B":       "gemma4-e2b",
    "Gemma4-E4B":       "gemma4-e4b",
    "Gemma4-12B":       "gemma4-12b",
    "Gemma4-31B":       "gemma4-31b",
    "Gemma4-26B-A4B":   "gemma4-26b-a4b",
    "DiffusionGemma":   "gemma4-26b-a4b",  # identical shapes to gemma4-26b-a4b
}
models = {}
for name, sub in MODELS.items():
    d = PREBUILT if sub == "." else os.path.join(PREBUILT, sub)
    files = glob.glob(os.path.join(d, "mul_mat_aie2_*_f32_1x*x*_gemv.xclbin"))
    by_dtype = {}          # dtype -> set(K)
    mshapes = []
    for f in files:
        m = pat.match(f)
        if not m:
            continue
        dt, K, N = m.group(1), int(m.group(2)), int(m.group(3))
        by_dtype.setdefault(dt, set()).add(K)
        mshapes.append({"dtype": dt, "K": K, "N": N})
    overlays_by_dtype = {dt: len(ks) for dt, ks in sorted(by_dtype.items())}
    models[name] = {
        "overlays_by_dtype": overlays_by_dtype,
        "max_overlays_any_dtype": max(overlays_by_dtype.values(), default=0),
        "shapes": sorted(mshapes, key=lambda s: (s["dtype"], s["K"], s["N"])),
    }

manifest = {
    "note": ("Single-hw_context overlay + per-shape ELF module set for the NPU decode "
             "(gemv, M=1) path. Register one overlay per (dtype,K); load each shape's "
             "ELF as an xrt::module into that context. QUANT (q4_0/q4k/q6k) overlays+ELFs "
             "now carry the 16-CORE kernel (gemv_mc16.py, 4 cols x 4 rows; memtile "
             "distribute/gather, b broadcast; cores mv_q4*.cc/mv_q6k.cc) -- matches the "
             "promoted standalone gemv xclbins, so GGML_XRT_OVERLAY can be re-enabled once "
             "validated. bf16 is single-column (mv.cc). N must be divisible by m*16=512. "
             "UNVALIDATED (compiled on Linux, not executed on NPU)."),
    "kernel_name": "MLIR_AIE",
    "opcode": 3,
    "core_note": ("quant overlays/ELFs = 16-core (4 cols x 4 rows); bf16 = 1-column scalar. "
                  "The overlay is the (dtype,K) 16-core program; the ELF is the per-N "
                  "instruction stream (N-independent overlay verified empirically at 16 "
                  "cores). Cores as of 2026-07-18: q6k=16core_simd2 (unrolled, 2 accum), "
                  "q4k=16core_loopSIMD, q4_0=16core_SIMD. Only the overlay xclbin carries "
                  "the core, so a core change re-emits the (dtype,K) overlays; the ELFs "
                  "(instruction stream) are unchanged if the fifo/tile layout is unchanged."),
    "host_call": "kernel(3, 0, 0, A_bo, B_bo, C_bo)  # instrs come from the module",
    "overlay_count": len(overlays),
    "shape_count": len(shapes),
    "overlays": overlays,
    "shapes": shapes,
    "models": models,
}
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"manifest: {len(overlays)} overlays, {len(shapes)} shapes, {len(models)} models")
worst = max((m["max_overlays_any_dtype"] for m in models.values()), default=0)
print(f"max overlays needed by any single (model,dtype): {worst}  (must be <= 5)")
PY

echo ""
echo "Artifacts in ${OUT}:"
echo "  overlays: $(ls "${OUT}"/*_overlay.xclbin 2>/dev/null | wc -l)"
echo "  elfs:     $(ls "${OUT}"/*_gemv.elf 2>/dev/null | wc -l)"
[ "$FAIL" -eq 0 ] && echo "BUILD OK" || { echo "BUILD HAD FAILURES"; exit 1; }
