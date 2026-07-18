# Grafting ggml-xrt into llama.cpp

How the `ggml-xrt` NPU backend (developed in this fork, the standalone ggml repo) is integrated
into a llama.cpp checkout so it runs under `llama-cli` / the full inference stack on Windows.
This is a **manual graft** (copy + wire), kept in sync by re-copying — see "Keeping in sync".

## Why a graft (not a submodule)

llama.cpp vendors its own copy of ggml under `llama.cpp/ggml/`. Our backend lives in the
standalone ggml fork (`kenneth-ge/ggml-xrt`, branch `ggml-xrt-backend`). We copy the backend +
build wiring into llama.cpp's vendored ggml. This works cleanly because the vendored ggml and our
fork are the **same version with a byte-identical backend interface** (see below).

## Target checkout

- **Clone:** `C:\Users\kennyge2\projects\llama.cpp` — upstream `ggml-org/llama.cpp`.
- **Commit at graft time:** `86a9c79` (2026-07-17).
- **Vendored ggml version:** **0.17.0** — matches the fork. Verified byte-identical:
  `ggml/src/ggml-backend-impl.h`, `ggml/include/ggml-backend.h`, `ggml/include/ggml.h`,
  `ggml/src/ggml-impl.h` are all `diff`-clean vs the fork. That identity is what makes the graft
  drop-in; if you graft onto a different llama.cpp commit, re-check these four headers first.
- **Not pushed:** the llama.cpp clone is local build scaffolding; the source of truth is the fork.
  Do NOT push llama.cpp. (An older checkout at `projects/using_npu/llama.cpp` vendors ggml 0.9.5 —
  interface drift — do not use it.)

## What is copied in (source only — not the prebuilt kernel data)

| From (fork) | To (llama.cpp) |
|---|---|
| `src/ggml-xrt/ggml-xrt.cpp` | `ggml/src/ggml-xrt/ggml-xrt.cpp` |
| `src/ggml-xrt/CMakeLists.txt` | `ggml/src/ggml-xrt/CMakeLists.txt` |
| `include/ggml-xrt.h` | `ggml/include/ggml-xrt.h` |

Prebuilt xclbins stay in the fork at `src/ggml-xrt/kernels/prebuilt/` and are pointed to at
runtime via `GGML_XRT_KERNEL_DIR` — they are not copied into llama.cpp.

## Wiring added (3 files, mirrors the fork)

1. `ggml/CMakeLists.txt` — add the option above the Vulkan one:
   `option(GGML_XRT "ggml: use XRT (AMD XDNA NPU via Windows/Linux XRT)" OFF)`
2. `ggml/src/CMakeLists.txt` — add to the backend list:
   `ggml_add_backend(XRT)` (maps `XRT`→dir `ggml-xrt`, defines `GGML_USE_XRT`).
3. `ggml/src/ggml-backend-reg.cpp` — three points:
   - include: `#ifdef GGML_USE_XRT \n #include "ggml-xrt.h" \n #endif`
   - register (in `get_reg()` init): `#ifdef GGML_USE_XRT register_backend(ggml_backend_xrt_reg()); #endif`
   - dynamic load list: `ggml_backend_load_best("xrt", silent, dir_path);`

The `ggml-xrt/CMakeLists.txt` itself: finds XRT via `find_package(XRT)` or `-DXILINX_XRT=<path>`
(`find_path` for `xrt/xrt_device.h` + `find_library` for `xrt_coreutil`), and adds
`/Zc:__cplusplus` on MSVC (so XRT headers pick `std::any` over `boost::any`).

## Configure & build

XRT only:
```
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_XRT=ON -DXILINX_XRT=C:/dev/xrt-sdk \
  -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
```
XRT + Vulkan (the hybrid; needs the Vulkan SDK, add its `\Bin` to PATH for `glslc`):
```
set VULKAN_SDK=C:\VulkanSDK\1.4.350.0
cmake -B build -G "Visual Studio 17 2022" -A x64 -DGGML_XRT=ON -DXILINX_XRT=C:/dev/xrt-sdk \
  -DGGML_VULKAN=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build --target llama-cli --config Release
```
Expected configure lines: `ggml-xrt: found XRT headers ... coreutil ...`, `Including XRT backend`,
`Vulkan found` / `Including Vulkan backend`. `llama-cli --list-devices` then shows
`XRT0: NPU Phoenix`.

## Runtime requirements (Windows)

- PATH must include the **driver-store dir** so `xrt_coreutil.dll` + `xrt_core.dll` load:
  `C:\Windows\System32\DriverStore\FileRepository\kipudrv.inf_amd64_7b0051e064968f34`
  (plus the build's `bin\Release`, and `C:\VulkanSDK\1.4.350.0\Bin` for Vulkan).
- `GGML_XRT_KERNEL_DIR` → the fork's `src/ggml-xrt/kernels/prebuilt`.
- The XRT backend is an **ACCEL** device: llama.cpp does NOT `-ngl`-offload weights to it, but the
  scheduler still routes matching ops to it (like BLAS). So `-ngl 0` = NPU takes CPU-resident
  conformant ops; partial `-ngl` = 3-way split with Vulkan.
- **Do NOT run `llama-cli` inside Claude Code** (it can crash the session; also llama.cpp has a
  benign teardown hang on exit). Run inference in a normal terminal.

## Keeping in sync

The backend DLLs are dynamic, so after editing the fork's `src/ggml-xrt/ggml-xrt.cpp`:
1. copy it to `llama.cpp/ggml/src/ggml-xrt/ggml-xrt.cpp`,
2. `cmake --build build --target ggml-xrt --config Release` (rebuilds just `ggml-xrt.dll`; the
   already-linked `llama-cli` picks up the new DLL — no relink needed),
3. kill any lingering `llama-cli.exe` first (the teardown hang leaves it holding `ggml-xrt.dll`,
   which locks the rebuild): `Stop-Process -Name llama-cli -Force`.

## MSVC gotchas (all handled, listed for re-graft)

- `/Zc:__cplusplus` is required (in `ggml-xrt/CMakeLists.txt`) — MSVC otherwise reports
  `__cplusplus == 199711L`, and XRT headers then pull `boost::any` instead of `std::any`.
- XRT dev SDK is hand-staged at `C:\dev\xrt-sdk` (headers from the XRT source tree + an import lib
  regenerated from the **driver-store** `xrt_coreutil.dll`; the RyzenAI SDK copy is the wrong
  version). See the environment memory / step 1 of the handoff.
