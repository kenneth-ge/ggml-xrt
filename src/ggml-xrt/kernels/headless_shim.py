# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
"""Headless XRT shim for offline AIE kernel compilation (no NPU present).

`import aie.iron` eagerly constructs ``CachedXRTRuntime()`` (see
``mlir_aie/python/aie/utils/__init__.py``), which opens ``pyxrt.device(0)`` and
fails on a machine without an NPU (e.g. WSL). Kernel *compilation* does not need a
live device — the target architecture comes from the IRON design, not the device —
so this shim replaces ``pyxrt.device`` with a stub that reports a Phoenix (npu1)
device, letting MLIR generation proceed.

Usage: place this directory on ``PYTHONPATH`` as ``sitecustomize.py`` so every
Python subprocess in the build applies it automatically:

    mkdir -p /tmp/xrtshim && cp headless_shim.py /tmp/xrtshim/sitecustomize.py
    export PYTHONPATH=/tmp/xrtshim:$PYTHONPATH
"""

try:
    import pyxrt

    class _FakeDevice:
        """Minimal stand-in for xrt::device sufficient for offline compilation."""

        def __init__(self, *args, **kwargs):
            pass

        def get_info(self, *args, **kwargs):
            # Must contain a string matched by XRTHostRuntime.NPU_MODELS["npu1"].
            return "RyzenAI-npu1 Phoenix"

        def __getattr__(self, name):
            def _stub(*args, **kwargs):
                return None

            return _stub

    pyxrt.device = _FakeDevice
except Exception:
    # If pyxrt is absent, aie.utils already degrades to CPU-only; nothing to do.
    pass
