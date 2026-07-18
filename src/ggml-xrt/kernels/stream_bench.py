# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie passthrough_dmas (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt DECODE bandwidth microbenchmark: pure DDR->AIE weight STREAMING, no compute.
# DMAs a large buffer from DDR into the array and DISCARDS it (core just acquires/releases;
# emits one token so the host can time completion). This isolates the achievable DDR READ
# bandwidth on this silicon with zero dequant/MAC in the way — the ceiling every real decode
# kernel is measured against. Report:  GB/s = total_bytes / wall_time.
#
#   --cols C : stream across C columns (1..4) in parallel, each column its own shim->core
#              chain moving total_bytes/C. C=1 = what the current gemv path does; C=4 tests
#              whether parallel shim channels saturate LPDDR5 (the real ceiling question).
#   -B bytes : total bytes streamed (e.g. 2621440 = 2.5 MiB, 10485760 = 10 MiB).
#   --tile   : per-transfer L1 tile bytes (<= ~L1/depth). Big tiles amortize core loop
#              overhead so the run is DMA-bound.
#   --depth  : object_fifo depth (>=2 for DMA/consume overlap).
#
# No dequant, no output beyond a token: measures the same DDR read traffic a weight-only
# gemv would incur (weight bytes dominate; activation vector is negligible at decode).
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_


def stream_bench(dev, total_bytes, cols, tbytes, depth):
    assert total_bytes % cols == 0, "total_bytes must divide across cols"
    per_col = total_bytes // cols
    assert per_col % tbytes == 0, "per-col bytes must be a multiple of --tile"
    # Shim DMA wrap limits: d0 (contiguous tile) <= 1023 addr-gen units (=4092 B for uint8),
    # d1 <= 1023, iteration (d3) <= 64. So express the per-col stream as
    #   tbytes (d0) x inner (d1) x reps (iteration),  tbytes*inner*reps = per_col.
    assert tbytes % 4 == 0 and tbytes // 4 <= 1023, "tile must be <=4092 B, mult of 4"
    ntiles = per_col // tbytes
    reps = 1
    for r in range(min(64, ntiles), 0, -1):
        if ntiles % r == 0 and (ntiles // r) <= 1023:
            reps = r
            break
    inner = ntiles // reps
    assert reps * inner == ntiles and inner <= 1023 and reps <= 64, \
        f"cannot factor {ntiles} tiles into inner<=1023 x reps<=64"
    niter = ntiles

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            tile_ty = np.ndarray[(tbytes,), np.dtype[np.uint8]]
            tok_ty = np.ndarray[(1,), np.dtype[np.int32]]

            shims = [tile(c, 0) for c in range(cols)]
            cores = [tile(c, 2) for c in range(cols)]

            def build_col(c):  # own scope so the core closure binds this column's fifos
                inF = object_fifo(f"in{c}", shims[c], cores[c], depth, tile_ty)
                tokF = object_fifo(f"tok{c}", cores[c], shims[c], 2, tok_ty)

                @core(cores[c])
                def core_body():
                    t = tokF.acquire(ObjectFifoPort.Produce, 1)
                    for _ in range_(niter):
                        inF.acquire(ObjectFifoPort.Consume, 1)
                        inF.release(ObjectFifoPort.Consume, 1)
                    t[0] = 1
                    tokF.release(ObjectFifoPort.Produce, 1)

                return inF, tokF

            inFs, tokFs = [], []
            for c in range(cols):
                inF, tokF = build_col(c)
                inFs.append(inF)
                tokFs.append(tokF)

            in_flat = np.ndarray[(total_bytes,), np.dtype[np.uint8]]
            tok_flat = np.ndarray[(cols,), np.dtype[np.int32]]

            @runtime_sequence(in_flat, tok_flat)
            def sequence(IN, TOK):
                for c in range(cols):
                    # d0=tbytes (contiguous), d1=inner, iteration=reps; d2 unused on shim.
                    npu_dma_memcpy_nd(
                        metadata=inFs[c], bd_id=c * 2 + 1, mem=IN,
                        offsets=[0, 0, 0, c * per_col],
                        sizes=[reps, 1, inner, tbytes],
                        strides=[inner * tbytes, 0, tbytes, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=tokFs[c], bd_id=c * 2 + 2, mem=TOK,
                        offsets=[0, 0, 0, c], sizes=[1, 1, 1, 1], strides=[0, 0, 0, 1],
                    )
                for c in range(cols):
                    dma_wait(tokFs[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt DDR->AIE streaming bandwidth microbenchmark")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-B", type=int, required=True, help="total bytes to stream")
    p.add_argument("--cols", type=int, default=1, help="parallel columns (1..4)")
    p.add_argument("--tile", type=int, default=8192, help="L1 tile bytes")
    p.add_argument("--depth", type=int, default=4, help="object_fifo depth")
    a, _ = p.parse_known_args()
    stream_bench(a.dev, a.B, a.cols, a.tile, a.depth)
