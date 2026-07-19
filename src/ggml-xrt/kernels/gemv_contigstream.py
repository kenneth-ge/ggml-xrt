# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on stream_probe.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt STEP-2 CONTIGUOUS-STREAM BANDWIDTH PROBE for the M=1 q6k weight-DMA wall.
#
# Streams the EXACT q6k weight bytes of the shipping down 6144x2048 16-core gemv (10.4 MB total,
# 4 columns x 2.6 MB each) through 4 shim->memtile->core chains with NO dequant and NO MAC — the
# cores only acquire/release (pure drain, like stream_probe). Wall time == pure weight-DMA time.
#
# The ONLY thing that differs from the shipping weight path is the DDR access pattern of the
# weight npu_dma_memcpy_nd:
#   --pattern strided : sizes=[M_div_m, K_div_k, m, REC], contiguous burst = REC(=212) bytes,
#                       then a (K/256)*REC(=5088) byte stride to the next output. This reproduces
#                       the shipping gemv's 212-byte-burst / 5088-byte-stride DDR pattern (the
#                       object_fifo distribute groups one k-tile across all outputs of a tile).
#   --pattern contig  : sizes=[1, 1, M_div_m*K_div_k, m*REC], strides=[0,0,m*REC,1] — every read is
#                       adjacent (stride == inner size) so each column is ONE fully-contiguous
#                       2.6 MB blast; contiguous burst = m*REC(=6784) bytes, 32x the strided run.
#
# Same 6784-byte object granule (a_ty=[m,REC]) and same 4-column/shim topology in BOTH cases, so
# strided-vs-contig isolates the access pattern alone. If contig hits ~23 GB/s (~0.45 ms) while
# strided caps ~11 GB/s (~0.95 ms), the 212-byte STRIDING is the lever (Step 3 worth it). If contig
# ALSO caps ~11, the memtile/shim aggregate BW is the wall (lever dead — STOP).
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212
QK = 256


def contigstream(dev, K, N, m, cols, depth, pattern):
    bf16 = str_to_dtype("bf16")
    K_div_k = K // QK
    N_col = N // cols            # outputs per column (512 for down 6144x2048)
    M_div_m = N_col // m         # output m-tiles per column (16)
    assert N % cols == 0 and N_col % m == 0
    Acol = N_col * K_div_k * REC  # contiguous weight bytes per column (2.6 MB)

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            zero = external_func("zero_scalar_f32", inputs=[c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [tile(c, 2) for c in range(cols)]

            col_fifos = []

            def build(c):
                memA = object_fifo(f"memA{c}", shims[c], mts[c], depth, a_ty)
                inA = object_fifo(f"inA{c}", mts[c], cores[c], depth, a_ty)
                object_fifo_link(memA, inA)
                outC = object_fifo(f"outC{c}", cores[c], shims[c], 2, c_ty)

                # One outC tile per infinite-loop iter (runtime feeds M_div_m tiles), draining
                # K_div_k weight elements per tile = M_div_m*K_div_k = Acol/6784 elements total.
                # Pure drain: NO matvec, NO dequant — wall == weight-DMA time.
                @core(cores[c], "mv_q6k.o", stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            inA.acquire(ObjectFifoPort.Consume, 1)
                            inA.release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                col_fifos.append((memA, outC))

            for c in range(cols):
                build(c)

            @runtime_sequence(
                np.ndarray[(cols * Acol,), np.dtype[np.uint8]],
                np.ndarray[(N,), np.dtype[np.float32]],
            )
            def sequence(A, C):
                bd = 1
                for c in range(cols):
                    memA, outC = col_fifos[c]
                    if pattern == "contig":
                        # fully contiguous: every dim stride == (inner size * inner count), so the
                        # DDR address stream is perfectly adjacent (no gaps) and the shim coalesces
                        # into max AXI bursts. Inner dim stays REC(=212, <=1023 BD wrap limit) but
                        # the NEXT step is +REC (adjacent) instead of the strided +5088, so column
                        # c's 2.6 MB is one contiguous blast. sizes chain: [1, m, M_div_m*K_div_k, REC].
                        assert (M_div_m * K_div_k) <= 1023
                        npu_dma_memcpy_nd(metadata=memA, bd_id=bd, mem=A,
                                          offsets=[0, 0, 0, c * Acol],
                                          sizes=[1, m, M_div_m * K_div_k, REC],
                                          strides=[0, REC * (M_div_m * K_div_k), REC, 1])
                    else:
                        # strided: reproduces the shipping gemv DDR pattern — 212 B contiguous,
                        # then (K/256)*REC = 5088 B stride to the next output within a k-tile.
                        npu_dma_memcpy_nd(metadata=memA, bd_id=bd, mem=A,
                                          offsets=[0, 0, 0, c * Acol],
                                          sizes=[M_div_m, K_div_k, m, REC],
                                          strides=[m * K_div_k * REC, REC, K_div_k * REC, 1])
                    bd += 1
                    npu_dma_memcpy_nd(metadata=outC, bd_id=bd, mem=C,
                                      offsets=[0, 0, 0, c * N_col],
                                      sizes=[1, 1, 1, N_col], strides=[0, 0, 0, 1])
                    bd += 1
                for c in range(cols):
                    dma_wait(col_fifos[c][1])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt contiguous-stream BW probe")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=6144)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--pattern", choices=["contig", "strided"], default="contig")
    a, _ = p.parse_known_args()
    contigstream(a.dev, a.K, a.N, a.m, a.cols, a.depth, a.pattern)
