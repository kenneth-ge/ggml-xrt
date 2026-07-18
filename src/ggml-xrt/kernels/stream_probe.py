# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on stream_gemv.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt MAX-BANDWIDTH derisking probe: find the NPU's true peak DDR-read rate. Pure weight
# streaming, no compute, no activation. Sweeps parallelism to see if any config beats the
# current 12.1 GB/s 4-col ceiling (CPU sustains 41.9 GB/s on the same DRAM). Go/no-go for
# whether NPU decode parity is physically possible.
#
# Knobs:
#   --cols C            : columns (distinct shim+memtile+core per column), 1..4
#   --streams S         : parallel weight DMAs PER column (1..2). The shim has 2 MM2S channels,
#                         so S=2 drives all 8 shim read-DMA channels across 4 cols.
#   --depth D           : object_fifo depth (DMA/consume overlap)
# Each (col,stream) drains STREAM_BYTES from DDR through its own shim->memtile->core chain.
# Aggregate GB/s = (cols * streams * STREAM_BYTES) / wall.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212
QK = 256


def stream_probe(dev, K, N, m, cols, streams, depth):
    bf16 = str_to_dtype("bf16")
    K_div_k = K // QK
    M_div_m = N // m
    Sper = N * (K // QK) * REC  # bytes per stream
    ntiles = M_div_m * K_div_k

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
                ins = []
                for s in range(streams):
                    memA = object_fifo(f"memA{c}_{s}", shims[c], mts[c], depth, a_ty)
                    inA = object_fifo(f"inA{c}_{s}", mts[c], cores[c], depth, a_ty)
                    object_fifo_link(memA, inA)
                    ins.append((memA, inA))
                outC = object_fifo(f"outC{c}", cores[c], shims[c], 2, c_ty)

                # Mirror the WORKING stream_gemv structure exactly: one outC tile per
                # infinite-loop iter (runtime feeds M_div_m tiles), draining K_div_k weight
                # elements per stream per tile. The prior version drained `ntiles` in a single
                # outC tile, so the outC DMA (sized for M_div_m tiles) completed early / hung.
                @core(cores[c], "mv_q6k.o", stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            for s in range(streams):
                                ins[s][1].acquire(ObjectFifoPort.Consume, 1)
                                ins[s][1].release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                col_fifos.append(([mA for mA, _ in ins], outC))

            for c in range(cols):
                build(c)

            @runtime_sequence(
                np.ndarray[(cols * streams * Sper,), np.dtype[np.uint8]],
                np.ndarray[(cols * N,), np.dtype[np.float32]],
            )
            def sequence(A, C):
                bd = 1
                for c in range(cols):
                    memAs, outC = col_fifos[c]
                    for s in range(streams):
                        idx = c * streams + s
                        npu_dma_memcpy_nd(metadata=memAs[s], bd_id=bd, mem=A,
                                          offsets=[0, 0, 0, idx * Sper],
                                          sizes=[M_div_m, K_div_k, m, REC],
                                          strides=[m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                        bd += 1
                    npu_dma_memcpy_nd(metadata=outC, bd_id=bd, mem=C,
                                      offsets=[0, 0, 0, c * N],
                                      sizes=[1, 1, 1, N], strides=[0, 0, 0, 1])
                    bd += 1
                for c in range(cols):
                    dma_wait(col_fifos[c][1])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt max-bandwidth streaming probe")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=6144)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--streams", type=int, default=2)
    p.add_argument("--depth", type=int, default=4)
    a, _ = p.parse_known_args()
    stream_probe(a.dev, a.K, a.N, a.m, a.cols, a.streams, a.depth)
