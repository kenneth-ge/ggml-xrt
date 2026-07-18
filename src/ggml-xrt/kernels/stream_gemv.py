# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_q6k.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt multi-column DDR->AIE streaming CEILING benchmark. This is the WORKING single-column
# gemv-stream design (gemv_q6k --mode stream, which measures 3.59 GB/s on HW) replicated across
# C columns in parallel, with the dequant+MAC removed (core only zero + acquire/release). The
# from-scratch stream_bench harness hung; gutting a known-good gemv sidesteps whatever the
# minimal scaffold was missing (per the measuring agent's suggestion).
#
# Each column streams its own K x N q6_K weight (== the gemv memA path, 3 buffers A/B/C, real
# fifo depth, mv_q6k.o linked for zero). Aggregate GB/s = (C * per-col weight bytes) / wall.
#   C=1 must reproduce ~3.59 GB/s; C=4 answers whether parallel columns multiply DDR bandwidth
#   (the go/no-go for CPU-parity decode).
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212
QK = 256


def stream_gemv(dev, K, N, m, cols):
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    M_div_m = N // m
    Aper = N * (K // QK) * REC  # weight bytes streamed per column
    assert N % m == 0 and K % QK == 0

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [tile(c, 2) for c in range(cols)]

            def build(c):  # exact working gemv-stream structure, one per column
                memA = object_fifo(f"memA{c}", shims[c], mts[c], 2, a_ty)
                inA = object_fifo(f"inA{c}", mts[c], cores[c], 2, a_ty)
                object_fifo_link(memA, inA)
                inB = object_fifo(f"inB{c}", shims[c], cores[c], 2, b_ty)
                outC = object_fifo(f"outC{c}", cores[c], shims[c], 2, c_ty)

                @core(cores[c], "mv_q6k.o", stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            inA.acquire(ObjectFifoPort.Consume, 1)
                            inB.acquire(ObjectFifoPort.Consume, 1)
                            inA.release(ObjectFifoPort.Consume, 1)
                            inB.release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                return memA, inB, outC

            fs = [build(c) for c in range(cols)]

            @runtime_sequence(
                np.ndarray[(cols * Aper,), np.dtype[np.uint8]],
                np.ndarray[(cols * K,), np.dtype[bf16]],
                np.ndarray[(cols * N,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for c in range(cols):
                    memA, inB, outC = fs[c]
                    npu_dma_memcpy_nd(metadata=inB, bd_id=c * 3 + 2, mem=B,
                                      offsets=[0, 0, 0, c * K],
                                      sizes=[M_div_m, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=memA, bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, c * Aper],
                                      sizes=[M_div_m, K_div_k, m, REC],
                                      strides=[m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                    npu_dma_memcpy_nd(metadata=outC, bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * N],
                                      sizes=[1, 1, 1, N], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(fs[c][2])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt multi-column gemv-stream ceiling benchmark")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=6144)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=1)
    a, _ = p.parse_known_args()
    stream_gemv(a.dev, a.K, a.N, a.m, a.cols)
