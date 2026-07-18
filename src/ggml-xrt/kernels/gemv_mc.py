# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_q6k.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt MULTI-COLUMN decode gemv (q4_0 / q4_K / q6_K). Splits the output N across `cols`
# columns (one compute core each), so decode gets ~cols x the cores AND rides the measured
# multi-column DMA scaling (4col ~= 3x a single column's bandwidth on Phoenix). Same host
# buffers/ABI as the single-column gemv (A = full [N][K/qk][rec] weight, B = [K] activation
# shared/broadcast to every column, C = full [N] output) - a drop-in throughput upgrade.
#
# Column c computes output rows [c*N/cols : (c+1)*N/cols] with the validated vectorized core
# (matvec_*_f32 from mv_*.o). N must be divisible by cols*m.
#
#   --qtype q4_0|q4k|q6k   --cols C   -M N(output)  -K K(contraction)  -m tile
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

# qtype -> (superblock/k-tile, record bytes, core object, matvec symbol)
QMAP = {
    "q4_0": (32, 20, "mv_q4_32x32.o", "matvec_q4_0_f32"),
    "q4k": (256, 148, "mv_q4k.o", "matvec_q4k_f32"),
    "q6k": (256, 212, "mv_q6k.o", "matvec_q6k_f32"),
}


def gemv_mc(dev, qtype, M, K, m, cols):
    QK, REC, obj, mvsym = QMAP[qtype]
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    assert M % (m * cols) == 0, "N must be divisible by cols*m"
    assert K % QK == 0
    Mc = M // cols                 # output rows per column
    M_div_m_c = Mc // m
    Aper = Mc * (K // QK) * REC     # weight bytes per column

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func(mvsym, inputs=[a_ty, b_ty, c_ty])
            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [tile(c, 2) for c in range(cols)]

            def build(c):
                memA = object_fifo(f"memA{c}", shims[c], mts[c], 2, a_ty)
                inA = object_fifo(f"inA{c}", mts[c], cores[c], 2, a_ty)
                object_fifo_link(memA, inA)
                inB = object_fifo(f"inB{c}", shims[c], cores[c], 2, b_ty)
                outC = object_fifo(f"outC{c}", cores[c], shims[c], 2, c_ty)

                @core(cores[c], obj, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            a = inA.acquire(ObjectFifoPort.Consume, 1)
                            b = inB.acquire(ObjectFifoPort.Consume, 1)
                            matvec(a, b, elem_out)
                            inA.release(ObjectFifoPort.Consume, 1)
                            inB.release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                return memA, inB, outC

            fs = [build(c) for c in range(cols)]

            @runtime_sequence(
                np.ndarray[(M * (K // QK) * REC,), np.dtype[np.uint8]],
                np.ndarray[(K,), np.dtype[bf16]],
                np.ndarray[(M,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for c in range(cols):
                    memA, inB, outC = fs[c]
                    # activation is shared - every column reads the full b[K] (offset 0).
                    npu_dma_memcpy_nd(metadata=inB, bd_id=c * 3 + 2, mem=B,
                                      sizes=[M_div_m_c, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=memA, bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, c * Aper],
                                      sizes=[M_div_m_c, K_div_k, m, REC],
                                      strides=[m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                    npu_dma_memcpy_nd(metadata=outC, bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * Mc],
                                      sizes=[1, 1, 1, Mc], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(fs[c][2])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt multi-column decode gemv")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--qtype", choices=list(QMAP), required=True)
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True, help="contraction K")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    a, _ = p.parse_known_args()
    gemv_mc(a.dev, a.qtype, a.M, a.K, a.m, a.cols)
