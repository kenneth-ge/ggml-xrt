# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_q6k.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt DECODE bottleneck DECOMPOSITION for the q6k 6144x2048 gemv (down_proj, 10.2 MB
# repacked weight, the current worst kernel: 5641 ms). Three variants of the SAME shape,
# identical DMA/objectfifo structure, differing ONLY in what the core does — so the timing
# gap between them attributes the cost to streaming vs compute vs per-dispatch overhead.
#
#   --mode full    : the current kernel (matvec each k-tile). Reference == c).
#   --mode stream  : core DRAINS every weight + activation tile (acquire/release), NO dequant/
#                    MAC. Same full 10.2 MB DDR weight traffic, zero compute.  == (a)
#   --mode compute : DMA delivers only ONE weight tile (near-zero DDR weight traffic); the core
#                    runs the scalar dequant+MAC the SAME number of times (M_div_m*K_div_k) over
#                    that resident tile. Isolates compute cost.  == (b)
#
# Interpretation:  b~=c  -> compute-bound (vectorize) ;  a~=c<<b -> streaming-bound ;
#                  c >> a+b -> per-dispatch / per-descriptor overhead dominates.
# Report each as wall time; the host also computes GB/s = 10.2 MB / time for full & stream.
#
# Weight repack unchanged (q6_K [N][K/256][212]); links mv_q6k.o (matvec_q6k_f32 + zero).
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212
QK = 256


def my_gemv_q6k_bench(dev, M, K, m, mode):
    n_cores = 1
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    M_div_m = M // (m * n_cores)
    ntiles = M_div_m * K_div_k  # total matvec/k-tile invocations in the real kernel
    A_sz = M * (K // QK) * REC
    B_sz = K
    C_sz = M

    assert M % (m * n_cores) == 0
    assert K % QK == 0

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func("matvec_q6k_f32", inputs=[a_ty, b_ty, c_ty])

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            Core = tile(0, 2)

            memA = object_fifo("memA", ShimTile, MemTile, 2, a_ty)
            inA = object_fifo("inA", MemTile, Core, 2, a_ty)
            object_fifo_link(memA, inA)
            inB = object_fifo("inB", ShimTile, Core, 2, b_ty)
            outC = object_fifo("outC", Core, ShimTile, 2, c_ty)

            @core(Core, "mv_q6k.o")
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                    zero(elem_out)
                    if mode == "compute":
                        # One resident weight tile + activation; loop matvec ntiles times
                        # (== full's total) with NO further DDR weight traffic.
                        a = inA.acquire(ObjectFifoPort.Consume, 1)
                        b = inB.acquire(ObjectFifoPort.Consume, 1)
                        for _ in range_(ntiles):
                            matvec(a, b, elem_out)
                        inA.release(ObjectFifoPort.Consume, 1)
                        inB.release(ObjectFifoPort.Consume, 1)
                    else:
                        # full + stream: consume every weight k-tile from DDR.
                        for _ in range_(K_div_k):
                            a = inA.acquire(ObjectFifoPort.Consume, 1)
                            b = inB.acquire(ObjectFifoPort.Consume, 1)
                            if mode == "full":
                                matvec(a, b, elem_out)
                            inA.release(ObjectFifoPort.Consume, 1)
                            inB.release(ObjectFifoPort.Consume, 1)
                    outC.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[np.uint8]],
                np.ndarray[(B_sz,), np.dtype[bf16]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                if mode == "compute":
                    # Deliver ONE weight tile, one activation k-tile, one output tile only.
                    npu_dma_memcpy_nd(metadata=inB, bd_id=2, mem=B,
                                      sizes=[1, 1, 1, k], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=memA, bd_id=1, mem=A,
                                      sizes=[1, 1, m, REC], strides=[0, 0, REC, 1])
                    npu_dma_memcpy_nd(metadata=outC, bd_id=0, mem=C,
                                      sizes=[1, 1, 1, m], strides=[0, 0, 0, 1])
                    dma_wait(outC)
                    return
                # full + stream: the real kernel's DMA (full 10.2 MB weight stream).
                npu_dma_memcpy_nd(metadata=inB, bd_id=2, mem=B,
                                  sizes=[M_div_m, 1, 1, K], strides=[0, 0, 0, 1])
                npu_dma_memcpy_nd(metadata=memA, bd_id=1, mem=A,
                                  sizes=[M_div_m, K_div_k, m, REC],
                                  strides=[m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                npu_dma_memcpy_nd(metadata=outC, bd_id=0, mem=C,
                                  sizes=[1, 1, 1, C_sz], strides=[0, 0, 0, 1])
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt q6k gemv bottleneck decomposition")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True, help="contraction K (mult of 256)")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--mode", choices=["full", "stream", "compute"], required=True)
    a, _ = p.parse_known_args()
    my_gemv_q6k_bench(a.dev, a.M, a.K, a.m, a.mode)
