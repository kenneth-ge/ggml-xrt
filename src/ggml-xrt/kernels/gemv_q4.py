# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie matrix_multiplication/matrix_vector (Apache-2.0 WITH LLVM-exception).
#
# Q4_0 on-chip-dequant decode gemv: C[M] = dequant(A_q4[M,K]) . B[K], M = output N.
# Weight repacked host-side into ONE buffer of 20-byte block records (16 nibble bytes
# + f32 scale per 32-elem block), row-major [M][K/32][20]; see mv_q4.cc. Two input
# DMAs (weight, activation) + one output (C, f32) — fits the shim's DMA channels.
#
# UNVALIDATED scaffold — compiled on Linux, verify on-device before enabling dispatch.
import argparse

import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 20  # bytes per q4_0 block record: 16 nibbles + f32 scale


def my_gemv_q4(dev, M, K, m, k):
    n_cores = 1
    bf16 = str_to_dtype("bf16")
    assert k == 32, "this design streams one 32-elem q4_0 block per k-tile"
    K_div_k = K // k
    M_div_m = M // (m * n_cores)
    A_sz = M * (K // 32) * REC
    B_sz = K
    C_sz = M

    assert M % (m * n_cores) == 0, "M must be divisible by m*n_cores"
    assert K % 32 == 0, "K must be a multiple of 32"

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]   # one block-record per row
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func("matvec_q4_0_f32", inputs=[a_ty, b_ty, c_ty])

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            Core = tile(0, 2)

            memA = object_fifo("memA", ShimTile, MemTile, 2, a_ty)
            inA = object_fifo("inA", MemTile, Core, 2, a_ty)
            object_fifo_link(memA, inA)
            inB = object_fifo("inB", ShimTile, Core, 2, b_ty)
            outC = object_fifo("outC", Core, ShimTile, 2, c_ty)

            @core(Core, "mv_q4_32x32.o")
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

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[np.uint8]],
                np.ndarray[(B_sz,), np.dtype[bf16]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inB, bd_id=2, mem=B,
                    sizes=[M_div_m, 1, 1, K], strides=[0, 0, 0, 1],
                )
                # A: per m-tile, K_div_k block-records of REC bytes for each of m rows
                npu_dma_memcpy_nd(
                    metadata=memA, bd_id=1, mem=A,
                    sizes=[M_div_m, K_div_k, m, REC],
                    strides=[m * (K // 32) * REC, REC, (K // 32) * REC, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=outC, bd_id=0, mem=C,
                    sizes=[1, 1, 1, C_sz], strides=[0, 0, 0, 1],
                )
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt Q4_0 dequant gemv design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True, help="contraction dim K")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    a, _ = p.parse_known_args()
    my_gemv_q4(a.dev, a.M, a.K, a.m, a.k)
