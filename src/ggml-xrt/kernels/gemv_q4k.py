# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie matrix_multiplication/matrix_vector (Apache-2.0 WITH LLVM-exception).
#
# Q4_K on-chip-dequant decode gemv: C[M] = dequant(A_q4K[M,K]) . B[K], M = output N.
# Weight repacked host-side into ONE buffer of 148-byte superblock records
# (qs[128] + scales[12] + f32 d + f32 dmin), row-major [M][K/256][148]; see mv_q4k.cc.
# k-tile == one 256-elem q4_K superblock. Two input DMAs (weight, activation) + C (f32).
#
# UNVALIDATED scaffold — compiled on Linux, verify on-device before enabling dispatch.
import argparse

import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 148  # bytes per q4_K superblock record: qs[128] + scales[12] + f32 d + f32 dmin
QK = 256   # q4_K superblock elements


def my_gemv_q4k(dev, M, K, m):
    n_cores = 1
    bf16 = str_to_dtype("bf16")
    k = QK  # one superblock per k-tile
    K_div_k = K // k
    M_div_m = M // (m * n_cores)
    A_sz = M * (K // QK) * REC
    B_sz = K
    C_sz = M

    assert M % (m * n_cores) == 0, "M must be divisible by m*n_cores"
    assert K % QK == 0, "K must be a multiple of 256 (q4_K superblock)"

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]   # one superblock record per row
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func("matvec_q4k_f32", inputs=[a_ty, b_ty, c_ty])

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            Core = tile(0, 2)

            memA = object_fifo("memA", ShimTile, MemTile, 2, a_ty)
            inA = object_fifo("inA", MemTile, Core, 2, a_ty)
            object_fifo_link(memA, inA)
            inB = object_fifo("inB", ShimTile, Core, 2, b_ty)
            outC = object_fifo("outC", Core, ShimTile, 2, c_ty)

            @core(Core, "mv_q4k.o", stack_size=0x2000)
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
                npu_dma_memcpy_nd(
                    metadata=memA, bd_id=1, mem=A,
                    sizes=[M_div_m, K_div_k, m, REC],
                    strides=[m * (K // QK) * REC, REC, (K // QK) * REC, 1],
                )
                npu_dma_memcpy_nd(
                    metadata=outC, bd_id=0, mem=C,
                    sizes=[1, 1, 1, C_sz], strides=[0, 0, 0, 1],
                )
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt Q4_K dequant gemv design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True, help="contraction dim K (mult of 256)")
    p.add_argument("-m", type=int, default=32)
    a, _ = p.parse_known_args()
    my_gemv_q4k(a.dev, a.M, a.K, a.m)
