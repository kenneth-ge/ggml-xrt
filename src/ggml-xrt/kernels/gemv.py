# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie programming_examples/basic/matrix_multiplication/matrix_vector
# (Apache-2.0 WITH LLVM-exception, (c) 2025 AMD Inc.).
#
# Parameterized matrix-vector (gemv) design for the ggml-xrt decode path.
#
# Computes C[M] = A[M,K] · B[K] on aie2/Phoenix. Mapping to a ggml MUL_MAT with a
# single token (decode): the design's **M is the output dim N**, A is the WEIGHT
# stored [N,K] row-major (== ggml's native weight layout, so NO transpose needed —
# unlike the tiled matmul), B is the K-length activation vector, C is the N output.
#
# Emits MLIR to stdout (like the stock example). Sizes are CLI args so one design
# serves any (K,N). Uses the scalar matvec (the vectorized path is marked erroneous
# upstream); switch to vectorized once validated on hardware for more throughput.
#
# NOTE: compiled on Linux/WSL, NOT executed here (no NPU). Correctness is validated
# on the Windows side via the existing mulmat/gemv check harness.
import argparse

import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype


def my_gemv(dev, M, K, m, k, dtype_in_str, dtype_out_str):
    """Emit MLIR for C[M] = A[M,K]·B[K]. M is the output (ggml N); A is the weight."""
    n_cores = 1

    dtype_in = np.dtype[str_to_dtype(dtype_in_str)]
    dtype_out = np.dtype[str_to_dtype(dtype_out_str)]

    A_sz = M * K
    B_sz = K
    C_sz = M
    C_sz_div_n_cores = C_sz // n_cores

    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k

    m_x_K = m * K

    vectorized = False  # scalar matvec; upstream vectorized path is erroneous

    assert M % (m * n_cores) == 0, "M must be divisible by (m * n_cores)"
    assert K % k == 0, "K must be divisible by k"

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            inA_ty = np.ndarray[(m * k,), dtype_in]
            inB_ty = np.ndarray[(k,), dtype_in]
            outC_ty = np.ndarray[(m,), dtype_out]
            A_ty = np.ndarray[(m, k), dtype_in]

            func_type = "vectorized" if vectorized else "scalar"
            zero = external_func(f"zero_{func_type}_{dtype_out_str}", inputs=[outC_ty])
            matvec = external_func(
                f"matvec_{func_type}_{dtype_in_str}_{dtype_out_str}",
                inputs=[A_ty, inB_ty, outC_ty],
            )

            ShimTile0 = tile(0, 0)
            ShimTile1 = tile(1, 0)
            ShimTiles = [ShimTile0, ShimTile1]
            MemTile0 = tile(0, 1)
            MemTiles = [MemTile0]
            ComputeTile0 = tile(0, 2)
            cores = [ComputeTile0]

            memA_fifos = []
            inA_fifos = []
            outC_fifos = []

            for i in range(n_cores):
                memA_fifos.append(object_fifo(f"memA{i}", ShimTiles[i], MemTiles[i], 2, inA_ty))
                inA_fifos.append(object_fifo(f"inA{i}", MemTiles[i], cores[i], 2, A_ty))
                object_fifo_link(memA_fifos[i], inA_fifos[i])
                outC_fifos.append(object_fifo(f"outC{i}", cores[i], ShimTiles[i], 2, outC_ty))

            inB_fifo = object_fifo("inB", ShimTiles[1 % n_cores], cores[0:n_cores], 2, inB_ty)

            for i in range(n_cores):

                @core(cores[i], f"mv_{m}x{k}.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC_fifos[i].acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                            matvec(elem_in_a, elem_in_b, elem_out)
                            inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                            inB_fifo.release(ObjectFifoPort.Consume, 1)
                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(A_sz,), dtype_in],
                np.ndarray[(B_sz,), dtype_in],
                np.ndarray[(C_sz,), dtype_out],
            )
            def sequence(A, B, C):
                npu_dma_memcpy_nd(
                    metadata=inB_fifo, bd_id=2, mem=B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K], strides=[0, 0, 0, 1],
                )
                for i in range(n_cores):
                    A_offset = i * M_div_m_div_n_cores * m * K
                    C_offset = i * M_div_m_div_n_cores * m
                    npu_dma_memcpy_nd(
                        metadata=memA_fifos[i], bd_id=1, mem=A, offsets=[0, 0, 0, A_offset],
                        sizes=[M_div_m_div_n_cores, K_div_k, m, k], strides=[m_x_K, k, K, 1],
                    )
                    npu_dma_memcpy_nd(
                        metadata=outC_fifos[i], bd_id=0, mem=C, offsets=[0, 0, 0, C_offset],
                        sizes=[1, 1, 1, C_sz_div_n_cores], strides=[0, 0, 0, 1],
                    )
                dma_wait(*outC_fifos)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt parameterized gemv (matrix-vector) design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True, help="contraction dim K")
    p.add_argument("-m", type=int, default=32, help="M tile")
    p.add_argument("-k", type=int, default=32, help="K tile")
    p.add_argument("--dtype_in", default="bf16")
    p.add_argument("--dtype_out", default="f32")
    a, _ = p.parse_known_args()
    my_gemv(a.dev, a.M, a.K, a.m, a.k, a.dtype_in, a.dtype_out)
