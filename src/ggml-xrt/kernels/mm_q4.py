# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie matrix_multiplication/single_core (Apache-2.0 WITH LLVM-exception).
#
# Q4_0 on-chip-dequant FUSED tiled matmul (prefill AND padded-decode) for aie2/Phoenix.
#   C[M,N] += A_act[M,K] . dequant(W_q4K)      C f32, A bf16, W packed q4_0.
# Single compute tile. B (weight) is delivered RAW (no memB transform); the core dequants
# each 20-byte q4_0 block (8 per 256 k-tile) and scatters bf16 into a design-owned L1 `aie.buffer`
# (Bl1) in the mmul B sub-tile layout, then runs the validated matmul_vectorized_4x4 MAC.
# (A core-local Bl1 array ICEs llvm-aie — the scratch MUST be a design buffer; that is the
# whole point of this design vs. mm_q4.cc alone.)
#
# k-tile == 256 == eight 32-elem q4_0 blocks per weight column.
# Weight repack (host, see build-q4-mm.sh / mv_q4.cc): ONE buffer row-major
#   [N][K/32][20] : 16 nibble bytes (block_q4_0.qs) + f32 d  (per 32-elem block).
# A bf16 [M,K]; C f32 [M,N]; weight NOT transposed. ABI: kernel(op=3, instr, ninstr, A,B,C).
#
# UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device (q4_gemv_check, mm
# mode) before enabling the fused-quant prefill dispatch.
import argparse
import sys

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 160  # per k-tile (256): 8 q4_0 blocks x 20 B (16 nibbles + f32 d)
QK = 256   # forced k-tile = 8 q4_0 blocks (block=32)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul_q4(dev, M, K, N, m, n):
    # r,s,t: aie2 bf16 MMUL dims (this one chip only).
    r, s, t = 4, 8, 4
    k = QK  # k-tile is one superblock; scales must not split across tiles

    assert M % m == 0, "M must be divisible by m"
    assert K % k == 0, "K must be a multiple of 256 (q4_0 superblock)"
    assert N % n == 0, "N must be divisible by n"
    assert m % r == 0 and k % s == 0 and n % t == 0

    bf16 = str_to_dtype("bf16")

    A_sz = M * K
    B_sz = N * (K // QK) * REC  # packed weight, bytes
    C_sz = M * N

    M_div_m = M // m
    K_div_k = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1_1col if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, k), np.dtype[bf16]]
            b_ty = np.ndarray[(n, REC), np.dtype[np.uint8]]  # RAW quant: n superblock recs
            c_ty = np.ndarray[(m, n), np.dtype[np.float32]]
            bl1_ty = np.ndarray[(k, n), np.dtype[bf16]]      # L1 dequant scratch

            zero = external_func("zero_f32", inputs=[c_ty])
            # matmul_q4_0_f32(qB, A, Bl1, C)  (see aie2/mm_q4.cc)
            matmul = external_func(
                "matmul_q4_0_f32", inputs=[b_ty, a_ty, bl1_ty, c_ty]
            )

            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile = tile(0, 2)

            # Input A: activation, standard mmul A sub-tile transform on mem->core.
            inA = object_fifo("inA", shim_tile, mem_tile, 2, a_ty)
            memA = object_fifo(
                "memA", mem_tile, compute_tile, 2, a_ty,
                [(m // r, r * k), (k // s, s), (r, k), (s, 1)],
            )
            object_fifo_link(inA, memA)

            # Input B: RAW packed q4_0, NO transform (core dequants+scatters into Bl1).
            inB = object_fifo("inB", shim_tile, mem_tile, 2, b_ty)
            memB = object_fifo("memB", mem_tile, compute_tile, 2, b_ty)
            object_fifo_link(inB, memB)

            # Output C: standard mmul C sub-tile transform on core->mem.
            memC = object_fifo("memC", compute_tile, mem_tile, 2, c_ty)
            outC = object_fifo(
                "outC", mem_tile, shim_tile, 2, c_ty,
                [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
            )
            object_fifo_link(memC, outC)

            # Design-owned L1 dequant scratch (the ICE fix): core writes it, then MACs.
            Bl1 = buffer(compute_tile, bl1_ty, "Bl1_q4")

            @core(compute_tile, "mm_q4.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                            elem_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_b, elem_a, Bl1, elem_out)
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)
                        memC.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[bf16]],
                np.ndarray[(B_sz,), np.dtype[np.uint8]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                rows_per_block = 4
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    for pingpong in [0, 1]:
                        C_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            break
                        npu_dma_memcpy_nd(
                            metadata=outC, bd_id=bd_id_base, mem=C,
                            offsets=[0, 0, 0, C_row_offset],
                            sizes=[num_tile_rows, N_div_n, m, n],
                            strides=[m * N, n, N, 1],
                        )
                        for tile_row in range(num_tile_rows):
                            A_row_offset = (row_base + tile_row) * m * K
                            npu_dma_memcpy_nd(
                                metadata=inA, bd_id=bd_id_base + 2 * tile_row + 1, mem=A,
                                offsets=[0, 0, 0, A_row_offset],
                                sizes=[N_div_n, K_div_k, m, k],
                                strides=[0, k, K, 1],
                            )
                            # RAW q4_0 gather: per (N-tile, k-superblock) deliver n records.
                            # DDR record(nc,kc) at (nc*(K/256)+kc)*REC bytes.
                            npu_dma_memcpy_nd(
                                metadata=inB, bd_id=bd_id_base + 2 * tile_row + 2, mem=B,
                                sizes=[N_div_n, K_div_k, n, REC],
                                strides=[n * (K // QK) * REC, REC, (K // QK) * REC, 1],
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outC)
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt Q4_0 fused dequant matmul design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="tokens (mult of m)")
    p.add_argument("-K", type=int, required=True, help="contraction K (mult of 256)")
    p.add_argument("-N", type=int, required=True, help="output channels (mult of n)")
    p.add_argument("-m", type=int, default=32, help="token sub-tile")
    p.add_argument("-n", type=int, default=32, help="output-channel sub-tile")
    a, _ = p.parse_known_args()
    my_matmul_q4(a.dev, a.M, a.K, a.N, a.m, a.n)
