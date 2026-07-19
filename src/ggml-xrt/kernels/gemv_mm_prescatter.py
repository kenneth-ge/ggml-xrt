# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on mlir-aie matrix_multiplication/single_core (Apache-2.0 WITH LLVM-exception).
#
# PRE-SCATTERED bf16 FUSED-nothing tiled matmul (spec-decode M=N VERIFY) for aie2/Phoenix.
#   C[M,N] += A_act[M,K] . B_prescattered[K,N]      C f32, A bf16, B bf16.
# Single compute tile. The "host pre-scatters the weight" experiment: the weight arrives
# ALREADY dequantized to bf16 AND already in the mmul B systolic-tile ("Bl1") order, so the
# core does NO on-core dequant and NO scatter — just the raw matmul_vectorized_4x4 MAC.
# This measures the pure-mmul ceiling with dequant+scatter entirely removed (vs mm_q6k.py,
# which dequants each 212-byte q6_K superblock and scatters 8192 bf16 into an L1 Bl1 scratch).
#
# There is NO Bl1 L1 scratch and NO memB transform here: B is streamed straight to the core
# and used DIRECTLY as the mmul B operand.
#
# k-tile == 256 (parity with the q6_K superblock k-tile). A bf16 [M,K]; C f32 [M,N].
#
# ===================== HOST PRE-SCATTER SPEC (produce buffer B) =====================
# Let n = N-sub-tile (default 32), k = 256, r,s,t = 4,8,4. The DRAM buffer B is bf16 and is
# laid out as  [N/n][K/k][k*n]  row-major:  one contiguous k*n (=DIM_K*DIM_N) block per
# (N-tile, k-tile), each block in the mmul B systolic-tile ("Bl1") order. Total size N*K bf16.
#
# To fill B from a q6_K weight W of shape [N rows][K cols] (e.g. Qwen3 down 6144x2048):
#   for ktile in range(K // 256):          # k-tile index, one q6_K superblock along K
#     for nc in range(N):                  # global output column (weight row)
#       col = dequantize_row_q6_K(W[nc], superblock=ktile)   # 256 f32 -> cast bf16
#       ntile = nc // n ;  lc = nc % n     # which N-tile, and local column within it
#       nt = lc // t ;     ti = lc % t
#       block_base = (ntile * (K // 256) + ktile) * (256 * n)   # bf16 elements
#       for kk in range(256):
#         ks = kk // s ;  si = kk % s
#         within = ((ks * (n // t) + nt) * s + si) * t + ti
#         B[block_base + within] = bf16(col[kk])
# The inner `within` expression is EXACTLY the Bl1 scatter target from aie2/mm_q6k_vecdq.cc,
# so on-device C matches the fused-q6k kernel bit-for-bit (modulo host-vs-onchip dequant
# rounding). NOTE: this DOUBLES weight memory (bf16 N*K vs packed q6_K) — it is a ceiling
# experiment / an MoE active-expert path where the active expert is pre-expanded per step.
# ===================================================================================
#
# UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
import argparse
import sys

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

QK = 256   # forced k-tile (parity with q6_K superblock)


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul_prescatter(dev, M, K, N, m, n, serialize=False):
    # r,s,t: aie2 bf16 MMUL dims (this one chip only).
    r, s, t = 4, 8, 4
    k = QK  # k-tile (256)

    assert M % m == 0, "M must be divisible by m"
    assert K % k == 0, "K must be a multiple of 256"
    assert N % n == 0, "N must be divisible by n"
    assert m % r == 0 and k % s == 0 and n % t == 0

    bf16 = str_to_dtype("bf16")

    A_sz = M * K
    B_sz = N * K  # pre-scattered bf16 dense weight (doubles memory vs q6_K)
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
            b_ty = np.ndarray[(k, n), np.dtype[bf16]]   # pre-scattered mmul B tile (Bl1 order)
            c_ty = np.ndarray[(m, n), np.dtype[np.float32]]

            zero = external_func("zero_f32", inputs=[c_ty])
            # matmul_prescatter_f32(A, B, C)  (see aie2/mm_prescatter_bf16.cc)
            matmul = external_func(
                "matmul_prescatter_f32", inputs=[a_ty, b_ty, c_ty]
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

            # Input B: pre-scattered bf16 weight, NO transform (used DIRECTLY as mmul B).
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

            @core(compute_tile, "mm_prescatter.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):
                        elem_out = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k) if K_div_k > 1 else range(1):
                            elem_a = memA.acquire(ObjectFifoPort.Consume, 1)
                            elem_b = memB.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_a, elem_b, elem_out)
                            memA.release(ObjectFifoPort.Consume, 1)
                            memB.release(ObjectFifoPort.Consume, 1)
                        memC.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[bf16]],
                np.ndarray[(B_sz,), np.dtype[bf16]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                # Pre-scattered B DRAM layout: [N/n][K/k][k*n] bf16, one Bl1-order tile per
                # (N-tile, k-tile), contiguous. Block(nt,kc) at (nt*K_div_k+kc)*(k*n) bf16.
                if serialize:
                    for mt in range(M_div_m):
                        npu_dma_memcpy_nd(
                            metadata=outC, bd_id=0, mem=C,
                            offsets=[0, 0, 0, mt * m * N],
                            sizes=[1, N_div_n, m, n], strides=[m * N, n, N, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=inA, bd_id=1, mem=A,
                            offsets=[0, 0, 0, mt * m * K],
                            sizes=[N_div_n, K_div_k, m, k], strides=[0, k, K, 1],
                        )
                        npu_dma_memcpy_nd(
                            metadata=inB, bd_id=2, mem=B,
                            sizes=[N_div_n, K_div_k, k, n],
                            strides=[K_div_k * k * n, k * n, n, 1],
                        )
                        dma_wait(outC)
                    return
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
                            # Pre-scattered B gather: per (N-tile, k-tile) deliver one
                            # k*n (=DIM_K*DIM_N) Bl1-order tile, contiguous in DRAM.
                            npu_dma_memcpy_nd(
                                metadata=inB, bd_id=bd_id_base + 2 * tile_row + 2, mem=B,
                                sizes=[N_div_n, K_div_k, k, n],
                                strides=[K_div_k * k * n, k * n, n, 1],
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outC)
                dma_wait(outC)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt pre-scattered bf16 matmul design")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="tokens (mult of m)")
    p.add_argument("-K", type=int, required=True, help="contraction K (mult of 256)")
    p.add_argument("-N", type=int, required=True, help="output channels (mult of n)")
    p.add_argument("-m", type=int, default=16, help="token sub-tile")
    p.add_argument("-n", type=int, default=32, help="output-channel sub-tile")
    p.add_argument("--serialize-mtiles", action="store_true",
                   help="dma_wait between m-tiles (fixes >8 MB back-to-back B streams)")
    a, _ = p.parse_known_args()
    my_matmul_prescatter(a.dev, a.M, a.K, a.N, a.m, a.n, a.serialize_mtiles)
