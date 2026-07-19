# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mm.py (16-core fan-out) + mm_q6k.py (mmul core + DMA transforms) +
# whole_array.py (broadcast/distribute/gather). Apache-2.0 WITH LLVM-exception.
#
# ggml-xrt 16-CORE whole_array-style Q6_K on-chip-dequant FUSED matmul, spec-decode M=N VERIFY:
#   C[M, N] = A[M, K] . dequant(W_q6k[N, K])     A bf16, W packed q6_K, C f32.
# All 16 Phoenix compute tiles (4 cols x 4 rows). N is split across the 16 cores (N/16 each);
# every core runs the SAME matmul_vectorized_4x4 mmul (DIM_M=M=16, DIM_N=n=32, DIM_K=256) on its
# N-slice, with the vectorized dequant+scatter core (aie2/mm_q6k_wa.cc).
#
# Data movement (per column c of 4):
#   A  (activation): shim -> memtile -> BROADCAST to the column's 4 cores, with the mmul A
#      sub-tile transform. A[M,K] is tiny; every core needs the full A. Re-streamed per n-tile.
#   W  (weight q6_K, RAW): shim -> memtile -> DISTRIBUTE (object_fifo_link dstOffsets) the
#      column's N/4 channels across its 4 cores (N/16 each). Core dequants+scatters into Bl1.
#   C  (f32): each core writes its [M,n] tile through the mmul C sub-tile transform (-> row-major
#      [M,n] in L2), GATHERED (srcOffsets) up the column, then scattered out to C[M,N].
#
# 4 shim columns => 4x weight-DMA BW (and dodges the >8MB single-stream corruption in mm_q6k.py).
# Bl1 is a design-owned L1 aie.buffer PER CORE (a core-local array ICEs llvm-aie).
#
# UNVALIDATED scaffold — compiled on Linux (no NPU). Verify on-device before dispatch.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212  # q6_K superblock record: ql[128] + qh[64] + scales[16] + f32 d
QK = 256   # q6_K superblock / forced k-tile


def gemv_mm16(dev, M, K, N, m, n, rows, a_oncore=True):
    # r,s,t: aie2 bf16 MMUL dims (this one chip only). k-tile == one q6_K superblock.
    # a_oncore: True  -> A broadcast RAW shim->cores, re-tiled to mmul layout ON-CORE
    #                    (memtile stays 5+5; core needs an Al1 buffer -> only fits M<=16).
    #           False -> A pre-tiled by the memtile DMA transform (memtile 6+6; NO Al1 -> the
    #                    L1-lean path required for M=32, where 2x A buffers would overflow 64KB).
    r, s, t = 4, 8, 4
    k = QK
    cols = 4
    ncores = cols * rows

    assert M == m, "M must equal the token sub-tile m (single m-tile; host pads draft tokens)"
    assert m % r == 0 and k % s == 0 and n % t == 0
    assert m % 16 == 0 and n % 16 == 0, "matmul_vectorized_4x4 needs DIM_M,DIM_N %16"
    assert K % QK == 0, "K must be a multiple of 256 (q6_K superblock)"
    assert N % (n * ncores) == 0, "N must be divisible by n*cols*rows"

    bf16 = str_to_dtype("bf16")
    K_div_k = K // k
    Nc = N // ncores            # output channels per core
    Ndt = Nc // n               # n-tiles per core (mmul C tiles per core)
    Ncol = N // cols            # channels per column
    Bcol = Ncol * (K // QK) * REC  # weight bytes per column

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, k), np.dtype[bf16]]              # activation, RAW row-major
            b_ty = np.ndarray[(n, REC), np.dtype[np.uint8]]        # RAW quant: n superblock recs
            c_ty = np.ndarray[(m, n), np.dtype[np.float32]]        # mmul C tile
            al1_ty = np.ndarray[(m, k), np.dtype[bf16]]            # L1 re-tiled mmul A (per core)
            bl1_ty = np.ndarray[(k, n), np.dtype[bf16]]            # L1 dequant scratch (per core)
            # per-column L2 staging
            bL2_ty = np.ndarray[(rows * n, REC), np.dtype[np.uint8]]
            cL2_ty = np.ndarray[(rows * m * n,), np.dtype[np.float32]]

            # mmul sub-tile DMA transforms (mem-tile, 4 dims OK; see whole_array.py / mm_q6k.py).
            a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
            c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]

            zero = external_func("zero_f32", inputs=[c_ty])
            if a_oncore:
                # matmul_q6k_f32(qB, Aplain, Al1, Bl1, C)  (see aie2/mm_q6k_wa.cc)
                matmul = external_func("matmul_q6k_f32",
                                       inputs=[b_ty, a_ty, al1_ty, bl1_ty, c_ty])
            else:
                # matmul_q6k_f32(qB, A, Bl1, C)  (see aie2/mm_q6k_wa_mt.cc)
                matmul = external_func("matmul_q6k_f32", inputs=[b_ty, a_ty, bl1_ty, c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [[tile(c, 2 + rr) for rr in range(rows)] for c in range(cols)]

            bA = [None] * cols        # a_oncore: shim->cores plain broadcast
            inA = [None] * cols       # not a_oncore: shim->memtile
            memA = [None] * cols      # not a_oncore: memtile->cores broadcast + a_dims transform
            wB_l3l2 = [None] * cols
            wB_l2l1 = [[None] * rows for _ in range(cols)]
            cC_l1l2 = [[None] * rows for _ in range(cols)]
            cC_l2l3 = [None] * cols
            Al1 = [[None] * rows for _ in range(cols)]
            Bl1 = [[None] * rows for _ in range(cols)]

            def build_col(c):
                if a_oncore:
                    # A: RAW row-major [m,k] BROADCAST shim -> the column's 4 cores DIRECTLY (no
                    #    memtile stage -> memtile stays 5+5). Cores re-tile to mmul A on-chip.
                    bA[c] = object_fifo(f"bA_{c}", shims[c],
                                        [cores[c][rr] for rr in range(rows)], 2, a_ty)
                else:
                    # A: shim -> memtile, then memtile -> BROADCAST to the 4 cores WITH the mmul A
                    #    sub-tile transform (delivers A pre-tiled; NO on-core Al1). memtile -> 6+6.
                    #    memA depth 1 (its cons buffers live on the core; depth 2 would overflow
                    #    L1 at M=32). inA depth 2 lives on the memtile (512KB, free).
                    inA[c] = object_fifo(f"inA_{c}", shims[c], mts[c], 2, a_ty)
                    memA[c] = object_fifo(f"memA_{c}", mts[c],
                                          [cores[c][rr] for rr in range(rows)], 1, a_ty, a_dims)
                    object_fifo_link(inA[c], memA[c])

                # W: shim -> memtile (rows*n recs), DISTRIBUTE the 4 rows' N/16 slices (RAW).
                wB_l3l2[c] = object_fifo(f"wB_l3l2_{c}", shims[c], mts[c], 2, bL2_ty)
                for rr in range(rows):
                    wB_l2l1[c][rr] = object_fifo(f"wB_l2l1_{c}_{rr}", mts[c], cores[c][rr], 2, b_ty)
                object_fifo_link(wB_l3l2[c], [wB_l2l1[c][rr] for rr in range(rows)],
                                 [], [n * REC * j for j in range(rows)])

                # C: each core -> memtile (plain [m,n], no transform — a compute-tile DMA only
                #    supports 3 dims). GATHER the 4 rows up the column, then memtile -> shim WITH
                #    the mmul C sub-tile transform (mem-tile DMA, 4 dims OK) -> row-major [m,n].
                for rr in range(rows):
                    cC_l1l2[c][rr] = object_fifo(f"cC_l1l2_{c}_{rr}", cores[c][rr], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty, c_dims)
                object_fifo_link([cC_l1l2[c][rr] for rr in range(rows)], cC_l2l3[c],
                                 [m * n * j for j in range(rows)], [])

                for rr in range(rows):
                    if a_oncore:
                        Al1[c][rr] = buffer(cores[c][rr], al1_ty, f"Al1_{c}_{rr}")
                    Bl1[c][rr] = buffer(cores[c][rr], bl1_ty, f"Bl1_{c}_{rr}")
                    make_core(c, rr)

            def make_core(cc, rr):  # own scope so the no-arg core closure binds cc,rr
                aFifo = bA[cc] if a_oncore else memA[cc]
                wB = wB_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]
                a1 = Al1[cc][rr]
                b1 = Bl1[cc][rr]

                @core(cores[cc][rr], "mm_q6k.o", stack_size=0x1800)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        for _ in range_(Ndt) if Ndt > 1 else range(1):
                            eo = cC.acquire(ObjectFifoPort.Produce, 1)
                            zero(eo)
                            for _ in range_(K_div_k):
                                ea = aFifo.acquire(ObjectFifoPort.Consume, 1)
                                eb = wB.acquire(ObjectFifoPort.Consume, 1)
                                if a_oncore:
                                    matmul(eb, ea, a1, b1, eo)
                                else:
                                    matmul(eb, ea, b1, eo)
                                aFifo.release(ObjectFifoPort.Consume, 1)
                                wB.release(ObjectFifoPort.Consume, 1)
                            cC.release(ObjectFifoPort.Produce, 1)

            for c in range(cols):
                build_col(c)

            @runtime_sequence(
                np.ndarray[(M * K,), np.dtype[bf16]],                    # A[M,K]
                np.ndarray[(N * (K // QK) * REC,), np.dtype[np.uint8]],  # W[N][K/256][REC]
                np.ndarray[(M * N,), np.dtype[np.float32]],              # C[M,N]
            )
            def sequence(A, W, C):
                for c in range(cols):
                    # A broadcast: re-stream the full A[M,K] per n-tile, per k-tile [m,k]. dim0
                    # (n-tiles) stride 0 = broadcast. a_oncore -> RAW to bA (cores re-tile on-chip);
                    # else -> RAW to inA (memtile transform pre-tiles it to mmul A layout).
                    npu_dma_memcpy_nd(metadata=(bA[c] if a_oncore else inA[c]),
                                      bd_id=c * 3 + 1, mem=A,
                                      sizes=[Ndt, K_div_k, m, k],
                                      strides=[0, k, K, 1])
                    # W distribute: column c's N/4 channels, channel-tile-major so the row-split
                    # (contiguous rows*n records) hands each row its N/16 slice.
                    #   record(channel, ktile) at (channel*(K/256)+ktile)*REC.
                    npu_dma_memcpy_nd(metadata=wB_l3l2[c], bd_id=c * 3 + 2, mem=W,
                                      offsets=[0, 0, 0, c * Bcol],
                                      sizes=[Ndt, K_div_k, rows * n, REC],
                                      strides=[rows * n * (K // QK) * REC, REC,
                                               (K // QK) * REC, 1])
                    # C gather+scatter: fifo delivers [n-tile i][row r][tok][nn] (each core tile
                    # row-major after the C transform); scatter to C[tok, c*(N/4)+i*rows*n+r*n+nn].
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * Ncol],
                                      sizes=[Ndt, rows, m, n],
                                      strides=[rows * n, n, N, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core whole_array Q6_K verify matmul")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="tokens (== m, single m-tile)")
    p.add_argument("-K", type=int, required=True, help="contraction K (mult of 256)")
    p.add_argument("-N", type=int, required=True, help="output channels (mult of n*16)")
    p.add_argument("-m", type=int, default=16, help="token sub-tile (== M)")
    p.add_argument("-n", type=int, default=32, help="output-channel sub-tile per core-call")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--a-memtile", action="store_true",
                   help="deliver A pre-tiled via the memtile DMA transform (no on-core Al1); "
                        "required for M=32 (on-core Al1 overflows L1). Default: on-core A re-tile.")
    a, _ = p.parse_known_args()
    gemv_mm16(a.dev, a.M, a.K, a.N, a.m, a.n, a.rows, a_oncore=not a.a_memtile)
