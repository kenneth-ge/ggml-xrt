# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py (Apache-2.0 WITH LLVM-exception).
#
# STEP-3a PROOF: 16-core DENSE bf16 M=1 gemv with NO horizontal reduce (outputs-in-lanes).
# Same 16-core (4 cols x 4 rows) whole_array partition, same activation broadcast, same C
# gather, and same per-k-tile (k=256) matvec-call cadence as the shipping q6k gemv
# (gemv_mc16.py) so the reduce-elimination win is measured in ISOLATION and comparably.
#
# DIFFERENCE vs gemv_mc16.py: the weight is dense bf16 laid out K-MAJOR. For one output-tile
# (m=32 outputs) and one k-tile (k=256 columns), the weight is [k][m] bf16 -- element (col,
# output n) at k*m + n -- so a given column's m outputs are contiguous (one load_v<m> per
# rank-1 update in mv_bf16_noreduce.cc, no DMA transpose). Host repack (dense, no quant):
#   A[c][i][t][r][k][m]  (c=col 0..3, i=output-tile-per-core 0..Mdm-1, t=k-tile 0..K_div_k-1,
#                         r=row 0..rows-1, k=256 cols, m=32 outputs) -- fully contiguous in
#   kernel-consumption order, so the shim->memtile DMA is a plain contiguous split and the
#   memtile->row link is contiguous k*m slices.
#
# bf16 weight is 2 B/wt (vs q6k ~0.83 B/wt) => ~2.4x more weight DMA; this build isolates the
# compute (reduce) effect, DMA-size tradeoff assessed separately in the report.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype


def gemv_mc16_noreduce(dev, M, K, m, rows):
    OBJ = "mv_bf16.o"
    MVSYM = "matvec_bf16_f32"
    bf16 = str_to_dtype("bf16")
    k = 256                      # columns per k-tile (matches q6k QK)
    K_div_k = K // k
    cols = 4
    ncores = cols * rows
    assert M % (m * ncores) == 0, "N must be divisible by m*cols*rows"
    assert K % k == 0, "K must be divisible by 256"
    Mc = M // ncores             # output rows per core
    Mdm = Mc // m                # output m-tiles per core
    km = k * m                   # a_ty (one matvec call) bf16 elements
    Acol = (M // cols) * K       # weight bf16 elements per column

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(k, m), np.dtype[bf16]]      # K-major weight, one k-tile
            b_ty = np.ndarray[(k,), np.dtype[bf16]]        # activation slice
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]  # output tile
            # per-column L2 staging: one (output-tile,k-tile) block for the column's `rows` rows
            aL2_ty = np.ndarray[(rows * k, m), np.dtype[bf16]]
            cL2_ty = np.ndarray[(rows * m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func(MVSYM, inputs=[a_ty, b_ty, c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [[tile(c, 2 + r) for r in range(rows)] for c in range(cols)]

            wA_l3l2 = [None] * cols
            wA_l2l1 = [[None] * rows for _ in range(cols)]
            bB = [None] * cols
            cC_l1l2 = [[None] * rows for _ in range(cols)]
            cC_l2l3 = [None] * cols

            def build_col(c):
                # weight: shim->memtile (one [rows,k,m] block), distribute k*m to each of the 4 rows
                wA_l3l2[c] = object_fifo(f"wA_l3l2_{c}", shims[c], mts[c], 2, aL2_ty)
                for r in range(rows):
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], 2, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [km * j for j in range(rows)])
                # activation: shim->memtile->broadcast to 4 rows
                bB[c] = object_fifo(f"bB_{c}", shims[c], [cores[c][r] for r in range(rows)], 2, b_ty)
                # C: 4 rows -> memtile (gather) -> shim
                for r in range(rows):
                    cC_l1l2[c][r] = object_fifo(f"cC_l1l2_{c}_{r}", cores[c][r], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty)
                object_fifo_link([cC_l1l2[c][r] for r in range(rows)], cC_l2l3[c],
                                 [m * j for j in range(rows)], [])

                for r in range(rows):
                    make_core(c, r)

            def make_core(cc, rr):  # own scope so the no-arg core closure binds cc,rr
                wA = wA_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]
                bBc = bB[cc]

                @core(cores[cc][rr], OBJ, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        eo = cC.acquire(ObjectFifoPort.Produce, 1)
                        zero(eo)
                        for _ in range_(K_div_k):
                            av = wA.acquire(ObjectFifoPort.Consume, 1)
                            bv = bBc.acquire(ObjectFifoPort.Consume, 1)
                            matvec(av, bv, eo)
                            wA.release(ObjectFifoPort.Consume, 1)
                            bBc.release(ObjectFifoPort.Consume, 1)
                        cC.release(ObjectFifoPort.Produce, 1)

            for c in range(cols):
                build_col(c)

            @runtime_sequence(
                np.ndarray[(M * K,), np.dtype[bf16]],   # dense bf16 weight, K-major repack
                np.ndarray[(K,), np.dtype[bf16]],       # activation
                np.ndarray[(M,), np.dtype[np.float32]], # output
            )
            def sequence(A, B, C):
                for c in range(cols):
                    # weight for column c: A[c][i][t][r][k][m], contiguous in consumption order.
                    # element = one (output-tile i, k-tile t) block = [rows*k, m] bf16.
                    # Fully contiguous read of column c's slice. NB: EVERY DMA dim must
                    # be <=1023 (the innermost only escapes that when outer dims are size-1,
                    # as for the activation broadcast). The per-(output-tile,k-tile) element
                    # is rows*k*m elements contiguous; split it into [mid, inner] so both
                    # stay <=1023. dims 0,1 iterate the Mdm*K_div_k elements.
                    elem = rows * km                     # aL2_ty elements per DMA element
                    inner = 512
                    mid = elem // inner
                    assert elem % inner == 0 and mid <= 1023, "retune DMA split"
                    npu_dma_memcpy_nd(metadata=wA_l3l2[c], bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, c * Acol],
                                      sizes=[Mdm, K_div_k, mid, inner],
                                      strides=[K_div_k * elem, elem, inner, 1])
                    npu_dma_memcpy_nd(metadata=bB[c], bd_id=c * 3 + 2, mem=B,
                                      sizes=[Mdm, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * (M // cols)],
                                      sizes=[1, 1, 1, M // cols], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core dense bf16 NO-REDUCE gemv")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    a, _ = p.parse_known_args()
    gemv_mc16_noreduce(a.dev, a.M, a.K, a.m, a.rows)
