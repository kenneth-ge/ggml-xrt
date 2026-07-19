# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc.py + whole_array distribute (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt 16-CORE decode gemv: use ALL of Phoenix's compute tiles (4 cols x 4 rows), not just
# one row. Output N is split across 16 cores (N/16 each); decode is ~90x below the DMA ceiling
# so 16 cores get ~4x the compute of the current 4-core gemv_mc. Per column: one shim->memtile
# weight stream (N/4) DISTRIBUTED to the column's 4 core rows (N/16 each) via object_fifo_link
# offsets; activation b[K] BROADCAST to the 4 rows; C GATHERED from the 4 rows -> memtile -> shim.
#
# Weight stream is OUTPUT-tile-major (each output tile's full K contiguous) so the row-distribute
# is a clean contiguous split; the core accumulates over K per output tile. Same repack contract
# ([N][K/256][REC]) and ABI. q4_0/q4k/q6k via --qtype.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

QMAP = {
    "q4_0": (32, 20, "mv_q4_32x32.o", "matvec_q4_0_f32"),
    "q4k": (256, 148, "mv_q4k.o", "matvec_q4k_f32"),
    "q6k": (256, 212, "mv_q6k.o", "matvec_q6k_f32"),
    "q6k_pd": (256, 276, "mv_q6k.o", "matvec_q6k_f32"),  # pre-dequant int8 record (mv_q6k_predq.cc)
}


def gemv_mc16(dev, qtype, M, K, m, rows, wdepth=2):
    # wdepth: object_fifo depth on the WEIGHT stream ONLY (shim->memtile AND memtile->core).
    #   2 = shipping (double-buffered, DMA overlaps compute). 1 = single-buffered (no overlap).
    # Activation (bB) and C fifos are left at depth 2 in all cases -> clean A/B on weight DMA.
    QK, REC, obj, mvsym = QMAP[qtype]
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    cols = 4
    ncores = cols * rows
    assert M % (m * ncores) == 0, "N must be divisible by m*cols*rows"
    Mc = M // ncores            # output rows per core
    Mdm = Mc // m               # output m-tiles per core
    Acol = (M // cols) * (K // QK) * REC  # weight bytes per column
    Acore = Mc * (K // QK) * REC          # weight bytes per core

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            # per-column L2 staging types (rows x per-core)
            aL2_ty = np.ndarray[(rows * m, REC), np.dtype[np.uint8]]
            cL2_ty = np.ndarray[(rows * m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            matvec = external_func(mvsym, inputs=[a_ty, b_ty, c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [[tile(c, 2 + r) for r in range(rows)] for c in range(cols)]

            wA_l3l2 = [None] * cols
            wA_l2l1 = [[None] * rows for _ in range(cols)]
            bB = [None] * cols
            cC_l1l2 = [[None] * rows for _ in range(cols)]
            cC_l2l3 = [None] * cols

            def build_col(c):
                # weight: shim->memtile (N/4), distribute to the 4 rows (N/16 each)
                wA_l3l2[c] = object_fifo(f"wA_l3l2_{c}", shims[c], mts[c], wdepth, aL2_ty)
                for r in range(rows):
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], wdepth, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [m * REC * j for j in range(rows)])
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

                # N-independent core (no Mdm baked): one output tile per infinite-loop iter,
                # runtime feeds Mdm tiles per core. Keeps the overlay per (dtype,K) - ELF per N.
                @core(cores[cc][rr], obj, stack_size=0x2000)
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
                np.ndarray[(M * (K // QK) * REC,), np.dtype[np.uint8]],
                np.ndarray[(K,), np.dtype[bf16]],
                np.ndarray[(M,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for c in range(cols):
                    # weight for column c's N/4 outputs, output-tile-major so the row-distribute
                    # (contiguous m*REC chunks) hands each row its N/16 slice.
                    # element = [rows*m records, REC] for one (output-tile i, k-tile t):
                    #   record for output n=i*(rows*m)+j at k-tile t -> n*(K/256)*REC + t*REC.
                    # dim2 (j, the rows*m outputs) stride = (K/256)*REC; dim1 (t) stride = REC;
                    # dim0 (i) stride = rows*m*(K/256)*REC.
                    npu_dma_memcpy_nd(metadata=wA_l3l2[c], bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, c * Acol],
                                      sizes=[Mdm, K_div_k, rows * m, REC],
                                      strides=[rows * m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                    npu_dma_memcpy_nd(metadata=bB[c], bd_id=c * 3 + 2, mem=B,
                                      sizes=[Mdm, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * (M // cols)],
                                      sizes=[1, 1, 1, M // cols], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core decode gemv")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--qtype", choices=list(QMAP), required=True)
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--wdepth", type=int, default=2, help="weight-stream object_fifo depth (1 or 2)")
    a, _ = p.parse_known_args()
    gemv_mc16(a.dev, a.qtype, a.M, a.K, a.m, a.rows, a.wdepth)
