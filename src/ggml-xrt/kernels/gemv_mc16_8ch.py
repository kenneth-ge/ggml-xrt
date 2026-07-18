# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py + whole_array distribute/broadcast (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt 16-CORE decode gemv, 8-WEIGHT-CHANNEL variant. Same 4 cols x 4 rows compute layout and
# same per-core matvec as gemv_mc16.py; ONLY the DMA structure changes so the WEIGHT stream uses
# BOTH shim MM2S channels per column instead of one.
#
# Phoenix (aie2/XDNA1) shim has exactly 2 MM2S (DDR->AIE) channels per column = 8 total. gemv_mc16
# spends 1 MM2S/column on weight (4 channels, ~13 GB/s regime) and the other on the activation b.
# This variant frees weight to use both:
#
#   --stream (weight-only ceiling): NO activation. Every column splits its N/4 weight into TWO
#       shim->memtile streams (each N/8) over its two MM2S channels; the memtile distributes each
#       half to 2 core rows. All 8 MM2S carry weight -> measures the ~23 GB/s ceiling. Cores are
#       gutted (zero + acquire/release, no matvec) so this isolates pure weight DMA rate.
#
#   full (drop-in decode kernel): weight wants all 8 MM2S but the activation b must still enter
#       from DDR over one MM2S. b is tiny (K bf16) so we load it ONCE and cross-column broadcast
#       it to all 16 cores via column-0's memtile (the whole_array A-broadcast idiom). That costs
#       column 0 one MM2S, so columns 1-3 run dual weight streams (2 MM2S each = 6) and column 0
#       runs a single weight stream (1 MM2S) + the b broadcast (1 MM2S) = 7 weight channels + b.
#       Compute-bound today (won't speed up until the matvec drops), but drop-in: natural C[N],
#       same repack contract ([N][K/256][REC]) and ABI as gemv_mc16.
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
}


def gemv_mc16_8ch(dev, qtype, M, K, m, rows, stream):
    QK, REC, obj, mvsym = QMAP[qtype]
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    cols = 4
    ncores = cols * rows
    assert rows == 4, "8ch weight split feeds 2 rows per MM2S; needs rows==4"
    assert M % (m * ncores) == 0, "N must be divisible by m*cols*rows"
    Mc = M // ncores            # output rows per core
    Mdm = Mc // m               # output m-tiles per core
    Acol = (M // cols) * (K // QK) * REC  # weight bytes per column
    rm = rows * m               # outputs per column output-tile-index (== distribute span)

    # Weight streams per column: list of row-groups. Each group is fed by ONE shim MM2S channel
    # and the memtile distributes it to that group's rows. stream: every column dual (8 MM2S for
    # weight). full: col 0 single (frees a MM2S for the b broadcast), cols 1-3 dual -> 7 weight
    # channels + b.
    def groups_for(c):
        if stream:
            return [[0, 1], [2, 3]]
        return [[0, 1, 2, 3]] if c == 0 else [[0, 1], [2, 3]]

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            cL2_ty = np.ndarray[(rows * m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            if not stream:
                matvec = external_func(mvsym, inputs=[a_ty, b_ty, c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [[tile(c, 2 + r) for r in range(rows)] for c in range(cols)]

            wA_grp = [[] for _ in range(cols)]           # per col: list of (fifo, group)
            wA_l2l1 = [[None] * rows for _ in range(cols)]
            cC_l1l2 = [[None] * rows for _ in range(cols)]
            cC_l2l3 = [None] * cols
            bB = None       # shim->memtile0 (full only)
            bBcast = None   # memtile0->16 cores broadcast (full only)

            def build_col(c):
                grps = groups_for(c)
                # weight: one shim->memtile stream per row-group, distributed to the group's rows.
                for gi, g in enumerate(grps):
                    gL2_ty = np.ndarray[(len(g) * m, REC), np.dtype[np.uint8]]
                    f = object_fifo(f"wA_l3l2_{c}_{gi}", shims[c], mts[c], 2, gL2_ty)
                    wA_grp[c].append((f, g))
                    for r in g:
                        wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], 2, a_ty)
                    object_fifo_link(f, [wA_l2l1[c][r] for r in g],
                                     [], [m * REC * i for i in range(len(g))])
                # C: 4 rows -> memtile (gather) -> shim
                for r in range(rows):
                    cC_l1l2[c][r] = object_fifo(f"cC_l1l2_{c}_{r}", cores[c][r], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty)
                object_fifo_link([cC_l1l2[c][r] for r in range(rows)], cC_l2l3[c],
                                 [m * j for j in range(rows)], [])

            def build_b():
                # activation loaded ONCE and cross-column broadcast to all 16 cores via memtile 0.
                nonlocal bB, bBcast
                bB = object_fifo("bB", shims[0], mts[0], 2, b_ty)
                bBcast = object_fifo("bBcast", mts[0],
                                     [cores[c][r] for c in range(cols) for r in range(rows)],
                                     2, b_ty)
                object_fifo_link(bB, bBcast)

            def make_core(cc, rr):  # own scope so the no-arg core closure binds cc,rr
                wA = wA_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]

                @core(cores[cc][rr], obj, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        eo = cC.acquire(ObjectFifoPort.Produce, 1)
                        zero(eo)
                        for _ in range_(K_div_k):
                            av = wA.acquire(ObjectFifoPort.Consume, 1)
                            if not stream:
                                bv = bBcast.acquire(ObjectFifoPort.Consume, 1)
                                matvec(av, bv, eo)
                                bBcast.release(ObjectFifoPort.Consume, 1)
                            wA.release(ObjectFifoPort.Consume, 1)
                        cC.release(ObjectFifoPort.Produce, 1)

            for c in range(cols):
                build_col(c)
            if not stream:
                build_b()
            for c in range(cols):
                for r in range(rows):
                    make_core(c, r)

            @runtime_sequence(
                np.ndarray[(M * (K // QK) * REC,), np.dtype[np.uint8]],
                np.ndarray[(K,), np.dtype[bf16]],
                np.ndarray[(M,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for c in range(cols):
                    # weight per row-group, output-tile-major. For group g the element carries
                    # len(g)*m records = column-local outputs i*rm + g[0]*m + [0,len(g)*m); the
                    # distribute hands each row its contiguous m-record slice.
                    for gi, (f, g) in enumerate(wA_grp[c]):
                        npu_dma_memcpy_nd(
                            metadata=f, bd_id=c * 4 + gi, mem=A,
                            offsets=[0, 0, 0, c * Acol + g[0] * m * (K // QK) * REC],
                            sizes=[Mdm, K_div_k, len(g) * m, REC],
                            strides=[rm * (K // QK) * REC, REC, (K // QK) * REC, 1])
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 4 + 2, mem=C,
                                      offsets=[0, 0, 0, c * (M // cols)],
                                      sizes=[1, 1, 1, M // cols], strides=[0, 0, 0, 1])
                if not stream:
                    npu_dma_memcpy_nd(metadata=bB, bd_id=3, mem=B,
                                      sizes=[Mdm, 1, 1, K], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core decode gemv (8 weight channels)")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--qtype", choices=list(QMAP), required=True)
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--stream", action="store_true",
                   help="gut the matvec (weight-only DMA ceiling; all 8 MM2S carry weight)")
    a, _ = p.parse_known_args()
    gemv_mc16_8ch(a.dev, a.qtype, a.M, a.K, a.m, a.rows, a.stream)
