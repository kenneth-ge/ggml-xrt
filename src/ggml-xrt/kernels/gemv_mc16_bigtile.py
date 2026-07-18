# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt 16-core decode gemv, LARGER-TILE variant. Same 16-core layout / DMA / distribute /
# gather / repack contract / ABI as gemv_mc16.py; the ONLY change is the core's inner loop
# batches its object_fifo acquire: one acquire/release pair per KK k-blocks instead of per
# block. This implements the OllamaAMDNPU transferable lever ("halve the instruction sequence
# length") by cutting the per-256-block semaphore acquire/release + loop overhead by ~KK, WITHOUT
# touching the (delicate) DMA-distribute offsets. The matvec kernel is unchanged (still one
# 256-block per call); KK matvec calls run inside one acquired batch.
#
# If decode is dequant/MAC-compute-bound this changes little; if it's acquire/loop-overhead-bound
# (which the 3-way isolation experiment tells us) this is a direct win. --kk sweeps the tile depth.
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


def gemv_mc16_bigtile(dev, qtype, M, K, m, rows, kk):
    QK, REC, obj, mvsym = QMAP[qtype]
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    assert K_div_k % kk == 0, "K/QK must be divisible by --kk"
    cols = 4
    ncores = cols * rows
    assert M % (m * ncores) == 0, "N must be divisible by m*cols*rows"
    Mc = M // ncores            # output rows per core
    Mdm = Mc // m               # output m-tiles per core
    Acol = (M // cols) * (K // QK) * REC  # weight bytes per column
    # BD budget is 48/memtile: keep L3->L2 staging at 2, size the core-side fifos to hold one
    # batch (kk) + 1 for prefetch. depth=2*kk exhausts BDs across the 4 cols' 16 core fifos.
    cdepth = kk + 1            # core-side (wA_l2l1, bB): enough for a batch + overlap

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
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
                wA_l3l2[c] = object_fifo(f"wA_l3l2_{c}", shims[c], mts[c], 2, aL2_ty)
                for r in range(rows):
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], cdepth, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [m * REC * j for j in range(rows)])
                bB[c] = object_fifo(f"bB_{c}", shims[c], [cores[c][r] for r in range(rows)], cdepth, b_ty)
                for r in range(rows):
                    cC_l1l2[c][r] = object_fifo(f"cC_l1l2_{c}_{r}", cores[c][r], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty)
                object_fifo_link([cC_l1l2[c][r] for r in range(rows)], cC_l2l3[c],
                                 [m * j for j in range(rows)], [])

                for r in range(rows):
                    make_core(c, r)

            def make_core(cc, rr):
                wA = wA_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]
                bBc = bB[cc]

                @core(cores[cc][rr], obj, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        eo = cC.acquire(ObjectFifoPort.Produce, 1)
                        zero(eo)
                        # one acquire/release per KK k-blocks (vs per block) -> fewer instrs
                        for _ in range_(K_div_k // kk):
                            av = wA.acquire(ObjectFifoPort.Consume, kk)
                            bv = bBc.acquire(ObjectFifoPort.Consume, kk)
                            for j in range(kk):  # compile-time unroll (static subview index)
                                matvec(av[j], bv[j], eo)
                            wA.release(ObjectFifoPort.Consume, kk)
                            bBc.release(ObjectFifoPort.Consume, kk)
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
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core decode gemv (larger-tile)")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--qtype", choices=list(QMAP), required=True)
    p.add_argument("-M", type=int, required=True, help="output dim (ggml N)")
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--kk", type=int, default=2, help="k-blocks per acquire (instruction-seq lever)")
    a, _ = p.parse_known_args()
    gemv_mc16_bigtile(a.dev, a.qtype, a.M, a.K, a.m, a.rows, a.kk)
