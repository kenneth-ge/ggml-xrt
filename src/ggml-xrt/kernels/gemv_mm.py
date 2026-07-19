# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt 16-CORE M=N quantized "mat-MAT" for the speculative-decode VERIFY kernel:
#   C[MT, N_out] = A[MT, K] . dequant(W[N_out, K])   (q6k weights, bf16 activations)
# N_out is split across 16 cores (N_out/16 each) exactly like gemv_mc16. The weight
# stream + dequant is amortized over the MT draft tokens: each core dequants its output
# rows ONCE per k-block and MACs all MT tokens against them (see aie2/mv_mm_q6k.cc).
#
# vs. gemv_mc16 the two changes are the extra MT (draft-token) dimension on the
# activation and the output:
#   - activation A[MT, K] is broadcast per k-block as A[MT, 256] (MT strips of K, stride K)
#   - output C[MT, N_out] is gathered per core as C[MT, m] and scattered out with an
#     MT-stride (N_out) so token mt lands in output row mt.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

# qtype -> (QK, REC bytes, core .o, matvec sym, zero sym)
QMAP = {
    "q6k": (256, 212, "mv_mm_q6k.o", "matvec_mm_q6k_f32", "zero_mm_f32"),
    "q4k": (256, 148, "mv_mm_q4k.o", "matvec_mm_q4k_f32", "zero_mm_f32"),
}


def gemv_mm(dev, qtype, M, K, m, rows, MT):
    QK, REC, obj, mvsym, zerosym = QMAP[qtype]
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    cols = 4
    ncores = cols * rows
    assert M % (m * ncores) == 0, "N_out must be divisible by m*cols*rows"
    Mc = M // ncores            # output rows per core
    Mdm = Mc // m               # output m-tiles per core
    Acol = (M // cols) * (K // QK) * REC  # weight bytes per column

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]        # weight records, m rows
            b_ty = np.ndarray[(MT, k), np.dtype[bf16]]             # MT tokens x 256-k-block
            c_ty = np.ndarray[(MT, m), np.dtype[np.float32]]       # MT tokens x m out rows
            # per-column L2 staging types
            aL2_ty = np.ndarray[(rows * m, REC), np.dtype[np.uint8]]
            cL2_ty = np.ndarray[(rows * MT * m,), np.dtype[np.float32]]

            zero = external_func(zerosym, inputs=[c_ty])
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
                # weight: shim->memtile (N/4), distribute to the 4 rows (N/16 each). Unchanged
                # from gemv_mc16 - the weight stream does not depend on MT.
                wA_l3l2[c] = object_fifo(f"wA_l3l2_{c}", shims[c], mts[c], 2, aL2_ty)
                for r in range(rows):
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], 2, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [m * REC * j for j in range(rows)])
                # activation A[MT,256]: shim->broadcast to 4 rows
                bB[c] = object_fifo(f"bB_{c}", shims[c], [cores[c][r] for r in range(rows)], 2, b_ty)
                # C: 4 rows -> memtile (gather MT*m per row, contiguous) -> shim
                for r in range(rows):
                    cC_l1l2[c][r] = object_fifo(f"cC_l1l2_{c}_{r}", cores[c][r], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty)
                object_fifo_link([cC_l1l2[c][r] for r in range(rows)], cC_l2l3[c],
                                 [MT * m * j for j in range(rows)], [])

                for r in range(rows):
                    make_core(c, r)

            def make_core(cc, rr):  # own scope so the no-arg core closure binds cc,rr
                wA = wA_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]
                bBc = bB[cc]

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
                np.ndarray[(M * (K // QK) * REC,), np.dtype[np.uint8]],  # W repacked [N][K/256][REC]
                np.ndarray[(MT * K,), np.dtype[bf16]],                   # A[MT, K]
                np.ndarray[(MT * M,), np.dtype[np.float32]],             # C[MT, N_out]
            )
            def sequence(A, B, C):
                for c in range(cols):
                    # weight for column c's N/4 outputs, output-tile-major (unchanged from gemv_mc16):
                    #   record for output n=i*(rows*m)+j at k-tile t -> n*(K/256)*REC + t*REC.
                    npu_dma_memcpy_nd(metadata=wA_l3l2[c], bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, c * Acol],
                                      sizes=[Mdm, K_div_k, rows * m, REC],
                                      strides=[rows * m * (K // QK) * REC, REC, (K // QK) * REC, 1])
                    # activation A[MT, K], per k-block deliver A[MT, 256] (broadcast, re-streamed
                    # per output tile). Object = MT strips of 256; strip mt stride K, within strip 1.
                    # dims [Mdm(repeat), k-tile, MT, k] -> strides [0, k, K, 1].
                    npu_dma_memcpy_nd(metadata=bB[c], bd_id=c * 3 + 2, mem=B,
                                      sizes=[Mdm, K_div_k, MT, k],
                                      strides=[0, k, K, 1])
                    # C[MT, N_out]: fifo delivers [tile i][row r][mt][mrow]; scatter to
                    # C[mt][ c*(N/4) + (i*rows+r)*m + mrow ]. dims [i, r, mt, mrow].
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c * (M // cols)],
                                      sizes=[Mdm, rows, MT, m],
                                      strides=[rows * m, m, M, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core M=N quantized mat-MAT (verify)")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--qtype", choices=list(QMAP), default="q6k")
    p.add_argument("-M", type=int, required=True, help="output dim N_out (ggml N)")
    p.add_argument("-K", type=int, required=True)
    p.add_argument("-m", type=int, default=32, help="output m-tile per core-call")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--mt", type=int, default=8, help="draft tokens M")
    a, _ = p.parse_known_args()
    gemv_mm(a.dev, a.qtype, a.M, a.K, a.m, a.rows, a.mt)
