# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt FUSED FFN SwiGLU decode gemv (16-core). Computes y[N_ff] = silu(gate . x) * (up . x) in
# ONE dispatch, where gate = Wg . x and up = Wu . x are both q4k decode matvecs. This removes the
# NPU->GPU->NPU round-trip that today runs silu*mul on the iGPU between the up and down matmuls -
# the real per-token cost is a per-dispatch/handoff penalty, so fewer NPU<->GPU boundaries is the
# lever. Each of 16 cores owns N_ff/16 outputs; per output it accumulates BOTH gate and up over K
# into core-local buffers, then applies silu(gate)*up and writes the fused slice.
#
# Weight is an INTERLEAVED per-record stream (REC=296 = gate_q4k(148) ++ up_q4k(148)) so the DMA is
# 4-D identical to gemv_mc16 and fits Phoenix's 2-MM2S/column budget (weight + activation + output).
#   A = [N_ff][K/256][296] (gate++up per record) ; B = shared x[K] bf16 ; C = fused y[N_ff] f32.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

QK = 256
RECq = 148          # one q4k record
REC = 2 * RECq      # interleaved gate++up record = 296


def gemv_swiglu(dev, M, K, m, rows):
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    cols = 4
    ncores = cols * rows
    assert M % (m * ncores) == 0, "N_ff must be divisible by m*cols*rows"
    Mc = M // ncores            # output rows per core
    Mdm = Mc // m               # output m-tiles per core
    Acol = (M // cols) * (K // QK) * REC   # interleaved weight bytes per column

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]     # interleaved gate++up records
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            aL2_ty = np.ndarray[(rows * m, REC), np.dtype[np.uint8]]
            cL2_ty = np.ndarray[(rows * m,), np.dtype[np.float32]]

            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            fused = external_func("fused_gate_up_q4k", inputs=[a_ty, b_ty, c_ty, c_ty])
            silu_mul = external_func("silu_mul_f32", inputs=[c_ty, c_ty, c_ty])

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
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], 2, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [m * REC * j for j in range(rows)])
                bB[c] = object_fifo(f"bB_{c}", shims[c], [cores[c][r] for r in range(rows)], 2, b_ty)
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
                gbuf = buffer(cores[cc][rr], np.ndarray[(m,), np.dtype[np.float32]],
                              name=f"gbuf_{cc}_{rr}")
                ubuf = buffer(cores[cc][rr], np.ndarray[(m,), np.dtype[np.float32]],
                              name=f"ubuf_{cc}_{rr}")

                @core(cores[cc][rr], "mv_swiglu.o", stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        eo = cC.acquire(ObjectFifoPort.Produce, 1)
                        zero(gbuf)
                        zero(ubuf)
                        for _ in range_(K_div_k):
                            av = wA.acquire(ObjectFifoPort.Consume, 1)
                            bv = bBc.acquire(ObjectFifoPort.Consume, 1)
                            fused(av, bv, gbuf, ubuf)
                            wA.release(ObjectFifoPort.Consume, 1)
                            bBc.release(ObjectFifoPort.Consume, 1)
                        silu_mul(gbuf, ubuf, eo)
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
    p = argparse.ArgumentParser(prog="ggml-xrt fused SwiGLU decode gemv (16-core)")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, required=True, help="N_ff (ffn intermediate size)")
    p.add_argument("-K", type=int, required=True, help="hidden size")
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    a, _ = p.parse_known_args()
    gemv_swiglu(a.dev, a.M, a.K, a.m, a.rows)
