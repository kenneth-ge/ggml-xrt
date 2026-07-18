# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc16.py (16-core distribute/gather) + gemv_qkv_fused.py (q/k/v pack/ABI).
# (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt 16-CORE FUSED q/k/v decode gemv. q_proj, k_proj, v_proj all contract the SAME
# layer-input activation b[K], so compute all three in ONE dispatch (one activation upload,
# one hw_context). Unlike the 4-core gemv_qkv_fused (1 tile/proj-column -> compute regression),
# this uses ALL 16 compute tiles: 4 columns x 4 rows, exactly like gemv_mc16.
#
# KEY INSIGHT: assign a WHOLE column (4 core-rows) to one proj so each column is dtype-uniform
# (uniform record size), so the per-column memtile distribute works exactly like gemv_mc16:
#   col0 (4 cores) = q rows    0..1023  (q4k, REC=148)
#   col1 (4 cores) = q rows 1024..2047  (q4k, REC=148)
#   col2 (4 cores) = k rows    0..1023  (q4k, REC=148)
#   col3 (4 cores) = v rows    0..1023  (q6k, REC=212)
# Per column: shim->memtile weight stream (out_rows) DISTRIBUTED to its 4 core-rows
# (out_rows/4 each) via object_fifo_link dstOffsets; activation b[K] BROADCAST to the 4 rows;
# C GATHERED from the 4 rows -> memtile -> shim (srcOffsets). Per-column weight source offset
# into A and per-column output offset into C.
#
# Frozen 3-buffer ABI:
#   A = q_weight(q4k, Nq*Kt*148) ++ k_weight(q4k, Nkv*Kt*148) ++ v_weight(q6k, Nkv*Kt*212)
#   B = x[K] bf16  (shared - every column reads offset 0, broadcast)
#   C = q_out[Nq] ++ k_out[Nkv] ++ v_out[Nkv]  = f32[Nq + 2*Nkv]
#
# Qwen3-1.7B default: K=2048, Nq=2048 (q4k), Nkv=1024 (k q4k, v q6k) -> total out 4096.
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

QK = 256
# dtype -> (record bytes, core object, matvec symbol)
QMAP = {
    "q4k": (148, "mv_q4k.o", "matvec_q4k_f32"),
    "q6k": (212, "mv_q6k.o", "matvec_q6k_f32"),
}


def qkv16(dev, K, Nq, Nkv, m, qk_dtype, v_dtype, rows):
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    Kt = K // QK
    cols = 4
    assert K % QK == 0
    # Whole-column-per-proj design: q split over 2 cols (Nq/2 each), k on 1, v on 1.
    assert Nq % 2 == 0 and (Nq // 2) % (rows * m) == 0, "Nq/2 must be divisible by rows*m"
    assert Nkv % (rows * m) == 0, "Nkv must be divisible by rows*m"

    RECqk = QMAP[qk_dtype][0]
    RECv = QMAP[v_dtype][0]
    q_w = Nq * Kt * RECqk
    k_w = Nkv * Kt * RECqk
    # per-column: (dtype, out_rows, A byte offset, C f32 offset)
    cols_cfg = [
        (qk_dtype, Nq // 2, 0,                        0),           # q, first half
        (qk_dtype, Nq // 2, (Nq // 2) * Kt * RECqk,   Nq // 2),     # q, second half
        (qk_dtype, Nkv,     q_w,                      Nq),          # k
        (v_dtype,  Nkv,     q_w + k_w,                Nq + Nkv),    # v
    ]
    A_sz = q_w + k_w + Nkv * Kt * RECv
    C_sz = Nq + 2 * Nkv

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            b_ty = np.ndarray[(k,), np.dtype[bf16]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            cL2_ty = np.ndarray[(rows * m,), np.dtype[np.float32]]

            # zero_scalar_f32 is defined in every core object; declare once.
            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            mv = {}
            for dt in set([qk_dtype, v_dtype]):
                rec, obj, mvsym = QMAP[dt]
                a_ty = np.ndarray[(m, rec), np.dtype[np.uint8]]
                mv[dt] = external_func(mvsym, inputs=[a_ty, b_ty, c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [[tile(c, 2 + r) for r in range(rows)] for c in range(cols)]

            wA_l3l2 = [None] * cols
            wA_l2l1 = [[None] * rows for _ in range(cols)]
            bB = [None] * cols
            cC_l1l2 = [[None] * rows for _ in range(cols)]
            cC_l2l3 = [None] * cols
            # per-column runtime params: (rec, a_off, c_off, out_rows, Mdm)
            rt = [None] * cols

            def make_core(cc, rr, obj, dt):  # own scope binds cc,rr,obj,dt for the no-arg closure
                wA = wA_l2l1[cc][rr]
                cC = cC_l1l2[cc][rr]
                bBc = bB[cc]
                mvf = mv[dt]

                @core(cores[cc][rr], obj, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        eo = cC.acquire(ObjectFifoPort.Produce, 1)
                        zero(eo)
                        for _ in range_(K_div_k):
                            av = wA.acquire(ObjectFifoPort.Consume, 1)
                            bv = bBc.acquire(ObjectFifoPort.Consume, 1)
                            mvf(av, bv, eo)
                            wA.release(ObjectFifoPort.Consume, 1)
                            bBc.release(ObjectFifoPort.Consume, 1)
                        cC.release(ObjectFifoPort.Produce, 1)

            def build_col(c):
                dt, out_rows, a_off, c_off = cols_cfg[c]
                rec, obj, mvsym = QMAP[dt]
                a_ty = np.ndarray[(m, rec), np.dtype[np.uint8]]
                aL2_ty = np.ndarray[(rows * m, rec), np.dtype[np.uint8]]
                Mc = out_rows // rows       # output rows per core
                Mdm = Mc // m               # output m-tiles per core
                rt[c] = (rec, a_off, c_off, out_rows, Mdm)

                # weight: shim->memtile (out_rows), distribute to the 4 rows (out_rows/4 each)
                wA_l3l2[c] = object_fifo(f"wA_l3l2_{c}", shims[c], mts[c], 2, aL2_ty)
                for r in range(rows):
                    wA_l2l1[c][r] = object_fifo(f"wA_l2l1_{c}_{r}", mts[c], cores[c][r], 2, a_ty)
                object_fifo_link(wA_l3l2[c], [wA_l2l1[c][r] for r in range(rows)],
                                 [], [m * rec * j for j in range(rows)])
                # activation: shim->memtile->broadcast to 4 rows
                bB[c] = object_fifo(f"bB_{c}", shims[c], [cores[c][r] for r in range(rows)], 2, b_ty)
                # C: 4 rows -> memtile (gather) -> shim
                for r in range(rows):
                    cC_l1l2[c][r] = object_fifo(f"cC_l1l2_{c}_{r}", cores[c][r], mts[c], 2, c_ty)
                cC_l2l3[c] = object_fifo(f"cC_l2l3_{c}", mts[c], shims[c], 2, cL2_ty)
                object_fifo_link([cC_l1l2[c][r] for r in range(rows)], cC_l2l3[c],
                                 [m * j for j in range(rows)], [])

                for r in range(rows):
                    make_core(c, r, obj, dt)

            for c in range(cols):
                build_col(c)

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[np.uint8]],
                np.ndarray[(K,), np.dtype[bf16]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for c in range(cols):
                    rec, a_off, c_off, out_rows, Mdm = rt[c]
                    # weight for this column's out_rows outputs, output-tile-major so the row
                    # distribute (contiguous m*rec chunks) hands each row its out_rows/4 slice.
                    npu_dma_memcpy_nd(metadata=wA_l3l2[c], bd_id=c * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, a_off],
                                      sizes=[Mdm, K_div_k, rows * m, rec],
                                      strides=[rows * m * Kt * rec, rec, Kt * rec, 1])
                    npu_dma_memcpy_nd(metadata=bB[c], bd_id=c * 3 + 2, mem=B,
                                      sizes=[Mdm, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=cC_l2l3[c], bd_id=c * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c_off],
                                      sizes=[1, 1, 1, out_rows], strides=[0, 0, 0, 1])
                for c in range(cols):
                    dma_wait(cC_l2l3[c])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt 16-core fused q/k/v decode gemv")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("--Nq", type=int, default=2048)
    p.add_argument("--Nkv", type=int, default=1024)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--qk", default="q4k", choices=list(QMAP))
    p.add_argument("--v", default="q6k", choices=list(QMAP))
    a, _ = p.parse_known_args()
    qkv16(a.dev, a.K, a.Nq, a.Nkv, a.m, a.qk, a.v, a.rows)
