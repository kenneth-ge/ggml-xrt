# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_mc.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt FUSED q/k/v decode gemv. q_proj, k_proj, v_proj all contract the SAME layer-input
# activation b[K], so compute all three in ONE dispatch (one activation upload, one hw_context)
# instead of three. Packed into the frozen 3-buffer ABI:
#   A = q_weight ++ k_weight ++ v_weight   (concatenated, each its own dtype/record)
#   B = activation b[K]                     (shared - every column reads offset 0)
#   C = q_out[Nq] ++ k_out[Nkv] ++ v_out[Nkv]
# 4 columns: q split over 2 (Nq/2 each), k on 1, v on 1. q/k link the q4k SIMD core, v links
# the q6k SIMD core (different @core objects per tile is fine - separate per-core ELFs).
#
# Qwen3-1.7B default: K=2048, Nq=2048 (q4k), Nkv=1024 (k q4k, v q6k).
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


def qkv_fused(dev, K, Nq, Nkv, m, qk_dtype, v_dtype):
    bf16 = str_to_dtype("bf16")
    k = QK
    K_div_k = K // k
    Kt = K // QK
    assert Nq % (2 * m) == 0 and Nkv % m == 0 and K % QK == 0

    RECqk = QMAP[qk_dtype][0]
    RECv = QMAP[v_dtype][0]
    # per-column: (dtype, out_rows, A byte offset, C f32 offset)
    q_w = Nq * Kt * RECqk
    k_w = Nkv * Kt * RECqk
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

            # zero_scalar_f32 is defined in every core object; declare once.
            zero = external_func("zero_scalar_f32", inputs=[c_ty])
            mv = {}
            for dt in set([qk_dtype, v_dtype]):
                rec, obj, mvsym = QMAP[dt]
                a_ty = np.ndarray[(m, rec), np.dtype[np.uint8]]
                mv[dt] = external_func(mvsym, inputs=[a_ty, b_ty, c_ty])

            shims = [tile(c, 0) for c in range(4)]
            mts = [tile(c, 1) for c in range(4)]
            cores = [tile(c, 2) for c in range(4)]

            fifos = []

            def build(ci):
                dt, rows, a_off, c_off = cols_cfg[ci]
                rec, obj, mvsym = QMAP[dt]
                a_ty = np.ndarray[(m, rec), np.dtype[np.uint8]]
                M_div_m_c = rows // m
                memA = object_fifo(f"memA{ci}", shims[ci], mts[ci], 2, a_ty)
                inA = object_fifo(f"inA{ci}", mts[ci], cores[ci], 2, a_ty)
                object_fifo_link(memA, inA)
                inB = object_fifo(f"inB{ci}", shims[ci], cores[ci], 2, b_ty)
                outC = object_fifo(f"outC{ci}", cores[ci], shims[ci], 2, c_ty)
                mvf = mv[dt]

                @core(cores[ci], obj, stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            av = inA.acquire(ObjectFifoPort.Consume, 1)
                            bv = inB.acquire(ObjectFifoPort.Consume, 1)
                            mvf(av, bv, elem_out)
                            inA.release(ObjectFifoPort.Consume, 1)
                            inB.release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                return (memA, inB, outC, rec, a_off, c_off, M_div_m_c)

            for ci in range(4):
                fifos.append(build(ci))

            @runtime_sequence(
                np.ndarray[(A_sz,), np.dtype[np.uint8]],
                np.ndarray[(K,), np.dtype[bf16]],
                np.ndarray[(C_sz,), np.dtype[np.float32]],
            )
            def sequence(A, B, C):
                for ci in range(4):
                    memA, inB, outC, rec, a_off, c_off, M_div_m_c = fifos[ci]
                    npu_dma_memcpy_nd(metadata=inB, bd_id=ci * 3 + 2, mem=B,
                                      sizes=[M_div_m_c, 1, 1, K], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=memA, bd_id=ci * 3 + 1, mem=A,
                                      offsets=[0, 0, 0, a_off],
                                      sizes=[M_div_m_c, K_div_k, m, rec],
                                      strides=[m * Kt * rec, rec, Kt * rec, 1])
                    npu_dma_memcpy_nd(metadata=outC, bd_id=ci * 3 + 0, mem=C,
                                      offsets=[0, 0, 0, c_off],
                                      sizes=[1, 1, 1, M_div_m_c * m], strides=[0, 0, 0, 1])
                for ci in range(4):
                    dma_wait(fifos[ci][2])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt fused q/k/v decode gemv")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("--Nq", type=int, default=2048)
    p.add_argument("--Nkv", type=int, default=1024)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--qk", default="q4k", choices=list(QMAP))
    p.add_argument("--v", default="q6k", choices=list(QMAP))
    a, _ = p.parse_known_args()
    qkv_fused(a.dev, a.K, a.Nq, a.Nkv, a.m, a.qk, a.v)
