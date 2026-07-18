# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPATIALLY-PACKED decode-attention overlay: all THREE attention primitives
# (QK^T gemv, soft_max_ext, scores*V gemv) declared in ONE aie.device (aie2/Phoenix)
# so they share a SINGLE hw_context. Mirrors rms_rope_packed.py (which packs rms+rope
# into one overlay) but with three cores on three columns.
#
#   * Core A (QK^T)     : bf16 matvec on tile(0,2) [col0], K=head_dim(128), M=n_kv(free).
#                         C[M] = A[M,K].B[K]; A = K^T rows [n_kv, head_dim], B = query[128].
#   * Core B (softmax)  : soft_max_ext on tile(1,2) [col1], streams scores[N]+mask[N]->probs[N].
#   * Core C (scores*V) : bf16 matvec on tile(2,2) [col2], M=head_dim(128), K=n_kv(bucket).
#                         C[M] = A[M,K].B[K]; A = V^T rows [head_dim, n_kv], B = probs[n_kv].
#
#   Each core has its OWN shim/memtile/objectfifos; ALL three cores + all objectfifos are
#   ALWAYS declared, so the device/core config (= the overlay) is IDENTICAL regardless of
#   `--op`. `--op {qkt,softmax,sv}` controls ONLY the runtime_sequence (which op's DMAs
#   run). => ONE overlay xclbin (3 cores, 1 hw_context) + per-op ELF modules. Idle cores
#   simply block on their (never-filled) input objectfifo acquire - harmless.
#
# This is LEVEL-A "co-located, separately-dispatched" packing: NO on-chip chaining between
# the three cores yet (that is a later step); they are independent, each dispatched by its
# own per-op ELF loaded against the shared overlay.
#
# Low-level/placed style (like gemv.py / rms_rope_packed.py): the high-level IRON API
# refuses an ObjectFifo whose producer endpoint is never driven, which blocks the
# "declare all cores, drive one" per-op-ELF pattern; the placed API lets us declare all
# endpoints explicitly and emit DMA for only the active op.
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Windows validates on hardware.
import argparse

import numpy as np
from ml_dtypes import bfloat16
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def make_gemv_core(ct, obj, inA_fifo, inB_fifo, outC_fifo, K_div_k, zero_fn, matvec_fn):
    """Factory (avoids default-kwarg closures, which @core rejects) building a single-core
    bf16 gemv body: one m-tile of C per outer iter, accumulating K/k inner tiles."""

    @core(ct, obj, stack_size=0x2000)
    def core_body():
        for _ in range_(0xFFFFFFFF):
            elem_out = outC_fifo.acquire(ObjectFifoPort.Produce, 1)
            zero_fn(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = inA_fifo.acquire(ObjectFifoPort.Consume, 1)
                elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                matvec_fn(elem_in_a, elem_in_b, elem_out)
                inA_fifo.release(ObjectFifoPort.Consume, 1)
                inB_fifo.release(ObjectFifoPort.Consume, 1)
            outC_fifo.release(ObjectFifoPort.Produce, 1)

    return core_body


def packed(dev, op, M_qk, K_qk, N_sm, M_sv, K_sv, m, k):
    dtype_in = np.dtype[bfloat16]
    dtype_out = np.dtype[np.float32]

    # gemv tile granularity (shared by both matvec cores; m=k=32 -> mv_32x32.o)
    assert M_qk % m == 0 and K_qk % k == 0, "QK^T: M%m and K%k"
    assert M_sv % m == 0 and K_sv % k == 0, "scores*V: M%m and K%k"
    assert N_sm % 16 == 0, "softmax N must be a multiple of SM_VEC_LEN (16)"

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # ---- shared gemv tile types (m,k identical for both matvec cores) -------
            inA_ty = np.ndarray[(m * k,), dtype_in]     # memtile element (flat)
            A_ty = np.ndarray[(m, k), dtype_in]         # core element (2-D tile)
            inB_ty = np.ndarray[(k,), dtype_in]         # activation vector tile
            outC_ty = np.ndarray[(m,), dtype_out]       # output m-tile
            sm_ty = np.ndarray[(N_sm,), dtype_in]       # softmax whole (padded) vector

            # ---- external kernel funcs (declared ONCE; reused by both gemv cores) ---
            zero_fn = external_func("zero_scalar_f32", inputs=[outC_ty])
            matvec_fn = external_func(
                "matvec_scalar_bf16_f32", inputs=[A_ty, inB_ty, outC_ty]
            )
            softmax_fn = external_func(
                "softmax_ext_bf16", inputs=[sm_ty, sm_ty, sm_ty]
            )

            # ---- tiles: one column per op (self-contained shim/mem/core) ------------
            shim_qk = tile(0, 0); mem_qk = tile(0, 1); ct_qk = tile(0, 2)   # QK^T
            shim_sm = tile(1, 0);                       ct_sm = tile(1, 2)   # softmax
            shim_sv = tile(2, 0); mem_sv = tile(2, 1); ct_sv = tile(2, 2)   # scores*V

            # ---- Core A: QK^T gemv (col0) ------------------------------------------
            memA_qk = object_fifo("memA_qk", shim_qk, mem_qk, 2, inA_ty)
            inA_qk = object_fifo("inA_qk", mem_qk, ct_qk, 2, A_ty)
            object_fifo_link(memA_qk, inA_qk)
            inB_qk = object_fifo("inB_qk", shim_qk, ct_qk, 2, inB_ty)
            outC_qk = object_fifo("outC_qk", ct_qk, shim_qk, 2, outC_ty)
            make_gemv_core(ct_qk, "mv_32x32.o", inA_qk, inB_qk, outC_qk,
                           K_qk // k, zero_fn, matvec_fn)

            # ---- Core B: soft_max_ext (col1) ---------------------------------------
            sm_scores = object_fifo("sm_scores", shim_sm, ct_sm, 2, sm_ty)
            sm_mask = object_fifo("sm_mask", shim_sm, ct_sm, 2, sm_ty)
            sm_probs = object_fifo("sm_probs", ct_sm, shim_sm, 2, sm_ty)

            @core(ct_sm, "softmax_ext.o", stack_size=0x2000)
            def core_sm():
                for _ in range_(0xFFFFFFFF):
                    es = sm_scores.acquire(ObjectFifoPort.Consume, 1)
                    em = sm_mask.acquire(ObjectFifoPort.Consume, 1)
                    eo = sm_probs.acquire(ObjectFifoPort.Produce, 1)
                    softmax_fn(es, em, eo)
                    sm_scores.release(ObjectFifoPort.Consume, 1)
                    sm_mask.release(ObjectFifoPort.Consume, 1)
                    sm_probs.release(ObjectFifoPort.Produce, 1)

            # ---- Core C: scores*V gemv (col2) --------------------------------------
            memA_sv = object_fifo("memA_sv", shim_sv, mem_sv, 2, inA_ty)
            inA_sv = object_fifo("inA_sv", mem_sv, ct_sv, 2, A_ty)
            object_fifo_link(memA_sv, inA_sv)
            inB_sv = object_fifo("inB_sv", shim_sv, ct_sv, 2, inB_ty)
            outC_sv = object_fifo("outC_sv", ct_sv, shim_sv, 2, outC_ty)
            make_gemv_core(ct_sv, "mv_32x32.o", inA_sv, inB_sv, outC_sv,
                           K_sv // k, zero_fn, matvec_fn)

            # ================= runtime_sequence: one op's dataflow ==================
            if op == "qkt":
                _gemv_seq(M_qk, K_qk, m, k, dtype_in, dtype_out,
                          memA_qk, inB_qk, outC_qk)
            elif op == "sv":
                _gemv_seq(M_sv, K_sv, m, k, dtype_in, dtype_out,
                          memA_sv, inB_sv, outC_sv)
            else:  # softmax

                @runtime_sequence(sm_ty, sm_ty, sm_ty)
                def seq_sm(S, Msk, O):
                    npu_dma_memcpy_nd(metadata=sm_scores, bd_id=2, mem=S,
                                      sizes=[1, 1, 1, N_sm], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=sm_mask, bd_id=1, mem=Msk,
                                      sizes=[1, 1, 1, N_sm], strides=[0, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=sm_probs, bd_id=0, mem=O,
                                      sizes=[1, 1, 1, N_sm], strides=[0, 0, 0, 1])
                    dma_wait(sm_probs)

        print(ctx.module)


def _gemv_seq(M, K, m, k, dtype_in, dtype_out, memA_fifo, inB_fifo, outC_fifo):
    """Emit a single-core gemv runtime_sequence (== gemv.py sequence, n_cores=1)."""
    A_sz = M * K
    M_div_m = M // m
    K_div_k = K // k
    m_x_K = m * K

    @runtime_sequence(
        np.ndarray[(A_sz,), dtype_in],
        np.ndarray[(K,), dtype_in],
        np.ndarray[(M,), dtype_out],
    )
    def sequence(A, B, C):
        npu_dma_memcpy_nd(metadata=inB_fifo, bd_id=2, mem=B,
                          sizes=[M_div_m, 1, 1, K], strides=[0, 0, 0, 1])
        npu_dma_memcpy_nd(metadata=memA_fifo, bd_id=1, mem=A,
                          sizes=[M_div_m, K_div_k, m, k], strides=[m_x_K, k, K, 1])
        npu_dma_memcpy_nd(metadata=outC_fifo, bd_id=0, mem=C,
                          sizes=[1, 1, 1, M], strides=[0, 0, 0, 1])
        dma_wait(outC_fifo)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt packed decode-attention overlay")
    p.add_argument("-d", "--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--op", required=True, choices=["qkt", "softmax", "sv"])
    # QK^T: M=n_kv(free), K=head_dim
    p.add_argument("--M_qk", type=int, default=512)
    p.add_argument("--K_qk", type=int, default=128)
    # softmax: padded n_kv
    p.add_argument("--N_sm", type=int, default=512)
    # scores*V: M=head_dim, K=n_kv(bucket)
    p.add_argument("--M_sv", type=int, default=128)
    p.add_argument("--K_sv", type=int, default=512)
    p.add_argument("-m", type=int, default=32, help="gemv M tile")
    p.add_argument("-k", type=int, default=32, help="gemv K tile")
    o, _ = p.parse_known_args()
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu) for this packed overlay")
    packed(o.dev, o.op, o.M_qk, o.K_qk, o.N_sm, o.M_sv, o.K_sv, o.m, o.k)
