# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# LEVEL-B on-chip-CHAINED decode-attention overlay: QK^T -> soft_max_ext ->
# scores*V wired CORE-TO-CORE through object_fifos so the WHOLE attention runs in
# ONE dispatch with NO L3 round-trips between the three sub-ops. This is the step
# that removes the NPU<->CPU handoffs (the measured ~130ms/token bottleneck) that
# Level-A's three-separate-dispatch packing (attn_packed.py) still paid.
#
#   Core A (QK^T,     tile(0,2)): scores[n_kv] = K[n_kv,head_dim] . Q[head_dim]
#       (bf16 matvec, M=n_kv tiled by m, K=head_dim whole). Q broadcast in; K
#       streamed from L3. Output scores are written bf16 (== soft_max input dtype)
#       in m-tiles into an object_fifo that AGGREGATES (via a col1 memtile) into
#       the FULL n_kv vector for Core B. scores NEVER touch L3.
#   Core B (soft_max, tile(1,2)): consumes the WHOLE scores[n_kv] + mask[n_kv],
#       runs softmax_ext_bf16 (scale=1/sqrt(head_dim) baked in), produces the
#       WHOLE probs[n_kv] core-to-core to Core C. probs NEVER touch L3.
#   Core C (scores*V, tile(2,2)): out[head_dim] = Vt[head_dim,n_kv] . probs[n_kv]
#       (bf16->f32 matvec, M=head_dim tiled by m, K=n_kv whole). probs held whole
#       and reused across the m-tiles; V streamed from L3; out[head_dim] DMA'd out.
#
# THE SOFTMAX BARRIER: soft_max reduces over the ENTIRE n_kv, so B needs ALL
# scores before it starts and C needs ALL probs. Both chain fifos therefore carry
# the FULL n_kv vector as ONE logical element:
#   * A->B: a size-ratio object_fifo_link (Core A releases M/m tiles of `m` bf16;
#     the col1 memtile aggregates them into ONE n_kv-element the softmax acquires
#     whole). Verified to lower via --aie-objectFifo-stateful-transform.
#   * B->C: a 1:1 same-size (n_kv bf16) direct core-to-core object_fifo.
#
# ONE runtime_sequence drives it: DMA K,Q,V,mask in; dma_wait on out. Only K,Q,V,
# mask,out cross L3; scores and probs stay on-chip (A->B->C via fifos).
#
# Low-level/placed style (like gemv.py / rms_rope_packed.py): the placed API lets
# us declare all endpoints and the size-ratio link explicitly.
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Windows validates on HW.
import argparse

import numpy as np
from ml_dtypes import bfloat16
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def chained(dev, n_kv, head_dim, m):
    dtype_in = np.dtype[bfloat16]
    dtype_out = np.dtype[np.float32]

    # QK^T   : M = n_kv (tiled by m), K = head_dim (whole).
    # scores*V: M = head_dim (tiled by m), K = n_kv (whole).
    assert n_kv % m == 0, "n_kv must be divisible by m (QK^T output tiling)"
    assert head_dim % m == 0, "head_dim must be divisible by m (scores*V tiling)"
    assert n_kv % 16 == 0, "n_kv must be a multiple of SM_VEC_LEN (16) for softmax"

    M_qk = n_kv          # QK^T output rows
    K_qk = head_dim      # QK^T contraction (whole)
    M_sv = head_dim      # scores*V output rows
    K_sv = n_kv          # scores*V contraction (whole)

    qk_tiles = M_qk // m  # number of scores m-tiles Core A produces
    sv_tiles = M_sv // m  # number of out m-tiles Core C produces

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            # ---- types --------------------------------------------------------
            Q_ty = np.ndarray[(K_qk,), dtype_in]              # query vector (whole)
            memK_ty = np.ndarray[(m * K_qk,), dtype_in]       # K weight tile (flat, L3->L2)
            K_ty = np.ndarray[(m, K_qk), dtype_in]            # K weight tile (2-D, L2->core)
            score_tile_ty = np.ndarray[(m,), dtype_in]        # one scores m-tile (bf16)
            vec_ty = np.ndarray[(n_kv,), dtype_in]            # full n_kv vector (bf16)
            memV_ty = np.ndarray[(m * K_sv,), dtype_in]       # V weight tile (flat, L3->L2)
            V_ty = np.ndarray[(m, K_sv), dtype_in]            # V weight tile (2-D, L2->core)
            out_tile_ty = np.ndarray[(m,), dtype_out]         # one out m-tile (f32)

            # ---- external kernel funcs ---------------------------------------
            # Core A: bf16-OUT matvec (float accumulate) so scores stay bf16.
            zero_bf16 = external_func("zero_scalar_bf16", inputs=[score_tile_ty])
            matvec_qk = external_func(
                "matvec_scalar_bf16_bf16", inputs=[K_ty, Q_ty, score_tile_ty]
            )
            # Core B: soft_max_ext (scale baked, DIM_N=n_kv baked).
            softmax_fn = external_func(
                "softmax_ext_bf16", inputs=[vec_ty, vec_ty, vec_ty]
            )
            # Core C: bf16->f32 matvec (final output precision).
            zero_f32 = external_func("zero_scalar_f32", inputs=[out_tile_ty])
            matvec_sv = external_func(
                "matvec_scalar_bf16_f32", inputs=[V_ty, vec_ty, out_tile_ty]
            )

            # ---- tiles: one column per op ------------------------------------
            shim_a = tile(0, 0); mem_a = tile(0, 1); ct_a = tile(0, 2)   # QK^T
            shim_b = tile(1, 0); mem_sc = tile(1, 1); ct_b = tile(1, 2)  # softmax + scores agg
            shim_c = tile(2, 0); mem_c = tile(2, 1); ct_c = tile(2, 2)   # scores*V

            # ---- Core A inputs: Q (broadcast) + K weights (L3->L2->core) ------
            inQ = object_fifo("inQ", shim_a, ct_a, 2, Q_ty)
            memK = object_fifo("memK", shim_a, mem_a, 2, memK_ty)
            inK = object_fifo("inK", mem_a, ct_a, 2, K_ty)
            object_fifo_link(memK, inK)

            # ---- CHAIN A->B: scores m-tiles aggregate into full n_kv ---------
            # Core A releases qk_tiles x (m bf16); the col1 memtile aggregates
            # them into ONE (n_kv bf16) element the softmax acquires whole.
            scores_a = object_fifo("scores_a", ct_a, mem_sc, 2, score_tile_ty)
            scores_b = object_fifo("scores_b", mem_sc, ct_b, 2, vec_ty)
            object_fifo_link(scores_a, scores_b)

            # ---- Core B input: mask (L3->core), full n_kv --------------------
            inMask = object_fifo("inMask", shim_b, ct_b, 2, vec_ty)

            # ---- CHAIN B->C: full probs[n_kv] core-to-core (1:1) -------------
            probs = object_fifo("probs", ct_b, ct_c, 2, vec_ty)

            # ---- Core C input: V weights (L3->L2->core) + output (core->L3) --
            # inV (on-core V tile) is (m, n_kv) bf16 = large (32*512*2 = 32KB at
            # the 512 bucket); single-buffer it (depth 1) so Core C's L1 stays
            # under 64KB (double-buffering it would need 64KB for V alone).
            memV = object_fifo("memV", shim_c, mem_c, 2, memV_ty)
            inV = object_fifo("inV", mem_c, ct_c, 1, V_ty)
            object_fifo_link(memV, inV)
            outO = object_fifo("outO", ct_c, shim_c, 2, out_tile_ty)

            # ================= cores ==========================================
            @core(ct_a, "mv_qkt.o", stack_size=0x2000)
            def core_a():
                for _ in range_(0xFFFFFFFF):
                    eQ = inQ.acquire(ObjectFifoPort.Consume, 1)  # hold Q whole
                    for _ in range_(qk_tiles):
                        es = scores_a.acquire(ObjectFifoPort.Produce, 1)
                        zero_bf16(es)
                        eK = inK.acquire(ObjectFifoPort.Consume, 1)
                        matvec_qk(eK, eQ, es)
                        inK.release(ObjectFifoPort.Consume, 1)
                        scores_a.release(ObjectFifoPort.Produce, 1)
                    inQ.release(ObjectFifoPort.Consume, 1)

            @core(ct_b, "softmax_ext.o", stack_size=0x2000)
            def core_b():
                for _ in range_(0xFFFFFFFF):
                    es = scores_b.acquire(ObjectFifoPort.Consume, 1)  # full n_kv
                    em = inMask.acquire(ObjectFifoPort.Consume, 1)
                    ep = probs.acquire(ObjectFifoPort.Produce, 1)     # full n_kv
                    softmax_fn(es, em, ep)
                    scores_b.release(ObjectFifoPort.Consume, 1)
                    inMask.release(ObjectFifoPort.Consume, 1)
                    probs.release(ObjectFifoPort.Produce, 1)

            @core(ct_c, "mv_sv.o", stack_size=0x2000)
            def core_c():
                for _ in range_(0xFFFFFFFF):
                    ep = probs.acquire(ObjectFifoPort.Consume, 1)  # hold probs whole
                    for _ in range_(sv_tiles):
                        eo = outO.acquire(ObjectFifoPort.Produce, 1)
                        zero_f32(eo)
                        eV = inV.acquire(ObjectFifoPort.Consume, 1)
                        matvec_sv(eV, ep, eo)
                        inV.release(ObjectFifoPort.Consume, 1)
                        outO.release(ObjectFifoPort.Produce, 1)
                    probs.release(ObjectFifoPort.Consume, 1)

            # ================= ONE runtime_sequence ===========================
            # Inputs cross L3: K[n_kv,head_dim], Q[head_dim], V[head_dim,n_kv],
            # mask[n_kv]. Output: out[head_dim]. scores/probs stay on-chip.
            @runtime_sequence(
                np.ndarray[(M_qk * K_qk,), dtype_in],  # K weights
                np.ndarray[(K_qk,), dtype_in],         # Q
                np.ndarray[(M_sv * K_sv,), dtype_in],  # V weights (transposed)
                np.ndarray[(n_kv,), dtype_in],         # mask
                np.ndarray[(M_sv,), dtype_out],        # out
            )
            def sequence(K, Q, V, Msk, Out):
                # Q broadcast once (Core A holds it across all scores m-tiles).
                npu_dma_memcpy_nd(metadata=inQ, bd_id=4, mem=Q,
                                  sizes=[1, 1, 1, K_qk], strides=[0, 0, 0, 1])
                # K weights streamed as m-tiles (K_div_k=1: head_dim whole).
                npu_dma_memcpy_nd(metadata=memK, bd_id=3, mem=K,
                                  sizes=[qk_tiles, 1, m, K_qk],
                                  strides=[m * K_qk, K_qk, K_qk, 1])
                # mask (full n_kv, all-zeros for decode).
                npu_dma_memcpy_nd(metadata=inMask, bd_id=2, mem=Msk,
                                  sizes=[1, 1, 1, n_kv], strides=[0, 0, 0, 1])
                # V weights (transposed Vt[head_dim,n_kv]) streamed as m-tiles.
                npu_dma_memcpy_nd(metadata=memV, bd_id=1, mem=V,
                                  sizes=[sv_tiles, 1, m, K_sv],
                                  strides=[m * K_sv, K_sv, K_sv, 1])
                # out[head_dim] gathered from Core C's m-tiles.
                npu_dma_memcpy_nd(metadata=outO, bd_id=0, mem=Out,
                                  sizes=[1, 1, 1, M_sv], strides=[0, 0, 0, 1])
                dma_wait(outO)

        print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt chained decode-attention overlay")
    p.add_argument("-d", "--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--n_kv", type=int, default=512, help="KV cache length bucket")
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("-m", type=int, default=32, help="gemv M tile")
    o, _ = p.parse_known_args()
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu) for this chained overlay")
    chained(o.dev, o.n_kv, o.head_dim, o.m)
