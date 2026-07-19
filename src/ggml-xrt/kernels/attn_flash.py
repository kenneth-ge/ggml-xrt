# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# FLASH-ATTN-EXT-TARGETING on-chip-CHAINED decode-attention overlay.
#
# This is attn_chain.py re-pinned to ggml's FUSED GGML_OP_FLASH_ATTN_EXT node
# I/O so the ggml-xrt backend can claim that ONE op directly (keeping the graph's
# flash-attn ON -> fast prefill), instead of falling back to the -fa-off 3-op
# chain (soft_max/mul_mat/mul_mat). Same three chained cores + same on-chip
# QK^T->softmax->scores*V pipeline (ONE dispatch, scores/probs never touch L3);
# the L3 DMA layouts + softmax + scores*V kernel are adapted to the flash
# contract. Qwen3-1.7B decode: FULL-CAUSAL, bf16 KV, NO mask.
#
# ---- ggml_flash_attn_ext CONTRACT (verified against ggml.h + ggml.c +
#      ggml-cpu/ops.cpp:ggml_compute_forward_flash_attn_ext_f16_one_chunk) ------
#   q   : ne=[head_dim, n_batch, n_head,    ne3]  contiguous dim = head_dim
#   k   : ne=[head_dim, n_kv,    n_head_kv, ne3]  contiguous dim = head_dim
#   v   : ne=[head_dim, n_kv,    n_head_kv, ne3]  contiguous dim = head_dim  !! NOT transposed !!
#   mask: ne=[n_kv,     n_batch, ne32,      ne33] type = F16 (HARD ASSERT), contiguous
#   out : ne=[head_dim, n_head,  n_batch,   ne3]  type = F32,  permute(0,2,1,3)
#   params (op_params f32[0..2]): scale, max_bias, logit_softcap
#   op_params i32[3] = prec (GGML_PREC_DEFAULT/F32; both -> F32 accumulate)
#   GQA broadcast: n_head % n_head_kv == 0 ; Q head h uses KV head h//(n_head/n_head_kv)
#   DECODE: n_batch (=neq1=N) = 1.
#
# ---- WHAT THIS IMPLEMENTS (Qwen3-1.7B decode) ------------------------------
#   scale = 1/sqrt(head_dim) baked into softmax (-DSCALE). max_bias = 0 (no ALiBi).
#   logit_softcap = 0 (Qwen3 has none -> TODO fold s=softcap*tanh(s/softcap) if a
#   softcap model is targeted). No sinks. FULL-CAUSAL -> NO mask (see below).
#
# ---- NO MASK (full-causal) + INTERNAL PADDING ------------------------------
#   Qwen3-1.7B decode is full-causal (LLAMA_SWA_TYPE_NONE): the flash kq_mask is
#   all-0 for every valid past position, so it adds nothing -- and ggml
#   HARD-forces that mask to F16 (aie2 has NO native f16). So this overlay reads
#   NO mask at all (no mask fifo / no mask DMA) -> the f16 problem is gone.
#   The mask's OTHER role (padding the bucket tail) is handled IN-KERNEL: the
#   overlay is built for a fixed bucket n_kv (512/1024/2048) >= the actual n_kv;
#   n_valid is passed as a runtime int32 scalar (nv[0]) and the no-mask softmax
#   writes a large-negative sentinel into scores[n_valid:bucket) before the
#   softmax, so padded KV lanes contribute 0. Any bucket >= n_valid is correct.
#
# ---- LAYOUT MAPPING q/k/v/out -> the cores (NO TRANSPOSE ON K OR V) ---------
#   Q   : flash q decode = [n_head, head_dim] head-major, head_dim contiguous
#         -> feeds inQ directly (head rides BD dim3, elem dim0). NO transpose.
#   K   : flash k per KV head is a CONTIGUOUS [n_kv, head_dim] block (element
#         (d,j) at j*head_dim + d, i.e. each n_kv row is a contiguous head_dim
#         vector). The QK^T gemv wants exactly that tile (row = n_kv index j,
#         col = head_dim d, d contiguous). So flash K feeds memK DIRECTLY (same
#         strides attn_chain used). NO transpose.  [The original spec guessed K
#         needs a transpose -- it does NOT; that was an ne0-contiguous notation
#         confusion. In linear memory ggml ne=[head_dim,n_kv] IS row-major
#         [n_kv, head_dim].]
#   V   : flash v per KV head has element (d,j) at j*head_dim + d (head_dim
#         contiguous) -- each n_kv position j owns a contiguous head_dim vector.
#         The row-major matvec (mv.cc) would need V n_kv-contiguous (a bf16
#         element transpose), which the shim DMA REJECTS (a 1-element bf16 stride
#         = 2 bytes is not divisible by 4). So instead Core C uses a COLUMN-
#         ACCUMULATE kernel (mv_vt.cc) that reads V in its NATIVE head_dim-
#         contiguous layout and does out[d] += sum_j probs[j]*V(d,j). Result:
#         V is ZERO-COPY with NO transpose either.  [The spec guessed V matches
#         v_trans / needs no transpose for the OLD kernel -- with the old row-
#         major kernel V would have needed the (illegal) transpose; the fix is
#         the new native-layout kernel here.]
#   out : flash out decode = [n_head, head_dim] head-major, head_dim contiguous
#         (permute(0,2,1,3), n_batch=1) -> matches outO. NO change.
#
# The 4-wave head drive (shim DMA task queue ~4-deep) is UNCHANGED.
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU. Windows validates on HW.
import argparse

import numpy as np
from ml_dtypes import bfloat16
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_


def flash(dev, n_kv, head_dim, m, kc, n_head, n_head_kv):
    dtype_in = np.dtype[bfloat16]
    dtype_out = np.dtype[np.float32]
    dtype_nv = np.dtype[np.int32]

    # QK^T   : M = n_kv (tiled by m), K = head_dim (whole).
    # scores*V: out = head_dim (WHOLE), contraction = n_kv (K-CHUNKED by kc rows).
    assert n_kv % m == 0, "n_kv must be divisible by m (QK^T output tiling)"
    assert n_kv % 16 == 0, "n_kv must be a multiple of SM_VEC_LEN (16) for softmax"
    assert n_kv % kc == 0, "n_kv must be divisible by kc (scores*V n_kv chunking)"
    assert n_head % n_head_kv == 0, "n_head must be divisible by n_head_kv (GQA group)"
    gqa = n_head // n_head_kv  # Q heads sharing one KV head (GQA group size)

    M_qk = n_kv          # QK^T output rows
    K_qk = head_dim      # QK^T contraction (whole)

    qk_tiles = M_qk // m   # number of scores m-tiles Core A produces
    kc_chunks = n_kv // kc  # number of n_kv chunks Core C accumulates over

    # ---- head WAVES (shim DMA task-queue overflow guard) --------------------
    # K/V are one BD per (KV-)head (their gather uses all 4 descriptor dims, so
    # the head dim can't ride a BD repeat dim); the AIE2 (Phoenix) shim DMA task
    # queue is only ~4 deep. Drive the pipe in WAVES of wkv KV-heads (W = wkv*gqa
    # Q-heads); dma_wait(out) drains each wave and frees the BD ids. ONE dispatch.
    Q_SAFE = 4               # AIE2 shim DMA task-queue depth (BDs per channel)
    wkv = max(1, min(n_head_kv, Q_SAFE // gqa))   # KV-heads per wave
    while n_head_kv % wkv != 0:                    # keep waves evenly sized
        wkv -= 1
    W = wkv * gqa            # Q-heads per wave
    n_waves = n_head_kv // wkv
    assert W <= Q_SAFE or gqa > Q_SAFE, "wave head count must fit the task queue"

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
            probs_kc_ty = np.ndarray[(kc,), dtype_in]         # one probs kc-chunk (bf16)
            memV_ty = np.ndarray[(kc * head_dim,), dtype_in]  # V chunk tile (flat, L3->L2)
            V_ty = np.ndarray[(kc, head_dim), dtype_in]       # V chunk tile (KC rows n_kv, HD cols) native
            out_full_ty = np.ndarray[(head_dim,), dtype_out]  # whole out[head_dim] (f32)
            nv_ty = np.ndarray[(1,), dtype_nv]                # n_valid runtime scalar

            # ---- external kernel funcs ---------------------------------------
            # Core A: bf16-OUT matvec (float accumulate) so scores stay bf16.
            zero_bf16 = external_func("zero_scalar_bf16", inputs=[score_tile_ty])
            matvec_qk = external_func(
                "matvec_scalar_bf16_bf16", inputs=[K_ty, Q_ty, score_tile_ty]
            )
            # Core B: NO-MASK soft_max_ext (scale baked, DIM_N=bucket baked); n_valid
            # scalar (nv[0]) -> sentinel-pad scores[n_valid:bucket). NO mask read.
            softmax_fn = external_func(
                "softmax_ext_nomask_bf16", inputs=[vec_ty, vec_ty, nv_ty]
            )
            # Core C: ZERO-COPY column-accumulate scores*V (native flash V layout,
            # NO transpose). out[head_dim] += sum_j probs[j]*V(d,j), K-chunked (kc).
            zero_hd = external_func("zero_scalar_f32_hd", inputs=[out_full_ty])
            matvec_sv = external_func(
                "matvec_vt_scalar_bf16_f32", inputs=[V_ty, probs_kc_ty, out_full_ty]
            )

            # ---- tiles: one column per op ------------------------------------
            shim_a = tile(0, 0); mem_a = tile(0, 1); ct_a = tile(0, 2)   # QK^T
            shim_b = tile(1, 0); mem_sc = tile(1, 1); ct_b = tile(1, 2)  # softmax + scores agg
            shim_c = tile(2, 0); mem_c = tile(2, 1); ct_c = tile(2, 2)   # scores*V
            # V's L3->L2 uses column-3's (otherwise-unused) shim so its per-head BD
            # loop (up to n_head BDs) does not collide with out's BD on shim_c.
            shim_v = tile(3, 0)

            # ---- Core A inputs: Q (broadcast) + K weights (L3->L2->core) ------
            inQ = object_fifo("inQ", shim_a, ct_a, 2, Q_ty)
            memK = object_fifo("memK", shim_a, mem_a, 2, memK_ty)
            inK = object_fifo("inK", mem_a, ct_a, 2, K_ty)
            object_fifo_link(memK, inK)

            # ---- CHAIN A->B: scores m-tiles aggregate into full n_kv ---------
            scores_a = object_fifo("scores_a", ct_a, mem_sc, 2, score_tile_ty)
            scores_b = object_fifo("scores_b", mem_sc, ct_b, 2, vec_ty)
            object_fifo_link(scores_a, scores_b)

            # ---- Core B input: n_valid runtime scalar (no mask) --------------
            inNV = object_fifo("inNV", shim_b, ct_b, 2, nv_ty)

            # ---- CHAIN B->C: whole probs[n_kv] -> col2 memtile -> kc-tiles ----
            probs_bc = object_fifo("probs_bc", ct_b, mem_c, 2, vec_ty)
            probs_c = object_fifo("probs_c", mem_c, ct_c, 2, probs_kc_ty)
            object_fifo_link(probs_bc, probs_c)

            # ---- Core C input: V chunk tiles (L3->L2->core) + output ----------
            memV = object_fifo("memV", shim_v, mem_c, 2, memV_ty)
            inV = object_fifo("inV", mem_c, ct_c, 2, V_ty)
            object_fifo_link(memV, inV)
            outO = object_fifo("outO", ct_c, shim_c, 2, out_full_ty)

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

            @core(ct_b, "softmax_ext_nomask.o", stack_size=0x2000)
            def core_b():
                for _ in range_(0xFFFFFFFF):
                    es = scores_b.acquire(ObjectFifoPort.Consume, 1)  # full n_kv
                    env = inNV.acquire(ObjectFifoPort.Consume, 1)     # n_valid scalar
                    ep = probs_bc.acquire(ObjectFifoPort.Produce, 1)  # full n_kv
                    softmax_fn(es, ep, env)
                    scores_b.release(ObjectFifoPort.Consume, 1)
                    inNV.release(ObjectFifoPort.Consume, 1)
                    probs_bc.release(ObjectFifoPort.Produce, 1)

            @core(ct_c, "mv_vt.o", stack_size=0x2000)
            def core_c():
                for _ in range_(0xFFFFFFFF):
                    eo = outO.acquire(ObjectFifoPort.Produce, 1)  # whole head_dim
                    zero_hd(eo)
                    for _ in range_(kc_chunks):
                        ep = probs_c.acquire(ObjectFifoPort.Consume, 1)  # kc probs
                        eV = inV.acquire(ObjectFifoPort.Consume, 1)      # (kc, head_dim) native
                        matvec_sv(eV, ep, eo)
                        inV.release(ObjectFifoPort.Consume, 1)
                        probs_c.release(ObjectFifoPort.Consume, 1)
                    outO.release(ObjectFifoPort.Produce, 1)

            # ================= ONE runtime_sequence (ALL heads) ===============
            # Inputs cross L3 in the FLASH layout: q[n_head,head_dim] (head-major),
            # k[n_head_kv,n_kv,head_dim] (head_dim contiguous, NO transpose),
            # v[n_head_kv,n_kv,head_dim] (head_dim contiguous, NON-transposed flash
            # V, NO transpose), n_valid (int32 scalar). Output: out[n_head,head_dim]
            # f32. scores/probs stay on-chip at every head. NO mask.
            @runtime_sequence(
                np.ndarray[(n_head_kv * n_kv * head_dim,), dtype_in],  # K (flash: [n_head_kv,n_kv,head_dim])
                np.ndarray[(n_head * head_dim,), dtype_in],            # Q (flash: [n_head,head_dim])
                np.ndarray[(n_head_kv * n_kv * head_dim,), dtype_in],  # V (flash: [n_head_kv,n_kv,head_dim] NON-transposed)
                np.ndarray[(1,), dtype_nv],                            # n_valid (runtime scalar)
                np.ndarray[(n_head * head_dim,), dtype_out],           # out (flash: [n_head,head_dim] f32)
            )
            def sequence(K, Q, V, NV, Out):
                # Heads fed in n_waves waves of W = wkv*gqa Q-heads (see wave note).
                # Q/nv/out ride the BD iteration/repeat dim (ONE push per wave);
                # K/V are one BD per (KV-)head. dma_wait(out) drains each wave.
                for w in range(n_waves):
                    h0 = w * W            # first Q head of this wave
                    kv0 = w * wkv         # first KV head of this wave
                    # Q: W vectors; head rides d3 (+head_dim), elem d0 (+1). flash q
                    # decode = [n_head, head_dim] head-major, head_dim contiguous.
                    npu_dma_memcpy_nd(metadata=inQ, bd_id=0, mem=Q,
                                      offsets=[0, 0, 0, h0 * K_qk],
                                      sizes=[W, 1, 1, K_qk],
                                      strides=[K_qk, 0, 0, 1])
                    # K: NO transpose. flash k per KV head is a contiguous
                    # [n_kv, head_dim] block (element (j,d) at j*head_dim + d), so
                    # the QK^T tile (row=n_kv +head_dim, col=head_dim +1) reads it
                    # directly. GQA replay rides d3 (stride 0, size gqa).
                    #   d3 (gqa replica): +0 ; d2 (m-tile t): +m*head_dim ;
                    #   d1 (row j): +head_dim ; d0 (col d): +1 ; offset = j*n_kv*head_dim.
                    for kk in range(wkv):
                        j = kv0 + kk
                        npu_dma_memcpy_nd(metadata=memK, bd_id=1 + kk, mem=K,
                                          offsets=[0, 0, 0, j * n_kv * head_dim],
                                          sizes=[gqa, qk_tiles, m, K_qk],
                                          strides=[0, m * K_qk, K_qk, 1])
                    # n_valid: 1 int32, SHARED -> replayed W times via a stride-0
                    # OUTERMOST dim (Core B re-acquires it per head).
                    npu_dma_memcpy_nd(metadata=inNV, bd_id=0, mem=NV,
                                      sizes=[W, 1, 1, 1],
                                      strides=[0, 0, 0, 0])
                    # V: NO transpose (native flash layout). flash v per KV head has
                    # element (d,j) at j*head_dim + d (head_dim contiguous). Core C's
                    # column-accumulate kernel (mv_vt.o) wants a (kc rows of n_kv,
                    # head_dim cols) row-major tile V[jj*head_dim + d] -- which is
                    # EXACTLY the native flash slab. Innermost dim d0 = head_dim
                    # (stride 1, contiguous); all outer strides are multiples of
                    # head_dim (>=256 bytes) so the DMA is legal (no 2-byte stride).
                    #   d3 (chunk c'): +kc*head_dim ; d1 (row jj): +head_dim ;
                    #   d0 (col d): +1 ; offset = (p//gqa)*n_kv*head_dim.
                    for pp in range(W):
                        p = h0 + pp
                        npu_dma_memcpy_nd(metadata=memV, bd_id=pp, mem=V,
                                          offsets=[0, 0, 0, (p // gqa) * n_kv * head_dim],
                                          sizes=[kc_chunks, 1, kc, head_dim],
                                          strides=[kc * head_dim, 0, head_dim, 1])
                    # out: head_dim per head (head rides d3), gathered from Core C.
                    #   d3 (head): +head_dim ; d0: +1 ; offset = h0*head_dim.
                    npu_dma_memcpy_nd(metadata=outO, bd_id=0, mem=Out,
                                      offsets=[0, 0, 0, h0 * head_dim],
                                      sizes=[W, 1, 1, head_dim],
                                      strides=[head_dim, 0, 0, 1])
                    # Drain this wave (out written) -> frees the BD ids for reuse.
                    dma_wait(outO)

        print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt flash_attn_ext decode-attention overlay")
    p.add_argument("-d", "--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--n_kv", type=int, default=512, help="KV cache length bucket")
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("-m", type=int, default=32, help="gemv M tile (QK^T output)")
    p.add_argument("--kc", type=int, default=64,
                   help="scores*V n_kv chunk (rows of V per Core C call; V tile = kc*head_dim)")
    p.add_argument("--n_head", type=int, default=16,
                   help="number of Q heads processed in ONE dispatch (1 = single head)")
    p.add_argument("--n_head_kv", type=int, default=8,
                   help="number of KV heads (GQA: Q head h uses KV head h//(n_head/n_head_kv))")
    o, _ = p.parse_known_args()
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu) for this flash overlay")
    flash(o.dev, o.n_kv, o.head_dim, o.m, o.kc, o.n_head, o.n_head_kv)
