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
#       (bf16->f32 matvec, M=head_dim tiled by m, K=n_kv K-CHUNKED by KC). Core C
#       loops n_kv/KC k-chunks, accumulating out[head_dim] over the chunks. Per
#       chunk it loads V (m, KC)=32*256*2=16KB (double-bufferable=32KB, fits L1)
#       and one KC-slice of probs; V streamed from L3, out[head_dim] DMA'd out.
#
# WHY K-CHUNK CORE C: Core C's V tile was (m, n_kv) = 32*n_kv*2 bytes -> 32KB at
# n_kv=512 (single-buffered to fit) and 64KB at n_kv=1024 which OVERFLOWS L1.
# Tiling the n_kv contraction into KC=256 chunks caps the V tile at 32*256*2=16KB
# regardless of n_kv, so the same overlay builds at 512/1024/2048. The out[m]
# accumulator (matvec c[row]+=) is zeroed once per m-tile then summed across the
# n_kv/KC chunks. Because the acquired-objectfifo memref only supports scalar
# indexing (no pointer math to sub-slice probs in the core), probs is delivered
# to Core C in KC-tiles via a memtile size-ratio link (B -> mem_c(whole probs) ->
# C(KC-tiles)), mirroring how gemv_mc16 distributes its weight k-tiles.
#
# LOOP NEST: Core C runs k-chunks OUTER, output m-tiles INNER, holding all
# head_dim/m out accumulators live at once (multi-acquire). This lets the WHOLE
# probs[n_kv] be consumed exactly ONCE per token (each KC probs-chunk is reused
# across all m-tiles inside its acquire) -- no probs replay needed, so Core B
# still produces probs just once.
#
# THE SOFTMAX BARRIER: soft_max reduces over the ENTIRE n_kv, so B needs ALL
# scores before it starts and C needs ALL probs. The chain fifos therefore carry
# the FULL n_kv vector:
#   * A->B: a size-ratio object_fifo_link (Core A releases M/m tiles of `m` bf16;
#     the col1 memtile aggregates them into ONE n_kv-element the softmax acquires
#     whole). Verified to lower via --aie-objectFifo-stateful-transform.
#   * B->C: Core B produces the WHOLE probs[n_kv] to the col2 memtile; a
#     size-ratio object_fifo_link SPLITS it into n_kv/KC KC-tiles for Core C
#     (the inverse of the A->B aggregate). probs NEVER touch L3.
#
# ONE runtime_sequence drives it: DMA K,Q,V,mask in; dma_wait on out. Only K,Q,V,
# mask,out cross L3; scores and probs stay on-chip (A->B->C via fifos).
#
# MULTI-HEAD (n_head heads, GQA n_head:n_head_kv) IN ONE DISPATCH: the three cores
# keep their infinite loops -- each iteration already does exactly ONE head's work,
# so we simply feed n_head head-iterations through the SAME fifos (scores/probs stay
# on-chip, reused per head). The head loop lives in the runtime_sequence's DMA
# descriptors, NOT in the cores:
#   * Q  : Q[n_head, head_dim]      -> one nd-descriptor, head as outer dim0.
#   * K  : K[n_head_kv, n_kv, head_dim] -> one nd-descriptor. GQA is folded in as a
#          [n_head_kv, gqa(stride 0), ...] split: Q head h reads KV head h//gqa, so
#          each KV head's contiguous K block is replayed `gqa` times (stride-0 rep).
#   * mask: mask[n_kv] SHARED across heads -> one nd-descriptor, n_head copies via a
#          stride-0 outer dim (Core B acquires the same mask each head iteration).
#   * out : out[n_head, head_dim]   -> one nd-descriptor, head as outer dim0.
#   * V  : Vt[n_head_kv, head_dim, n_kv] -- the ONLY input whose single-head gather
#          already uses all 4 descriptor dims (chunk,m-tile,row,col), so head+GQA
#          cannot be folded into one descriptor. V is issued as a PER-HEAD loop of
#          npu_dma_memcpy_nd (offset = (h//gqa)*head_dim*n_kv). To stay within the
#          16-BD-per-shim budget (16 V BDs would collide with out on shim col2), V's
#          L3->L2 stream is routed through the otherwise-unused COLUMN-3 shim, so V
#          owns its shim's BDs alone while Q/K/mask/out keep cols 0-2.
# Result: ONE dispatch for all n_head heads (n_head=1 reproduces the single-head
# overlay). scores/probs never touch L3 at any head.
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


def chained(dev, n_kv, head_dim, m, kc, n_head, n_head_kv):
    dtype_in = np.dtype[bfloat16]
    dtype_out = np.dtype[np.float32]

    # QK^T   : M = n_kv (tiled by m), K = head_dim (whole).
    # scores*V: M = head_dim (tiled by m), K = n_kv (K-CHUNKED by kc).
    assert n_kv % m == 0, "n_kv must be divisible by m (QK^T output tiling)"
    assert head_dim % m == 0, "head_dim must be divisible by m (scores*V tiling)"
    assert n_kv % 16 == 0, "n_kv must be a multiple of SM_VEC_LEN (16) for softmax"
    assert n_kv % kc == 0, "n_kv must be divisible by KC (scores*V k-chunking)"
    assert n_head % n_head_kv == 0, "n_head must be divisible by n_head_kv (GQA group)"
    gqa = n_head // n_head_kv  # Q heads sharing one KV head (GQA group size)

    M_qk = n_kv          # QK^T output rows
    K_qk = head_dim      # QK^T contraction (whole)
    M_sv = head_dim      # scores*V output rows
    K_sv = n_kv          # scores*V contraction (K-chunked by kc)

    qk_tiles = M_qk // m  # number of scores m-tiles Core A produces
    sv_tiles = M_sv // m  # number of out m-tiles Core C produces
    kc_chunks = K_sv // kc  # number of n_kv k-chunks Core C accumulates over

    # ---- head WAVES (shim DMA task-queue overflow guard) --------------------
    # V and K must be issued as one BD PER HEAD / PER KV-HEAD (their single-head
    # gather already uses all 4 descriptor dims, so the head dim can't ride the
    # BD's iteration/repeat dim like Q/mask/out do). Each such BD is a separate
    # push onto the shim DMA channel's TASK QUEUE, which is only ~4 deep on AIE2
    # (Phoenix): enqueuing all n_head V BDs (and n_head_kv K BDs) up front
    # OVERFLOWS the queue and silently DROPS the later heads (observed: only the
    # first ~5 heads produced, rest all-zero). Fix: drive the pipe in WAVES of
    # `wkv` KV-heads (W = wkv*gqa Q-heads) so no channel ever has more than the
    # queue depth of BDs outstanding; a dma_wait(out) between waves drains the
    # pipe and frees the BD ids to be re-pushed. Still ONE dispatch (one
    # runtime_sequence / one hw_context) -- the cores' infinite loops just see
    # n_head head-iterations fed across n_waves waves.
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
            probs_kc_ty = np.ndarray[(kc,), dtype_in]         # one probs KC-chunk (bf16)
            memV_ty = np.ndarray[(m * kc,), dtype_in]         # V k-chunk tile (flat, L3->L2)
            V_ty = np.ndarray[(m, kc), dtype_in]              # V k-chunk tile (2-D, L2->core)
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
            # Core C: bf16->f32 matvec (final output precision), K-CHUNKED (DIM_K=kc).
            # c[row]+= per chunk -> accumulates the full n_kv contraction across
            # kc_chunks calls (zero once per out m-tile, then sum the chunks).
            zero_f32 = external_func("zero_scalar_f32", inputs=[out_tile_ty])
            matvec_sv = external_func(
                "matvec_scalar_bf16_f32", inputs=[V_ty, probs_kc_ty, out_tile_ty]
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
            # Core A releases qk_tiles x (m bf16); the col1 memtile aggregates
            # them into ONE (n_kv bf16) element the softmax acquires whole.
            scores_a = object_fifo("scores_a", ct_a, mem_sc, 2, score_tile_ty)
            scores_b = object_fifo("scores_b", mem_sc, ct_b, 2, vec_ty)
            object_fifo_link(scores_a, scores_b)

            # ---- Core B input: mask (L3->core), full n_kv --------------------
            inMask = object_fifo("inMask", shim_b, ct_b, 2, vec_ty)

            # ---- CHAIN B->C: whole probs[n_kv] -> col2 memtile -> KC-tiles ----
            # Core B produces the WHOLE probs[n_kv] to mem_c; a size-ratio link
            # SPLITS it into n_kv/kc KC-tiles for Core C (inverse of the A->B
            # aggregate). Chunking via the DMA/link, NOT pointer math in the core.
            probs_bc = object_fifo("probs_bc", ct_b, mem_c, 2, vec_ty)
            probs_c = object_fifo("probs_c", mem_c, ct_c, 2, probs_kc_ty)
            object_fifo_link(probs_bc, probs_c)

            # ---- Core C input: V k-chunk tiles (L3->L2->core) + output --------
            # V tile is now (m, kc) bf16 = 32*256*2 = 16KB REGARDLESS of n_kv, so
            # inV double-buffers (depth 2 = 32KB) and Core C's L1 stays < 64KB at
            # every n_kv bucket. V streamed k-chunk-major (chunk outer, m-tile
            # inner) to match Core C's loop nest.
            memV = object_fifo("memV", shim_v, mem_c, 2, memV_ty)
            inV = object_fifo("inV", mem_c, ct_c, 2, V_ty)
            object_fifo_link(memV, inV)
            # out[head_dim] gathered from Core C's sv_tiles m-tiles (direct to
            # shim, multi-acquired so all accumulators stay live across k-chunks).
            outO = object_fifo("outO", ct_c, shim_c, sv_tiles, out_tile_ty)

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
                    ep = probs_bc.acquire(ObjectFifoPort.Produce, 1)  # full n_kv
                    softmax_fn(es, em, ep)
                    scores_b.release(ObjectFifoPort.Consume, 1)
                    inMask.release(ObjectFifoPort.Consume, 1)
                    probs_bc.release(ObjectFifoPort.Produce, 1)

            @core(ct_c, "mv_sv.o", stack_size=0x2000)
            def core_c():
                for _ in range_(0xFFFFFFFF):
                    # Hold all sv_tiles out m-tile accumulators live across the
                    # k-chunk loop; zero each once, then accumulate the chunks.
                    eo = outO.acquire(ObjectFifoPort.Produce, sv_tiles)
                    for t in range(sv_tiles):
                        zero_f32(eo[t])
                    for _ in range_(kc_chunks):
                        # one probs KC-chunk, reused across ALL out m-tiles
                        ep = probs_c.acquire(ObjectFifoPort.Consume, 1)
                        for t in range(sv_tiles):
                            eV = inV.acquire(ObjectFifoPort.Consume, 1)  # (m, kc)
                            matvec_sv(eV, ep, eo[t])
                            inV.release(ObjectFifoPort.Consume, 1)
                        probs_c.release(ObjectFifoPort.Consume, 1)
                    outO.release(ObjectFifoPort.Produce, sv_tiles)

            # ================= ONE runtime_sequence (ALL heads) ===============
            # Inputs cross L3 (all-heads): Q[n_head,head_dim], K[n_head_kv,n_kv,
            # head_dim], V[n_head_kv,head_dim,n_kv], mask[n_kv] (shared). Output:
            # out[n_head,head_dim]. scores/probs stay on-chip at every head.
            # The three cores' infinite loops each do one head's work per iteration,
            # so the DMAs simply feed n_head head-iterations through the same fifos.
            @runtime_sequence(
                np.ndarray[(n_head_kv * M_qk * K_qk,), dtype_in],  # K (all KV heads)
                np.ndarray[(n_head * K_qk,), dtype_in],            # Q (all heads)
                np.ndarray[(n_head_kv * M_sv * K_sv,), dtype_in],  # V (all KV heads, transposed)
                np.ndarray[(n_kv,), dtype_in],                     # mask (shared)
                np.ndarray[(n_head * M_sv,), dtype_out],           # out (all heads)
            )
            def sequence(K, Q, V, Msk, Out):
                # NATURAL head order: the cores process Q heads p = 0..n_head-1, and
                # head p uses KV head p//gqa (GQA). Heads are fed in n_waves WAVES of
                # W = wkv*gqa heads (wkv KV-heads) so the shim DMA task queue never
                # overflows (see the wave-size note above). Q/mask/out ride the BD
                # iteration/repeat dim (ONE push per wave each); K/V are one BD per
                # KV-head / per head within the wave. dma_wait(out) drains each wave
                # and frees the BD ids for the next wave. bd ids are per shim tile:
                #   col0: inQ=0 (ch0), memK=1..wkv (ch1) ; col1: inMask=0 ;
                #   col2: outO=0 ; col3: memV=0..W-1 -- all <= queue depth per wave.
                for w in range(n_waves):
                    h0 = w * W            # first Q head of this wave
                    kv0 = w * wkv         # first KV head of this wave
                    # Q: W vectors (head p in [h0, h0+W)); head dim rides d3.
                    #   d3 (head): +K_qk ; d0 (elem): +1 ; offset = h0*K_qk.
                    npu_dma_memcpy_nd(metadata=inQ, bd_id=0, mem=Q,
                                      offsets=[0, 0, 0, h0 * K_qk],
                                      sizes=[W, 1, 1, K_qk],
                                      strides=[K_qk, 0, 0, 1])
                    # K: one BD per KV-head in the wave. Each KV head's K block is
                    # streamed as qk_tiles m-tiles of (m, head_dim) -- object shape
                    # kept (innermost = K_qk) so no wrap dim exceeds the 1023 BD
                    # limit (flattening to m*K_qk=4096 would). GQA replay rides the
                    # OUTERMOST dim (stride 0, size gqa; the only dim where stride 0
                    # is legal) so KV head j emits heads gqa*j .. gqa*j+gqa-1 in
                    # NATURAL order.
                    #   d3 (gqa replica): +0 ; d2 (m-tile t): +m*K_qk ;
                    #   d1 (row): +K_qk ; d0: +1 ; offset = j*n_kv*head_dim.
                    for kk in range(wkv):
                        j = kv0 + kk
                        npu_dma_memcpy_nd(metadata=memK, bd_id=1 + kk, mem=K,
                                          offsets=[0, 0, 0, j * n_kv * head_dim],
                                          sizes=[gqa, qk_tiles, m, K_qk],
                                          strides=[0, m * K_qk, K_qk, 1])
                    # mask: full n_kv (zeros for decode), SHARED -> replayed W times
                    # via a stride-0 OUTERMOST dim (Core B re-acquires it per head).
                    npu_dma_memcpy_nd(metadata=inMask, bd_id=0, mem=Msk,
                                      sizes=[W, 1, 1, n_kv],
                                      strides=[0, 0, 0, 1])
                    # V (transposed Vt[n_head_kv,head_dim,n_kv]) streamed as (m, kc)
                    # k-chunk tiles, CHUNK-major (d3=chunk j') then m-tile (d2=t),
                    # matching Core C's k-chunk-outer / m-tile-inner loop nest. One
                    # BD per head in the wave (all 4 dims used by the gather). Head
                    # p's KV head is p//gqa; V rides column-3's shim.
                    #   tile(j',t) = Vt[t*m:(t+1)*m, j'*kc:(j'+1)*kc]
                    #   d3 (j'): +kc ; d2 (t): +m*n_kv ; d1 (row): +n_kv ; d0: +1
                    for pp in range(W):
                        p = h0 + pp
                        npu_dma_memcpy_nd(metadata=memV, bd_id=pp, mem=V,
                                          offsets=[0, 0, 0, (p // gqa) * M_sv * K_sv],
                                          sizes=[kc_chunks, sv_tiles, m, kc],
                                          strides=[kc, m * K_sv, K_sv, 1])
                    # out: head_dim per head (head rides d3), gathered from Core C.
                    #   d3 (head): +M_sv ; d0: +1 ; offset = h0*M_sv.
                    npu_dma_memcpy_nd(metadata=outO, bd_id=0, mem=Out,
                                      offsets=[0, 0, 0, h0 * M_sv],
                                      sizes=[W, 1, 1, M_sv],
                                      strides=[M_sv, 0, 0, 1])
                    # Drain this wave (out written) -> frees the BD ids for reuse and
                    # keeps each channel's task queue within its depth.
                    dma_wait(outO)

        print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt chained decode-attention overlay")
    p.add_argument("-d", "--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("--n_kv", type=int, default=512, help="KV cache length bucket")
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("-m", type=int, default=32, help="gemv M tile")
    p.add_argument("--kc", type=int, default=256,
                   help="scores*V n_kv k-chunk size (caps Core C V tile at m*kc)")
    p.add_argument("--n_head", type=int, default=16,
                   help="number of Q heads processed in ONE dispatch (1 = single head)")
    p.add_argument("--n_head_kv", type=int, default=8,
                   help="number of KV heads (GQA: Q head h uses KV head h//(n_head/n_head_kv))")
    o, _ = p.parse_known_args()
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu) for this chained overlay")
    chained(o.dev, o.n_kv, o.head_dim, o.m, o.kc, o.n_head, o.n_head_kv)
