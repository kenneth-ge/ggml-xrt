# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# SPATIALLY-PACKED rms_norm + rope in ONE aie.device (aie2/Phoenix) so both ops share a
# SINGLE hw_context. Low-level/placed style (like gemv.py) — the high-level IRON API
# refuses an ObjectFifo whose producer endpoint is never driven, which blocks the
# "declare both cores, drive one" per-op-ELF pattern; the placed API lets us declare all
# endpoints explicitly and simply emit DMA/RTP for only the active op.
#
#   * rms_norm core on tile(0,2) [column 0], rope core on tile(1,2) [column 1]; each has
#     its own shim DMA path, fixed max L1 buffers, RTP scalar (cols/dims) and a lock used
#     as the runtime->core "start" barrier (exactly what rms_norm_rtp.py lowers to).
#   * BOTH cores + all objectfifos + both RTP buffers + both locks are ALWAYS declared, so
#     the device/core config (= the overlay) is IDENTICAL regardless of `--op`.
#   * `--op {rms,rope}` controls ONLY the runtime_sequence: which RTP is written, which
#     lock is set, and which op's DMAs run. => that all lives in the ELF. The idle op's
#     core blocks on its (never-set) lock — harmless.
# => ONE overlay xclbin (both cores) + per-(op,shape) ELF modules. Host registers the
#    overlay once (ONE hw_context) and loads the rms ELF or the rope ELF to run either op
#    independently on that shared context.
#
# HOST ABI (per op, from manifest): rows uploaded padded to the op's LMAX (real data in
# [0:cols|dims]); kernel touches only the valid prefix.
#   rms : buffers (in, out)      -> kernel(3,0,0, IN_bo, OUT_bo)
#   rope: buffers (in, lut, out) -> kernel(3,0,0, IN_bo, LUT_bo, OUT_bo)
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU.
import argparse
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16

LMAX_RMS = 6144
LMAX_ROPE = 256


def packed(dev, op, seq, n):
    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            rms_row = np.ndarray[(LMAX_RMS,), np.dtype[bfloat16]]
            rope_row = np.ndarray[(LMAX_ROPE,), np.dtype[bfloat16]]

            shim0 = tile(0, 0)
            shim1 = tile(1, 0)
            ct_rms = tile(0, 2)
            ct_rope = tile(1, 2)

            # --- objectfifos (endpoints explicit; always declared) ---
            rms_in = object_fifo("rms_in", shim0, ct_rms, 2, rms_row)
            rms_out = object_fifo("rms_out", ct_rms, shim0, 2, rms_row)
            rope_in = object_fifo("rope_in", shim1, ct_rope, 2, rope_row)
            rope_lut = object_fifo("rope_lut", shim1, ct_rope, 2, rope_row)
            rope_out = object_fifo("rope_out", ct_rope, shim1, 2, rope_row)

            rms_fn = external_func("rms_norm", inputs=[rms_row, rms_row, np.int32])
            rope_fn = external_func("rope", inputs=[rope_row, rope_row, rope_row, np.int32])

            rtp_cols = buffer(ct_rms, T.memref(1, T.i32()), "rtp_cols", use_write_rtp=True)
            rtp_dims = buffer(ct_rope, T.memref(1, T.i32()), "rtp_dims", use_write_rtp=True)
            lock_rms = lock(ct_rms, init=0)
            lock_rope = lock(ct_rope, init=0)

            @core(ct_rms, "rms_norm.o")
            def core_rms():
                use_lock(lock_rms, LockAction.Acquire, value=1)
                c = rtp_cols[0]
                for _ in range_(0xFFFFFFFF):
                    ei = rms_in.acquire(ObjectFifoPort.Consume, 1)
                    eo = rms_out.acquire(ObjectFifoPort.Produce, 1)
                    rms_fn(ei, eo, c)
                    rms_in.release(ObjectFifoPort.Consume, 1)
                    rms_out.release(ObjectFifoPort.Produce, 1)

            @core(ct_rope, "rope.o")
            def core_rope():
                use_lock(lock_rope, LockAction.Acquire, value=1)
                d = rtp_dims[0]
                for _ in range_(0xFFFFFFFF):
                    ei = rope_in.acquire(ObjectFifoPort.Consume, 1)
                    el = rope_lut.acquire(ObjectFifoPort.Consume, 1)
                    eo = rope_out.acquire(ObjectFifoPort.Produce, 1)
                    rope_fn(ei, el, eo, d)
                    rope_in.release(ObjectFifoPort.Consume, 1)
                    rope_lut.release(ObjectFifoPort.Consume, 1)
                    rope_out.release(ObjectFifoPort.Produce, 1)

            if op == "rms":
                @runtime_sequence(np.ndarray[(seq * LMAX_RMS,), np.dtype[bfloat16]],
                                  np.ndarray[(seq * LMAX_RMS,), np.dtype[bfloat16]])
                def seq_fn(A, C):
                    rtp_cols[0] = n
                    set_lock(lock_rms, 1)
                    npu_dma_memcpy_nd(metadata=rms_in, bd_id=1, mem=A,
                                      sizes=[seq, 1, 1, LMAX_RMS], strides=[LMAX_RMS, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=rms_out, bd_id=0, mem=C,
                                      sizes=[seq, 1, 1, LMAX_RMS], strides=[LMAX_RMS, 0, 0, 1])
                    dma_wait(rms_out)
            else:
                @runtime_sequence(np.ndarray[(seq * LMAX_ROPE,), np.dtype[bfloat16]],
                                  np.ndarray[(seq * LMAX_ROPE,), np.dtype[bfloat16]],
                                  np.ndarray[(seq * LMAX_ROPE,), np.dtype[bfloat16]])
                def seq_fn(A, B, C):
                    rtp_dims[0] = n
                    set_lock(lock_rope, 1)
                    npu_dma_memcpy_nd(metadata=rope_in, bd_id=2, mem=A,
                                      sizes=[seq, 1, 1, LMAX_ROPE], strides=[LMAX_ROPE, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=rope_lut, bd_id=1, mem=B,
                                      sizes=[seq, 1, 1, LMAX_ROPE], strides=[LMAX_ROPE, 0, 0, 1])
                    npu_dma_memcpy_nd(metadata=rope_out, bd_id=0, mem=C,
                                      sizes=[seq, 1, 1, LMAX_ROPE], strides=[LMAX_ROPE, 0, 0, 1])
                    dma_wait(rope_out)

        print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu")
    p.add_argument("--op", required=True, choices=["rms", "rope"])
    p.add_argument("-s", "--seq", type=int, required=True)
    p.add_argument("-n", "--n", type=int, required=True, help="rms cols / rope head_dim")
    o = p.parse_args()
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu)")
    packed(o.dev, o.op, o.seq, o.n)
