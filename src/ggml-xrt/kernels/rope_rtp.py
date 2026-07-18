# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# RoPE (GPT-J / adjacent-pair, cos/sin LUT), aie2/Phoenix, restructured for OVERLAY+ELF
# collapse — same technique as rms_norm_rtp.py:
#   * FIXED max L1 buffers (LMAX bf16) for in / lut / out  -> L1 alloc shape-independent.
#   * `dims` (real head_dim) is a RUNTIME PARAMETER (RTP), not a baked arith.constant.
#   * core loops range_(0xFFFFFFFF) so the ROW COUNT (seq) isn't baked.
# => AIE array config + core program (= the overlay) is IDENTICAL for every (seq, dims);
#    only the runtime_sequence (DMA counts + the RTP-write value) differs. One overlay +
#    cheap per-shape ELFs, like silu / rms_norm_rtp.
#
# HOST ABI: single-core (npu1_1col). in/lut/out rows are fixed LMAX objects; the host
# uploads rows padded to LMAX (real data [0:dims]); the kernel touches only [0:dims].
# rope.cc is GPT-J (adjacent pairs) taking a full cos/sin LUT (see wishlist §4); NEOX is
# handled host-side by permutation, unchanged.
#
# UNVALIDATED: compiled on Linux/WSL, NOT executed on NPU.

import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer, WorkerRuntimeBarrier
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1
from aie.iron.controlflow import range_
from ml_dtypes import bfloat16


def rope_rtp(dev, seq, dims, lmax):
    assert dims <= lmax, f"dims {dims} must be <= lmax {lmax}"
    tensor_ty = np.ndarray[(seq * lmax,), np.dtype[bfloat16]]
    row_ty = np.ndarray[(lmax,), np.dtype[bfloat16]]

    of_in = ObjectFifo(row_ty, name="in0")
    of_lut = ObjectFifo(row_ty, name="lut0")
    of_out = ObjectFifo(row_ty, name="out0")

    rope_kernel = Kernel("rope", "rope.o", [row_ty, row_ty, row_ty, np.int32])

    rtp = Buffer(
        np.ndarray[(1,), np.dtype[np.int32]],
        name="rtp_dims",
        initial_value=np.array([0], dtype=np.int32),  # placeholder; real value via inline_ops
        use_write_rtp=True,
    )
    barrier = WorkerRuntimeBarrier()

    def core_fn(of_in, of_lut, of_out, kernel, my_rtp, bar):
        bar.wait_for_value(1)
        d = my_rtp[0]
        for _ in range_(0xFFFFFFFF):
            ei = of_in.acquire(1)
            el = of_lut.acquire(1)
            eo = of_out.acquire(1)
            kernel(ei, el, eo, d)
            of_in.release(1)
            of_lut.release(1)
            of_out.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_in.cons(), of_lut.cons(), of_out.prod(), rope_kernel, rtp, barrier],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, b_lut, c_out):
        rt.start(worker)

        def set_rtp(r):
            r[0] = dims

        rt.inline_ops(set_rtp, [rtp])
        rt.set_barrier(barrier, 1)

        tg = rt.task_group()
        rt.fill(of_in.prod(), a_in, task_group=tg)
        rt.fill(of_lut.prod(), b_lut, task_group=tg)
        rt.drain(of_out.cons(), c_out, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu")
    p.add_argument("-s", "--seq", type=int, required=True, help="number of rows")
    p.add_argument("-e", "--dims", type=int, required=True, help="real head_dim")
    p.add_argument("-L", "--lmax", type=int, default=256, help="fixed max L1 row buffer")
    o = p.parse_args(sys.argv[1:])
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu)")
    print(rope_rtp(NPU1Col1(), o.seq, o.dims, o.lmax))
