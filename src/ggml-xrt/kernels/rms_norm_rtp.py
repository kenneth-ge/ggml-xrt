# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# RMS-norm, aie2/Phoenix, restructured for OVERLAY+ELF collapse:
#   * FIXED max L1 buffer (LMAX bf16 elements) -> L1 allocation is shape-independent.
#   * `cols` (the real row length) is a RUNTIME PARAMETER (RTP buffer written by the
#     runtime sequence via inline_ops), NOT an arith.constant baked in the core.
#   * the core loops range_(0xFFFFFFFF) (gemv pattern) so the ROW COUNT (seq) is not
#     baked either; the runtime DMA decides how many rows actually flow.
# Result: the AIE array config + core program (= the overlay) is IDENTICAL for every
# (seq, cols); only the runtime_sequence (DMA counts + the RTP-write value) differs, so
# every shape collapses to ONE overlay + a cheap instruction ELF, like silu.
#
# HOST ABI: single-core (npu1_1col). Each row is a fixed LMAX-element object, so the
# host uploads rows padded to LMAX (real data in [0:cols]); the kernel reduces over the
# first `cols` only, so padding is ignored and the mean/divisor stay correct.
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


def rmsnorm_rtp(dev, seq, cols, lmax):
    assert cols <= lmax, f"cols {cols} must be <= lmax {lmax}"
    tensor_ty = np.ndarray[(seq * lmax,), np.dtype[bfloat16]]   # DDR: seq rows padded to lmax
    row_ty = np.ndarray[(lmax,), np.dtype[bfloat16]]            # fixed L1 object

    of_in = ObjectFifo(row_ty, name="in0")
    of_out = ObjectFifo(row_ty, name="out0")

    rms_kernel = Kernel("rms_norm", "rms_norm.o", [row_ty, row_ty, np.int32])

    # initial_value is a fixed placeholder (0) so the compile-time device config is
    # cols-INDEPENDENT; the real `cols` is written at runtime by inline_ops below (that
    # write lives in the runtime_sequence -> the ELF, not the overlay).
    rtp = Buffer(
        np.ndarray[(1,), np.dtype[np.int32]],
        name="rtp_cols",
        initial_value=np.array([0], dtype=np.int32),
        use_write_rtp=True,
    )
    barrier = WorkerRuntimeBarrier()

    def core_fn(of_in, of_out, kernel, my_rtp, bar):
        bar.wait_for_value(1)
        c = my_rtp[0]
        for _ in range_(0xFFFFFFFF):
            ei = of_in.acquire(1)
            eo = of_out.acquire(1)
            kernel(ei, eo, c)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_fn,
        fn_args=[of_in.cons(), of_out.prod(), rms_kernel, rtp, barrier],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(worker)

        def set_rtp(r):
            r[0] = cols

        rt.inline_ops(set_rtp, [rtp])
        rt.set_barrier(barrier, 1)

        tg = rt.task_group()
        rt.fill(of_in.prod(), a_in, task_group=tg)
        rt.drain(of_out.cons(), c_out, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dev", default="npu")
    p.add_argument("-s", "--seq", type=int, required=True, help="number of rows")
    p.add_argument("-e", "--cols", type=int, required=True, help="real row length (embedding_dim)")
    p.add_argument("-L", "--lmax", type=int, default=6144, help="fixed max L1 row buffer")
    o = p.parse_args(sys.argv[1:])
    if o.dev != "npu":
        raise ValueError("aie2/Phoenix only (npu)")
    print(rmsnorm_rtp(NPU1Col1(), o.seq, o.cols, o.lmax))
