# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on kernels/attn_softmax.py (single-core 2-in-1-out streaming) and
# programming_examples/ml/eltwise_add/eltwise_add.py (Apache-2.0 WITH
# LLVM-exception).
#
# ggml-xrt RESIDUAL-ADD (single compute tile, one column).
# Streams two f32 vectors a[N] + b[N] in and one f32 vector c[N] out, computing
# on ONE core:
#
#   c = a + b   (pure elementwise, f32; the residual stream is f32)
#
# Used for the two per-layer residual adds (x += attn_out, x += ffn_out) that
# keep the residual stream resident on the NPU as part of the q..o block.
#
# N is n_embd (Qwen3-1.7B = 2048) and must match the kernel's -DDIM_N and be a
# multiple of the v16 vector width (16).
import argparse

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1


def eltwise_add(dev, N):
    dtype = np.float32
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    tile_ty = np.ndarray[(N,), np.dtype[dtype]]

    # Data movement: two inputs (a, b) in, one output (c) out.
    of_a = ObjectFifo(tile_ty, name="a")
    of_b = ObjectFifo(tile_ty, name="b")
    of_out = ObjectFifo(tile_ty, name="c")

    # AIE core function: DIM_N baked at compile time.
    add_f32 = Kernel("add_f32", "eltwise_add.o", [tile_ty, tile_ty, tile_ty])

    def core_body(in_a, in_b, out_c, kernel):
        # One shot: whole vector is one tile.
        elem_a = in_a.acquire(1)
        elem_b = in_b.acquire(1)
        elem_o = out_c.acquire(1)
        kernel(elem_a, elem_b, elem_o)
        in_a.release(1)
        in_b.release(1)
        out_c.release(1)

    worker = Worker(
        core_body,
        [of_a.cons(), of_b.cons(), of_out.prod(), add_f32],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_a.prod(), A)
        rt.fill(of_b.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt residual eltwise add")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-N", type=int, required=True, help="n_embd (multiple of 16)")
    a, _ = p.parse_known_args()
    if a.N % 16 != 0:
        raise ValueError(f"N ({a.N}) must be a multiple of VEC_LEN (16)")
    dev = NPU1Col1() if a.dev == "npu" else NPU2Col1()
    print(eltwise_add(dev, a.N))
