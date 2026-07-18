# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on programming_examples/ml/eltwise_add/eltwise_add.py and
# programming_examples/ml/softmax/softmax.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt DECODE-ATTENTION soft_max_ext (single compute tile, one column).
# Streams the attention score vector scores[N] and an additive mask[N] in, and
# streams probs[N] (bf16) out, computing on ONE core:
#
#   probs = softmax(scores*scale + mask)   (scale = 1/sqrt(head_dim), baked in
#                                            the kernel via -DSCALE)
#
# softmax is a reduction over the WHOLE vector (max then sum), so - unlike the
# stock batched softmax example that chops N into independent 1024 rows - here
# the tile IS the full (padded) vector: one core, one kernel call over N.
#
# PAD REQUIREMENT: N is the padded n_kv and must match the kernel's -DDIM_N and
# be a multiple of SM_VEC_LEN (16). Caller pads tail lanes of scores/mask with a
# large-negative sentinel (see softmax_ext.cc).
import argparse

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2Col1


def attn_softmax(dev, N):
    dtype = bfloat16
    tensor_ty = np.ndarray[(N,), np.dtype[dtype]]
    tile_ty = np.ndarray[(N,), np.dtype[dtype]]

    # Data movement: two inputs (scores, mask) in, one output (probs) out.
    of_scores = ObjectFifo(tile_ty, name="scores")
    of_mask = ObjectFifo(tile_ty, name="mask")
    of_out = ObjectFifo(tile_ty, name="probs")

    # AIE core function: scale + mask baked/streamed; DIM_N baked at compile.
    softmax_ext = Kernel(
        "softmax_ext_bf16", "softmax_ext.o", [tile_ty, tile_ty, tile_ty]
    )

    def core_body(in_scores, in_mask, out_probs, kernel):
        # One shot: whole (padded) vector is one tile.
        elem_s = in_scores.acquire(1)
        elem_m = in_mask.acquire(1)
        elem_o = out_probs.acquire(1)
        kernel(elem_s, elem_m, elem_o)
        in_scores.release(1)
        in_mask.release(1)
        out_probs.release(1)

    worker = Worker(
        core_body,
        [of_scores.cons(), of_mask.cons(), of_out.prod(), softmax_ext],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (S, M, O):
        rt.start(worker)
        rt.fill(of_scores.prod(), S)
        rt.fill(of_mask.prod(), M)
        rt.drain(of_out.cons(), O, wait=True)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt decode-attention soft_max_ext")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-N", type=int, required=True, help="padded n_kv (multiple of 16)")
    a, _ = p.parse_known_args()
    if a.N % 16 != 0:
        raise ValueError(f"N ({a.N}) must be a multiple of SM_VEC_LEN (16)")
    dev = NPU1Col1() if a.dev == "npu" else NPU2Col1()
    print(attn_softmax(dev, a.N))
