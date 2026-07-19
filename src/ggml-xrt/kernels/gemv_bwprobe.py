# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# Based on gemv_contigstream.py + stream_probe.py (Apache-2.0 WITH LLVM-exception).
#
# ggml-xrt WEIGHT-DMA BANDWIDTH PROBES that vary the DATAFLOW (not the DDR access pattern) to
# decide whether the M=1 q6k gemv's ~11 GB/s weight-DMA is a HARDWARE DRAM wall or a
# dataflow/channel-config limit with headroom.
#
# ALL variants read the FULL q6k weight of the shipping down 6144x2048 16-core gemv (10.4 MB =
# N*(K/256)*REC bytes, REC=212) from DRAM with a FULLY-CONTIGUOUS access pattern (each shim column
# fires one adjacent-stride blast — contiguity is adjacent strides, NOT a big inner dim). NO
# dequant, NO MAC: the cores only acquire/release (pure drain, like stream_probe / contigstream),
# so wall time == pure weight-DMA time. The access pattern is held FIXED (contiguous) across every
# variant; only the dataflow is swept:
#
#   --cols {1,2,4}    CHANNEL SCALING: same 10.4 MB total, split across 1/2/4 shim columns (1 shim
#                     + 1 memtile + 1 core per column). If aggregate BW scales ~linearly with cols
#                     -> channel/shim-engine limited (headroom); if flat (c1~=c2~=c4~=11) -> a shared
#                     DRAM/NoC wall saturated at one column. (Phoenix = 4 shim columns max.)
#   --nomt            MEMTILE BYPASS: DMA weight shim->core DIRECTLY (one object_fifo, no memtile
#                     object_fifo_link stage) vs the default shim->memtile->core. Faster => memtile
#                     routing is the bottleneck (dataflow fix); same => memtile isn't it.
#   --depth D         MORE IN-FLIGHT: weight object_fifo depth (default 2). depth 4/8 tests whether
#                     more outstanding DMA raises BW. Higher depth => the 2-deep fifo was starving
#                     the DMA engine (dataflow); flat => already saturated.
#
# Same 6784 B (=m*REC) object granule and same per-column contiguous blast in every case, so the
# ONLY thing that changes is the dataflow (# channels / memtile presence / in-flight depth).
import argparse

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

REC = 212
QK = 256


def factor_dims(T, nd, lim=1023):
    """Factor T into exactly nd dims (product == T), each <= lim, for a contiguous npu_dma_memcpy_nd.
    Contiguity comes from the strides chaining (each dim's stride == inner size * inner count), so
    ANY product decomposition is a single adjacent-stride blast; we only need each count <= 1023."""
    primes = []
    n = T
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.append(d)
            n //= d
        d += 1
    if n > 1:
        primes.append(n)
    primes.sort(reverse=True)
    dims = [1] * nd
    for p in primes:
        if p > lim:
            raise ValueError(f"prime factor {p} > {lim} for T={T}; cannot make contiguous BD")
        placed = False
        for i in sorted(range(nd), key=lambda i: dims[i]):
            if dims[i] * p <= lim:
                dims[i] *= p
                placed = True
                break
        if not placed:
            raise ValueError(f"cannot pack T={T} into {nd} dims each <= {lim}")
    return dims


def bwprobe(dev, K, N, m, cols, depth, nomt):
    K_div_k = K // QK
    N_col = N // cols                 # outputs per column
    assert N % cols == 0 and N_col % m == 0
    Acol = N_col * K_div_k * REC      # contiguous weight bytes per column
    nrec = Acol // REC                # REC-byte records per column (contiguous)

    # weight DMA: 3 outer dims (each <=1023) x inner REC -> one contiguous 2.6/5.2/10.4 MB blast/col
    wd = factor_dims(nrec, 3)
    w_sizes = [wd[0], wd[1], wd[2], REC]
    w_strides = [REC * wd[2] * wd[1], REC * wd[2], REC, 1]
    # output C drain: N_col f32, contiguous (split into <=1023 dims for cols=1/2 where N_col>1023)
    od = factor_dims(N_col, 3)
    c_sizes = [1, od[0], od[1], od[2]]
    c_strides = [0, od[1] * od[2], od[2], 1]

    with mlir_mod_ctx() as ctx:
        dev_ty = AIEDevice.npu1 if dev == "npu" else AIEDevice.npu2

        @device(dev_ty)
        def device_body():
            a_ty = np.ndarray[(m, REC), np.dtype[np.uint8]]
            c_ty = np.ndarray[(m,), np.dtype[np.float32]]
            zero = external_func("zero_scalar_f32", inputs=[c_ty])

            shims = [tile(c, 0) for c in range(cols)]
            mts = [tile(c, 1) for c in range(cols)]
            cores = [tile(c, 2) for c in range(cols)]

            col_fifos = []

            def build(c):
                if nomt:
                    # MEMTILE BYPASS: weight streams shim->core DIRECTLY (single object_fifo, no
                    # memtile, no object_fifo_link). DMA targets this fifo; core consumes it.
                    inA = object_fifo(f"inA{c}", shims[c], cores[c], depth, a_ty)
                    memA = inA          # DMA target is the shim->core fifo itself
                else:
                    # default: shim->memtile->core (memtile L2 staging + link), as the shipping path.
                    memA = object_fifo(f"memA{c}", shims[c], mts[c], depth, a_ty)
                    inA = object_fifo(f"inA{c}", mts[c], cores[c], depth, a_ty)
                    object_fifo_link(memA, inA)
                # C drains core->shim DIRECTLY in every variant (memtile bypass here is intentional
                # and constant; only the WEIGHT path's memtile presence is the --nomt lever).
                outC = object_fifo(f"outC{c}", cores[c], shims[c], 2, c_ty)

                # Pure drain: one outC tile per infinite-loop iter (runtime feeds N_col/m tiles),
                # draining K_div_k weight objects per tile. NO matvec, NO dequant -> wall == DMA.
                @core(cores[c], "mv_q6k.o", stack_size=0x2000)
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_out = outC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_out)
                        for _ in range_(K_div_k):
                            inA.acquire(ObjectFifoPort.Consume, 1)
                            inA.release(ObjectFifoPort.Consume, 1)
                        outC.release(ObjectFifoPort.Produce, 1)

                col_fifos.append((memA, outC))

            for c in range(cols):
                build(c)

            @runtime_sequence(
                np.ndarray[(cols * Acol,), np.dtype[np.uint8]],
                np.ndarray[(N,), np.dtype[np.float32]],
            )
            def sequence(A, C):
                bd = 1
                for c in range(cols):
                    memA, outC = col_fifos[c]
                    # fully-contiguous weight blast: column c's Acol bytes, adjacent strides.
                    npu_dma_memcpy_nd(metadata=memA, bd_id=bd, mem=A,
                                      offsets=[0, 0, 0, c * Acol],
                                      sizes=w_sizes, strides=w_strides)
                    bd += 1
                    npu_dma_memcpy_nd(metadata=outC, bd_id=bd, mem=C,
                                      offsets=[0, 0, 0, c * N_col],
                                      sizes=c_sizes, strides=c_strides)
                    bd += 1
                for c in range(cols):
                    dma_wait(col_fifos[c][1])

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="ggml-xrt weight-DMA dataflow BW probe")
    p.add_argument("--dev", choices=["npu", "npu2"], default="npu")
    p.add_argument("-K", type=int, default=6144)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--depth", type=int, default=2, help="weight object_fifo depth (in-flight DMA)")
    p.add_argument("--nomt", action="store_true", help="bypass memtile: shim->core direct weight DMA")
    a, _ = p.parse_known_args()
    bwprobe(a.dev, a.K, a.N, a.m, a.cols, a.depth, a.nomt)
