# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
#
# STEP-3 host repacker + round-trip verifier for the CONTIGUOUS q6k weight layout used by
# gemv_mc16_contig.py. This is a REORDER-ONLY repack (no byte changes to any record), so the
# dequant result is IDENTICAL to the shipping [N][K/256][REC] layout by construction.
#
# Shipping layout  (produced by ggml_xrt_repack_quant_weight in ggml-xrt.cpp):
#     Aorig[n][b][REC]   record for output n, k-block b at  (n*K_div_k + b)*REC
#
# Contiguous layout (this repack; consumed by gemv_mc16_contig.py's fully-contiguous weight DMA):
#     Anew[c][i][t][jj][REC]  at linear record index  (((c*Mdm + i)*K_div_k + t)*(rows*m) + jj)
#     holding Aorig's record for  n = c*(rows*m*Mdm) + i*(rows*m) + jj,  b = t
#   where cols=4, rows=4, m=32 (the 16-core overlay), Mdm = (N/(cols*rows))/m, K_div_k = K/256.
#
# The order (c,i,t,jj,byte) is EXACTLY the traversal order of gemv_mc16_contig.py's weight DMA
# (sizes=[Mdm,K_div_k,rows*m,REC], contiguous strides), so the memtile receives, per object, one
# k-tile's [rows*m,REC] block — same as the shipping strided DMA delivers from [N][K/256][REC] —
# hence distribute/kernel/C-order are all unchanged; only the DRAM byte order (and DMA strides)
# differ, turning 212 B strided reads into a single contiguous per-column blast.
import argparse
import numpy as np

REC = 212
QK = 256


def repack_contig(Aorig, N, K, m=32, cols=4, rows=4):
    """Aorig: uint8[N*K_div_k*REC]  ([N][K/256][REC]).  Returns uint8[same size] contiguous layout."""
    K_div_k = K // QK
    ncores = cols * rows
    assert N % (m * ncores) == 0, "N must be divisible by m*cols*rows"
    Nc = N // cols                 # outputs per column (rows*m*Mdm)
    Mc = N // ncores               # outputs per core
    Mdm = Mc // m                  # output m-tiles per core
    rm = rows * m
    A = Aorig.reshape(N, K_div_k, REC)
    out = np.empty((cols, Mdm, K_div_k, rm, REC), dtype=np.uint8)
    for c in range(cols):
        for i in range(Mdm):
            for jj in range(rm):
                n = c * Nc + i * rm + jj
                out[c, i, :, jj, :] = A[n, :, :]        # all k-tiles for this output
    return out.reshape(-1)


def dequant_q6k_record(rec):
    """Reference dequant of one 212 B q6k record -> float32[256]. Canonical ggml q6_K layout
    (see ggml-quants.c dequantize_row_q6_K). Used only for the round-trip equality check."""
    sc = rec[192:208].view(np.int8).astype(np.int64)   # 16 group scales (int8)
    d = rec[208:212].view(np.float32)[0].astype(np.float64)
    out = np.empty(256, dtype=np.float64)
    for n in range(2):                                  # n over 2 blocks of 128
        for l in range(32):
            is_ = 8 * n
            ql0 = rec[64 * n + l].astype(np.int64) & 0xF
            ql1 = rec[64 * n + l].astype(np.int64) >> 4
            ql32_0 = rec[64 * n + l + 32].astype(np.int64) & 0xF
            ql32_1 = rec[64 * n + l + 32].astype(np.int64) >> 4
            qhb = rec[128 + 32 * n + l].astype(np.int64)
            q1 = (ql0 | ((qhb & 0x03) << 4)) - 32
            q2 = (ql32_0 | (((qhb >> 2) & 0x03) << 4)) - 32
            q3 = (ql1 | (((qhb >> 4) & 0x03) << 4)) - 32
            q4 = (ql32_1 | (((qhb >> 6) & 0x03) << 4)) - 32
            base = 128 * n
            out[base + l] = d * sc[is_ + 0] * q1
            out[base + l + 32] = d * sc[is_ + 2] * q2
            out[base + l + 64] = d * sc[is_ + 4] * q3
            out[base + l + 96] = d * sc[is_ + 6] * q4
    return out


def roundtrip_check(N, K, seed=0, m=32, cols=4, rows=4):
    K_div_k = K // QK
    rng = np.random.default_rng(seed)
    Aorig = rng.integers(0, 256, size=N * K_div_k * REC, dtype=np.uint8)
    # write a valid f32 d into each record so dequant is finite (not required for byte-equality)
    Anew = repack_contig(Aorig, N, K, m, cols, rows)
    assert Anew.size == Aorig.size

    # 1) byte-level: every (c,i,t,jj) record in Anew equals its source Aorig record.
    ncores = cols * rows
    Nc = N // cols
    Mc = N // ncores
    Mdm = Mc // m
    rm = rows * m
    Ao = Aorig.reshape(N, K_div_k, REC)
    An = Anew.reshape(cols, Mdm, K_div_k, rm, REC)
    ok = True
    for c in range(cols):
        for i in range(Mdm):
            for t in range(K_div_k):
                for jj in range(rm):
                    n = c * Nc + i * rm + jj
                    if not np.array_equal(An[c, i, t, jj], Ao[n, t]):
                        ok = False
    print(f"byte-reorder round-trip: {'PASS' if ok else 'FAIL'}  (N={N} K={K})")

    # 2) dequant equality on a sample of records (dequant(new slot) == dequant(orig n,b)).
    deq_ok = True
    for _ in range(64):
        c = rng.integers(cols); i = rng.integers(Mdm); t = rng.integers(K_div_k); jj = rng.integers(rm)
        n = c * Nc + i * rm + jj
        if not np.array_equal(dequant_q6k_record(An[c, i, t, jj]),
                              dequant_q6k_record(Ao[n, t])):
            deq_ok = False
    print(f"dequant round-trip (64 samples): {'PASS' if deq_ok else 'FAIL'}")
    return ok and deq_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="q6k contiguous-layout repacker + round-trip")
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-K", type=int, default=6144)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--rows", type=int, default=4)
    a, _ = p.parse_known_args()
    ok = roundtrip_check(a.N, a.K, m=a.m, cols=a.cols, rows=a.rows)
    raise SystemExit(0 if ok else 1)
