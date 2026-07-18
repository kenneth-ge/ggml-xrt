#!/usr/bin/env python3
# Local (Linux, no NPU) sanity model for the decode-attention chain's LAYOUT + GQA contract.
# It CANNOT run the AIE kernel (no chess sim), but it verifies the exact DMA-offset / GQA /
# Vt-layout contract the multi-head kernel's runtime_sequence must implement -- catching the
# LIKELY bug class (wrong head/GQA index, wrong offset, wrong V transpose) BEFORE the device
# round-trip. It also emits the reference out[n_head,128] the Windows harness should match.
#
# Contract under test (per the validated single-head chain + KV-cache report):
#   Q[n_head, head_dim]            offset(h)      = h*head_dim
#   K[n_head_kv, n_kv, head_dim]   offset(h)      = (h//gqa)*n_kv*head_dim   (native K-cache)
#   V[n_head_kv, head_dim, n_kv]   offset(h)      = (h//gqa)*head_dim*n_kv   (native v_trans = Vt)
#   out[n_head, head_dim]          offset(h)      = h*head_dim
#   scale = 1/sqrt(head_dim) (folded in softmax); GQA: Q head h uses KV head h//gqa.
import argparse
import numpy as np


def ref_and_contract(n_head, n_head_kv, head_dim, n_kv, seed):
    rng = np.random.default_rng(seed)
    gqa = n_head // n_head_kv
    scale = 1.0 / np.sqrt(head_dim)
    # flat activation buffers exactly as the kernel sees them (element-indexed like the DMA offsets)
    Q = rng.standard_normal((n_head, head_dim)).astype(np.float32)
    Kc = rng.standard_normal((n_head_kv, n_kv, head_dim)).astype(np.float32)   # [kv][j][d]
    Vt = rng.standard_normal((n_head_kv, head_dim, n_kv)).astype(np.float32)   # [kv][d][j] (v_trans)
    Qf, Kf, Vf = Q.ravel(), Kc.ravel(), Vt.ravel()

    ref = np.zeros((n_head, head_dim), np.float32)
    kern = np.zeros((n_head, head_dim), np.float32)
    for h in range(n_head):
        kv = h // gqa
        # ---- reference (clean numpy) ----
        s = (Kc[kv] @ Q[h]) * scale                 # scores[n_kv]
        s = np.exp(s - s.max()); s /= s.sum()        # softmax
        ref[h] = Vt[kv] @ s                          # out[head_dim]
        # ---- kernel-view: reconstruct ONLY from flat buffers + the contract's offsets ----
        qoff, koff, voff = h * head_dim, kv * n_kv * head_dim, kv * head_dim * n_kv
        q_h = Qf[qoff:qoff + head_dim]
        K_h = Kf[koff:koff + n_kv * head_dim].reshape(n_kv, head_dim)   # [n_kv, head_dim]
        Vt_h = Vf[voff:voff + head_dim * n_kv].reshape(head_dim, n_kv)  # [head_dim, n_kv]
        sk = (K_h @ q_h) * scale
        sk = np.exp(sk - sk.max()); sk /= sk.sum()
        kern[h] = Vt_h @ sk
    return ref, kern, gqa


def main():
    p = argparse.ArgumentParser(description="attention layout/GQA contract sanity (no NPU)")
    p.add_argument("--n_head", type=int, default=16)
    p.add_argument("--n_head_kv", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--n_kv", type=int, default=1024)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    ref, kern, gqa = ref_and_contract(a.n_head, a.n_head_kv, a.head_dim, a.n_kv, a.seed)
    # the contract check: kernel-view offsets must reconstruct the reference exactly (f32, so ~0)
    err = np.abs(ref - kern).max()
    rms = np.sqrt(((ref - kern) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-9)
    print(f"n_head={a.n_head} n_head_kv={a.n_head_kv} gqa={gqa} head_dim={a.head_dim} "
          f"n_kv={a.n_kv} seed={a.seed}")
    print(f"  layout/GQA contract: max_abs_err={err:.3e} NRMSE={rms:.3e}  -> "
          f"{'PASS (offsets/GQA/Vt correct)' if err < 1e-4 else 'FAIL (offset/index/transpose bug)'}")
    print("  DMA offsets the multi-head runtime_sequence MUST emit (elements):")
    for h in [0, 1, 2, a.n_head - 1]:
        kv = h // gqa
        print(f"    h={h:2d} kv={kv}: Q@{h*a.head_dim}  K@{kv*a.n_kv*a.head_dim}  "
              f"Vt@{kv*a.head_dim*a.n_kv}  out@{h*a.head_dim}")


if __name__ == "__main__":
    main()
