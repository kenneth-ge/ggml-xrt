# Resident q..o block — design + wiring spec (Qwen3-1.7B decode)

**Goal:** make a decoder layer's q..o region NPU-resident so the ggml scheduler groups it into
ONE split (≈1 NPU↔CPU transition/layer instead of ~7), closing the measured ~130 ms handoff
tail. Grouping comes ONLY from scheduler *assignment* (claim ops via `supports_op` → they land
in one split when contiguous) — `graph_optimize` is post-split and can't merge. Proven by the
SwiGLU 1288-split log (`mul_mat×2 + glu` grouped into one split). So: **claim every op in the
region + keep them contiguous → one groupable region.**

## Decode layer op sequence (Qwen3-1.7B: n_embd=2048, n_head=16, n_head_kv=8, head_dim=128)
```
  rms_norm(x)                         [A] input norm
  q=Wq·, k=Wk·, v=Wv·                 [M] qkv matmul (weight)  -> can fuse (gemv_qkv16) or keep 3
  q_norm(q per head), k_norm(k)       [A] Qwen3 QK-norm (RMSNorm over head_dim, per head)
  rope(q), rope(k)                    [R] NEOX rope
  attention: QK^T->softmax->scores·V  [T] the chained attn kernel (built, validated)
  o = Wo·attn                         [M] o matmul (weight)
  x = x + o                           [+] residual add
  rms_norm(x) -> swiglu -> down       [FFN] (SwiGLU already wired)
  x = x + down                        [+] residual add
```

## NPU-kernel status per op
| op | kernel | overlay | claimed in supports_op? | gap |
|----|--------|---------|----|-----|
| rms_norm | rms_norm.cc | rms_rope_overlay | needs claim (MATMUL_ONLY rejects today) | claim |
| qkv matmul | mv_q4k/q6k noscratch | (dtype,K) gemv overlay | YES | — |
| q_norm/k_norm | rms_norm.cc (c=head_dim=128) | rms_rope_overlay | needs claim + a c=128 ELF | build ELF + claim |
| rope | rope.cc | rms_rope_overlay (rope_s1_d128) | needs claim | claim |
| attention | attn_chain (built) | attn overlay | new — needs claim + wire | wire |
| o matmul | mv_* noscratch | gemv overlay | YES | — |
| residual add | **GAP** | — | — | **build add kernel** |
| swiglu/down | mv_swiglu / mv_* | swiglu + gemv | GLU claimed (SwiGLU wired) | — |

## Context budget (Phoenix ~5 hw_contexts) — fits
- gemv weights (K∈{2048,6144}) = 1–2
- rms_rope_overlay (rms_norm + q/k norm + rope, all packed) = 1
- attention overlay = 1
- swiglu = shares gemv path / 1
→ **≤5.** The rms+rope+qknorm all live in ONE packed overlay (rms_rope_packed pattern); attention in ONE (attn_packed/chain). Comfortable.

## Gaps to build (my side), staged by ROI
1. **residual-add kernel** (`aie2/eltwise_add.cc` — bf16/f32 elementwise `c=a+b`, len=n_embd=2048). Trivial (mlir-aie ml/eltwise_add template). Needed twice/layer. Build + op-overlay.
2. **q_norm/k_norm ELF** at c=head_dim=128 (the rms_norm kernel already exists; just needs the s1_c128 ELF in rms_rope_overlay — may already be covered by rms_norm_s1_c128.elf; verify).
3. **multi-head attention chain** (in flight) — the [T] op.

## Windows wiring (per SwiGLU pattern — proven)
1. **Flip the region's ops to claimed** in `ggml_backend_xrt_device_supports_op` (relax MATMUL_ONLY for this set): RMS_NORM, ROPE, the attention op (a fused GGML op or a recognized subgraph), residual ADD — each only if its artifact exists. Weightless ops place on XRT with no offload nudge (proven by GLU).
2. **Contiguity:** ensure nothing CPU-only sits between these nodes in the graph (reshape/permute for rope is host today — move the permute onto the rope kernel or make it a no-op claim, else it breaks the contiguous region and the split won't group).
3. **Dispatch:** each claimed op runs its overlay/ELF; the scheduler groups the contiguous claimed run into one split → ~1 transition/layer.
4. **KV:** `-ctk/-ctv bf16` (attention reads cache zero-copy).

## Staged rollout (measure transitions after each — expect a clear win each stage)
- **Stage 1: attention on NPU** (wire the chain; claim it). Merges the attn region; removes the QK/softmax/KQV CPU excursion. Biggest single cut.
- **Stage 2: claim rope + qk-norm** (kernels exist). Merges q..attn contiguous.
- **Stage 3: claim input rms_norm + residual add** (add kernel new). Closes the region to ~1 split.
- **Stage 4: chain on-chip** (endgame): fuse the region into fewer dispatches (max(compute,DMA)); optional once splits are grouped.

Each stage is independently measurable and independently valuable; do 1→4, re-measure transitions/token after each.
```
