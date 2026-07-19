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

## STAGING CONTRACT (for backend wiring — attention validated all-16-heads)
Qwen3-1.7B decode layer op sequence (n_embd=2048, n_head=16, n_head_kv=8, head_dim=128).
Compute order (top→bottom); STAGING order = which ops to claim/wire first (attention → rope+qknorm → norm+add).

```
  op#  op                              tensors (consume -> produce)                       kernel / status
  1    rms_norm(inpL)*attn_norm_w      inpL[2048] f32 -> cur[2048]                         rms_norm.cc (exists, claim @ stage3)
  2    Wq·cur, Wk·cur, Wv·cur          cur[2048] -> Qcur[16,128] Kcur[8,128] Vcur[8,128]   gemv q4k/q6k (claimed already)
  3    q_norm(Qcur), k_norm(Kcur)      per-head rms_norm over head_dim=128                 rms_norm.cc c=128 (exists, stage2)
  4    rope(Qcur,pos), rope(Kcur,pos)  [.,128] -> roped                                    rope_s1_d128 (exists, stage2)
  5    write Kcur,Vcur -> KV cache     (host/cache mgmt)                                   —
  6    kq = mul_mat(Kcache, Qroped)    K[8,n_kv,128], Q[16,128] -> kq[16,n_kv]             ]
  7    kq = soft_max_ext(kq,mask,1/√128) kq[16,n_kv], mask[n_kv] -> probs[16,n_kv]         ] attn_chain_mh (BUILT+VALIDATED)
  8    kqv = mul_mat(Vcache, probs)    V[8,128,n_kv], probs -> kqv[16,128]                 ]  = STAGE 1
  9    kqv reshape -> [2048]                                                               —
  10   cur = Wo·kqv                    kqv[2048] -> cur[2048]                              gemv (claimed already)
  11   inpL = inpL + cur               inpL[2048]+cur[2048] -> inpL[2048]                  eltwise_add (BUILDING, stage3)
  12   rms_norm(inpL)*ffn_norm_w       -> cur[2048]                                        rms_norm.cc (stage3)
  13   swiglu(gate,up)                 cur -> ffn[6144]                                    SwiGLU (WIRED)
  14   down·ffn                        ffn[6144] -> cur[2048]                              gemv (claimed)
  15   inpL = inpL + cur               -> inpL                                             eltwise_add (stage3)
```

### STAGE 1 — attention on NPU (wire FIRST, biggest single cut)
Claim the subgraph {op6 kq=mul_mat(Kcache,Q), op7 soft_max_ext, op8 kqv=mul_mat(Vcache,kq)} and dispatch `attn_chain_mh_{n_kv-bucket}.xclbin` in its place.
- CONSUMES: Qroped[16,128] bf16 (from op4, still CPU/GPU at this stage), Kcache[8,n_kv,128] + Vcache[8,128,n_kv] bf16 (`-ctk/-ctv bf16`, zero-copy native layout), mask[n_kv] bf16.
- PRODUCES: kqv[16,128] f32 (feeds op9 reshape → op10 Wo).
- SKIP: the 3 CPU/GPU ops (kq/softmax/kqv). ABI: kernel(3, instr@1, ninstr@2, K@3, Q@4, V@5, mask@6, out@7). n_kv bucket: pick 512/1024/2048 ≥ current seq (pad probs tail = -inf → 0).
- Result: removes the attention CPU excursion. Validate kqv vs CPU, then measure transitions.

### STAGE 2 — rope + qk-norm on NPU (extend the claimed region backward)
Claim op3 (q_norm/k_norm, `rms_norm` c=128, per-head) + op4 (rope, `rope_s1_d128`). Now op2→op8 is a contiguous NPU run (qkv→qknorm→rope→attention) → the scheduler groups it into one split.
- CONSUMES/PRODUCES: op3 Qcur[16,128]→Qnormed (k likewise); op4 Qnormed+cos/sin→Qroped.
- CAVEAT (from the design): the host permute currently in `ggml_backend_xrt_rope` must move onto the rope kernel or be a claimed no-op, else it breaks contiguity and the split won't group.

### STAGE 3 — input norm + residual adds on NPU (close the region → ~1 split/layer)
Claim op1 (input rms_norm), op11 + op15 (residual adds via `eltwise_add`), op12 (post-attn rms_norm).
- op11/op15 CONSUME inpL[2048] f32 + cur[2048] f32 → inpL[2048] f32 (`add_f32`, DIM_N=2048).
- With op1..op15 all claimed + contiguous, the whole q..o(..down) region groups into ~1 split → ~1 NPU↔CPU transition/layer (from ~7). Closes the ~130 ms tail.

### WIRING PATTERN (same as SwiGLU, proven)
Per stage: claim the ops in `supports_op` (relax MATMUL_ONLY for that set, artifact-gated), dispatch the fused/op kernel, skip the covered CPU/GPU ops. Weightless nodes (attention/norm/rope/add) place on XRT with no offload nudge (GLU proved it); unified-buft groups the contiguous claimed run into one split. Measure transitions/token after EACH stage — each is independently valuable + a clear cut.
