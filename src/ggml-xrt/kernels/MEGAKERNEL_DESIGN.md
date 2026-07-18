# ggml-xrt megakernel — design & staging plan

**Goal:** fuse a decode layer's ops into resident, software-pipelined array programs so cores
never re-init/re-sync between dispatches and the next weight tile's DMA overlaps the current
compute (time → `max(compute, DMA)` instead of `sum`). Removes per-dispatch launch/sync cost,
kills intermediate L3 round-trips, and is the substrate that turns compute-bound decode into
DMA-bound decode.

## Reality check (why this is staged, not urgent)
Decode is currently **~12× compute-bound** (16-core q6k down = 9.55 ms compute vs ~0.8 ms
weight DMA). The megakernel's headline wins — dispatch-overhead removal and DMA/compute overlap
— only *show* once compute drops to ~DMA range. So this is **prep for the DMA-bound regime**;
build it in parallel, but the near-term needle-mover is compute throughput (#1 producer/consumer,
C repack-preassemble). Two megakernel fragments do have modest near-term value (they remove ops
+ round-trips even while compute-bound) and are the right first steps:

## Fragment 1 — fused SwiGLU (gate + up + silu·mul), the first build
`ffn: down( silu(gate·x) * (up·x) )`. Today that's 4 dispatches (gate matmul, up matmul, silu,
mul) + gate/up L3 round-trips. Fuse gate+up+silu+mul into ONE 16-core dispatch:
- Each core computes its `N_ff/16` slice of BOTH `gate = Wg·x` and `up = Wu·x` (two matvec
  accumulations over K into `gate_buf[Mc]`, `up_buf[Mc]`), then applies `silu(gate)*up`
  elementwise over `Mc` and writes the FUSED slice. Output is `[N_ff]` (not gate+up separately)
  → halves output DMA, removes silu+mul dispatches, keeps gate/up on-chip.
- Reuse the vector silu from `mlir-aie/aie_kernels/aie2/silu.cc`/`swiglu.cc`:
  `sigmoid(x)=0.5*(getTanhBf16(x/2)+1)`, `silu(x)=x*sigmoid(x)`.
- Core: two weight streams (Wg, Wu) via the memtile distribute; x broadcast; fused out gathered.
- Near-term value while compute-bound: −2 dispatches, −gate/up round-trips, −½ output DMA.
- Files (planned): `aie2/mv_swiglu_q4k.cc` (2-accumulate + swiglu epilogue), `gemv_swiglu.py`.

## Fragment 2 — fused RMSNorm-scale into the following matmul
`RMS_NORM` then `MUL(gamma)` then `MUL_MAT` → fold the norm's per-row scale + gamma into the
matmul's input read (on-core epilogue of the norm, or prologue of the matmul). Removes an op +
a round-trip per norm.

## Fragment 3 — q/k/v fused already exists (gemv_qkv_fused) — promote once context-bound
Shelved earlier (net time loss while compute-bound), but it's a megakernel fragment: one
dispatch, shared activation, 3→1 hw_context. Re-enable if we go context-bound.

## Endgame — whole-layer resident + weight prefetch
Once the fragments exist and compute is near DMA-bound: chain a full layer (attn + ffn) as one
resident program, double-buffer weights so layer N+1's DMA runs under layer N's compute
(`fifo_depth ≥ 2` on the weight L3→L2), and never return to the host mid-layer. This is the
`time → max(compute, DMA)` payoff and the real end state.

## Dependencies / ordering
1. #1 producer/consumer + C (compute throughput) — bring compute toward the DMA ceiling FIRST.
2. Fragment 1 (SwiGLU) — build now as prep (modest near-term win, exercises 2-weight fusion).
3. 8-channel-weight gemv (separate, in progress) — the DMA half of the eventual overlap.
4. Endgame whole-layer + prefetch — once 1–3 land and compute ≈ DMA.

Status: design only + Fragment 1 core prototype in progress. UNVALIDATED.
