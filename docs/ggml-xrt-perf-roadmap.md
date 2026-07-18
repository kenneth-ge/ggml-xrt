# ggml-xrt performance roadmap — three levers to CPU-competitive low-power decode

Status baseline (2026-07-18, Qwen3-1.7B-Q4_K_M on Phoenix NPU):
decode ~1.5 t/s (~637 ms/token) = matmul ~452 ms (71%) + tail ~167 ms (26%) + host ~21 ms (3%).
CPU-only: ~37 t/s (~27 ms/token). Prefill already fast on iGPU (~40 t/s).

## The reframe (all measured, not assumed)

Decode is a stack of GEMVs (M=1). **Each weight is read exactly once per token — zero reuse — so decode speed IS the weight-streaming rate.** Compute (dequant/scale/MAC) only matters if it stalls the stream.

- **NPU weight-DMA ceiling = ~23 GB/s (HARD).** Pure-stream kernel, down shape: 1col 22.6, 2col 24.0, 4col 23.4 GB/s — saturates at ONE column, does NOT scale with channels. This is the shim DMA limit; the 89.6 GB/s LPDDR5 and the NPU's TOPS are not reachable through this path.
- CPU streams at ~42 GB/s. **So NPU-alone single-stream decode caps at ~1 GB/token ÷ 23 = ~44 ms = ~20 t/s < CPU's 37.** Raw-speed parity via bandwidth is impossible.
- **But the real gemv is far below its own ceiling:** down (q6k, noscratch) 9.1 GB/s; gate/up (q4k) 2.8 GB/s — vs the 23 ceiling. 2.5–8× headroom, wasted because compute stalls the DMA pipeline.
- MAC units run at ~0.6% of peak during decode — irrelevant while memory-bound. TOPS only pays off in PREFILL (M>1, reuse) and SPEC-DECODE VERIFY (M=N).

Consequence: the three levers are (1) close the gemv→ceiling gap [biggest, pure win], (2) kill the handoff tail [residency], (3) escape the bandwidth wall entirely [spec decode — the only path to *beat* CPU, by amortizing streams and shifting to the compute-bound regime where the NPU's TOPS actually win].

---

## Lever 1 — saturate the weight-DMA pipeline  (matmul 452 → ~70 ms)

**Problem.** gemv achieves 9.1 GB/s (q6k) / 2.8 GB/s (q4k) vs the 23 GB/s ceiling. The pure-stream kernel proves 23 is reachable; the real gemv falls short because dequant+scale+MAC execute *between* DMA bursts, stalling the stream. noscratch removed the sbuf L1 round-trip for q6k (4→9 GB/s) but did NOT help q4k (clang unroll ICE → small-array fallback, still 2.8 GB/s).

**Mechanism (NOT more channels — those are saturated).**
1. **Double-buffer / software-pipeline** the tile loop: ping-pong two tile buffers so the DMA of weight-tile N+1 overlaps the compute of tile N. Goal: DMA runs continuously at ~23 GB/s, compute fully hidden under it.
2. **q4k unroll fix (prerequisite).** q4k's data path is the worst (2.8 GB/s) and gate/up are the single biggest matmul cost (~137 ms/token). Fix the clang full-unroll ICE so q4k gets real inline-register scale (like q6k), not the small-array fallback.
3. Per-shape tile sizing so the pipeline depth covers compute latency.

**Tasks (kernel agent).** pipelined DMA descriptors + double-buffer scratch; q4k unroll fix; per-shape tuning. **Backend:** none (same ABI).

**Validation (my side).** `gemv_bench` GB/s per shape → target ≥18 GB/s (78% of ceiling). `stream_bench` already confirms 23 is achievable with no compute. Correctness via `q4_gemv_check` (NRMSE < 0.05).

**Projection.** all shapes ~18–23 GB/s → matmul **452 → ~60–70 ms/token**. This alone: ~1.5 → ~4 t/s.

---

## Lever 2 — full-layer NPU residency  (tail 167 → ~40 ms)

**Problem (confirmed).** The scheduler dispatches ONE matmul per split — `graph_compute` logs `1 op` each, 196 splits/token, ~0.66 ms NPU↔CPU handoff apiece = ~130 ms. Structural: with MATMUL_ONLY every matmul pair has a non-matmul op (rope/attention/add/silu) between them, so no two matmuls ever share a split.

**Mechanism.** Put the in-between ops on the NPU so consecutive matmuls share a split and whole layers run without returning to CPU. Target 196 → ~28 transitions (one per layer) or fewer.

Per-layer split map today (each `|` is an NPU↔CPU round-trip):
`add,norm(CPU) | q(NPU) | rope(CPU) | k(NPU) | rope(CPU) | v(NPU) | attention(CPU) | o(NPU) | add,norm(CPU) | gate,up(NPU) | silu·mul(CPU) | down(NPU)`

**Fragments, ranked by transitions removed/layer:**
1. **SwiGLU (BUILT, wire it).** Fuses gate+up+silu·mul+ (feeds down) — removes the up→iGPU-silu·mul→down round-trip. −1 transition/layer, compute parity (4.47 vs 4.64 ms). ABI: A=[Nff][K/256][296] interleaved gate_q4k(148)++up_q4k(148), B=bf16[K], C=f32[Nff].
2. **rope+attention on NPU.** Merges q|k|v|o into one resident block — the biggest single cut (~3 transitions/layer). Needs rope (overlay exists) + attention (KQ/softmax/KQV) kernels + KV-cache residency.
3. **residual-add + RMSNorm-fold.** Fold norm into the next matmul (B=[x[K]++gamma[K]], rms_inv on-core, x'=x·gamma·rsqrt(mean(x²)+eps)); keep the residual-add on NPU so raw_x is produced on-chip. NOTE: fold alone does NOT cut a transition (the input handoff survives) — it only pays off once the residual-add is also resident. Build it *with* residency, not standalone.

**Backend (my side).** `supports_op` claims the fused subgraph (e.g. recognize matmul src1 = mul(rms_norm(raw_x), gamma); recognize gate/up/silu/down as SwiGLU); `graph_compute` executes the fused unit, extracting raw_x + gamma / interleaving weights. Context budget: reuse the overlay context scheme; a fused-layer megakernel is 1 context.

**Validation.** re-run with `GGML_XRT_ENABLE_LOG=1`, confirm `graph_compute` op-counts >1 and the split count drops; per-token tail from the instrument.

**Projection.** 196 → ~28 transitions → tail **167 → ~40 ms**. With Lever 1: token ~70+40+21 ≈ 130 ms → **~7–9 t/s**.

---

## Lever 3 — speculative decode  (break the bandwidth wall → beat CPU at low power)

**Why this is the only path past CPU.** NPU is bandwidth-outclassed (23 < 42 GB/s), so single-stream it caps ~20 t/s < CPU 37. Spec decode changes the game two ways:
1. **Amortization.** A tiny draft proposes N tokens; the NPU verifies all N in ONE weight-stream. Throughput decouples from per-token streaming — the multiplier that pushes effective t/s *above* the raw bandwidth limit.
2. **Regime shift.** Verify is an M=N matmul (weight reuse) → COMPUTE-bound, not memory-bound → finally uses the NPU's TOPS advantage (the prefill mm kernels, MACs no longer idle).

**Mechanism.** Draft = 0.5B model (or n-gram / Medusa heads) running on iGPU/CPU; NPU verifies the N-token block as a batched forward (prefill-style mm path already exists). Accept the longest matching prefix; typical acceptance 2–3×.

**Needs.** Lever 1 first (so the base is DMA-bound and the verify is the bottleneck worth amortizing); the M=N verify path (prefill mm kernels — vectorize if needed); llama.cpp speculative support (`--model-draft` / lookahead) wired through the sched so verify lands on XRT; acceptance-rate measurement per task.

**Projection.** base (post 1+2) ~9 t/s × ~2.5 acceptance ≈ **~22 t/s effective**; if the base is pushed to the ~20 t/s DMA ceiling, ×2.5 ≈ **50 t/s > CPU**, at a fraction of CPU power. Honest caveat: acceptance is model/task dependent; the multiplier is real but variable, and verify-kernel efficiency (M=N mm) must be good or it eats the gain.

---

## Combined trajectory

`0.7 (start) → 1.5 (overlay fix) → ~1.7 (q6k noscratch) → ~2.2 (q4k noscratch) → ~4 (lever 1 DMA pipeline) → ~7-9 (lever 2 residency) → ~20+ (lever 3 spec decode)`

Raw-speed CPU parity is impossible on bandwidth alone (23 < 42). The credible endgame is **lever 3: match/beat CPU *effective* throughput at ~1/5 the power** — which is the actual goal (always-on, power-efficient local agent). Levers 1+2 are prerequisites that also stand on their own (~7-9 t/s at very low power).

Ownership: Lever 1 = kernel agent (+ my validation). Lever 2 = kernel agent (op/fused kernels) + me (backend fusion recognition). Lever 3 = me (llama spec-decode wiring + sched routing) + kernel agent (M=N verify kernel).

---

## Lever 4 — big MoE (frontier quality at active-param speed, NPU power)

The payoff direction. A Mixture-of-Experts model (e.g. **Qwen3-30B-A3B**: 30B total, ~3B active/token) decodes by streaming ONLY the routed experts (~3B ≈ 1.6 GB Q4/token), not all 30B. So per token it costs like a ~3B dense model — while delivering 30B-class quality.

**Why the NPU stops being disadvantaged here.** Decode is memory-bound (weight streaming). A MoE is memory-bound on the ~1.6 GB active/token *for every accelerator* — the iGPU included. The NPU's only deficit vs the iGPU is bandwidth (23 vs 42 GB/s ≈ 2×); it is NOT compute-limited (MACs idle). So the NPU runs a 30B MoE at ~7–11 t/s (1.6 GB ÷ 23 GB/s ceiling), ~half the iGPU's raw rate but at a fraction of the power — and **spec-decode (Lever 3) stacks on top** (~2.5× → ~15–20 t/s effective). Frontier quality, small-active speed, NPU power.

**Fits the machine:** 30B-A3B Q4 ≈ 17 GB; host has 27.8 GB (free the ~13 GB in use → fits at Q4, comfortable at Q3). And Qwen3-30B-A3B shares the Qwen3 vocab, so the **Qwen3-0.6B draft (Lever 3) drafts it directly** — the spec-decode work transfers verbatim.

**Blocker (new kernel arc, but overlapping):** MoE experts use `GGML_OP_MUL_MAT_ID` (gather the top-k selected experts, then quant matmul), not plain `MUL_MAT`. Needs: (a) a gather-selected-experts + quant matmul NPU kernel, (b) backend `MUL_MAT_ID` support + the router (top-k over the gate), (c) streaming only the selected experts' weights per token. **Heavy overlap with Lever 3's M=N verify kernel** — both are efficient multi-row quant matmul (verify = M=N tokens; expert = n_tokens × selected experts). Build the M=N kernel MoE-aware and it becomes most of the expert kernel.

**Sequence:** Lever 2 (residency) → Lever 3 (spec-decode, forces the fast M=N mm) → Lever 4 (MoE, reuses that mm + adds gather/router). Each de-risks the next; the Qwen3-0.6B draft serves both the 1.7B dense and the 30B-A3B MoE.

Ownership: Lever 4 = kernel agent (MUL_MAT_ID gather-matmul, generalized from the M=N verify kernel) + me (backend MUL_MAT_ID support, router, expert-weight streaming/caching, model wiring).
