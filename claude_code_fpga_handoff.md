# PHAT-JeT FPGA Synthesis — Claude Code Handoff

## Context

PHAT-JeT is a compact transformer for real-time jet tagging at the CMS Level-1 trigger, under review at NeurIPS 2026 (submission 28409). All three reviewers asked for hardware evidence beyond FLOP counts. The rebuttal deadline is in ~1 week. The goal of this task is **Vitis HLS C-synthesis resource/latency estimates** (NOT place-and-route, NOT a bitstream) for the PHAT-JeT components, to be reported in a rebuttal table.

The claim we need to support: PHAT-JeT is FPGA-*compatible* (built from operations with established FPGA implementations), with resource/latency estimates comparable to JEDI-Linear (the deployed baseline at the same FLOP count). We are NOT claiming a validated deployment.

## Deliverable

A markdown table with one row per component and a total row:

| Component | LUT | FF | DSP | BRAM | Latency (cycles) | Latency (ns @ 200 MHz) | II |
|---|---|---|---|---|---|---|---|
| GMP (scatter + DWConv + gather + pointwise) | | | | | | | |
| Local patch attention (per patch, P=10) | | | | | | | |
| Local patch attention (all 15 patches, parallel) | | | | | | | |
| Patch-token pooling + global MHSA (15 tokens) | | | | | | | |
| Broadcast + residual | | | | | | | |
| FFN + classifier head | | | | | | | |
| **Total (pipeline)** | | | | | | | |

Plus a reference row from JEDI-Linear's published FPGA numbers (Que et al., arXiv:2508.15468) for context.

Everything labeled explicitly: "Vitis HLS 2023.x C-synthesis estimates, pre-place-and-route, ap_fixed<16,6>, target xcvu13p (Ultrascale+), 200 MHz (5 ns clock)."

## Model configuration (from the paper, Appendix E)

- Input: N=150 particles (zero-padded), 3 features each (pT, Δη, Δϕ)
- Hidden dimension: d_model = 16
- Attention heads: 4 (head dim 4)
- Single PHAT block (one attention layer total)
- Patch size P = 10 → N_P = 15 patches
- GMP grid spacing δ = 0.2; grid dims determined by jet cone (roughly
  H×W ≈ 8×8 to 16×16 cells; confirm from code — see repo)
- GMP: quantize (η,ϕ) to grid → sum-scatter features per cell →
  3×3 depthwise conv → gather back per particle → pointwise Dense(16)
  → LayerNorm → residual add
- Patch tokenizer: mean pool (default config)
- Global stage: MHSA over 15 patch tokens, then optional Dense(16)
  projection (message_proj=True), broadcast + residual
- FFN: Dense(16→~32-64)→GELU→Dense(→16); confirm expansion from code
- Classifier: global mean pool over particles → MLP → 5 classes
- Total: ~6.7K params, ~1.31M FLOPs

## Code

- Anonymous repo: https://anonymous.4open.science/r/PHAT-JeT-540B
- The Keras/TF (or PyTorch — check) model definition contains the exact
  layer shapes. Extract them first; do not trust the summary above where
  it conflicts with code.

## Approach (recommended order)

### Step 1 — Extract exact shapes (1-2 hrs)
Load the model definition from the repo, print every layer's input/output
shape and parameter count. Write these to `shapes.md`. All subsequent HLS
work keys off this file.

### Step 2 — Try the hls4ml path first (0.5-1 day)
hls4ml supports Dense, Conv2D (depthwise may need the SeparableConv2D
path or a recent version — check hls4ml >= 1.0), pooling, and softmax.
Multi-head attention support has landed for the transformer work
(look for `hls4ml` MHA layer support merged for the L1 transformer
effort; Chang Sun knows the exact branch if the public release doesn't
have it).

- Convert what converts cleanly. Component-by-component conversion is
  fine — we need per-component numbers anyway.
- Config: ap_fixed<16,6> everywhere as a first pass; Resource strategy;
  io_parallel for the small components.
- If a component fails conversion, fall through to Step 3 for that
  component only.

### Step 3 — Hand-written HLS for non-converting components (1-2 days)
Expected problem children:

**(a) GMP scatter/gather.** No hls4ml primitive. But: N=150 is FIXED and
grid dims are FIXED, so this is static index routing, not dynamic memory.
Implementation sketch:
```cpp
// scatter: accumulate features into grid
// u[i], v[i] computed on-chip from eta/phi (2 subs, 2 mults, 2 floors)
for (int i = 0; i < 150; i++) {      // pipeline II=1
    for (int c = 0; c < 16; c++) {
        grid[u[i]][v[i]][c] += X[i][c];   // BRAM accumulate
    }
}
// DWConv 3x3 over grid: standard, hls4ml can do this part
// gather: one BRAM read per particle
for (int i = 0; i < 150; i++)
    for (int c = 0; c < 16; c++)
        Z[i][c] = grid_conv[u[i]][v[i]][c];
```
Watch for: BRAM write conflicts when multiple particles land in the same
cell in the same cycle — either serialize the scatter loop (II=1 over
particles is fine, 150 cycles) or partition the grid array. Serializing
is acceptable; 150 cycles @ 5ns = 750 ns for the scatter, which is fine
for a component estimate. Note the actual L1 latency budget is O(1 µs)
for the full algorithm — flag if the naive total exceeds ~2 µs, since
then we should report a partitioned/parallel variant.

**(b) 10×10 softmax attention within a patch.** Small enough to fully
unroll or partially unroll. QK^T is 10×10×4 MACs per head; softmax over
10 elements via the hls4ml softmax LUT approach (or invert-exp table).
15 patches are independent → report both the single-patch instance and
×15 parallel replication (resource ×15, same latency) and a serialized
variant (resource ×1, latency ×15). Give both rows; the deployment
choice is a resource/latency knob, which itself is a nice rebuttal point.

**(c) Global MHSA over 15 tokens.** Same structure as (b) but 15×15;
one instance.

### Step 4 — Synthesize + collect (0.5 day)
`vitis_hls -f build.tcl` per component; parse the csynth reports
(`*_csynth.rpt`) for the table. Script the parsing — we will iterate.

### Step 5 — Sanity checks before reporting
- DSP count roughly consistent with MAC count / II (order of magnitude).
- Total latency: components in dataflow → max of stage latencies for II,
  sum for end-to-end. State both.
- Compare vs JEDI-Linear published numbers — we expect same ballpark.
  If we're 10× worse on anything, investigate before reporting (likely
  an unroll/partition config issue, not a real architectural cost).

## Fallback (if synthesis fights us past day 3)

Analytical estimate, still per-component:
- DSP: MACs per component / II (state II assumption per component)
- BRAM: weight storage + grid buffer + activation buffers at ap_fixed<16>
- Latency: dataflow graph depth in cycles at stated unroll factors
Deliver the same table shape, labeled "analytical estimates at stated
unroll/II configuration." This still answers the reviewers.

## What NOT to do

- No place-and-route, no timing closure, no bitstream. C-synthesis only.
- No quantization-aware retraining (HGQ). Post-training ap_fixed<16,6>
  is fine for estimates; note in caption that QAT would tighten this.
- Do not report accuracy of the quantized model unless it's free to
  check — the rebuttal claim is resource/latency, not quantized accuracy.
  (If a quick fixed-point inference check is easy via hls4ml's
  `numerical` profiling, do it and report the accuracy delta — but only
  if it costs <2 hrs.)
- Do not silently change the model config to make synthesis easier.
  If a change is needed (e.g., GELU→ReLU for a supported activation),
  flag it prominently — reviewers may catch config mismatches vs paper.

## Escalation

- hls4ml MHA support questions / branch access: Chang Sun.
- Model config ambiguities vs paper: Zihan.
- If GMP scatter turns out genuinely ugly in HLS by Day 2, stop and
  report — we pivot the rebuttal framing to "dense grid layout with
  fixed N makes this static routing" + analytical numbers for GMP only,
  synthesized numbers for everything else.

## Definition of done

1. `shapes.md` — exact layer shapes from the repo code.
2. `results/` — csynth reports (or analytical derivation) per component.
3. `synthesis_table.md` — the deliverable table with caption text ready
   to paste into the rebuttal, including the JEDI-Linear reference row
   and the pre-P&R disclaimer.
4. `notes.md` — any config deviations from the paper, problem areas,
   and the parallel-vs-serialized patch attention tradeoff numbers.
