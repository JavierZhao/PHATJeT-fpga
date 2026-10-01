# notes.md — handoff definition-of-done item 4

Config deviations, problem areas, and trade-offs. This file is the index; the
full evidence trail is `DECISIONS.md` (43 entries) and `synthesis_table.md`.

## Config deviations from the paper (all disclosed)

| Deviation | Why | Where recorded |
|---|---|---|
| N = 128/64 instead of 150 | matched comparison vs JEDI-linear (reports 16/32/64/128) | D40 |
| P = 8 instead of 10 | P must divide N=128 | D40 |
| No LayerNorm; tanh pre-norm LUT instead | runtime div/sqrt hostile to II=1; tanh dominated BN in ablation | D14 |
| FFN 16→16 instead of 16→64 | 4x→1x measured better on both axes at the cheap end | D37/D41 |
| Mean patch tokenizer (sum-folded 1/P) | paper Table 16 + measured; exact reparameterisation | D-mean rows |
| Fixed GMP grid (one-hot bins) | da4ml requires static routing | original port |
| core7 7×7 non-uniform grid (under test) | Aaron's physics study: +2.0 acc at lower cost | D43 |
| Per-region quantizer pressure (attn_l1/ffn_l1) | the main resource lever, saturates at 1e-5..3e-5 | D22/D33/D35/D37/D41 |

## Problem areas / open items

- **Estimates, not synthesis.** No Vivado anywhere. RTL + `build_vivado_prj.tcl`
  are generated; the ask to Chang Sun is one command. Pre-P&R vs JEDI-linear's
  post-P&R asymmetry favours them.
- **Single seed everywhere.** Margins vs JEDI-linear (+1.1 to +1.5 at small N)
  exceed within-arm scatter but are not established against seed variance.
- **EBOPs is a misleading ranking metric** — trace before ranking (D24/D36/D38).
- **RTL generated before 2026-07-25 has illegal module names** and will not build
  in Vivado/Verilator; regenerate with current `extract_hw.py` (fixed).
- **Two-stage QAT init not yet ported** (Aaron's package: float 81.59% → QAT
  start 81.21%; needs wide i0 or it wrap-collapses silently). Largest remaining
  accuracy lever. D43.

## Trade-off tables

The parallel-vs-serialised patch attention question in the original handoff is
superseded: the design is fully unrolled at II=1 throughout, and the measured
trade-off surface is area/latency/accuracy via quantizer pressure
(`synthesis_table.md`, DECISIONS D34/D42).
