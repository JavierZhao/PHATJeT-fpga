# Final push-day report — campaign closed

## TLDR

The last day's first-principles re-evaluation delivered the campaign's best
row and its honest error bar. **Final verified matched-N state (all
RTL-emulated bit-exact):**

| Row | Result | vs JEDI-linear |
|---|---|---|
| **N=128 cheap champion (NEW)** | **0.7961 @ 385,471 LUT (22.3%), 89.9 ns**, FF 270,104 | **3.9×** |
| N=128 flagship (accuracy) | 0.7990 @ 478,214 (27.7%), 93.2 ns | 4.9× |
| N=64 headline | 0.8029 @ 487,451 (28.2%), 136.5 ns | 6.9× |
| N=64 cheap (NEW) | 0.8019 @ 451,836 (26.1%) | 6.4× |

Where the campaign started three nights ago: 6.8× / 9.2×. **The final N=128
row is sub-4× and sub-90 ns simultaneously.**

## Today's chain

1. **N=64 harvest** — the tail of the sparsity run held better points than
   the mid-run trace; a harvest continuation banked 0.8019 @ 451,836 (D118).
2. **Sparsity + PID (never combined before)** — with 62% of quantized weights
   zero, PID pressure re-narrowed the datalanes the dead columns no longer
   feed (LUT/EBOP 0.98) and dropped a pipeline stage: 385,471 LUT at 89.9 ns
   (D119). This one row is the campaign finale.
3. **Sparse-recipe seed replication** — a second seed reached 0.7947 @
   496,540 vs 0.7990 @ 478,214: **~0.4-pt / 4%-LUT cross-seed bar (n=2)**,
   now stated verbatim in the rebuttal draft (replacing the pre-sparsity
   0.75-pt figure that didn't apply).
4. **Codex round 9** red-teamed everything and directly edited the draft and
   decision log (caveat corrections — reviewed and kept): per-row sparsity
   fractions, "topology unchanged but blocks functionally inactive",
   "best tested λ" not "optimal", split-identity phrasing.

## Deliverables (all in ~/outputs/phatjet/, all current)

- `rebuttal_draft.md` — final numbers (3.9–4.9× / 6.4–6.9×, deficits
  1.00/0.61, Linformer 2.4× at +0.49), sparse seed bar, split-identity
  claim with selection caveat, objection responses, red-team notes.
- `synthesis_table.md` — all verified rows including today's two.
- `pareto_fpga.pdf/.png` + `make_pareto_fig.py` — frontier figure, red-team
  hardened (frontier-only mains, log-scale labeled, methodology split per
  series, caveats in-figure).
- `VIVADO_HANDOFF.md` (also at `hw/README_VIVADO.md` on the PVC) — turnkey
  package for Chang Sun: 5 prioritized RTL dirs, expected numbers, notes.
  **Note: dirs from before today's finale — add
  `hw/final-day/verilog_epoch_565_*` (the 385k row) as new priority 1 when
  sending.**
- `DECISIONS.md` — D111–D119 today/tonight; 44 entries across the campaign.

## What remains (external / your call)

1. **Send the Vivado package to Chang Sun** — the single highest-value
   remaining step; upgrades every number from estimate to synthesis result.
2. Submit-day polish of the response text (it is currently
   submission-shaped; Codex and I have both audited it).
3. Board is empty; no cluster jobs running. Nothing left that GPU-nights
   can improve before the deadline.
