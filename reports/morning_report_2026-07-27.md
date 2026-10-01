# Morning report — overnight session 3 (first-principles restart)

## TLDR

The first-principles rethink found the real wall had moved — and broke through
it. **Both matched-N headline rows improved again, all bit-exact:**

| Row | Yesterday | **This morning** | vs JEDI-linear |
|---|---|---|---|
| **N=128** | 0.7988 @ 513k LUT | **0.7990 @ 478,214 (27.7%)** — strictly dominates | **4.9×** (was 5.2×) |
| N=128 cheap | — | **0.7957 @ 404,852 (23.4%)**, FF −19% | **4.1×** |
| **N=64** | 0.8036 @ 544k | **0.8029 @ 487,451 (28.2%)** | **6.9×** (was 7.7×) |
| N=128 latency | 93.2 ns | **89.9 ns** (hard_dc=1, +0.35% LUT) | sub-90 ns |

Across the three overnight sessions the matched-N multiples have gone
**6.8×/9.2× → 4.1–4.9× / 6.9×**.

## The first-principles chain that got there

1. **Re-attributed the CONTINUED checkpoints** (nobody had, since the
   continuations transformed them): softmax — the old villain at 32% — is now
   ~3%; the wall is **constant-weight Q/K/V projections at ≈71% of LUT**
   (D111). The old attention-datalane story was obsolete.
2. **JEDI's own paper shows their weights are mostly zero/≤3-bit** — value
   sparsity, which we had never targeted (only bit-widths, D92). Implemented
   **ProxL1**: proximal soft-thresholding of kernel values after each
   optimizer step → exact zeros → CMVM adders vanish. Verified the
   assumption first (float zeros survive quantization as exact zeros).
3. **The quantization lattice amplifies it**: 15% float zeros became **62%
   quantized zeros (74% in Q/K/V)** — weights under half an LSB round to
   zero. Result: −21% LUT at accuracy parity (N=128), −10% (N=64), all
   emulated bit-exact. λ-ladder resolved: 1e-3 is the sweet spot; 3e-4 too
   weak, 3e-3 collapses (N=64) — N=128's 3e-3 arm lands in ~1h and can only
   add.
4. **Striking side-finding:** at the sparse knee the optimizer self-ablates
   the global-attention Q/K matrices, ffn1, and both GMP kernels to 100%
   quantized-zero while holding 0.796 — the architecture's over-provisioning
   made visible, layer by layer.

## Two rebuttal-hardening results (beyond the rows)

- **Our evaluation set is PROVEN to be the canonical hls4ml 260k test
  partition** — the same one JEDI-linear reports RTL-sim accuracy on.
  Proof is calibration-free (label sequences identical under per-event
  leading-pT ordering, 260k/260k; the two stores differ only by an exact
  ×34.831 linear pT recalibration). The rebuttal draft now uses the strong
  wording, keeping the selection-leakage caveat.
- **Sub-90 ns at N=128** from a free solver-setting sweep (hard_dc=1:
  27 stages, 89.9 ns, +0.35% LUT, bit-exact). Tightening the pipeline cutoff
  went the wrong way, exactly as Codex predicted.

## Codex's role tonight (round 8)

Its first-principles audit set the plan: ranked value-sparsity-first (over
low-rank QKV, Q=K tying, solver searches — all rejected for this window with
reasons), supplied the 11-point promotion checklist every traced row passed
(config integrity, quantized-zero accounting, whole-model LUT only, bit-exact
mandatory, no EBOPs-as-proxy), demanded the event-level split proof (class
counts weren't enough — it was right), and predicted the cutoff-sweep outcome.
It also flagged that ProxL1 thresholds ALL kernels, not just QKV — accepted
as a known limitation; the results didn't need the narrower variant.

## Updated deliverables (all in ~/outputs/phatjet/)

- `synthesis_table.md` — sparsity rows at both N, latency variant, split note.
- `rebuttal_draft.md` — strong split wording, 89.9 ns variant. **Numbers in
  the draft's main text still cite the pre-sparsity rows (513k/5.2×,
  544k/7.7×) — update to 478k/4.9× and 487k/6.9× before submitting, or I'll
  do it when n128-wv30 settles the final N=128 row.**
- `DECISIONS.md` — D111–D115 tonight; 40 entries across the three sessions.

## In flight / next

- `n128-wv30` (λ=3e-3 at N=128, ~e876) — lands within the hour; traced if its
  front beats 404,852 LUT at ≥0.795.
- Recommended next asks unchanged and now more valuable: **Vivado synthesis**
  (Chang Sun) — the RTL for every promoted row is on the PVC (`hw/wv10`,
  `hw/n64wv`, `hw/n128pid`, `hw/pid2`, `hw/latsweep`); the sparse rows'
  smaller netlists make synthesis fast.
- Optional if you want one more lever: QKV-only ProxL1 (Codex's follow-up)
  could push the N=64 row further — one GPU-night.

---

**Post-report addendum (wv30 resolved):** λ=3e-3 traced worse than λ=1e-3 at
identical accuracy (extra float zeros were already lattice-zeros) — the
λ-ladder is closed, 1e-3 optimal at both N (D116). The rebuttal draft's main
text now carries the final numbers (4.1–4.9× / 6.9×, deficits 1.00/0.61,
Linformer 2.4× at +0.49). Board is empty; session complete.
