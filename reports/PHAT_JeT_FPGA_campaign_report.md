# PHAT-JeT FPGA Evidence Campaign — Full Report

**NeurIPS 2026 submission 28409 rebuttal · 2026-07-24 → 2026-07-27 · three overnight sessions + two push days**

---

## 1. Executive summary

The goal was FPGA resource/latency evidence that PHAT-JeT — a patch-hierarchical
attention jet tagger for the CMS Level-1 trigger — is FPGA-compatible, compared
against JEDI-linear, the deployed-class linear GNN baseline. The campaign began
with a design that did not fit the target device at useful accuracy (431% of a
VU13P at the recipe's starting point) and a naive gap of ~76× in LUT at accuracy
parity.

**Final state: every promoted row fits in 22–31% of a VU13P at II=1 with 0 DSP
and 0 BRAM, is RTL-emulated bit-exact against its trained model, and is
evaluated on the same canonical 260k-jet test partition JEDI-linear reports on.**

| Row | Accuracy | LUT (da4ml est.) | % VU13P | Latency | FF | vs JEDI same N |
|---|---:|---:|---:|---:|---:|---:|
| **N=128 cheap champion** | 79.61% | **385,471** | **22.3%** | **89.9 ns** | 270,104 | **3.9×** |
| **N=128 flagship (accuracy)** | 79.90% | 478,214 | 27.7% | 93.2 ns | 358,023 | 4.9× |
| **N=64 headline** | 80.29% | 487,451 | 28.2% | 136.5 ns | 449,527 | 6.9× |
| **N=64 cheap** | 80.19% | 451,836 | 26.1% | 139.9 ns | 442,287 | 6.4× |
| N=32 mid-frontier | 77.04% | 642,141 | 37.2% | 139.9 ns | 535,932 | ~14× |
| N=16 (pre-existing rows stand) | 72.26–74.01% | 365,950–957,257 | 21–55% | 133–157 ns | — | ~8–22× |

Reference points: JEDI-linear (post-route, same device class, pT-sorted
3-feature) N=128: 80.9% @ 98k LUT / 82 ns; N=64: 80.9% @ 71k / 61 ns. The
closest FPGA transformer (Linformer, 64p, different FPGA): 79.8% @ 202k / 78 ns
— our N=64 headline is +0.49 pts at 2.4× its LUT.

**The honest claim** (as written in the rebuttal draft): PHAT-JeT is
accuracy-competitive but not resource-comparable — 3.9–6.9× JEDI-linear's LUT at
matched particle counts, 0.6–1.0 accuracy points below — while demonstrating
bit-exact II=1 RTL, comfortable device fit, a mapped and replicated
accuracy/area frontier, and a substantial narrowing of the "transformer on
FPGA" gap. The campaign's arc: **76× → 3.9×** at the best matched-N point.

---

## 2. How the gap closed: the five-stage recipe stack

Each stage was adopted only after controlled evidence, and each survived
adversarial audit (11 Codex review rounds):

1. **Synthesizable re-expression** (pre-campaign): static shapes, functional
   ops, GMP as one-hot indicators + einsums; tanh-only pre-norm (no LayerNorm,
   no BatchNorm); FFN 4×→1×. Float ceiling of this exact graph: 81.10%.
2. **Region-targeted quantizer pressure** (the single most important training
   mechanism): nested datalane L1 scopes on the attention (1e-5) and FFN (3e-5)
   construction sites, beta_scale 0.01. Global beta alone misleads — LUT/EBOP
   varies 1.22–1.92 across designs, so all ranking was done by trace, never by
   EBOPs.
3. **Continuation training** (D78–D86): the 1000-epoch runs were undertrained
   at their operating points. A second 1000 epochs (PID-held or constant beta —
   both converge to the same knee) sheds 17–26% EBOPs/LUT at held accuracy.
   The N=64 knee was independently replicated by two control schemes and a
   second seed; pushing below it collapses accuracy (a genuine wall at ~400k
   EBOPs).
4. **Weight-value sparsity** (D111–D116, the breakthrough): fresh attribution
   of the *continued* checkpoints showed the cost wall had moved — softmax
   (once 32%) was down to ~3%, and **constant-weight Q/K/V projections were
   ≈71% of LUT**. JEDI's own weight histograms show their weights mostly
   zero/≤3-bit. We added ProxL1 — proximal soft-thresholding of kernel values
   after each optimizer step, producing exact zeros that vanish from
   distributed-arithmetic netlists. The quantization lattice amplifies it:
   15% float zeros → **62% quantized zeros (74% in Q/K/V)** at N=128.
   Result: −21% LUT at accuracy parity. λ=1e-3 is the best tested setting at
   both N; weaker under-harvests, stronger adds only float zeros that were
   already lattice-zeros (and collapses at N=64).
5. **Sparsity + datalane pressure combined** (D119, the finale): with entire
   weight columns dead, PID pressure re-narrows the datalanes they no longer
   feed (LUT/EBOP 0.98) and the shallower carry chains drop a pipeline stage:
   **385,471 LUT at 89.9 ns — sub-4× and sub-90 ns in one row.**

---

## 3. Key scientific findings

- **The knee/wall structure is real and reproducible.** At N=64, two beta-control
  schemes and two seeds converge to the same operating point (~0.80 @ ~400–430k
  EBOPs); below it accuracy cliffs (0.73 @ 360k). The continuation gain is
  optimization time, not controller sophistication.
- **Cost attribution is dynamic.** The "attention datalanes dominate" story
  (66–74% of LUT at 9% of arithmetic) was true pre-compression and is the right
  *structural* explanation of the gap to a linear GNN — but after compression
  the residual cost is constant-CMVM projections, which value sparsity attacks.
  Both facts are in the rebuttal (the structural one held in reserve).
- **The quantization lattice amplifies value sparsity ~4×**: most small weights
  sit below half an LSB and round to zero once nudged. Float-zero counts
  understate hardware sparsity; all accounting was done on quantized kernels.
- **Self-ablation**: at the sparse knee the optimizer drives the global-attention
  Q/K matrices, the FFN first layer, and both GMP kernels to 100% quantized-zero
  while holding ~79.6% — the architecture's over-provisioning made visible
  layer-by-layer. Topology is unchanged; some learned blocks become functionally
  inactive (stated plainly in the draft).
- **Per-N design choices invert.** pT-sorted input beats kt at N=32 (+2 pts) but
  loses at N=16 (−3 pts) — at small N the sort decides *which* particles survive
  the head slice. P=4 patches beat P=8 at N=32 (compressed regime) but not at
  N=16. The modern recipe's gain scales with attention share: −23% (N=128),
  −17% (N=64), mid-frontier only (N=32), nil (N=16).
- **Seed structure**: end-of-schedule snapshots vary ~1.5 pts across seeds
  (undertrained points are noisy); continued knees vary 0.1–0.3 pts (N=64) and
  ~0.75 pts (N=128, pre-sparsity); the sparse recipe replicates at ~0.4 pts /
  4% LUT (n=2). The rebuttal quotes the applicable bar for each claim.
- **Evaluation-split identity**: our "validation" set is the canonical hls4ml
  260k test partition — proven by 100% label-sequence agreement under
  calibration-free per-event ordering (the stores differ only by an exact
  ×34.831 linear pT recalibration). We evaluate on the same jets as the
  baseline; the retained caveat is that this partition also drove checkpoint
  selection.

---

## 4. Verification standard

Every promoted row passed the same gate (formalized as the D112 checklist):
- Architecture rebuilt from recorded config (`run_config.json`); weights loaded
  by name into the matching graph; effective pod env verified at launch.
- Whole-model da4ml trace at `inputs_kif=(1,8,16)`, solver `hard_dc=2`
  (variants explicitly labeled), xcvu13p @ 3.33 ns target.
- **RTL emulation (Verilator) bit-exact against the Keras model: max|err| = 0,
  100% argmax agreement** — on every one of the ~15 promoted checkpoints. One
  historical non-bit-exact trace (an early relu checkpoint) was quarantined and
  never promoted.
- No EBOPs-based ranking anywhere (EBOPs does not credit value-zeros and its
  LUT ratio varies 0.95–1.92).
- Buildable Verilog (sanitized module names) archived on the PVC per row.

Standing caveats, stated wherever the numbers appear: all PHAT-JeT figures are
**da4ml pre-place-and-route estimates at a 300 MHz target** (JEDI's are
post-route); accuracies are operating-point estimates on a partition also used
for selection; the synthesized graph is a single PHAT block with enumerated
deviations from the full paper model; sparse rows carry an n=2-seed ~0.4-pt bar.

---

## 5. Everything tested — the closed ledger

Directions that failed or added nothing, each closed with recorded evidence
(DECISIONS.md D-numbers):

| Direction | Verdict |
|---|---|
| Sigmoid attention (naked, biased, hybrid) | Diverges or fails — 4 strikes (D63/D73/D80) |
| ReLU attention | Verified variant; softmax dominates both N after continuation (D69/D90) |
| Bounded relu (clamp) | Killed by 5-minute CPU screen: −0.9% LUT, accuracy collapse (D81) |
| GMP enrichment (5×5 kernel, 16×16 grid) | Both dominated; GMP is at capacity at 3×3/8×8 (D83/D88) |
| Weight-bit pressure (br) | No EBOPs reduction at any accuracy (D92) |
| Distillation (α=0.5, α=0.15, knee-tested) | KD term acts as an EBOPs floor; can't reach the knee (D95/D100) |
| Two-stage float init (proper 81.1% teacher) | Loses to from-scratch at the knee (D96) |
| Aaron's core7 non-uniform grid | Clean A/B on his own data: Δ0.0006 acc, +8% EBOPs (D79) |
| Aaron's recipe end-to-end | Replicated exactly, then shown dominated (his beta ramp collapses past his log's horizon) (D103) |
| Aaron's dataset under our recipe | Behind our store at matched schedule (D110) |
| d_model=12 | Unstable at per-head dim 3; killed by pre-registered criterion (D106) |
| Single-head, block-QKV, Linformer local, table pressure, AV-only pressure, SAT_SYM, pT mask | All closed in earlier sessions with evidence |
| λ=3e-4 / λ=3e-3 sparsity | Under-harvests / adds only non-lattice zeros (D114/D116) |
| Pipeline-cutoff tightening | More stages, worse latency (predicted and confirmed) (D113) |

Two levers worked besides the main stack: `hard_dc=1` pipelining (one stage
faster, area-neutral) and per-N sort/patch choices at N=32.

---

## 6. Incidents caught by process discipline

- **Teacher corruption (D89):** a cuDNN-killed float stage + skip-if-exists
  wrapper silently fed a half-trained (0.74) teacher to three arms. Caught by
  an audit-mandated teacher-accuracy print; fixed with a frozen-teachers
  convention + quality gate; all three arms rerun clean.
- **Five silent-sed failures** caught by the grep-back rule (D70) — including
  a missing `PJ_ATTN_ACT` and a missing `PJ_SORT` that would each have produced
  wrong-graph or wrong-data traces.
- **Extraction didn't reproduce the pT sort** (caught by Codex before any
  pT-trained checkpoint was traced) and a floor-only quantizer scope that was
  silently inert — both fixed.
- **Non-NaN divergences** (sigmoid family, distillation): TerminateOnNaN added
  after one arm burned 620 epochs on NaN; finite collapses still require
  reading the curves, which the monitoring loop did.
- **Mis-wired weight regularizer** (D77): `fr` vs `br` key silently dropped by
  the config scope — confirmed against library source, fixed, and the arm
  relaunched (also leaked pressure removed).

---

## 7. Codex's role (11 rounds)

Codex served as researcher, auditor, and red-team throughout: it caught the
mis-wired regularizer and the resume-overwrite bug before they cost nights,
demanded the event-level split proof (class counts weren't enough — correct),
ranked the lever queue each round (its value-sparsity-first ranking on the
final night was decisive), supplied the promotion checklist all rows passed,
predicted the cutoff-sweep outcome, red-teamed the figure and the draft
wording, and directly contributed the rebuttal text (round 6) and final caveat
edits (round 9). Its rejections mattered as much as its proposals: low-rank
QKV, Q=K tying, solver searches, and re-runs of closed directions were all
declined with reasons and stayed declined.

---

## 8. Deliverables (all in `~/outputs/phatjet/`)

| File | Contents |
|---|---|
| `rebuttal_draft.md` | Submission-shaped author-response text: main FPGA paragraph, four objection responses, red-team notes. Final numbers throughout. |
| `synthesis_table.md` | Every verified row with caveats, reference tables, limitations. |
| `pareto_fpga.pdf/.png` + `make_pareto_fig.py` | Frontier figure (red-team hardened) + generator for provenance. |
| `VIVADO_HANDOFF.md` | Turnkey synthesis package for Chang Sun — 5 prioritized RTL dirs with expected numbers (add `hw/final-day/verilog_epoch_565_*` as new priority 1). |
| `DECISIONS.md` | The complete decision log: 119 entries, every claim traceable to evidence. |
| `morning_report_2026-07-26.md`, `morning_report_2026-07-27.md`, `final_day_report_2026-07-27.md` | Session narratives. |
| PVC (`/data/phatjet-fpga/`) | All checkpoints, train logs, buildable RTL (`hw/*`), teachers, Aaron replication tree. |

---

## 9. What remains

1. **Vivado synthesis (external, Chang Sun)** — the only step that changes the
   evidence class from estimate to synthesis result. Package is ready.
2. **Submit-day polish** of the response text (audited twice; read it once
   fresh before submission).
3. Optional, post-rebuttal: QKV-only ProxL1 (cleaner attribution of the sparsity
   gain), a third sparse seed, and the N=64 lattice-harvest question (why its
   asymptote is shallower) — scientific loose ends, not rebuttal blockers.

---

*Every number in this report is traceable to a DECISIONS.md entry and a PVC
artifact; every promoted row's RTL exists and emulated bit-exact. Compiled
2026-07-27, end of campaign.*
