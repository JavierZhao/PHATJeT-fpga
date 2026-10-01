# PHAT-JeT on FPGA: resource, latency, and accuracy evidence

This is the working record of a campaign to produce FPGA evidence for PHAT-JeT
(Patch Hierarchical Attention Transformer for jet tagging) as a CMS Level-1
trigger tagger. It ran from 2026-07-24 to 2026-07-31 to support the
NeurIPS 2026 rebuttal. The toolchain was HGQ2 quantization-aware training
followed by the da4ml trace, which emits fully unrolled Verilog. The target
was `xcvu13p-flga2577-2-e` at 3.33 ns (300 MHz) with II = 1, matching
JEDI-linear (arXiv:2508.15468).

This README is the handoff. **`DECISIONS.md`** holds the full evidence trail
(D1–D175, append-only, newest at the bottom). Read this file first, then the
DECISIONS entries it cites.

---

## 1. State of play: read this before quoting any number

1. **For the rebuttal's deployment claim, this campaign's rows are superseded
   (D173).** A separate GMP-intact study by Zihan fits the real CTL2 envelope:
   ≤10% of the device's LUTs and <100 ns. Its N=64 row is 74.6% at 167,304 LUT
   (9.7%) and 96.6 ns. **That study's artifacts are not in this repo.** It
   still has one open check: with only 3.4% latency margin, confirm that its
   stage count holds when re-pipelined at `latency_cutoff = 3.33` (D173,
   bottom).
2. This campaign now contributes two things:
   - **Evidence that the accuracies are limited by training budget.** On the
     N=128 graph, float reaches 0.8126 and long-schedule QAT reaches 0.8108.
     The gap is 0.18 points (D146, D169).
   - **The verification method.** The DAIS screen predicts Verilator
     bit-exactness in seconds (D164).
3. **No number here comes from Vivado.** LUT and FF figures are da4ml
   pre-place-and-route estimates. DSP = 0, BRAM = 0 and II = 1 hold by
   construction; no synthesis report confirms them. 300 MHz is a target, not
   closed timing. The highest-value external request is still Vivado
   synthesis of the generated RTL (see `reports/VIVADO_HANDOFF.md`;
   contact Chang Sun).
4. **Latency correction (D172).** `extract_hw.py` defaults to
   `PJ_LATENCY_CUTOFF = 4.0` ns, but every latency is quoted at a 3.33 ns
   clock. A 4.0 ns stage cannot run at 300 MHz, so most latencies in the
   reports are about 20% optimistic. The combinational path
   (`comb_latency_ns`) is a floor that no pipelining choice can beat.
5. **⚠ Found while compiling this handoff: the "sub-100 ns" rows are not
   sub-100 ns either.** D172's floor argument was
   applied only to the accuracy rows. The same argument applies to the
   compressed rows quoted as deployable in D170, D173 and
   `VIVADO_HANDOFF.md`:

   | row | quoted latency | stages | `comb_latency_ns` (floor) |
   |---|--:|--:|--:|
   | `n128-wvpid` ep565, 0.7961 (Vivado primary candidate) | 89.9 ns | 27 | **106** |
   | `n128-wv10` ep64, 0.7990 (flagship) | 93.2 ns | 28 | **109** |
   | `n128-pid` ep901, 0.7988, `hard_dc=1` | 89.9 ns | 27 | **108** |

   **One point has already been measured.** D113 re-pipelined the
   `n128-pid` ep901 netlist at a 3.0 ns cutoff and got **37 stages =
   123.2 ns**. The netlist and LUT count were identical. D113 recorded this
   as "cutoff-tightening counterproductive". It is in fact the honest
   figure: only a cutoff ≤ 3.33 ns is consistent with a 300 MHz clock. At
   exactly 3.33 ns, expect roughly 33–34 stages, or about 110 ns. So the
   floor argument holds, and these rows are **above 100 ns** at 300 MHz.
   Still to do: trace the Vivado primary candidate (ep565) at
   `PJ_LATENCY_CUTOFF=3.33`, which is CPU-only and cheap, to get its exact
   figure. The `PJ_LATENCY_CUTOFF` default bug affects every row, not only
   the accuracy rows.

## 2. Headline results (all N = 128, da4ml estimates)

| row | acc | LUT | % VU13P | quoted latency | comb floor | verified | where |
|---|--:|--:|--:|--:|--:|---|---|
| PHAT, accuracy max (`g-ls-s3` ep412) | **0.8108** | 1,570,717 | 90.9% | 166.5 ns (50 stg @ 4.0 cutoff) | 200 ns | DAIS 2k + Verilator 512, exact | D169 |
| PHAT, efficient (`g-ls-s3` ep152) | 0.8101 | 1,461,880 | 84.6% | 166.5 ns | ~198 ns | exact, both | D168 |
| PHAT, independent schedule (`g-x3` ep167) | 0.8103 | 1,567,068 | 90.7% | 166.5 ns | — | exact, both | D169 |
| PHAT, sparsity flagship (`n128-wv10` ep64) | 0.7990 | 478,214 | 27.7% | 93.2 ns | 109 ns | exact | D114 |
| PHAT, sparsity + PID (`n128-wvpid` ep565) | 0.7961 | 385,471 | 22.3% | 89.9 ns | 106 ns | exact | D119 |
| JEDI-linear **in our pipeline**, Aaron's store (`n128-jediaz`) | 0.8109 | 189,048 | 10.9% | 46.6 ns | 52 ns | exact | D127–D128 |
| JEDI-linear **published** (Table II, N=128) | 0.809 | ~98k | — | 82 ns post-P&R | — | — | D7 |

What the table says:
- At accuracy parity PHAT costs about **8×** the LUTs of JEDI-linear traced
  in the same pipeline, and about **16×** the published JEDI figure. The
  "~76×" in `CLAUDE.md` comes from the BatchNorm-era design of D10 and is
  stale. The structural reason has changed too: attention datalanes and
  softmax used to dominate (D8, D20). After compression, the constant Q/K/V
  projections are about 71% of the LUTs (D111).
- GMP adds about +0.5 points and is nearly free in hardware, in both an
  attention host and a linear host (D122, D129). This is the most portable
  positive finding.
- The early accuracy gap to published JEDI was not a pipeline bug. It came
  from the constituent store, and separately from training budget (D127,
  D132).

The full table of every trace (107 rows) is in `evidence/all_hw_results.csv`.
The curated tables are in `reports/synthesis_table.md`.

### How the 0.8108 record was produced (D150–D169)
1. Resume `vu-long` ep773 (0.8077).
2. **Remove** the regional `PJ_ATTN_L1` and `PJ_FFN_L1` bit pressure.
   BetaPID cannot oppose it, because beta is clamped at ≥0.
3. Set LR to 3e-4.
4. Use `PJ_EPOCHS=600`, which gives 42-epoch cosine cycles.
5. Set label smoothing to 0.02.
6. Run BetaPID with a 1.7M EBOPs target and `PJ_INIT_BETA=1e-9`. The
   resume point (1.54M) is already under target, so beta stays near 1e-9.
   In effect the record trained with **no beta pressure**; only the global
   `MonoL1(1e-8)` remained. Some reports describe this step as "a positive
   init_beta near its operating range". The run configs say 1e-9 (see D151
   on why 1e-9 is inert far from target).
7. **Do not kill arms that look converged.** `g-ls-s3` sat flat at 0.8094 for
   about 130 epochs, then climbed to 0.8108.

The exact environment is in `evidence/pvc/runs/g-ls-s3/run_config.json`.
`PJ_SEED` is **not** recorded there. The `-s2/-s3/-s4` suffix is the seed by
convention (D158).

## 3. What was tried: closed directions

Each row below is closed; do not re-run it without a new reason. The verdict
is in the cited D entry.

| direction | verdict | D |
|---|---|---|
| Vitis HLS + hls4ml | replaced by HGQ2 + da4ml | D1 |
| pT validity mask on GMP scatter | rejected | D5 |
| BatchNorm on every Dense | dropped; tanh pre-norm only | D14 |
| Global bit-width reduction | per-region attention pressure is far better | D11, D22, D38 |
| Reducing N to cut attention cost | attention LUT does not scale with MAC count | D17, D21 |
| Ranking checkpoints by EBOPs | LUT/EBOP is not monotonic; select by traced LUT | D19, D24, D167, D174 |
| SAT_SYM overflow | does not transfer to the region-pressured recipe; keep wrap | D45, D49 |
| Aaron's core7 7×7 GMP grid | no gain under any port or store | D47, D55, D79 |
| Solver options (`hard_dc`) | LUT unchanged; `hard_dc=2` adopted (`=1` buys one stage on small designs) | D56, D113 |
| Table (LUT) pressure, floored or not | neutral to worse; unfloored version was reward-hacked | D59, D61, D72 |
| Sigmoid attention (naked, biased, hybrid) | diverges; family closed | D63, D73, D80 |
| ReLU attention | works, but is a latency/FF operating point, not a LUT win; softmax dominates at N=128 | D65, D69, D90 |
| Bounded ReLU | fails trace screen | D81 |
| Block-QKV factorisation | regression once the wiring bug was fixed | D66, D68 |
| AV-only pressure | refuted; ReLU's wide lanes are load-bearing | D74 |
| GMP 5×5 kernel / 16×16 grid | dominated / no benefit | D83, D88 |
| Weight-bit pressure | no benefit over region pressure | D92 |
| Distillation (α = 0.5, 0.15) | fails under EBOPs pressure; cannot reach the knee | D95, D98, D100 |
| Two-stage float initialisation | wins early, loses at the knee | D96 |
| d_model = 12 / d_model = 32 + ProxL1 | unstable / no compression traction; d16 is right | D106, D128 |
| Small-N modern recipe (N=16/32) | old rows stand; sort and patch effects invert between N | D108–D110 |
| λ-ladder for value sparsity | 1e-3 optimal at both N | D116 |
| Lane protection for ramp tolerance | failed pre-registered criterion | D125 |
| Structural changes to close the JEDI gap | gap survived all variants; attention is "prunable but not omittable" | D133 |
| Capacity route (d32/FFN×4, `g-cap`) | not reachable in time | D147 |
| Float warm-start into QAT | closed three ways | D154, D160 |
| Stripping quantizers to measure "quantization cost" | invalid for wrap-overflow HGQ2 (0.8077 → 0.3069) | D143 |

**Directions that worked, in order of impact:** per-region attention/FFN
bit pressure for area (D22, D33, D37), weight-value sparsity via ProxL1
(D114), BetaPID continuation (D78, D85), and GMP (D129). For accuracy at a
loose budget: removing regional pressure plus a low-LR long continuation
(D150, D168).

## 4. Open items, highest value first

1. **Re-pipeline at `PJ_LATENCY_CUTOFF=3.33`** for the compressed rows (§1.5)
   and for the GMP-intact study (D173). Then change the code default to the
   clock period. The code has deliberately not been changed yet (D172 action
   item).
2. **Vivado synthesis** of the RTL listed in `reports/VIVADO_HANDOFF.md`.
   This upgrades every estimate to a synthesis result. If §1.5 is confirmed,
   the handoff's primary candidate should be re-pipelined before synthesis.
3. **A PHAT arm on Aaron's raw store with zscore inputs.** This would show
   whether the +0.65-point store gain that JEDI got transfers to PHAT (D127
   open question).
4. **Interface/I/O story.** 8,832 input bits per jet per cycle is assumed to
   be an on-chip handoff (`reports/round15_absolute_slr_argument.md` §B(v)).

## 5. Statistical caveats that travel with every accuracy

- The evaluation partition is the standard hls4ml test split (2.6e5 jets;
  identity proven in D113). Binomial stderr is about 0.077 points. 0.8108 vs
  0.809 is about 0.2σ, which is **parity, not a win**.
- Checkpoint selection used the same partition, so quoted peaks are
  selection-biased maxima (D163).
- Seed spread at the knee is about 0.75 points at N=128 (D101) and about
  1.5 points end-of-schedule (D87, D93).
- "Bit-exact" means exact on the sampled jets: 2,000 for DAIS and 512 for
  Verilator. It is not a global proof (D163). About 40% of continuation
  checkpoints fail it; a failing checkpoint carries no accuracy claim (D70,
  D162).

## 6. Reproducing and continuing

**Environment:** `pip install "keras>=3.11" hgq2 da4ml jax[cpu] numpy torch`.
The package is **`hgq2`**. `pip install hgq` installs HGQ v1, which pins
`keras<3`.

**Pipeline:**
```
phat_jet_k3.build_phat_jet_k3(quantized=False) -> train_float.py -> float ceiling
                             (quantized=True)  -> train_qat.py   -> pareto/*.keras
       extract_hw.py <ckpt>  -> LUT / latency / Verilog / DAIS + Verilator bit-exactness
       component_costs.py    -> per-component LUT attribution (prefix differencing)
```
Everything is configured by `PJ_*` env vars. **The trace env must match the
checkpoint's training config exactly**, because weights load into a graph
rebuilt from env. Code defaults differ from the record recipe in ways that
fail silently (D141). See `evidence/run_config_env_map.md`.

**Cluster** (NRP Nautilus, namespace `cms-ml`, job prefix `zz-`). Source
ships as ConfigMap `zz-phatjet-src`, which matched this repo's code on
2026-09-30. The job generator was lost. `k8s/job_template.yaml` is a
**reconstructed, untested** replacement that encodes every gotcha:
`/bin/bash -c`, `set -o pipefail`, the torch-vs-jax trade-off, and
`PJ_DATA=""` for traces. See `CLAUDE.md` for the full list of gotchas.

**Data and artifacts** live on PVC `j-jepa-vol` under `/data/phatjet-fpga/`
(about 12 GB, not in this repo):

| path | contents |
|---|---|
| `jets_150x3_kt.npz` | our robust-normalised, kt-sorted store (`prep_data.py`) |
| `jets_128x3_aaron_ourunits_kt.npz` | Aaron's store in our units, kt-sorted; **used by the record runs** |
| `jets_128x3_aaron_*.npz` | other Aaron-store variants (D58, D62, D102) |
| `runs/<run>/` | `train.log`, `run_config.json`, `qat_hist.json`, `pareto/` (193 runs) |
| `hw/<tag>/` | traces: `hw_results.json`, RTL (`verilog_*/` with `build_vivado_prj.tcl`), logs (73 dirs, 3.7 GB) |
| `frozen/` | 62 checkpoints copied out of `pareto/`. **`ParetoFront` deletes dominated checkpoints**, so freeze anything you quote (D171). The ep152 0.8101 checkpoint was lost this way; its RTL survives in `hw/trace101/`. |
| `teachers/`, `aaron/` | float teachers; Aaron's shared files, including his raw `jets_128x3.npz` (the source of the `aaron_*` stores) |

**Data is not distributed with this repo; use the PVC copies.** Set `PJ_DATA`
to the file the source run used (its `run_config.json` records it). For the
record line that is `/data/phatjet-fpga/jets_128x3_aaron_ourunits_kt.npz`.
Trace-only jobs can run with `PJ_DATA=""` and still give LUT/latency, but
without data they skip accuracy and both bit-exactness checks.

## 7. Repository layout

| path | what |
|---|---|
| `DECISIONS.md` | decision log D1–D175: the primary record |
| `CLAUDE.md` | instructions for coding agents working in this workspace |
| `claude_code_fpga_handoff.md` | the original brief (toolchain diverged by D1) |
| `phat_jet_k3.py` | synthesizable model; float and quantized from one code path; also `build_jedi_linear` |
| `train_qat.py`, `train_float.py` | HGQ2 QAT / float reference |
| `prep_data.py` | builds the kt-sorted store |
| `extract_hw.py`, `component_costs.py` | trace to hardware numbers / per-component attribution |
| `calib_screen.py`, `float_eval_qat.py`, `sparsity_report.py` | D157 admission screen / D143 negative-result diagnostic / zero-fraction report |
| `analytical_estimates.py`, `extract_shapes.py` | op counts (Track A) |
| `make_pareto_fig.py`, `make_slr_fig.py`, `sweep_status.sh` | figures, job status |
| `reports/` | dated session reports, `synthesis_table.md` (curated tables), `VIVADO_HANDOFF.md`, figures |
| `evidence/all_hw_results.csv` | every trace flattened (`evidence/compile_hw_results.py` regenerates it) |
| `evidence/pvc/` | `run_config.json` / `qat_hist.json` for every run, plus `hw_results.json` / `component_costs.json` / RTL `metadata.json`, pulled from the PVC 2026-09-30 |
| `evidence/pvc_*_inventory.txt`, `evidence/train_log_last_epoch.txt` | what is on the PVC: frozen checkpoints, top Pareto points per run, RTL dirs, last logged epoch per run |
| `evidence/run_config_env_map.md` | run_config key → `PJ_*` env var, with defaults |
| `k8s/job_template.yaml` | reconstructed job spec |
| `UPSTREAM.md` | pinned upstream repos (PHAT-JeT, JEDI-linear, HGQ2-examples) |

**Left out of this public repo on purpose:** the submission PDFs, rebuttal and
reviewer-response drafts (which may quote confidential reviews), Aaron's
shared package, data and checkpoint, private messages, the 475 MB local npz,
and the RTL (about 130 MB for the headline designs; it stays on the PVC). The
original brief has had the review scores removed.

## 8. Contacts

- Chang Sun (`calad0i`): HGQ2/da4ml, Vivado synthesis.
- Zihan: model config vs paper, the GMP-intact study.
- Aaron Wang (`aaronw5`): upstream PHAT-JeT, data store.
