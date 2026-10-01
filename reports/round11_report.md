# Round 11 research advising: closing the PHAT–JEDI hardware gap

**Date:** 2026-07-28  
**Scope:** analysis only; no source or Kubernetes jobs were modified.

This report treats the traced rows in D122 as the current evidence and the
running `n128-wvpid2`, `n128-sat`, `n128-d32`, and `n128-d32wv` arms as
**pending**. No new result from those arms was found in the local workspace.
All LUT figures below are da4ml estimates unless stated otherwise.

## A. Ramp-tolerant attention

### Findings

**Finding — per-block beta is feasible in the installed HGQ2, but
`LayerConfigScope(beta0=...)` alone does not implement it.** The installed
`LayerConfigScope` accepts `beta0` and `enable_ebops`, and each Q-layer creates
its own non-trainable `_beta` when EBOP accounting is enabled. However, the
installed `BetaScheduler.on_epoch_begin()` walks every flattened model layer
and assigns the same scheduled beta to every layer with `_beta`. Thus a nested
`LayerConfigScope(beta0=0)` only controls initialization; the first scheduler
step erases the distinction. This is also how the current model is built:
`train_qat.py:238-239` places the entire graph under `LayerConfigScope(beta0=0)`,
then `train_qat.py:413-419` constructs one global schedule and
`train_qat.py:503-504` applies it to all Q-layers.

**Finding — the current “per-region beta” comments describe quantizer
regularization, not the layer EBOP beta.** `_attn_scope` changes the `kif`
fractional-bit regularizer/floor at the QK/AV construction sites
(`phat_jet_k3.py:96-128`); Q/K/V projections are created before that scope
(`phat_jet_k3.py:339-346`, `358-365`). This has been highly effective
(D22/D33/D34), but it is not selective scheduling of the EBOP loss.

**Finding — the protection boundary can be named exactly.** Local and global
data×data layers are `local_attn_qk`, `local_attn_softmax`, `local_attn_av`,
`patch_attn_qk`, `patch_attn_softmax`, and `patch_attn_av`; the fine attribution
cut list independently relies on those names (`component_costs.py:65-79`).
Protecting only QK/softmax/AV is the narrow test of “spare data lanes feeding
data×data products.” Protecting `*_wq`, `*_wk`, and `*_wv` too tests the broader
attention-block hypothesis, but it also forfeits pressure on the constant
projections that now account for roughly 71–75% of PHAT’s residual cost (D122).

**Hypothesis — a two-rate schedule can preserve attention while allowing the
full ramp to prune constant layers.** D6/D103 establish that PHAT collapses
under the full ramp, whereas D122 shows that the same ramp yields 63.7%
quantized zeros in JEDI without ProxL1. A selective schedule directly tests
causality. It is not yet a finding that attention beta, rather than some other
interaction, causes the collapse.

### Concrete implementation sketch

Prefer a selective callback over `enable_ebops=False`: disabling EBOPs would
also remove the protected layers from `FreeEBOPs`, corrupting the Pareto metric.

1. In `train_qat.py`, add environment knobs such as
   `PJ_ATTN_BETA_SCALE` (default `1.0`) and
   `PJ_ATTN_BETA_MODE={lanes,block}`.
2. Replace the stock `BetaScheduler` only when the attention scale differs.
   Subclass/copy its small callback and, at epoch begin:

   ```python
   beta = global_schedule(epoch)
   protected = lane_names if mode == "lanes" else block_names
   for layer in model._flatten_layers():
       if getattr(layer, "_beta", None) is None:
           continue
       scale = attn_beta_scale if layer.name in protected else 1.0
       layer._beta.assign(beta * scale)
   ```

   Log both `beta` and `attn_beta`. Use an explicit allow-list, not substring
   matching, and assert that all expected protected layers were found. The audit
   should also print a layer-name/beta table at epochs 0, the ramp breakpoint,
   and the last epoch.
3. `phat_jet_k3.py` need not change for the first experiment because the
   relevant layers already have stable names. If a construction-time version
   is desired, wrap only the `_einsum`/`_softmax` calls at
   `phat_jet_k3.py:345-349` and `364-368` in
   `LayerConfigScope(beta0=...)`; this still requires a selective scheduler.

Recommended first screen: two matched from-scratch 1000-epoch N=128 arms using
the current PHAT recipe, no ProxL1: control `PJ_BETA_SCALE=1.0`, versus
`PJ_BETA_SCALE=1.0 PJ_ATTN_BETA_MODE=lanes PJ_ATTN_BETA_SCALE=0.01`.
The protected value `0.01` is motivated by the successful PHAT global scale,
not claimed optimal. Only if lanes-only still collapses should the second arm
protect the full attention block. Do not launch a broad scale sweep first.

### Audit gate and recommendation

Promote “ramp tolerance” only if the protected arm (i) avoids the collapse at
the same epochs where the control collapses, (ii) reaches a matched accuracy
knee, (iii) shows materially more zero-bit/quantized-zero constant weights than
the existing `BETA_SCALE=0.01 + ProxL1` PHAT row, and (iv) passes full da4ml
trace and RTL emulation with `PJ_INPUTS_KIF=1,8,16` (the D30/D31 gate). Trace
LUT, not EBOPs, selects the winner (D24/D112). **Recommendation:** this is the
highest-leverage new training experiment, but implement and smoke-test the
selective callback before spending a full GPU allocation.

## B. Pipeline depth: structural versus tool-driven

### Findings

**Finding — PHAT contains two serial attention chains, while JEDI has only
affine/nonlinear/pooling chains.** Each PHAT MHA executes constant Q/K/V
projections, then data×data QK, normalization, data×data AV, and output
projection (`phat_jet_k3.py:334-370`). Local attention is followed by patch
tokenization and global attention, so these two chains are serial at graph
level. `analytical_estimates.py` makes the same first-principles decomposition:
softmax depth is exp-table → reduction tree → reciprocal-table → rescale, and
QK/AV add multiplier plus reduction depths. No scheduler can remove those
dependencies.

**Finding — the reported “stages” are da4ml timing partitions, not explicit
architectural registers.** `extract_hw.py` first creates a combinational trace
with `HWConfig(1,-1,-1)` and `comb_trace`, then calls
`to_pipeline(comb, latency_cutoff=4.0)` and reports
`len(pipe.solutions)` (`extract_hw.py:168-191`). The nominal output latency is
then stage count × 3.33 ns even though the partition cutoff defaults to 4.0 ns.
Consequently, stage count depends on graph dependency depth, learned bitwidths
and node delays, and the partition heuristic.

**Finding — solver/precision choices move stage count, but observed solver-only
upside is small.** D56 found `hard_dc=2` saved one stage with essentially
unchanged LUT; D113 found `hard_dc=1` saved one more stage on the flagship
(28→27), while tightening the cutoff to 3.0 ns increased stages to 37.
Conversely, stronger attention pressure reduced an earlier design from
45–48 stages to 32 (D34), and continuation reduced the N=128 row from 40 to 28
(D86). Thus much of the variation is quantized arithmetic delay, but the
remaining 28-stage PHAT versus 14-stage JEDI gap in D122 is consistent with
real serial structure.

**Hypothesis — a modest additional reduction without architecture change is
plausible; JEDI-like 14 stages is not.** One or a few stages might be available
from different learned widths, selective ramping, or a checkpoint-specific
`hard_dc` sweep. Halving 28 stages while retaining both exact attention chains
has no support in the current traces. Removing/approximating a normalization,
eliminating global attention, or fusing/replacing an attention stage would be an
architecture change.

### Audit gap and recommendation

The current component attribution records cumulative combinational latency but
not cumulative pipeline stages (`component_costs.py:11-30`), so it cannot
quantify exactly how many of 28 stages belong to QK, softmax, and AV. Add a
future **analysis script** that runs `to_pipeline` on every existing fine
prefix, with identical solver/cutoff, then report incremental stage counts with
the same CSE caveat as marginal LUT. This is cheap trace work, not training.
**Recommendation:** do not promise a shallower PHAT from solver tuning. Claim
the two serial attention chains as the supported structural explanation, and
label exact per-stage attribution unmeasured until the prefix-pipeline trace is
run.

## C. Controls required for the JEDI+GMP result

### Findings and ranked controls

The D122 pair is an excellent mechanism-generating screen: both rows have
0.8048 accuracy, while the hybrid is 266,382 versus 314,946 LUT, with identical
46.6 ns/14-stage latency, lower EBOPs, and 68.8% versus 63.7% quantized zeros.
Both are bit-exact. It is nevertheless a single-seed, 1000-epoch,
checkpoint-selected comparison; D87 and D119 show why seed behavior cannot be
assumed.

1. **Highest leverage: paired seed replication of both JEDI and JEDI+GMP.**
   Use identical seeds, schedules, ordering, data, and checkpoint/trace policy.
   Report paired deltas in accuracy, traced LUT, latency, and zero fractions.
   Replicating only the hybrid cannot distinguish GMP benefit from ordinary
   run-to-run variation. The already planned replication should therefore be
   explicitly paired.
2. **Next: PHAT ±GMP under the same current QAT protocol.** This completes the
   2×2 host × GMP factorial design. The current builder supports
   `use_gmp=False`, but `train_qat.py` does not expose/pass it, so a small
   `PJ_USE_GMP` plumbing change is required before launching. Compare both the
   ordinary PHAT recipe and, if it succeeds, the ramp-tolerant recipe. D114’s
   sparse checkpoint self-ablated GMP, but post-hoc zeroing is not a trained
   no-GMP control.
3. **Then: one 7000-epoch paired JEDI ±GMP replication, if time permits.**
   This tests whether the 1000-epoch inversion persists under the published
   training budget. It is less urgent than seeds because D122’s claim is about
   the matched 1000-epoch deployment point, not replication of the published
   80.9% row. Run both hosts; a long hybrid alone is uninterpretable.
4. **JetClass evaluation is valuable for transfer, not for validating the
   HLS4ML hardware claim.** The present validation events are already proven to
   be the canonical JEDI evaluation split (D113). A matched JetClass experiment
   requires a new data pipeline and should follow, not displace, the factorial
   and seed controls.

### Audit gate and recommendation

Pre-register a matched-accuracy comparison rule before reading replicates
(e.g., compare the best traced nondominated points within a fixed accuracy
window), trace the same selection policy, replay every `run_config.json`, and
require bit-exact RTL emulation. Report mean and individual paired runs; do not
turn `n=2` into a formal error bar. **Recommendation:** paired seeds plus the
PHAT no-GMP cell are the minimum evidence for a paper-level transferable GMP
claim.

## D. Strategic synthesis: what can GMP honestly anchor?

### Findings

**Finding — the current evidence supports GMP as a promising transferable
hardware-aware inductive bias, not yet as a settled central hardware
contribution.** In D122, adding GMP to JEDI is associated with the same 0.8048
accuracy and a 15.4% lower traced LUT estimate, while the component attribution
assigns GMP a negative marginal through cross-prefix CSE. The defensible
mechanism is not “GMP has negative hardware cost”; it is “the trained hybrid
reaches a more compressible solution, and whole-model trace is cheaper.”
`component_costs.py:22-30` explicitly warns that negative marginals can arise
from CSE and are not a component partition.

**Finding — the paper’s PHAT ablation supports accuracy transfer, but it does
not by itself establish the hardware mechanism.** The submitted paper reports
PHAT with GMP at 81.80±0.02 versus 81.51±0.11 without GMP, and says top sweep
configurations commonly improve roughly 4–5 points in average background
rejection. Those are not “+4.4 accuracy points.” Separately, the local D4
`+4.4 points` was caused by adding BN+tanh and was explicitly confounded before
ablation (D3/D4/D14); it must not be cited as a GMP gain. At the current sparse
PHAT knee, GMP kernels become entirely quantized-zero (D114), so the workspace
does not yet show that GMP improves both hosts at the hardware operating point.

**Finding — the deployment narrative is currently stronger than a
PHAT-is-best narrative.** At the matched protocol, JEDI+GMP dominates the
workspace’s PHAT rows in accuracy, LUT, and latency (D122). An honest paper can
separate (i) hierarchical attention as an accuracy-competitive scientific
architecture and (ii) JEDI+GMP as the present FPGA deployment point.

**Hypothesis — GMP supplies local context that lets global constant layers
compress more aggressively.** The hybrid’s lower EBOPs and larger zero fraction
at matched accuracy are consistent with this. Causality remains unproven with
one seed and no PHAT no-GMP training control.

### Decisive additional experiment and recommendation

The single most valuable experiment is the **paired-seed 2×2 factorial**:
PHAT/JEDI × GMP off/on, under one matched 1000-epoch QAT protocol, with
whole-model traced LUT and per-layer zero profiles. Its pre-registered
interaction questions are:

- Does GMP improve or preserve accuracy within each host?
- At matched accuracy, does GMP reduce whole-model LUT and increase
  constant-layer zeros within each host?
- Is any benefit consistent across paired seeds?

If ramp-tolerant PHAT succeeds, repeat the PHAT pair with that schedule because
it is the cleanest test of the proposed “GMP enables ramp compression”
mechanism. **Recommendation:** center GMP as a *transferable architectural
contribution* now; elevate it to the central *hardware* contribution only after
paired seeds plus the missing PHAT no-GMP cell confirm the whole-model effect.
Avoid “free GMP”; say “negligible direct attributed cost, with attribution
subject to CSE, and lower whole-model LUT after joint training.”

## E. Ranked actions for the remaining night

The commands below are proposed configurations, not commands executed in this
analysis round. Exact cluster submission syntax is intentionally omitted because
the workspace contains no authoritative job launcher and the directive forbids
job modification. Before any training launch, create a unique output directory,
save the effective environment, and run a one-batch graph/name audit.

1. **Harvest and audit the four already-running arms; spend no new GPU time.**
   For each completed arm, select candidates by accuracy first and trace LUT
   rather than trusting EBOPs (D24/D112). Replay its saved config:

   ```bash
   env $(jq -r 'to_entries[] | "PJ_\\(.key)=\\(.value|tostring)"' run_config.json) \
     PJ_DATA=data/jets_128x3.npz PJ_INPUTS_KIF=1,8,16 PJ_N_EMU=512 \
     PJ_OUT=<new-hw-dir> .venv/bin/python extract_hw.py <checkpoint.keras>
   ```

   Treat this as pseudocode because `run_config.json` keys are already uppercase
   Python names rather than necessarily direct `PJ_*` names; manually verify the
   echoed configuration against the training log. Gate: full evaluation,
   bit-exactness, sparsity report, and fine attribution. Do not infer the running
   results if artifacts are absent.
2. **Implement and smoke-test selective beta; launch one lanes-only
   ramp-tolerant PHAT arm and its full-ramp control.** Proposed matched env:

   ```bash
   PJ_ARCH=phat PJ_PARTICLES=128 PJ_PATCH=8 PJ_SORT=pt PJ_EPOCHS=1000 \
   PJ_BETA_SCALE=1.0 PJ_ATTN_BETA_MODE=lanes PJ_ATTN_BETA_SCALE=0.01 \
   PJ_OUT=<unique-ramp-lanes-dir> .venv/bin/python train_qat.py
   ```

   Control: identical, with attention beta scale `1.0`. Do **not** add ProxL1;
   the first test must isolate whether attention protection unlocks ramp-induced
   pruning. Before launch, assert the six protected names and print their beta
   values versus `embed`, `local_attn_wq`, and `head_out`.
3. **Use remaining capacity for paired JEDI seed replication, not an unpaired
   hybrid rerun.** For each new seed, run both:

   ```bash
   PJ_ARCH=jedi    PJ_SEED=<seed> PJ_PARTICLES=128 PJ_SORT=pt \
   PJ_EPOCHS=1000 PJ_BETA_SCALE=1.0 PJ_OUT=<unique-dir> .venv/bin/python train_qat.py
   PJ_ARCH=jedigmp PJ_SEED=<same-seed> PJ_PARTICLES=128 PJ_SORT=pt \
   PJ_EPOCHS=1000 PJ_BETA_SCALE=1.0 PJ_GMP_LO=-1.6 PJ_GMP_HI=1.6 \
   PJ_OUT=<unique-dir> .venv/bin/python train_qat.py
   ```

   Audit the exact effective GMP range because D120 identified it as a possible
   silent mismatch.
4. **Run trace-only pipeline attribution on existing champions.** Extend a copy
   of the fine-prefix analysis to call `to_pipeline` for each prefix at
   `hard_dc={1,2}` and cutoff 4.0 ns. This consumes no GPU and directly answers
   how much of the 28-versus-14-stage gap accumulates across QK, softmax, and AV.
   Do not interpret incremental stages as additive without noting CSE and
   repartitioning.
5. **Complete the PHAT ±GMP factorial plumbing and smoke test.** Expose
   `PJ_USE_GMP`, pass it to `build_phat_jet_k3`, and verify that the expected GMP
   layers are absent/present. Launch only after the running arms and paired JEDI
   replication are secured. The matched config should use the current PHAT
   flagship recipe; if selective ramping passes its gate, that becomes the more
   informative protocol.
6. **Only after the above, start a paired 7000-epoch JEDI ±GMP run.** With about
   eight GPU-hours available and no measured wall-time estimate in the local
   files, completion tonight is unverified. Do not consume the last slot on one
   member of the pair. JetClass is deferred beyond this window.

**Bottom line:** selective beta is technically feasible and is the best new
gap attack; PHAT’s remaining latency gap is mostly consistent with its two
serial attention chains, with only small demonstrated solver headroom; and GMP
is a compelling deployment hypothesis whose central hardware claim still needs
paired replication and the missing PHAT no-GMP cell.
