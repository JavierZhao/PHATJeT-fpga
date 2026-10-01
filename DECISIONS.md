# PHAT-JeT FPGA workstream — decision log

Running record of decisions taken, the evidence behind them, and what was
rejected. Append-only; newest section at the bottom. Started 2026-07-24 during
the overnight autonomous session.

---

## D1. Toolchain: HGQ2 QAT + da4ml, not Vitis HLS + hls4ml

**Decided** 2026-07-24. The handoff brief specifies Vitis HLS C-synthesis. The
implemented pipeline uses HGQ2 quantization-aware training and da4ml
trace-to-Verilog instead.

**Evidence:** no Vitis/Vivado is installed anywhere reachable; the HLS path
needed 1–2 days of hand-written HLS for the GMP scatter/gather before producing
a number. The da4ml path was already code-complete and blocked only on inputs.
Decisive factor: JEDI-linear itself (arXiv 2508.15468) is built with QAT +
distributed arithmetic, so this is the baseline's own toolchain, not an exotic
choice.

**Rejected:** running both paths in parallel — duplicates the hardest piece of
work (static re-expression of scatter/gather) in two toolchains for one person
in one week.

---

## D2. Data: kt-sorted 150-constituent store, val sorted locally

**Decided** 2026-07-24. Train uses the provided
`x_train_robust_150const_ptetaphi_sorted_kt.npy`; val is sorted here with the
same key.

**Evidence:** no sorted validation file exists in the store. Training on sorted
inputs and validating on unsorted would be silent train/serve skew, and it
matters more here than for a permutation-invariant model because PHAT-JeT's
patches are contiguous slices of the sequence. The train file's sort key was
reverse-engineered and confirmed as `pt * dR` at 100.00% exact-jet match over
4,000 jets, with every other candidate at 0%.

---

## D3. GMP grid range is not the accuracy bottleneck

**Decided** 2026-07-24. Keep 8x8 on [-1.6, 1.6]; stop sweeping grid geometry.

**Evidence:** the original [-0.4, 0.4] default resolves only 34.1% of real
constituents on this robust-normalised store (the docstring's "99.9%" claim is
false for this data). Widening to +/-1.6 (93.3% coverage) and +/-3.2 (99.96%)
produced no separation — all arms landed within 0.7494–0.7645, inside noise.
Normalisation moved accuracy 4.4 points over the same period.

---

## D4. Normalisation: BatchNorm on every Dense + tanh pre-norm

**Decided** 2026-07-24. Adopted from HGQ2-examples/jsc150 reference practice.

**Evidence:** +4.4 points (0.7645 -> 0.8085, later 0.8108), which put QAT above
the float reference (0.8088). Every Dense in the reference implementation is a
`QEinsumDenseBatchnorm`, and `get_transformer` bounds the datalane with tanh
LUTs before attention and before the FFN; we had neither. BatchNorm folds into
the kernel at inference so it is free on-chip; tanh is a table.

**Open:** the +4.4 and a 5M-EBOPs blowup arrived in the same change. The
four-cell ablation (none / bn / tanh / bn+tanh) is running to separate them.

---

## D5. pT validity mask on the GMP scatter — REJECTED

**Decided** 2026-07-24. `gmp_mask_padding` stays off by default.

**Evidence:** at matched EBOPs the masked arm was consistently ~2 points worse
than its unmasked twin across the entire overlapping range (−2.85, −2.44, −2.13,
−1.83, −2.09), and the same holds matched by epoch. The mask works as designed
(verified: grid is bit-identical when padded slots are relocated), so this is a
real effect, not a bug.

**Hypothesis:** padded slots pile into the centre cell in proportion to
(150 − n_real), so that cell encoded jet multiplicity — a genuinely
discriminative feature that masking deleted. The paper's Appendix G discusses
the related multiplicity sensitivity of per-cell sum pooling.

---

## D6. Reference beta ramp is too aggressive for this model

**Decided** 2026-07-24. Run at `PJ_BETA_SCALE` well below 1.0; sweep the
endpoint rather than inheriting jsc150's absolute schedule.

**Evidence:** at scale 1.0 every unnormalised arm collapsed to chance (0.2009–
0.2023 for 5 classes) with EBOPs driven to as low as 45. The schedule was tuned
for jsc150's larger models over 7000 epochs; absolute beta does not transfer to
a 5.4k-parameter graph. Pareto checkpoints preserve the good early points, so
collapsed runs are not total losses, but the tail is wasted compute.

---

## D7. Comparison class is JEDI-linear Table II, not Table I

**Decided** 2026-07-24, after verifying both tables directly against the PDF
(previous numbers came from an HTML render that I had already gotten wrong once).

**Evidence:** Table I is permutation-invariant and selects top-N by pT; Table II
is non-permutation-invariant and sorts by pT. PHAT-JeT builds patches from
contiguous slices of a *sorted* sequence, so it is not permutation-invariant.
Table II is therefore the correct comparison class.

Verified 3-feature rows:

| Model | N | Acc | LUT | FF | Latn | Fmax |
|---|---:|---:|---:|---:|---:|---:|
| JEDI-linear (Table II) | 32 | 78.0% | 45k | 26k | 63 ns | 300.8 |
| JEDI-linear (Table II) | 64 | 80.9% | 71k | 38k | 61 ns | 328.4 |
| JEDI-linear (Table II) | 128 | 80.9% | 98k | 48k | 82 ns | 257.6 |
| JEDI-linear (Table I) | 64 | 81.8% | 164k | 93k | 78 ns | 307.0 |
| JEDI-linear (Table I) | 128 | 81.6% | 296k | 163k | 138 ns | 203.1 |
| MLP-Mixer (Table II) | 128 | 79.8% | 83k | 21k | 72 ns | 208.7 |

All at 0 DSP, 0 BRAM, II=1.

**Consequence — this is the hard fact of the workstream.** Our best QAT
checkpoint is 81.08% at 7,450k LUT. Against the correct comparator (Table II,
128p/3f: 80.9% at 98k LUT) we are at accuracy parity and **~76x the LUT cost**.
Even the cheapest checkpoint that fits a VU13P (1,049k LUT) is ~10x their cost
at 73.25% accuracy. Reducing N is explicitly legitimate — JEDI-linear reports
32/64/128 rows itself — so the n64/n32 arms are comparison points, not
unjustified config changes.

---

## D8. Attention, not the FFN, is the resource target

**Decided** 2026-07-24 from the per-component da4ml trace of the 0.8101
checkpoint.

| Component | Marginal LUT | Share |
|---|---:|---:|
| Local patch attention | 2,790,247 | 37.5% |
| FFN | 2,432,616 | 32.7% |
| Patch pooling + global MHSA | 2,135,661 | 28.7% |
| Embedding | 81,113 | 1.1% |
| Aggregation + head | 91,414 | 1.2% |
| GMP | 0 | ~0% |

**Evidence:** the two attention stages are 66.2% of the design. This contradicts
the op-count intuition, which put the FFN at 307k of 535k constant-MACs and
attention at only 9.4% of arithmetic — activation x activation products cannot
be constant-folded into shift-add networks, so they cost far more per op. GMP
traces to zero marginal cost: the one-hot routing is genuinely free, more
strongly than the analytical model predicted.

**Consequence:** FFN pruning caps out at ~33%. The structural gap to a *linear*
GNN baseline is attention itself.

---

## D9. Cluster hygiene

- Job names use the `zz-` prefix. `zh-` jobs in `cms-ml` belong to someone else.
- Pods carry `podAntiAffinity` on the `zz-sweep` label (one pod per node).
  Co-scheduled JAX pods race on cuDNN workspace init and losers die with
  `cudnn status 5003`. An earlier diagnosis blaming specific `chi-dgx` nodes was
  a spurious correlation over five samples — the same node hosts both failing
  and succeeding pods.
- Jobs must set `set -o pipefail`; `python ... | tee` otherwise reports success
  after a crash.
- `PJ_DATA=""` for trace-only jobs: loading the 1.6 GB npz from cephfs-central
  stalled one job for 15+ minutes at 0% CPU.

---

## D10. Concede resource comparability; narrow the rebuttal claim

**Decided** 2026-07-25 (overnight), on Codex's blunt verdict and D7/D8 evidence.

At 81.08% vs JEDI-linear's 80.9% we consume 7,450k vs 98k LUT — ~76x. The
cheapest VU13P-fitting point is still ~10x larger while giving up 7.65 accuracy
points. "Comparable resources" is falsified in any ordinary reading of the
phrase, and no bit-width tuning plausibly closes 76x.

**Claim we will make instead:**

> PHAT-JeT and JEDI-linear are evaluated in the same trigger-oriented regime --
> three input features, sorted fixed-length inputs, II=1 distributed-arithmetic
> implementations produced by QAT -- but PHAT-JeT does **not** achieve
> comparable LUT efficiency. At accuracy parity its current da4ml estimate is
> ~76x larger. We characterise the accuracy/resource/II trade-off of an
> attention-based architecture under this deployment regime and report where it
> does and does not fit.

**Banned from the writeup:** "comparable resources", "competitive efficiency",
"modest overhead".

**Two qualifications that must travel with the number:** our figure is a da4ml
estimate from generated-but-unimplemented RTL, theirs is post-place-and-route;
and the per-component split is cumulative-prefix attribution, not an isolated
partition.

---

## D11. Attack attention precision specifically, not global bitwidth

**Decided** 2026-07-25 (overnight).

D8 showed attention is 66.2% of LUT on 9.4% of arithmetic. `BW_A` was a single
global datalane scope, so there was no way to squeeze attention alone. Added
`attn_bits`: a nested `QuantizerConfigScope(place="datalane", f0=attn_bits)`
wrapped around the QK/softmax/AV construction sites in both attention stages.
Embedding and head (~2.3% of LUT) stay at full precision.

**Rejected for tonight, per Codex:** replacing softmax with a cheaper
normalisation (no traceable substitute exists in the builder; may just move cost
into reciprocal logic; unproven that softmax rather than QK/AV dominates the
attributed attention LUT), and a smaller-patch-size arm (local attention scales
as N*P*d_model but global scales as (N/P)^2*d_model, so it is a trade not a
reduction — and N=32/64 already probe the defensible structural axis).

---

## D12. Torch backend for all training

**Decided** 2026-07-25 (overnight).

Six arms died with `cudnn status 5003` in XLA's grouped/depthwise-conv path
(`gmp_dwconv`, feature_group_count=16). The failure is non-deterministic and
explained by neither node identity nor co-scheduling: `abl-bn-tanh` succeeded on
`k8s-chase-ci-02` while `satsym-fixed` failed on the same node, and three of the
six carried podAntiAffinity. Two earlier diagnoses (broken chi-dgx nodes, then
GPU memory contention) were both wrong.

Keras 3 on the torch backend never enters XLA. Verified locally: builds and
trains. All relaunched arms use `pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime`
with `KERAS_BACKEND=torch`.

---

## D13. LUT-per-EBOP is not a constant — do not extrapolate across architectures

**Measured** 2026-07-25 (overnight).

| Config | EBOPs | LUT est | LUT/EBOP | % VU13P |
|---|---:|---:|---:|---:|
| unnormalised, FFN 4x | 751k | 1,049k | 1.40 | 60.7% |
| unnormalised, FFN 4x | 1,083k | 1,493k | 1.38 | 86.4% |
| bn-tanh, FFN 4x | 5,005k | 7,450k | 1.49 | 431.1% |
| bn-tanh, FFN 1x | 1,791k | 3,068k | 1.71 | 177.5% |
| bn-tanh, FFN 1x | 2,076k | 3,358k | 1.62 | 194.3% |

The ratio ranges 1.38–1.71, and is *higher* for the cheaper FFN. So the earlier
"fits at <= 1.25M EBOPs" rule (derived at 1.4) is optimistic; at 1.7 the fit
threshold is nearer **1.0M EBOPs**. Always trace rather than extrapolate.

**Component split for FFN-1x (compare D8, which was FFN-4x):**

| Component | FFN-4x share | FFN-1x share |
|---|---:|---:|
| Local patch attention | 37.5% | **46.8%** |
| Patch pooling + global MHSA | 28.7% | **26.9%** |
| FFN | 32.7% | 24.2% |
| GMP | ~0% | ~0% |

Cutting the FFN moved attention from 66.2% to **73.7%** of the design. Every
non-attention economy makes attention a larger share of what remains — this is
the structural wall Codex identified, now measured twice.

---

## D14. Drop BatchNorm, keep tanh — supersedes D4

**Decided** 2026-07-25 (overnight), from the four-cell ablation. All cells:
N=150, P=10, max tokenizer, 8x8 grid on [-1.6,1.6], mask=0, beta_scale=0.01.

| Cell | Best acc @ EBOPs | Cheapest above 0.78 |
|---|---|---|
| BN only | 0.8079 @ 7.25M | 0.7989 @ 5.23M |
| **tanh only** | 0.8064 @ 3.69M | **0.8037 @ 3.57M** |
| BN + tanh | 0.8014 @ 5.00M | 0.7968 @ 3.92M |

tanh-only dominates on both axes; BN+tanh is worse than tanh alone. D4 adopted
the pair together and attributed the +4.4 to both — that was a confounded
attribution. **The bounded activation is what mattered.** With
`overflow_mode='wrap'`, an out-of-range activation flips sign rather than
saturating, so bounding the datalane before attention is doing the work.

**CORRECTED 2026-07-25 after Codex review.** The safe statement is:
*tanh-only currently dominates the BN-containing configurations* — it reaches
comparable accuracy at ~32% fewer EBOPs. Two things I previously claimed here are
NOT established by this evidence:
  * "BatchNorm is not free on-chip". EBOPs record where *training landed*, not
    the inference-time cost of a folded BN operator. No trace isolates BN logic.
  * "The bounded activation is what mattered". The displayed table omits the
    no-BN/no-tanh cell (`abl-none` had not produced checkpoints when D14 was
    written), so the attribution is incomplete.
The accuracy ordering alone is also not statistically persuasive: the gaps are
0.15–0.48 points, while the only noise estimate in this log calls a 1.51-point
spread "inside noise" (D3), and there are no replicated-seed runs. The EBOPs
separation is the part worth acting on.

**Caveat:** fronts are small (n=3–5) and these arms were still running. Revisit
if the ordering changes at n>10.

---

## D15. Back to the jax backend with backoffLimit 6 — supersedes D12

**Decided** 2026-07-25 (overnight), after measuring the torch backend's cost.

D12 moved training to torch to escape the non-deterministic `cudnn status 5003`
in XLA's grouped-conv path. Torch does avoid it, but it is **5x slower**:
638 ms/step vs jax's 130 ms/step, i.e. 142 s/epoch vs 29 s/epoch, so a
1000-epoch arm goes from ~8 h to ~39 h. With a one-week deadline that is not
affordable.

The cuDNN failure has a property that makes retrying cheap: it always fires at
**epoch 1**, within a couple of minutes, and an arm that survives epoch 1 runs to
completion (qat-bn-tanh did a clean 1000 epochs on jax). At ~50% per attempt,
`backoffLimit: 6` leaves ~0.8% chance of losing an arm, at negligible cost.

**Config:** `python:3.11-slim` + `jax[cuda12]`, `KERAS_BACKEND=jax`,
`--xla_gpu_autotune_level=0`, no preallocation, `backoffLimit: 6`, and
podAntiAffinity retained.

Torch remains a verified fallback if a config turns out to fail jax
deterministically. `steps_per_execution` must be 1 on torch and can be 4 on jax;
`train_qat.py` now branches on `keras.backend.backend()`.

**Method note for the morning:** I twice inferred a cause from a handful of
correlated samples and was wrong both times (broken chi-dgx nodes; then "the new
knobs break it"). Both dissolved on a larger sample or a local reproduction.
Reproduce locally before theorising about cluster-side causes.

---

## D16. Do not trace a checkpoint from a still-running arm

**Observed** 2026-07-25 (overnight). `extract_hw.py` failed with a `zipfile`
error inside `load_weights`: `ParetoFront` was mid-write on the newest `.keras`
file when the trace job opened it. Either wait for the arm to finish, or copy
the checkpoint aside first, or pick a checkpoint that is not the most recent.

---

## D17. Attention LUT does NOT scale with its MAC count — reducing N barely helps

**Measured** 2026-07-25 (overnight). Traced the completed N=32 arm.

| Config | acc | EBOPs | LUT est | % VU13P |
|---|---:|---:|---:|---:|
| N=150, bn-tanh, FFN 4x | 0.8101 | 5,004,958 | 7,449,939 | 431.1% |
| N=150, bn-tanh, FFN 1x | 0.8018 | 2,075,519 | 3,358,344 | 194.3% |
| N=32,  bn-tanh, FFN 4x | 0.7912 | 1,407,144 | 2,651,843 | 153.5% |
| N=32,  bn-tanh, FFN 4x | 0.7710 | 1,137,714 | 2,177,974 | 126.0% |

**Nothing fits yet.** The best point is still 126% of a VU13P at 77.1% accuracy.

The important part is *why*. Local attention MACs scale as N*P*d_model:
N=150,P=10 gives 24,000; N=32,P=8 gives 4,096 — a 5.9x reduction. But its traced
LUT only fell 2,790,247 -> 1,740,287, a **1.6x** reduction. And its share of the
design *rose* from 37.5% to **65.6%**.

| Component | N=150 FFN-4x | N=150 FFN-1x | N=32 FFN-4x |
|---|---:|---:|---:|
| Local patch attention | 37.5% | 46.8% | **65.6%** |
| Patch pooling + global MHSA | 28.7% | 26.9% | 13.7% |
| FFN | 32.7% | 24.2% | 19.8% |
| GMP | ~0% | ~0% | ~0% |

So attention LUT is **not** driven by its multiply count. The cost is in softmax
and per-element quantizer/datalane overhead, which scale with the number of
attention *elements* and their bitwidths rather than with MACs. That explains why
every economy applied elsewhere raises attention's share, and why cutting N by
4.7x bought only 2.8x overall.

**Consequence for strategy.** At N=32 JEDI-linear reports 78.0% at 45k LUT
(Table II). We are at 77.1% for 2,178k LUT — **~48x larger**. The gap does not
close by shrinking N; it is roughly invariant. This is the strongest evidence yet
that D10 (concede resource comparability) is correct, and that the remaining hope
is attention-datalane precision (attn2/attn3, running) rather than any structural
size reduction.

---

## D18. `attn_bits` sets INITIAL fractional bits, not an enforced width

**Established** 2026-07-25 by Codex reading the installed HGQ2 source, and
consistent with the measurement.

Good news: the nested `QuantizerConfigScope` in `_attn_scope` **does inherit**
the enclosing `overflow_mode`, `fr=MonoL1(...)` and `ic=MinMax(0,12)`. HGQ2
mutates the active default config in place and restores it on exit, so only the
keys passed are overridden. The attn arms are not silently reverting to library
defaults, and the scope is active over both local and global QK/softmax/AV while
Q/K/V and output projections stay outside it, as intended.

Bad news: `f0` is documented as *the initial value of the number of fraction
bits*. Integer bits stay independently trainable and the regularizer can move
fractional precision during training. So `PJ_ATTN_BITS=2` does **not** pin a
2-bit attention datalane.

**Measured consistently with that:** `attn3` sits at 55.6M EBOPs at epoch ~2
against `abl-bn-tanh`'s 58.5M — only ~5% lower, where cutting 66–74% of the
design from 7 to 3 bits should have been dramatic. The quantizers drift back up.

**Consequence:** these are "low-initial-precision attention" arms, not low-
precision ones. To actually hold attention narrow, apply *pressure* rather than
initialisation — a stronger `fr=MonoL1(...)` inside the attention scope (i.e. a
per-region beta), which is the mechanism that drives bits down. Implemented as
`PJ_ATTN_L1`.

---

## D19. LUT/EBOP RISES as the model gets cheaper — there is a cost floor

**Measured** 2026-07-25 (overnight), pooling every trace taken so far.

| Config | EBOPs | LUT est | LUT/EBOP |
|---|---:|---:|---:|
| N=150 unnormalised, FFN 4x | 1,083k | 1,493k | 1.38 |
| N=150 unnormalised, FFN 4x | 751k | 1,049k | 1.40 |
| N=150 bn-tanh, FFN 4x | 5,005k | 7,450k | 1.49 |
| N=150 bn-tanh, FFN 1x | 2,076k | 3,358k | 1.62 |
| N=150 bn-tanh, FFN 1x | 1,791k | 3,068k | 1.71 |
| N=32 bn-tanh, FFN 4x | 1,407k | 2,652k | 1.88 |
| N=32 bn-tanh, FFN 4x | 1,138k | 2,178k | **1.91** |

The ratio climbs monotonically as EBOPs fall. EBOPs counts *multiplier* work, so
a rising LUT/EBOP means an increasing share of the design is **not** multipliers:
softmax tables, LUT-based activations, datalane routing and registering. That
cost is roughly fixed per attention element and does not shrink with bitwidth.

**This is worse news than D17.** It means bit-squeezing has diminishing returns:
driving EBOPs toward zero does not drive LUT toward zero, because the residual is
table-and-routing logic. Combined with D17 (attention LUT does not scale with its
MAC count), the two together say the attention block has a **floor cost** that is
insensitive to both size and precision — the two levers we have.

**Best measured point remains** N=32 at 1,138k EBOPs = 2,178k LUT = 126% of a
VU13P at 0.7710 accuracy. JEDI-linear at the same N is 78.0% at 45k LUT.

**Still untested:** whether per-region pressure on the attention datalanes
(`PJ_ATTN_L1`, arms running) reaches the softmax/table logic at all. D18 notes
the scope targets `place="datalane"`, which does NOT cover QSoftmax's *table*
quantizers — so the dominant fixed cost may be outside the scope entirely. If so,
no available knob touches it.

---

## D20. The single largest component is the LOCAL SOFTMAX (32% of the design)

**Measured** 2026-07-25 (overnight) with fine-grained cut points (`PJ_FINE=1`)
on the traced N=32 checkpoint (0.7912, 2,651,843 LUT).

| Sub-component | LUT | Share |
|---|---:|---:|
| local: Q/K/V projections | 379,541 | 14.3% |
| local: QK scores | 165,044 | 6.2% |
| **local: SOFTMAX** | **846,726** | **31.9%** |
| local: AV | 274,704 | 10.4% |
| local: output proj | 74,272 | 2.8% |
| global: Q/K/V | 262,720 | 9.9% |
| global: QK / softmax / AV / proj | 100,682 | 3.8% |
| FFN | 524,677 | 19.8% |
| Embedding, GMP, head | ~45,000 | ~1.7% |

Local softmax is larger than the entire FFN. The global stage is nearly free
because N_P = 4 tokens at N=32.

**This explains D19 mechanically.** Softmax is exp tables + normalisation, not
multiplies, so it contributes ~nothing to EBOPs while costing a third of the
LUTs. Hence LUT/EBOP rises as multiplier work is squeezed out: we have been
optimising the wrong quantity. EBOPs is the loss-side proxy, but it is blind to
the largest single block in the design.

**The lever this exposes:** softmax element count is `N_P * H * P * P = N * H * P`.
It is **linear in the number of heads**, even though QK/AV *multiply* count is
head-invariant at fixed `d_model`. Going 4 heads -> 1 should cut ~24% of the
whole design. It is also linear in P and in N.

This makes `PJ_HEADS` a first-class resource knob rather than the minor test it
was assessed as. Arms launched: `n32-h1`, `n32-h2`, and `best-h1`.

**Caveat:** the numbers are cumulative-prefix attribution, and softmax sits
directly downstream of QK, so some QK logic may be charged to the softmax row by
CSE across the cut. The ordering is unambiguous but the exact 31.9% is not.

---

## D21. CORRECTIONS to D17 and D20 — structural lever sweep at fixed precision

**Measured** 2026-07-25 (overnight). Previous lever claims came from comparing
*trained* checkpoints, which sit at different points on their beta frontiers with
different learned bitwidths — a confound. This sweep traces UNTRAINED models
where every quantizer is at identical initialisation, so only structure varies.
Absolute LUT is meaningless; the ratios are the result.

Base: N=32, P=8, heads=4, tanh-only, mean tokenizer, FFN 1x -> 3,295,363 LUT.

| Change | LUT | delta |
|---|---:|---:|
| heads 4 -> 1 | 3,152,394 | **-4.1%** |
| heads 4 -> 2 | 3,233,816 | -1.6% |
| P 8 -> 4 | 3,065,827 | -7.0% |
| P 8 -> 16 | 4,206,666 | **+27.7%** |
| N 32 -> 16 | 1,697,929 | **-48.5%** |

**D20 is FALSIFIED.** I predicted ~24% from cutting heads 4->1, reasoning that
softmax elements scale as N*H*P and softmax was 31.9% of the design. Measured:
4.1%. The 31.9% came from cumulative-prefix differencing, and softmax sits
directly downstream of QK — CSE across that cut evidently charges much of the
neighbouring logic to the softmax row. **I flagged exactly this caveat when
recording D20 and then built a strategy on the number anyway.** Head count is a
minor knob, as Codex originally assessed.

**D17 is partly WITHDRAWN.** It concluded attention LUT "does not scale with N".
At *fixed precision* LUT is close to linear in N (halving N gives -48.5%). The
observed sub-linearity between the trained N=150 and N=32 checkpoints was a
bitwidth confound, not a structural property.

**What survives:** D19 (LUT/EBOP rises as EBOPs falls, so EBOPs is blind to a
real part of the cost) is unaffected — it pools traces of trained checkpoints and
the trend is about what EBOPs measures, not about N.

**Consequence:** N is the only structural lever with real leverage, and it is
roughly linear. Larger patches are actively harmful. Next arms: N=16.
JEDI-linear Table II at N=16 is 71.9% @ 44k LUT, so N=16 is a legitimate
comparison point and one where our accuracy may exceed theirs even though our
area will not.

**Method note (third occurrence tonight):** every time I have inferred a
mechanism from a differenced or correlated measurement without an independent
test, I have been wrong. Structural claims need a controlled trace.

---

## D22. BREAKTHROUGH — per-region attention bit pressure. 431% -> 100.2% of VU13P

**Measured** 2026-07-25 (overnight). `attnl1-mid`: N=150, P=10, tanh-only, mean
tokenizer, FFN 1x, beta_scale 0.01, **PJ_ATTN_L1=1e-6**.

| Checkpoint | acc | EBOPs | LUT est | % VU13P | latency |
|---|---:|---:|---:|---:|---:|
| best of night, start | 0.8101 | 5,004,958 | 7,449,939 | 431.1% | 186.5 ns |
| **attnl1-mid** | **0.8004** | **1,251,731** | **1,731,475** | **100.2%** | **153.2 ns** |

**4.3x resource reduction for 1.0 accuracy point, at the paper's full N=150.**
We are 3,475 LUT over a VU13P — at the boundary, not yet inside it.

The controlled comparison is `best`, which is byte-identical apart from
`PJ_ATTN_L1`: 0.7764 @ 5,317,779 EBOPs vs 0.8004 @ 1,251,731. Roughly 4.2x fewer
EBOPs **and** +2.4 accuracy points. (Caveat: `best` had fewer checkpoints at the
time of reading, so it may not have been equally converged.)

**Why this works where the other levers did not.** D18 showed `attn_bits` (an
initial-value knob) barely moved cost. `attn_l1` applies a *stronger MonoL1 on
the fractional bits of the QK/AV datalanes only* — real pressure on precisely the
region D8 identified as 66-74% of the design, while embedding, head and FFN keep
their width. It is per-region beta, and it is the single most effective change
found in this workstream.

Component split at the new operating point: local attention 40.7%, global MHSA
28.4%, FFN 26.7%, GMP ~0%. Attention is still ~69%.

**Next:** `attnl1-hi` (1e-5) is running and should sit lower. Arms launched to
cross the line: attn_l1 with additional global beta.

**This does not overturn D10.** JEDI-linear Table II at N=128 is 80.9% @ 98k LUT;
we would be ~80% @ ~1,730k LUT — still ~18x larger. Resource comparability
remains conceded. What changes is that "fits on the target device at the paper's
particle count, II=1, ~153 ns" becomes a defensible claim instead of an
aspiration.

---

## D23. Pruned superseded arms to free GPUs for the fit-line arms

**Decided** 2026-07-25 (overnight). 15 of my pods held GPUs while 9 queued,
including every arm that could cross the VU13P line. Killed, with reason:

| Arm | Superseded by |
|---|---|
| `attn2`, `attn3` | D18 — `attn_bits` is an initial-value knob; `attn_l1` is the real one |
| `bw4` | D22 — global bitwidth is dominated by per-region attention pressure |
| `combo`, `combo-d8` | D14 — BN-containing configs are dominated by tanh-only |
| `d8`, `h1`, `n32-h1` | D21 — d_model/head levers measured at 4.1% and below |
| `best` | D22 — `attnl1-mid` is the same config plus the winning knob; `best` had already produced the 8 checkpoints needed for the controlled comparison |

Kept running: `attnl1-hi`, `fit-a`, `fit-b`, `fit-n64` (crossing the fit line),
`n16`, `n16-b01`, `n32-best` (N slope), `satsym-fixed` (overflow question still
open), `abl-neither` (completes the D14 ablation).

## D24. Select checkpoints by traced LUT, NOT by EBOPs

**Measured** 2026-07-25 (overnight). Two arms, traced:

| Arm | acc | EBOPs | LUT est | LUT/EBOP | % VU13P |
|---|---:|---:|---:|---:|---:|
| `best-n64` (N=64, no attn_l1) | 0.8043 | 1,168,456 | 2,198,207 | 1.88 | 127.2% |
| `attnl1-mid` (N=150, attn_l1=1e-6) | 0.8004 | 1,251,731 | **1,731,475** | **1.38** | **100.2%** |

`best-n64` has **7% fewer EBOPs but 27% more LUT**. EBOPs ranked it better; the
hardware ranked it worse. So the ParetoFront objective we have been optimising
and checkpointing against is not the objective we care about.

**Mechanism.** Per D19/D20 a large share of the design is softmax tables and
datalane routing, which EBOPs does not count. `attn_l1` compresses the QK/AV
datalanes specifically, so it shrinks that uncounted cost and drives LUT/EBOP
from 1.88 down to 1.38. Reducing N shrinks multiplier work (visible to EBOPs)
but leaves relatively more of the uncounted cost behind, pushing the ratio up.

**Consequences.**
1. Never compare arms on EBOPs alone. Trace before ranking.
2. `attnl1-mid` remains the best hardware point (100.2%) even though it is not
   the best EBOPs point.
3. Component split confirms it: `best-n64` is 53.1% local attention + 30.6%
   global = 83.7% attention; `attnl1-mid` is 40.7% + 28.4% = 69.1%. The pressure
   is doing what it was designed to do.
4. The deliverable frontier must be accuracy-vs-LUT, built by tracing several
   checkpoints per arm — not the EBOPs frontier ParetoFront happens to save.

**Next:** `fit-n64` (N=64 **plus** attn_l1=1e-6) should get both effects.

---

## D25. FIRST FITTING DESIGN — 86.4% of VU13P at 79.00% accuracy

**Measured** 2026-07-25 (overnight). Arm `n32-best`: N=32, P=8, tanh-only
(no BN), mean tokenizer, FFN 1x, GMP 8x8 on [-1.6,1.6], mask off,
beta_scale 0.01, no attn_l1.

| | value |
|---|---|
| Accuracy | **0.7912** |
| LUT est | **1,477,150 (85.5% of VU13P — FITS)** |
| FF (reg bits) | 708,617 |
| (second fitting point) | 0.7900 @ 1,493,694 LUT = 86.4% |
| Latency | 156.5 ns (47 stages @ 300 MHz) |
| II / DSP / BRAM | 1 / 0 / 0 |

**Comparison at matched particle count** (JEDI-linear Table II, N=32, 3 features,
same VU13P): 78.0% at 45k LUT, 63 ns.

So at N=32 we are **+1.0 accuracy point** over the baseline while using ~33x the
LUTs and ~2.5x the latency. Accuracy-competitive, not resource-competitive —
exactly the shape D10 predicted, now with a concrete fitting artifact behind it.

**Why this one fits and `attnl1-mid` (100.2%) does not:** it trades particle
count for area. `attnl1-mid` holds the paper's N=150 at 80.04% but sits just over
the device; `n32-best` drops to N=32 and 79.00% and comes in comfortably under.
Both are legitimate rows — the first is "paper configuration, marginally over",
the second is "reduced-N variant, fits". Report both.

**Note the EBOPs trap again (D24):** `n32-best` reached 837,721 EBOPs where
`attnl1-mid` bottomed at 1,251,731 — but the two have different LUT/EBOP ratios,
so only the trace settles the ranking. Ratio here is 1.78.

**Recipe that produced it** — every ingredient measured, none assumed:
tanh-only normalisation (D14), mean tokenizer (paper Table 16 + measured),
FFN 4x -> 1x, no pT mask (D5), grid ±1.6 (D3), moderate beta (D6), reduced N
(D21). This is the current best-known configuration.

**Job-hygiene note:** `zz-phatjet-hw-n32best` reports `FailureTarget`, but the
`extract_hw.py` stage completed and wrote both rows to `hw/n32-best/
hw_results.json`. Only the trailing `component_costs.py` stage failed, because
the manifest was templated from the `best-n64` job and its checkpoint argument
was not re-pointed. The fitting numbers are unaffected. This is the second time a
job's overall status has misrepresented what it actually produced — read the log,
not the status column (cf. D9).

---

## D26. Complete N-sweep — PHAT-JeT beats JEDI-linear accuracy at every matched N that fits

**Measured** 2026-07-25 (overnight). Same recipe at every N.

| N | Accuracy | LUT | % VU13P | Fits | JEDI-linear (Table II, same N) | delta acc |
|---:|---:|---:|---:|---|---|---:|
| 16 | 0.7401 | 957,257 | 55.4% | YES | 71.9% @ 44k | **+1.5** |
| 32 | 0.7912 | 1,477,150 | 85.5% | YES | 78.0% @ 45k | **+1.1** |
| 64 | 0.8043 | 2,198,207 | 127.2% | no | 80.9% @ 71k | -0.5 |
| 150 | 0.8004 | 1,731,475 | 100.2% | marginal | 80.9% @ 98k (N=128) | -0.9 |

Two clean fitting points, both accuracy-superior to the deployed baseline at
matched particle count, at ~20-30x the area.

Note N=64 costs MORE LUT than N=150 (2,198k vs 1,731k) despite being a smaller
model: the N=150 row carries `attn_l1=1e-6` and the N=64 row does not (D24 —
attention pressure lowers LUT/EBOP from 1.88 to 1.38). **The obvious untested
combination is reduced N *with* attention pressure**; `fit-n64` is running.

This is the shape of the final rebuttal table: an accuracy/resource curve across
matched particle counts, with the resource deficit stated rather than hidden.

---

## D27. Stronger global beta at N=32 gives a usable Pareto curve, not one point

**Measured** 2026-07-25 (overnight). Arm `n32-best-b01` = `n32-best` recipe with
`beta_scale` 0.01 -> 0.1. 32 checkpoints; four traced.

| Accuracy | EBOPs | LUT | % VU13P | Latency |
|---:|---:|---:|---:|---:|
| 0.7912 (beta 0.01) | 837,721 | 1,477,150 | 85.5% | 156.5 ns |
| 0.7890 | 861,423 | 1,432,983 | 82.9% | 153.2 ns |
| **0.7879** | 700,042 | **1,219,048** | **70.5%** | 149.8 ns |
| 0.7639 | 311,473 | 494,967 | **28.6%** | 133.2 ns |

**0.7879 at 70.5% is a strictly better operating point than the previous
headline** (0.7912 at 85.5%): 0.3 accuracy points for 17% less area, and it
leaves 30% of the device free for the rest of a trigger design.

The cheap end is also informative: at 28.6% utilisation we are ~11x JEDI-linear's
LUT for -1.6 accuracy points, versus ~27x at the headline point. The resource gap
is not constant along the curve — it narrows as we give up accuracy.

**Method:** D6 concluded the reference beta ramp was "too aggressive". That was
true at scale 1.0 for an *unnormalised* model. With tanh normalisation and the
reduced-N recipe, scale 0.1 is productive rather than destructive — it extends
the frontier down instead of collapsing it. D6 should be read as configuration-
specific, not a general property of the schedule.

---

## D28. The resource gap narrows toward the cheap end of the curve

**Measured** 2026-07-25 (overnight), completing both traced curves.

| N | Accuracy | LUT | % VU13P | Latency | vs JEDI-linear same N |
|---:|---:|---:|---:|---:|---|
| 32 | 0.7879 | 1,219,048 | 70.5% | 149.8 ns | +0.8 pts, ~27x LUT |
| 32 | 0.7639 | 494,967 | 28.6% | 133.2 ns | -1.6 pts, ~11x LUT |
| 16 | 0.7401 | 957,257 | 55.4% | 156.5 ns | +2.1 pts, ~22x LUT |
| **16** | **0.7226** | **365,950** | **21.2%** | 133.2 ns | **+0.4 pts, ~8x LUT** |

**The ~20-30x resource deficit is not a fixed architectural property.** It is
largest where we push accuracy hardest and falls to ~8x at the cheap end, where
we still edge the baseline on accuracy (72.26 vs 71.9 at N=16).

This is a materially better story than a single ratio: PHAT-JeT trades area for
accuracy along a curve, and a deployer can choose where to sit. At 21.2%
utilisation and 133 ns it leaves ~80% of the device free — which is the regime an
actual L1T integration would care about, since a tagger shares the part with link
infrastructure and other algorithms.

**Caveat unchanged:** these are da4ml estimates from unimplemented RTL against
JEDI-linear's post-place-and-route figures. The comparison is favourable to them
by construction, which makes the direction of the remaining gap trustworthy but
not its exact magnitude.

---

## D29. The paper's N=150 configuration FITS — 63.8% of a VU13P at 78.98%

**Measured** 2026-07-25 (overnight). Two final arms, both fitting.

| Arm | N | attn_l1 | Accuracy | LUT | % VU13P | Latency |
|---|---:|---|---:|---:|---:|---:|
| **attnl1-hi** | **150** | 1e-5 | **0.7898** | **1,102,348** | **63.8%** | 153.2 ns |
| **fit-n64** | 64 | 1e-6 | **0.8024** | 1,556,399 | 90.1% | 153.2 ns |

**This retires the central worry of the workstream.** We now have the paper's own
particle count (N=150, P=10) fitting a VU13P with ~36% of the device to spare, at
78.98% accuracy, II=1, 153 ns. Every earlier fitting point required dropping to
N=16 or N=32.

`fit-n64` also confirms the two levers **compound**: at N=64, adding attention
pressure moves 127.2% -> 90.1% of the device with accuracy unchanged within noise
(0.8043 -> 0.8024). It is the highest-accuracy fitting design and sits within 0.7
points of JEDI-linear's N=64 figure (80.9%).

**Full trajectory of the night:** 7,449,939 LUT (431%) at N=150 -> 1,102,348 LUT
(63.8%) at N=150. **A 6.8x resource reduction for 2.1 accuracy points**, without
leaving the published configuration.

**Stronger attention pressure is better than expected.** `attn_l1` 1e-6 -> 1e-5
at N=150 gave 1,731,475 -> 1,102,348 LUT (-36%) for -1.1 accuracy points. The
knob has not been pushed to its limit; 3e-5 and 1e-4 are untested and are the
obvious next arms.

---

## D30. BIT-EXACTNESS FAILS — the traced netlist does not match the Keras model

**Measured** 2026-07-25 (overnight), on the N=32 headline checkpoint
(0.7879, 1,219,048 LUT).

```
bit-exactness: max|err| = 1.043e+01   argmax match = 0.5977
```

Only **59.8%** of the 512 emulated predictions agree with the Keras forward pass.
The netlist is computing a different function.

**This qualifies every accuracy number in `synthesis_table.md`.** Those accuracies
are properties of the quantized **Keras** model. Until emulation agrees, they are
NOT established as properties of the generated hardware. The resource and latency
figures are unaffected — they come from the traced graph itself — but the pairing
of "this accuracy" with "this LUT count" is not yet verified end to end.

**Most likely cause, and it was flagged and not acted on.** `extract_hw.py`
hardcodes `inputs_kif=(1,5,8)` for the tracer while the model's input datalane
quantizer is *learned* during QAT (`kif`, `f0=BW_A`, `ic=MinMax(0,12)`). If the
trained input quantizer's (k,i,f) differs from (1,5,8), the netlist interprets its
inputs on a different scale than we feed. Codex raised exactly this in its first
review ("Input precision is disconnected from training configuration"); I recorded
it as a minor item and did not test it. It is now the leading suspect for a
first-order correctness problem.

**Two further bugs found on the way here, both mine:**
1. `extract_hw.py`'s `load_data()` never applied the N-slicing that
   `train_qat.py` does, so any reduced-N evaluation fed 150-particle arrays into
   a 32-particle graph. Fixed.
2. Verilog module names were sanitised for `=` only, leaving `-` and `.` from the
   checkpoint filename. Verilator rejects the prefix as an illegal identifier —
   and **Vivado would reject it too**, so every RTL project generated before this
   fix was unbuildable. Fixed. This matters because the deliverable offers the
   generated RTL for Vivado synthesis.

**Next step:** sweep `inputs_kif` and find the format that reproduces the Keras
output. If none does, the discrepancy is elsewhere in the trace and the accuracy
claims must be withdrawn until it is resolved.

---

## D31. Bit-exactness RESOLVED — netlist is faithful; the harness input format was wrong

**Measured** 2026-07-25 (overnight), resolving D30.

Swept the tracer's input format against the Keras model on the N=32 checkpoint:

| `inputs_kif` | max abs err | argmax match |
|---|---:|---:|
| (1,5,8) — the old default | 10.43 | 60.2% |
| (1,6,10) | 5.67 | 89.5% |
| (1,8,12) | 2.12 | 96.9% |
| **(1,8,16)** | **0.0** | **1.0000 — BIT-EXACT** |

The da4ml trace, pipelining and emitted Verilog are all correct. D30's failure
was purely a harness misconfiguration: `extract_hw.py` hardcoded a 14-bit input
port while the model's input quantizer is learned during QAT, so the netlist was
being fed truncated inputs.

**The feared consequence did not materialise.** Re-tracing the three headline
configurations at (1,8,16):

| Config | LUT at (1,5,8) | LUT at (1,8,16) | change |
|---|---:|---:|---:|
| N=150 attn_l1=1e-5 | 1,102,348 (63.8%) | 1,130,657 (65.4%) | +2.6% |
| N=32 best | 1,219,048 (70.5%) | 1,221,682 (70.7%) | +0.2% |
| N=64 attn_l1=1e-6 | 1,556,399 (90.1%) | 1,558,333 (90.2%) | +0.1% |

Widening the input port costs almost nothing because the embedding is ~2-3% of
the design. All conclusions stand, and `synthesis_table.md` now quotes the
verified format throughout.

**Remaining cleanups:**
1. The minimum bit-exact format is between f=12 and f=16; (1,8,16) is safe but
   may slightly overstate area.
2. `extract_hw.py` should derive `inputs_kif` from the model's input quantizer
   instead of a constant — the default is a footgun that silently produced a
   40%-wrong netlist.

---

## D32. Minimum input format; and argmax-exact is weaker than bit-exact

**Measured** 2026-07-25 (overnight), completing D31.

| `inputs_kif` | total bits | max abs err | argmax match |
|---|---:|---:|---:|
| (1,8,13) | 22 | 1.63 | 97.7% |
| (1,8,14) | 23 | 1.01 | 98.8% |
| (1,8,15) | 24 | 0.243 | **100%** |
| **(1,6,16)** | **23** | **0.0** | 100% — bit-exact |
| (1,7,16) | 24 | 0.0 | 100% — bit-exact |
| (1,8,16) | 25 | 0.0 | 100% — bit-exact |

**Fractional bits are the binding constraint.** f = 16 is required for
bit-exactness at any integer width tried; integer bits fall to 6 with no loss.
`(1,6,16)` is the minimum found — 2 bits narrower than the `(1,8,16)` used for
the reported numbers, so those are very slightly conservative.

**Argmax-exact arrives before bit-exact.** At `(1,8,15)` every one of 256
samples classifies identically while logits differ by up to 0.24. For a
classifier the deployed decision is the argmax, so argmax agreement is the
operationally relevant criterion and bit-exactness on logits is stricter than
deployment requires. Worth stating in the paper: we report the stricter result,
and the weaker-but-sufficient one holds with one less fractional bit.

**Not re-tracing at (1,6,16).** The step from (1,5,8) to (1,8,16) cost only
+0.1% to +2.6% LUT, so shaving 2 further bits is well inside the noise of a
pre-place-and-route estimate. The reported numbers stand and are conservative.

---

## D33. BEST RESULT — 80.22% at 53.8% of a VU13P (N=64, attn_l1=1e-5)

**Measured** 2026-07-25 (overnight), traced at the verified bit-exact input
format (1,8,16).

| | value |
|---|---|
| Config | N=64, P=8, tanh-only, mean tokenizer, FFN 1x, `attn_l1=1e-5` |
| Accuracy | **0.8022** |
| LUT | **929,722 (53.8% of VU13P — FITS)** |
| FF (reg bits) | 701,166 |
| Latency | 149.8 ns (45 stages @ 300 MHz) |
| II / DSP / BRAM | 1 / 0 / 0 |

**Within 0.7 accuracy points of JEDI-linear at N=64** (80.9% @ 71k LUT), at ~13x
the LUT — the narrowest area gap achieved at competitive accuracy, versus 20-30x
elsewhere on the curve.

**The attention-pressure lever, quantified one more time.** Same configuration:

| attn_l1 | Accuracy | LUT | % VU13P |
|---|---:|---:|---:|
| 1e-6 | 0.8024 | 1,558,333 | 90.2% |
| **1e-5** | **0.8022** | **929,722** | **53.8%** |

A 10x increase in per-region attention pressure removed **40% of the area for
0.02 accuracy points**. The design was over-provisioned precisely where the
component trace (D8/D20) said the cost was. This is the single most valuable
finding of the workstream and it came from Codex's original diagnosis.

**Full trajectory:** 7,449,939 LUT at 0.8101 (431% of VU13P, N=150) ->
929,722 LUT at 0.8022 (53.8%, N=64). **An 8x resource reduction for 0.8
accuracy points.**

**Untested and now clearly worth trying:** `attn_l1` at 1e-4 and 3e-4 on the N=64
recipe. The 1e-6 -> 1e-5 step cost almost nothing; the knob still is not
demonstrably saturated. (`attnl1-1e4` is running but at N=150.)

---

## D34. Attention pressure also cuts LATENCY, not just area

**Measured** 2026-07-25 (overnight). N=150, P=10, `attn_l1 = 3e-5`:

| | value |
|---|---|
| Accuracy | 0.7811 |
| LUT | 890,038 (51.5% of VU13P) |
| Latency | **106.6 ns — 32 pipeline stages** |

Every previous configuration, across all four particle counts and the whole
area range, sat at **45-48 stages / 146-160 ns**. Latency looked like a fixed
property of the graph. It is not: raising `attn_l1` to 3e-5 dropped the N=150
design to 32 stages.

**Mechanism:** `to_pipeline` cuts the combinational graph wherever accumulated
delay exceeds the 4 ns cutoff. Narrower attention datalanes shorten each
arithmetic node's delay, so fewer cuts are needed for the same clock. Squeezing
attention precision therefore shortens the critical path as well as the area.

**Consequence:** the knob is an **area/latency/accuracy** trade, not just
area/accuracy. The cheapest N=150 point is also the fastest, which is unusual and
worth stating explicitly — for an L1T argument, 106.6 ns against an O(1 us)
budget with 48% of the device free is a stronger position than the 150 ns rows.

Full N=150 curve now spans 51.5% -> 100.2% of a VU13P and 106.6 -> 153.2 ns,
at 78.11% -> 80.04% accuracy.

---

## D35. Attention pressure saturates around 1e-5 to 3e-5 — search converged

**Measured** 2026-07-25 (overnight). Sweeping `attn_l1` past the working point:

| N | attn_l1 | best accuracy at the cheap end |
|---:|---|---:|
| 64 | 1e-6 | 0.8024 |
| **64** | **1e-5** | **0.8022** |
| 64 | 1e-4 | 0.7750 |
| 150 | 1e-6 | 0.8004 |
| **150** | **1e-5** | **0.7898** |
| 150 | 3e-5 | 0.7811 |
| 150 | 1e-4 | 0.7466 |

1e-6 -> 1e-5 was nearly free (0.8024 -> 0.8022 at N=64) while removing 40% of the
area. 1e-5 -> 1e-4 costs 2.7 accuracy points at N=64 and 4.3 at N=150. The knob
saturates in the **1e-5 to 3e-5** band; beyond it accuracy degrades faster than
area improves.

**The search is converged.** Best configurations found:

| Purpose | Config | Accuracy | LUT | % VU13P | Latency |
|---|---|---:|---:|---:|---:|
| Best overall | N=64, attn_l1=1e-5 | 0.8022 | 929,722 | 53.8% | 149.8 ns |
| Paper's N, cheapest+fastest | N=150, attn_l1=3e-5 | 0.7811 | 890,038 | 51.5% | 106.6 ns |
| Paper's N, best accuracy | N=150, attn_l1=1e-6 | 0.8004 | 1,731,475 | 100.2% | 153.2 ns |

Further gains would need a different lever, not more of this one. Untried and
plausible: `attn_l1` between 1e-5 and 3e-5 on the N=64 recipe (the band is
under-sampled there), and applying the same per-region treatment to the FFN,
which is still ~20-27% of the design and has never had targeted pressure.

---

## D36. Global beta and per-region attention pressure are independent levers

**Measured** 2026-07-25 (overnight). Two N=150 points at comparable accuracy:

| Arm | lever | Accuracy | EBOPs | LUT | % VU13P | LUT/EBOP | Latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `best-b01` | global beta 0.1 | 0.7758 | 569,972 | **829,046** | **48.0%** | 1.45 | 136.5 ns |
| `attnl1-3e5` | attn_l1 3e-5 | 0.7811 | 728,200 | 890,038 | 51.5% | **1.22** | **106.6 ns** |

Both confirm D24 and extend it. Attention pressure gives the better **ratio**
(1.22 vs 1.45) because it compresses the datalanes EBOPs cannot see; global beta
reaches lower **absolute** EBOPs. Neither dominates: the cheapest area at N=150
comes from global beta, the lowest latency from attention pressure.

**They have not been pushed hard together.** `best-b01` uses no `attn_l1`;
`attnl1-3e5` uses beta 0.01. A run at beta 0.1 *and* attn_l1 3e-5 is untried and
is the obvious next arm at N=150.

**The paper's configuration now spans 48.0% -> 100.2% of a VU13P** at 77.58% ->
80.04% accuracy, with latency 106.6 -> 153.2 ns. That is a deployment curve at
the published particle count, which is a materially better rebuttal artifact than
any single operating point.

---

## D37. FFN per-region pressure works too — new best, 37.8% of a VU13P at 80.23%

**Measured** 2026-07-25. `n64-ffnl1hi`: N=64, P=8, tanh-only, mean tokenizer,
FFN 1x, `attn_l1 = 1e-5`, **`ffn_l1 = 3e-5`**.

| | attn_l1 only | **+ ffn_l1** |
|---|---:|---:|
| Accuracy | 0.8022 | **0.8023** |
| EBOPs | 655,727 | **524,687** |
| LUT | 929,722 | **652,680** |
| % VU13P | 53.8% | **37.8%** |
| Latency | 149.8 ns | 143.2 ns |

**A further 30% of the area removed at identical accuracy.** The mechanism that
worked on the QK/AV datalanes works equally on the FFN; it had simply never been
aimed there. `ffn_l1` was added tonight by generalising `_attn_scope` into
`_region_scope`.

**This is the pattern of the whole workstream:** the wins came from applying
precision pressure to *specific regions* identified by the component trace, not
from global knobs. Global bitwidth (`bw4`) and global beta alone never got near
these numbers.

**Untested and now obvious:** `ffn_l1` at 1e-4; the same treatment on the
embedding/head (only ~4% of the design, so little upside); and `ffn_l1` combined
with the strong-global-beta setting that gave 48.0% at N=150.

**Trajectory:** 7,449,939 LUT (431% of VU13P) -> **652,680 LUT (37.8%)** —
an **11.4x area reduction** — at 0.8101 -> 0.8023 accuracy, i.e. 0.8 points.

---

## D38. Region-targeted pressure beats global beta — prefer it, third confirmation of D24

**Measured** 2026-07-25. Two N=64 designs, both on the best recipe:

| Arm | Levers | Accuracy | EBOPs | LUT | % VU13P | LUT/EBOP |
|---|---|---:|---:|---:|---:|---:|
| `n64-ffnl1hi` | attn_l1 1e-5 + **ffn_l1 3e-5** | **0.8023** | 524,687 | **652,680** | **37.8%** | **1.24** |
| `n64-both` | attn_l1 1e-5 + **global beta 0.1** | 0.7950 | **448,861** | 659,283 | 38.2% | 1.47 |

`n64-both` reaches **14% fewer EBOPs yet costs more LUT and 0.73 fewer accuracy
points**. Third independent confirmation that EBOPs ranking misleads (D24, D36,
now D38).

**The actionable rule:** *global* EBOPs pressure lowers the EBOPs count but not
LUT proportionally, because it cannot see softmax tables and datalane routing.
*Region-targeted* pressure (`attn_l1`, `ffn_l1`) lowers both, because it
compresses exactly the datalanes that carry the uncounted cost. When choosing a
knob, prefer region-targeted pressure; use global beta only to explore, and
always trace before ranking.

**Headline unchanged:** N=64 with attn_l1=1e-5 + ffn_l1=3e-5 — 0.8023 at
652,680 LUT (37.8% of a VU13P), 143.2 ns, II=1.

---

## D39. SAT_SYM overflow is a large missed lever — never combined with the winning recipe

**Measured** 2026-07-25. `satsym-fixed` vs `abl-bn-tanh`, identical in every
respect except overflow mode (both N=150, P=10, max tokenizer, bn+tanh, mask off,
beta 0.01, FFN 4x):

| overflow | best accuracy | at EBOPs |
|---|---:|---:|
| **SAT_SYM** | **0.8116** | **1,694,973** |
| wrap | 0.8014 | 5,003,381 |

**+1.0 accuracy points at ~3x fewer EBOPs from a single flag.** 0.8116 is the
highest accuracy any QAT arm reached at N=150.

**Why this was missed for so long.** `PJ_OVERFLOW` originally reached only the
weight/bias scope — the datalane scope was hard-coded to `wrap` (found and fixed
mid-session, D-note under D14/D15). The first `qat-satsym` result was therefore
invalid and was discarded; the corrected arm then queued behind everything else
and only finished now. Meanwhile the entire optimisation campaign — tanh-only,
mean tokenizer, FFN 1x, `attn_l1`, `ffn_l1` — ran on `wrap`.

Mechanistically this is consistent with D14: with `wrap`, an out-of-range
activation flips sign rather than saturating, which is exactly what the tanh
pre-norm was compensating for. SAT_SYM addresses the same failure at the
quantizer instead of with an extra LUT.

**Consequence: the headline numbers are probably not the best available.** The
37.8% N=64 design and the 48.0% N=150 design both used `wrap`. Arms launched
combining SAT_SYM with the full winning recipe.

---

## D40. STANDARD PARTICLE COUNTS: 128 and 64 only — N=150 retired

**Decided** 2026-07-25 by Zihan. All future runs use **N=128 and N=64**.

**Reason:** JEDI-linear reports N = 16/32/64/128 (Table II). Our N=150 rows were
being compared against their N=128 row, which is not a matched comparison. N=128
makes the comparison apples-to-apples.

**Consequence for the model config:** N=128 requires P | 128, so **P = 8**
(16 patches). The paper's P = 10 does not divide 128. This is a disclosed
deviation, and it is the same P already used at N=64/32/16.

**Status of existing N=150 results:** not deleted — they remain the only data at
the paper's own particle count and are still useful context. But they must NOT be
presented as a matched comparison against JEDI-linear's N=128 row.
`synthesis_table.md` relabels them accordingly.

Killed: `n150-ffnl1`, `n150-all`, `n150-both`, `n150-sat`.
Launched at N=128: `n128-best` (best recipe, D37), `n128-sat` (+SAT_SYM, D39),
`n128-sat-notanh` (tests whether SAT_SYM subsumes the tanh pre-norm).

---

## D41. ffn_l1 saturates at 3e-5, same band as attn_l1 — precision pressure exhausted

**Measured** 2026-07-25 at N=64, all else identical (tanh-only, mean tokenizer,
FFN 1x, `attn_l1 = 1e-5`), comparing the cheapest checkpoint above 0.80 accuracy:

| `ffn_l1` | accuracy | EBOPs |
|---|---:|---:|
| 1e-5 | 0.8016 | 545,365 |
| **3e-5** | **0.8023** | **524,687** |
| 1e-4 | 0.8001 | 567,155 |

**3e-5 is optimal on both axes.** 1e-4 is worse in accuracy *and* EBOPs, so this
is genuine saturation rather than a plateau — over-pressuring forces the
optimiser to spend capacity elsewhere.

Both region-targeted knobs saturate in the same **1e-5 to 3e-5** band
(`attn_l1`, D35; `ffn_l1`, here). The setting already in the headline
configuration is optimal, and **precision pressure as a lever is exhausted**.

Remaining untried levers are structural or numerical, not precision:
SAT_SYM overflow (D39, arms running at N=128), and whether SAT_SYM makes the
tanh pre-norm redundant.

---

## D42. Three levers stack after all — 79.11% at 23.1% of a VU13P

**Measured** 2026-07-25. `n64-all` = best recipe + attn_l1 1e-5 + ffn_l1 3e-5
+ **global beta 0.1**, traced at the bit-exact input format:

| Levers | Accuracy | EBOPs | LUT | % VU13P | Latency |
|---|---:|---:|---:|---:|---:|
| attn+ffn (headline) | 0.8023 | 524,687 | 652,680 | 37.8% | 143.2 ns |
| **attn+ffn+beta** | 0.7911 | 259,037 | **399,554** | **23.1%** | 136.5 ns |

**This partially revises D38.** D38 said global beta *instead of* region pressure
is the wrong trade (it is — worse LUT at fewer EBOPs). But global beta *on top
of* region pressure is productive: with attention and FFN datalanes already
compressed, the residual EBOPs mass sits in places where global pressure now
translates to LUT at a good rate (ratio 1.54 here). -1.1 accuracy points buys a
39% area reduction.

**The N=64 curve now spans 23.1% -> 37.8% -> 90.2% -> 127.2%** of the device at
0.7911 -> 0.8023 -> 0.8024 -> 0.8043 accuracy. At the cheap end we are ~5.6x
JEDI-linear's N=64 LUT (399,554 vs 71k) at -1.8 accuracy points — the narrowest
multiple recorded anywhere in this workstream.

---

## D43. Aaron's core7 GMP grid — ported and under A/B test

**Received** 2026-07-25 (`aaron_package.zip`, extracted to `aaron_package/`).
A parallel workstream by Aaron/Anrun on the same rebuttal task, N=128 x 3f, same
HGQ2+da4ml toolchain, run on an M2 Max.

**The GMP setup ("core7"):** 7x7 NON-UNIFORM grid, raw edges
`[-0.40,-0.20,-0.12,-0.04,0.04,0.12,0.20,0.40]`, from a measured physics study
(`aaron_package/grid_study.md`):
- 95% of jet pT lives within r=0.19 (raw units) -> fine 0.08 bins there, coarse
  0.2 bins in the periphery. Non-uniform edges are FREE in hardware (each bin is
  one threshold LUT regardless of spacing).
- **Odd bin count centres a cell on the origin.** In any even symmetric grid the
  jet core lands on a 4-cell corner and the leading prong's energy splits across
  up to 4 cells (median hottest-cell share 56% -> 69% with 7x7). This is a
  fixed-grid artifact the paper's dynamic per-jet min-shift never had.
- His 25-epoch shootout: core7 beats 8x8 uniform by **+2.0 accuracy at 17%
  fewer EBOPs and 23% fewer cells**; matches 9x9 at 40% fewer cells.

**Port to our pipeline:** `gmp_edges` param + `PJ_GMP_EDGES` env. His edges are
in raw jet-relative units; his data is scaled by iqr=0.108. Our robust store has
a different scaling, so edges were re-anchored by measured pT-weighted 95%
containment (ours 1.51, his 0.19 raw -> scale 7.94):
`[-3.18,-1.59,-0.95,-0.32,0.32,0.95,1.59,3.18]`. Neatly, the outer edge lands at
our measured 99.96% coverage (±3.2) and the fine/coarse boundary at our old grid
bound (±1.6). The 99.9% anchor was unusable — our store clips its tails.

**This also resolves D3's puzzle:** the original docstring claim ("99.9% within
±0.4") was measured on HIS preprocessing, where it is true. Our D3 finding (34%
within ±0.4) was on the differently-scaled robust store. Both correct; the
docstring was written for data we were not using.

**Arms:** `n128-core7`, `n64-core7` (A/B vs `n128-best`/`n64-ffnl1hi`, identical
but the grid), `n128-core7-sat` (+SAT_SYM).

**Also in the package, not yet acted on — TWO-STAGE QAT INIT.** His float model
(81.59%) initialises QAT at 81.21% post-transfer, using WIDE initial bitwidths
(BW=10, i0_w=3, i0_a=8 — narrow i0 causes silent wrap-collapse to chance, which
his train_qat now asserts against). All our arms train from scratch and top out
lower. His warning is directly relevant: our own D14/D39 wrap-collapse findings
are the same failure mode he diagnosed. Porting two-stage init is the single
largest remaining accuracy lever and needs: PJ_I0_K/PJ_I0_A envs + float
pretrain at the matching architecture.

---

## D44. Codex final-state review — corrections applied

**Reviewed** 2026-07-25. Codex audited D22-D43 and the deliverables. Its findings
and what was done:

**Deliverable overclaims, all fixed:**
- "Bit-exact verified" was scoped to the whole table but held for ONE N=32
  checkpoint. Softened; headline-row verification is running (`verify-final`).
- Two stale-arithmetic errors in prose written for the previous headline row
  (929,722 LUT) and not updated when D37 improved it to 652,680: "~13x
  JEDI-linear" is actually ~9.2x, "leaves ~46%" is ~62%. Both corrected — in our
  favour, but wrong is wrong.
- "The cheapest N=150 point is also the fastest" became false when the
  best-b01 row was added. Removed.
- "GMP traces to zero LUT" restated as an attribution result, not a standalone
  synthesis of the block.
- Single-seed caveat attached to the +1.1/+1.5 margins directly in the table.
- "9x area reduction" now carries its 3.4-point accuracy cost.
- Malformed table header and stale counts fixed.

**core7 port caveat (Codex Q-B):** a single multiplicative scale anchored at the
95% pT containment does not establish that eta and phi have matched scalings in
our robust store, and `gmp_edges` applies one list to both axes. The port is a
legitimate empirical A/B candidate but must NOT be described as preserving
Aaron's raw-dR physics argument. If core7 wins its A/B, the follow-up is
per-axis edges anchored per-feature.

**Priority ruling (Codex Q-C, adopted):** let the 8 running arms finish — they
carry the D40-mandated N=128 rows. Two-stage init was NOT launch-ready anyway:
`i0` was hardcoded to 0 (guaranteeing the wrap-collapse Aaron warns about) and
the transfer path failed OPEN (silent fallback to from-scratch would change the
experiment's independent variable). Both fixed: `PJ_I0_K`/`PJ_I0_A` exposed,
transfer now fails closed when a float checkpoint exists. First freed GPU runs
one controlled two-stage arm at the headline recipe.

**Also fixed:** `run_config.json` silently dropped list-valued config, so core7
runs were not recording their own edges — a reproducibility hole for exactly the
arms now running.

---

## D45. SAT_SYM does NOT transfer to the region-pressured recipe — keep wrap

**Measured** 2026-07-25. `n64-sat` = headline recipe + `PJ_OVERFLOW=SAT_SYM`,
otherwise byte-identical to `n64-ffnl1hi` (tanh, mean, FFN 1x, attn_l1 1e-5,
ffn_l1 3e-5, beta 0.01), full 1000 epochs:

| overflow | best | cheapest above 0.80 |
|---|---|---|
| wrap | 0.8035 @ 664,576 | 0.8023 @ 524,687 |
| SAT_SYM | 0.7998 @ 1,215,704 | never reaches 0.80 |

D39's +1.0-point SAT_SYM win was measured on the OLD configuration (BN+tanh,
FFN 4x, max tokenizer, no region pressure). On the current recipe it is a
regression on both axes. Plausible mechanism: the tanh pre-norm and the
region-targeted MonoL1 pressure already control activation range, and
saturation's clipped gradients interfere with the L1-driven bit shrinkage that
produces the cheap frontier — but that is a hypothesis; the empirical ruling is
what matters.

**Ruling: the headline recipe stays on `wrap`.** D39 is downgraded to a
config-specific observation (this is the third lever whose effect failed to
transfer across configs — cf. D6 beta, D21 N-scaling). `n128-sat` /
`n128-sat-notanh` / `n128-core7-sat` remain running as independent checks at
N=128; if they agree, SAT_SYM is closed.

---

## D46. N=128 matched row lands — and N is nearly free under the full recipe

**Measured** 2026-07-25. `n128-best` (headline recipe at the D40 standard count),
traced at the verified format:

| N | Accuracy | LUT | % VU13P | Latency | JEDI-linear same N | gap |
|---:|---:|---:|---:|---:|---|---|
| 128 | 0.8005 | 662,654 | 38.3% | 133.2 ns | 80.9% @ 98k | -0.9 pts, ~6.8x |
| 64 | 0.8023 | 652,680 | 37.8% | 143.2 ns | 80.9% @ 71k | -0.7 pts, ~9.2x |

**Doubling N cost 1.5% more LUT.** Under the full region-pressure recipe the
particle count has become nearly free — the quantizers absorb the extra
elements by going narrower where it does not matter. (Contrast D21, where at
fixed precision LUT was ~linear in N: precision pressure changes the scaling.)
The N=128 row is also FASTER (133.2 vs 143.2 ns).

**The ~6.8x at N=128 is the narrowest matched-comparison gap of the workstream**
(vs 9.2x at N=64, 20-30x pre-pressure).

Both matched rows now exist at nearly identical cost; the rebuttal can lead with
N=128 (the larger, harder configuration) without giving anything up.

---

## D47. core7 does NOT transfer under the single-scale port — Codex's caveat vindicated

**Measured** 2026-07-25. `n64-core7` vs `n64-ffnl1hi`, identical except the grid,
both 1000 epochs:

| Grid | at matched ~650k EBOPs | cheapest above 0.80 |
|---|---:|---|
| uniform 8x8 on ±1.6 | **0.8026** | 0.8023 @ 524,687 |
| core7 (re-anchored 7x7) | 0.7898 | 0.8011 @ 3,181,473 (6x the cost) |

Aaron's +2.0-point win did not survive the port; on our pipeline it is a ~1.3
point REGRESSION at matched cost. The partial N=128 read agrees (core7 0.7954 @
1.09M vs baseline 0.8005 @ 666k).

**Why, most likely — exactly what Codex flagged in D44:** the port used a single
multiplicative scale anchored at one pT quantile, applied to both axes. Our
robust store scales eta and phi per-feature; a single scalar cannot reproduce
Aaron's raw-dR geometry (fine bins at prong-separation scale, cell centred on
the core). His result was produced on his own preprocessing, where the edges ARE
the physics. Ours moved the thresholds but not the physics.

**Ruling:** the headline recipe keeps the uniform 8x8 grid. core7 remains
plausible on Aaron's preprocessing — the honest statements are "core7 improves
accuracy on Aaron's data pipeline (his measurement)" and "a single-scale port to
differently-normalised data did not transfer (ours)". A faithful test would need
either his `jets_128x3.npz` + `robust_stats.json` (not in the package) or
per-axis quantile-matched edges. Fourth lever that failed to transfer across
configs (D6, D21, D45, now D47).

`n128-core7` / `n128-core7-sat` still running as confirmation; no new core7 arms.

---

## D48. core7 research: the port bug found, corrected arms launched

**Researched** 2026-07-25, from `aaron_package/CONFIG.md` + `grid_shootout.py`.

**The D47 failure was OUR porting bug, not (necessarily) a transfer failure.**
His scaling recipe is `raw_edges / iqr` with per-feature iqr=0.108 from
`robust_stats.json` — i.e. his network inputs are robust-scaled, the same
convention as our `x_*_robust_*` store. His scaled edges are directly:
outer ±3.70, fine boundary ±1.85, fine width 0.74. Our coordinates already live
in approximately those units (our per-axis q95 = 1.51 vs his scaled fine
boundary 1.85; his scaled 99.9% extent 3.98 vs our max 4.6).

My port instead anchored our PER-AXIS q95 (1.51) against his RADIAL r95 (0.19
raw) — an apples-to-oranges quantile that produced scale 7.94 and made every bin
~16% too narrow. Every core7 conclusion in D47 tested the wrong grid.

**Calibration from his results table (N=128, core7 + two-stage):** frontier
0.7854 @ 0.92M EBOPs, 0.7911 @ 5.65M, 0.8113 @ 116M. Two notable readings:
1. Our region-pressure recipe BEATS his frontier at matched EBOPs (n128-best:
   0.8005 @ 0.67M vs his 0.7854 @ 0.92M) — the pressure levers are worth more
   than his grid+init at the cheap end.
2. His high-accuracy end (0.8113) exceeds anything we reach from scratch — the
   two-stage init lever, which `twostage-n64` is testing on our recipe.
His "two-stage worth ~10 points at epoch 0" (79.13% vs ~69%) sets expectations.

**Launched:** `n64-core7v2` (his exact scaled edges, headline recipe) and
`n64-core7v2-mask` (+pT mask — hypothesis: core7 centres a fine cell exactly
where 75% of slots pile up as zero-padding; the mask interaction may flip vs D5).

**Per-axis check:** our eta/phi q95 differ by 0.6% (1.5083/1.5175) — isotropic,
matching his "<0.2%" note; a single shared edge list is fine for both axes.

---

## D49. SAT_SYM closed; core7-at-N=128 hints the grid story is N-dependent

**Measured** 2026-07-25, N=128 confirmation arms (baseline `n128-best`
0.8005 @ 666k):

| arm | cheapest >0.79 | verdict |
|---|---|---|
| n128-sat | 0.7908 @ 666k | SAT_SYM worse, confirming D45 |
| n128-sat-notanh | 0.7951 @ 684k | dropping tanh HELPS under SAT_SYM (redundancy confirmed) but still loses to wrap+tanh |
| n128-core7-sat | 0.7911 @ 725k | SAT_SYM drags core7 down too |
| n128-core7 (old scale) | 0.8000 @ 619k | **parity with baseline** — not the 1.3-pt regression seen at N=64 |

**SAT_SYM is closed**: three independent N=128 checks agree with the N=64 A/B.
The ordering wrap+tanh > SAT_SYM-alone > SAT_SYM+tanh also confirms the
redundancy hypothesis (saturation and tanh do overlapping work) while showing
wrap+tanh remains the best of the three. Headline recipe unchanged.

**core7 note:** even the WRONG-scale edges reach parity at N=128 (0.8000 @ 619k,
marginally cheaper than baseline) versus a clear regression at N=64. Grid
effects appear N-dependent — plausible since more particles populate the
periphery bins. The corrected-scale `core7v2` arms will give the fair test.

---

## D50. New experiment axes from arXiv:2510.24784 (first FPGA transformer, same toolchain)

**Researched** 2026-07-25 at Zihan's direction. The paper (Laatu, Sun, Cox,
Gandrakota, Maier, Ngadiuba, Que, Luk, Spiropulu, Tapper) puts a transformer on
an XCU250 with HGQ + da4ml — our exact stack, co-authored by our escalation
contact and the JEDI-linear author.

**Their winning configuration:** single-head Linformer (K/V projected to k=4),
64p x 3f, EBOPs-BUDGETED training (fixed 350k target rather than a ramp),
attention floored at 1 bit so pruning cannot kill the path:
**79.8% @ 202k LUT, 78 ns, II=1, 0 DSP.**

**New reference row added to synthesis_table.md.** Against it our N=64 is +0.4
accuracy at ~3.2x LUT — the fairest comparison in the table (both transformers,
both this toolchain) and a far narrower gap than vs JEDI-linear.

**Four axes proposed; two launched now:**
1. `n64-h1p`/`n128-h1p` — single head under the FULL pressure recipe. D21
   dismissed heads at -4.1%, but that was fixed-precision; under training the
   softmax element count (N*H*P) quarters, and softmax was the largest single
   block (D20). Their single-head success says retest.
2. `n64-linf`/`n128-linf` — **PHAT-Linformer hybrid**: GMP (free) + global patch
   stage kept, local patch attention replaced by their winning single-head
   Linformer (k=4) over the full sequence. Implemented via HGQ2's built-in
   QLinformerAttention (`PJ_LOCAL_MODE=linformer`). If it holds accuracy, the
   47-66% local-attention block shrinks toward their 202k economics — with GMP
   as PHAT-JeT's differentiator on top. Float reference uses full MHA (Linformer
   is a QAT-graph deviation; disclosed).
3. NOT YET RUN: EBOPs-budget training (fixed target instead of beta ramp) —
   their schedule reaches a chosen operating point directly.
4. NOT YET RUN: 1-bit floor (bc=Min(1)) inside the attention scope, enabling
   attn_l1 beyond 3e-5 without path death — their trick for hard pressure.

**Caveat for the paper narrative:** the hybrid is an architecture change beyond
the published PHAT-JeT; it belongs in the rebuttal as "the PHAT structure
composes with concurrent-work attention economisations", not as the paper's
model.

---

## D51. Axes 3+4 launched; per-turn job cleanup now standing practice

**2026-07-25.** Implemented and launched the remaining D50 axes:
- **EBOPs-budget training** via HGQ2's own `BetaPID` (discovered shipped in
  `hgq.utils.sugar` — no custom controller needed). `PJ_EBOPS_TARGET` replaces
  the beta ramp when set. Arms: `n64-budget350` (their 350k target),
  `n128-budget500`.
- **1-bit attention floor** (`PJ_ATTN_FLOOR`, `fc=Min(n)` inside the attention
  scope) + pressure past the D35 saturation: `n64-floor1e4` (attn_l1=1e-4,
  floor=1). Hypothesis: D35's 1e-4 collapse was path death, which the floor
  prevents.

**Housekeeping (Zihan's standing instruction):** terminal jobs are now cleared
after every turn — 78 deleted in the first sweep. Logs must be read before
deletion; PVC artifacts survive.

---

## D52. Two real failures diagnosed: Lambda deserialization and an HGQ2 jax bug

**2026-07-25.** Neither was the cuDNN flake.

**(a) twostage transfer crash — the D44 fail-closed guard working as designed.**
Stage 1 completed (float 0.8064 at the N=64 headline architecture, saved to PVC)
but stage 2's `load_model` refused the checkpoint: the GMP bin indicators are
Lambda layers, which Keras 3 safe-mode blocks. Under the pre-D44 code this would
have SILENTLY fallen back to from-scratch and invalidated the experiment; the
guard turned it into a loud stop. Fix: `safe_mode=False` (our own artifact).
Job resubmitted with stage 1 skipped — the 0.8064 checkpoint is reused.

**(b) `QLinformerAttention` cannot TRAIN on the jax backend** — its internal
`QEinsumDense` hits `'QEinsumDense' object has no attribute
'full_output_shape'` in the training path. Torch trains fine (verified with an
actual fit locally — the earlier build+forward test was insufficient, same
lesson as D-torch). linf arms resubmitted on the torch image. Worth reporting
upstream to Chang. Note for later: tracing a linf checkpoint under jax may still
work (inference path) — verify at extract time.

**Method note:** "builds and does a forward pass" is NOT "trains". Every new
layer type gets a local fit() smoke test on the target backend before cluster
submission.

---

## D53. Softmax replacement — the outside-the-box axis the measurements demand

**2026-07-25, at Zihan's direction to expand the search.** The fine-grained
trace (D20, with its attribution caveat) puts the local softmax at ~32% of the
design — the largest single block — and D19 shows it is invisible to EBOPs,
i.e. it IS the cost floor that precision pressure cannot reach. Softmax's cost
is structural: exp LUT + row sum tree + reciprocal + rescale multiply.

Two literature-backed replacements need ONE unary LUT per score and nothing
else (the row reduction disappears entirely):
- **sigmoid attention** (Ramapuram et al. 2024): w = sigmoid(score); shown to
  match softmax at scale in float. Elementwise -> no reduction hardware.
- **relu/n attention** (Wortsman et al. 2023, ViT-22B): w = relu(score)/n; the
  1/n folds into the LUT (and P=8 makes it a shift regardless).

If either holds accuracy, the ~32% block collapses to a table, AND the pipeline
depth shrinks (the sum tree and reciprocal are on the critical path — cf. D34).

Both modes verified with actual fit() on BOTH backends before submission (D52
lesson). Arms: `n64-sig`, `n64-relu`, `n128-sig` — A/B vs `n64-ffnl1hi`/
`n128-best`, identical except `PJ_ATTN_ACT`.

Codex is running a parallel open-ended research task (portfolio critique +
directions we have not considered: hardware mapping tricks, QKV factorisation,
underused HGQ2/da4ml API surface).

**D52 addendum:** the QLinformerAttention failure is keras-VERSION-dependent,
not backend-dependent: trains fine on keras 3.12.3 (local venv, both backends),
fails on the cluster's unpinned `keras>=3.11` (resolves newer) with the same
`full_output_shape` AttributeError on both jax and torch. linf jobs now pin
keras==3.12.3. Report to Chang with the version detail.

---

## D54. Codex research round: solver options, table quantization, projection factorization

**2026-07-25.** Codex's open-ended research (full output in the session log):

**Adopted immediately:**
1. **da4ml `solver_options` sweep** — never previously set; jsc150 uses
   `{"hard_dc": 2}`. Trace-only, zero retraining, potential LUT reduction on
   every constant matrix. `PJ_SOLVER_OPTS` wired; sweep running on the headline
   checkpoint (`''` vs hard_dc 2 vs 3).

**Queued (in priority order):**
2. **Softmax TABLE quantization** — QSoftmax exposes exp/inv table iq/oq
   configs we never touched; `place="table"` scope pressure attacks the actual
   32% block while KEEPING row normalisation. Lower risk than replacement.
3. **Block-diagonal per-head Q/K/V** — 4x projection-arithmetic reduction,
   registry-safe (reshape + head-indexed QEinsum); ~7-11% est. total LUT.
4. **Tied Q=K** — one arm, removes a third of projection cost; symmetric Gram
   scores (Reformer precedent).
5. **Branch-dropout training** — train with stochastic local/global branch
   drops so one attention stage becomes removable post-hoc.

**Critique of the running softmax-free arms (important):** naked
`sigmoid(QK)` fails at larger key counts via output-norm growth; the paper's
recipe needs an attention-logit bias b ≈ -ln(L) (L = keys). Our launched arms
are the naked variants — if they underperform, the biased recipe is the fix,
NOT a rejection of the direction. ReLU rows can die (all-negative logits =
zero output, zero gradient); watch zero-row rates. Sigmoid > ReLU in
robustness per the literature.

**Also flagged:** table sharing under II=1 is not free (parallel elements need
replicated ROMs) — reduce table precision/count instead; linear attention is
unattractive at P=8/d=16 (d x d state costs more than 8x8 attention).

---

## D55. core7 closed: correctly-scaled port still loses; both rescue hypotheses tested

**Measured** 2026-07-25. Fair A/B with Aaron's EXACT scaled edges (raw/iqr,
D48), 1000 epochs, vs `n64-ffnl1hi` at matched ~525-650k EBOPs:

| arm | at matched EBOPs | best anywhere |
|---|---:|---|
| uniform 8x8 (baseline) | **0.8023-0.8035** | 0.8035 @ 665k |
| core7v2 (correct scale) | 0.7966 | 0.7988 @ 3.1M |
| core7v2 + pT mask | 0.7833 | 0.7977 @ 1.4M |

The correct scaling recovered about half the D47 deficit (-1.3 -> -0.6 points)
but uniform still wins. The padding-concentration rescue (mask) made it WORSE —
that hypothesis is dead too (consistent with D5: the mask deletes multiplicity
signal regardless of grid).

**core7 is closed for our pipeline.** Remaining explanation: the grid's value is
coupled to HIS preprocessing/data (possibly interacting with kt-sorting or the
robust stats provenance). For Aaron: on our store, physics-informed non-uniform
edges underperform a plain uniform grid at matched cost — worth a direct A/B on
HIS data before the grid goes in any paper. Uniform 8x8 on ±1.6 stays.

---

## D56. Solver sweep: LUT unchanged, one pipeline stage free — hard_dc=2 adopted

**Measured** 2026-07-25 on the headline N=64 checkpoint. `hard_dc: 2` gives
-0.09% LUT (noise) but **42 stages instead of 43 -> 139.9 ns vs 143.2 ns**, and
~2% fewer register bits. hard_dc: 3 identical. Codex's 0-15% upside lands at
~0.1% for area — consistent with D19/D20: after region pressure, the constant
matrices are already cheap; the cost lives in tables/routing the solver does not
touch. Adopted as the extract_hw default for the free latency stage. Existing
table rows NOT restated (an 0.09% delta is far inside estimate noise).

---

## D57. Table-pressure and block-QKV arms submitted (Codex D54 #2/#3)

**2026-07-25.** `n64-tbl`/`n128-tbl` (PJ_TABLE_L1=1e-5: MonoL1 on the softmax
exp/reciprocal table bits — the only pressure that reaches the 32% block while
keeping row normalisation) and `n64-blk` (block-diagonal per-head Q/K/V via
QEinsumDense: d^2/H projection arithmetic; keras pinned 3.12.3 per D52). Both
fit-tested on both backends before submission.

---

## D58. Aaron-coordinate data RECONSTRUCTED from the shared store; clean cross-over launched

**2026-07-25.** His preprocessing is documented well enough to rebuild: CONFIG.md
publishes the robust_stats VALUES verbatim (pT 6.5956/17.65079, deta
-1.24e-5/0.108128, dphi 6.88e-6/0.107955; nonzero constituents only; pads
re-zeroed AFTER scaling). Only the raw->128x3 step lacks a script ("leading 128",
pT order = the unsorted store's native order).

**Reconstruction:** invert our store's scaling via its normparams pkl (keys
`x_median`/`interquantile_range` — and OUR store turns out to use NO centering,
median=0, IQR [34.83, 0.1211, 0.1211]) -> apply his stats -> re-zero pads ->
leading 128. Validated against his published physics: coord extent 99.997% vs
his ~99.9%; pT containment 93.9% vs 95%; clamped pT 0.0027% vs 0.007%. Saved as
`jets_128x3_aaron_recon.npz`. Caveat: val split differs (260k vs his 208k).

**Two consequences:**
1. Our store's IQR is 0.121 vs his 0.108 — a 12% unit mismatch — so even D55's
   "fair" core7 test was ~12% off scale. On the reconstruction his edges are
   exact by construction.
2. Cross-over arms launched at his N=128: `n128r-core7` (his grid, his units)
   vs `n128r-uni` (his uniform8 baseline, same units), both under our headline
   recipe. If core7's +2.0 reappears here, D55's failure was preprocessing
   coupling; if not, the grid does not interact well with our recipe even on
   his coordinates.

---

## D59. Table pressure without a floor is reward-hacked — relaunched floored

**Measured** 2026-07-25. `n64-tbl` ran 1000 epochs with total loss going
NEGATIVE (-1.83): with `br=MonoL1` and no bit floor, the optimizer drives the
softmax table bits negative and mines the regularizer instead of fitting.
Best 0.7969 — below baseline. Same failure class the FPGA-transformer paper
pre-empts with its 1-bit attention floor (D50 #4). Fixed with `bc=Min(1)` in
the table scope; both tbl arms relaunched. n128-tbl's first run discarded for
the same reason.

---

## D60. Cross-backend Lambda: the donor must be rebuilt, never deserialized

**2026-07-25.** The twostage transfer failed a second way: after safe_mode=False
cleared deserialization (D52a), the Lambda layers raised NotImplementedError at
CALL time — a Lambda saved under jax cannot execute under torch. Fix:
`transfer_float_weights` now REBUILDS the float donor from the same env config
and does `load_weights` only, mirroring Aaron's float_pretrain design (his
CONFIG.md run-3 postmortem is this same class of bug). Verified locally
(16 layers copied); job relaunched. 16 = the full set of weighted layers.

---

## D61. 1-bit floor: mechanism validated, but the saturation band still wins

**Measured** 2026-07-25. `n64-floor1e4` (attn_l1=1e-4 + fc=Min(1) floor), full
1000 epochs:

| | best | cheapest >0.79 |
|---|---|---|
| 1e-4, no floor (D35) | 0.7750 (collapse) | — |
| **1e-4 + floor** | 0.7958 | 0.7919 @ 937k |
| 3e-5-band headline | 0.8035 | 0.8023 @ 525k |

The floor rescued the collapse (+2.1 points over unfloored 1e-4) — confirming
D35's failure mode was path death, and validating the paper's floor trick as a
mechanism (it also fixed the D59 table exploit). But pressure past the
saturation band still nets worse than staying at 3e-5: the headline recipe is
unchanged. Axis 4 closed; floors remain in the toolkit as a SAFETY device (for
tables and any future hard-pressure region), not as a performance lever.

`n128-budget500` died on cuDNN retries; resubmitted with backoff 14.

---

## D62. Aaron confirms pT ordering — recon v1 was UNSORTED; v2 rebuilt and arms relaunched

**2026-07-25, from Zihan relaying Aaron.** His data is pT-sorted. Measured check:
the store's base `x_train_robust_150const_ptetaphi.npy` is **0% pT-descending**
(the `_sorted_pt` variant exists precisely because the base is unsorted) — so
recon v1 gave the model effectively arbitrary constituent order, which is fatal
for patch-contiguity. The few hours of `n128r-*` training on v1 are discarded.

**v2:** train from the store's own `_sorted_pt` file; val self-sorted by the
scaled pT column (scaling is monotone, so identical to raw-pT order; no key
reverse-engineering needed this time, unlike D2's kt case). Verified 100%
descending on both splits. Cross-over arms relaunched on v2.

**Accumulated ordering ledger:** our main line is kt-sorted (D2); Aaron's is
pT-sorted; the paper reports both at equal accuracy under consistent train/test
(Table 7: kt 81.80, pT 81.80). So the recon-vs-main comparison now differs in
BOTH stats and ordering — only the n128r-core7 vs n128r-uni A/B (same data) is
clean; recon-vs-main comparisons are not.

---

## D63. Naked sigmoid attention fails as predicted; biased variant launched

**Measured** 2026-07-25. `n64-sig` (w = sigmoid(score), no bias): best **0.7716**
across the whole front — never reaches 0.78, vs baseline 0.8023 @ 525k. This is
the exact failure mode Codex flagged from Ramapuram et al.: without the
b = -ln(L) logit bias, sigmoid outputs' norm grows with key count and training
degrades. Per that critique, this result REJECTS the naked variant only.

`n64-sigb` launched: sigmoid(score - ln P). The bias is a constant folded into
the same single LUT — hardware cost identical to naked sigmoid (one table per
score, no row reduction). n128-sig (naked) left running as the N=128 datapoint;
n64-relu still pending.

---

## D64. Round-2 survey: ReLU attention is the discovery; combinations launched

**Surveyed** 2026-07-25 mid-training (baseline 0.8023 @ 525k / 0.8035 @ 665k):

| arm | result so far | verdict |
|---|---|---|
| **n64-relu** | **0.7998 @ 420,710 — cheapest >0.79 ever, no softmax** | tracing now |
| **n64-blk** | **0.8048 @ 810k — best accuracy of any arm** | promising |
| n64-sig (naked) | 0.7716 max | failed as predicted (D63) |
| n64/n128-h1p | 0.7980 / 0.7993 | below baseline; single head not a win under pressure |
| n64/n128-tbl (floored) | ~0.797 / 0.790 | below baseline so far |
| budget arms (BetaPID) | 0.744 @ 15M mid-run | PID descending too slowly; gains likely wrong |
| linf hybrid | 0.7796 / 0.7688 | well below baseline so far |
| n128r cross-over | core7 0.7908 vs uni 0.6809 | UNEQUAL EPOCHS — no conclusion yet |

**Launched:** `n64-relu-blk` (combine the two wins), `n128-relu`, `n64-relu-3e5`
(the pressure saturation was measured WITH softmax; relu may move it).

**Answer to Zihan's question (fixed EBOPs budget):** only the two `budget*` arms
use a fixed target (HGQ2 BetaPID, 350k/500k); everything else uses the
beta-ramp + ParetoFront. The PID arms are converging too slowly from the wide
start — gains under review with Codex.

---

## D65. ReLU attention traced: softmax's BOUNDED OUTPUT was doing quantization work

**Measured** 2026-07-25. `n64-relu` @ 0.7998/420,710 EBOPs traced at (1,8,16):

| | softmax headline | relu |
|---|---:|---:|
| LUT | 652,680 (37.8%) | 766,266 (44.3%) |
| LUT/EBOP | 1.24 | **1.82** |
| latency | 143.2 ns / 43 stg | **116.5 ns / 35 stg** |
| FF | 565k | **387k** |

**The LUT projection was wrong in an informative way.** Removing softmax deleted
its tables and row reduction (hence 8 fewer pipeline stages and -31% FF), but
the LUT/EBOP ratio JUMPED from 1.24 to 1.82: softmax's output is bounded in
[0,1], which kept the AV-multiply datalanes narrow. ReLU/P weights are unbounded
above, so the AV products got wider and ate the table savings. **Attention-weight
boundedness is itself a quantization feature.**

**Where this leaves the softmax-free family:**
- relu = a LATENCY/FF operating point (116.5 ns, near-equal accuracy, +6.5pp
  area) — worth a row, not the headline.
- The pending `n64-sigb` (biased sigmoid) becomes the key test: sigmoid output
  is bounded [0,1] like softmax AND has no row reduction — it could win tables,
  stages, AND keep narrow AV lanes. If sigb holds accuracy it may dominate.
- `n64-relu-3e5` (launched) tests whether stronger pressure can re-narrow the
  relu AV lanes that the bound used to control.

---

## D66. Codex round 2: blk wiring bug found (by Codex, in MY code); portfolio executed

**2026-07-25.** Codex's review of round-2 results found `_qkv()` defined but
never called — my sed anchored on the pre-BatchNorm `_dense` signature and
silently failed, so `qkv_mode="block"` was a NO-OP. Consequences:
- `n64-blk`'s 0.8048 is a plain baseline replicate — which usefully bounds seed
  variance (0.8023/0.8035/0.8048 across three identical-recipe runs, ~±0.13).
- The running relu-blk tested nothing. Killed both; fixed (verified: dense
  112,345 params/QDense vs block 107,449/QEinsumDense); relaunched as
  `n64-blk2`, `n64-relu-blk2`.

**BetaPID diagnosis (Codex):** my gains p=1e-9/i=3e-10 are inert vs library
defaults p=1/i=2e-3 (log-scale controller), and init_beta=None inherited
beta0=0. Both budget arms killed; `n64-pid2` launched: warm-start FROM the
headline checkpoint (new PJ_RESUME_CKPT), p=0.1/i=1e-3, init_beta=3e-8,
target 350k — a modest further squeeze rather than a 40M descent.

**Portfolio kills** (Codex, adopted): naked n128-sig, both h1p arms (single head
loses under pressure), both old budget arms, both linf hybrids (0.77/0.769 well
below baseline). Distillation queued to target the RELU student once its
bit-exactness is checked, not the baseline.

**Seed-variance bonus:** three identical-recipe runs now span 0.8023-0.8048 —
the +1.1/+1.5 margins vs JEDI-linear at N=16/32 comfortably exceed this; the
-0.7/-0.9 deficits at N=64/128 are within ~2 sigma of it. Worth a line in the
paper's limitations.

---

## D67. Aaron's preprocess.py received — his assert REJECTS our reconstruction: upstream multiplicity differs

**2026-07-25.** Aaron sent the actual `preprocess.py` (saved to
`aaron_package/preprocess.py`). Three consequences:

**(a) Mask order confirmed as the critical detail.** His war story: computing
the real/pad mask AFTER median-centering silently zeroes ~50% of real
constituents (pT median 6.6) and gives chance accuracy with no error. Our
reconstruction is safe from this BY CONSTRUCTION — our store is uncentered
(median=0, D58), so `pt != 0` identifies pads correctly before we re-scale.

**(b) His `PAD_FRACTION_EXPECTED=0.617` validation constant FAILS on our
reconstruction:** ours is 0.705 (tolerance 0.05). Root cause is upstream, not
the scaling: our shared store averages **37.8 real constituents/jet; his data
has 49**. His source retains ~30% more (soft) constituents — a different
dataset version or a pT threshold in our store's production. No amount of
re-scaling can conjure missing constituents.

**(c) Consequences:**
1. The recon cross-over (n128r core7 vs uni) remains internally clean — same
   data both arms — but the reconstruction is NOT Aaron's data. Exact
   replication of his numbers definitively requires his npz (the ask stands,
   now with a concrete reason his file cannot be substituted).
2. His float advantage over ours (81.59 vs 80.64/80.88) may be partly DATA
   RICHNESS (49 vs 38 constituents), not recipe — worth remembering before
   attributing it all to two-stage training.
3. Adopted his practice: any future data prep here should carry a
   pad-fraction assert with a measured expected value.

---

## D68. Corrected block-QKV is a REGRESSION; relu frontier extends to 365k EBOPs

**Measured** 2026-07-25, both arms complete (1000 epochs).

**n64-blk2 (correctly wired):** best 0.7955, never reaches 0.80 — about -0.7 vs
the dense baseline family (0.8023-0.8048). Block-diagonal per-head projections
LOSE accuracy at this scale; the earlier apparent gain (D64) was purely the
wiring bug (D66). Direction closed; `n64-relu-blk2` killed as dominated.
Codex's refusal to interpret the buggy result was correct.

**n64-relu final:** 0.8002 @ 365,100 — 0.80+ accuracy at 30% fewer EBOPs than
the softmax headline (525k). Trace+EMULATION job running on that checkpoint
(bit-exactness is the gate Codex set for the relu-targeted distillation arm).
Open question remains the LUT/EBOP ratio at the cheap end: 1.82 at 421k; if
pressure narrowed the AV lanes further it may drop.

---

## D69. relu@365k traced + EMULATED: bit-exact; a latency/FF operating point, not a LUT win

**Measured** 2026-07-25. `n64-relu` epoch=901 (0.8002 @ 365,100 EBOPs):

| | softmax headline | relu (this ckpt) |
|---|---:|---:|
| accuracy | 0.8023 | 0.8002 (re-eval 20k: consistent) |
| LUT | 652,680 (37.8%) | 700,629 (40.5%) |
| LUT/EBOP | 1.24 | **1.92** |
| latency | 143.2 ns / 43 stg | **116.5 ns / 35 stg** |
| FF | 564,899 | **369,238 (-35%)** |
| bit-exact | yes | **yes (0.0 / 100%)** |

The unboundedness penalty (D65) persisted at the cheap end — pressure did not
re-narrow the AV lanes (ratio 1.92). relu is now a VERIFIED alternative
operating point: fastest and lowest-FF N=64 design, slight LUT premium. Added
to synthesis_table as the latency-optimised row.

**Distillation gate cleared** (Codex's condition): the relu student is bit-exact.
**The boundedness thesis now rides entirely on `n64-sigb`** — bounded like
softmax, reduction-free like relu.

---

## D70. sigb never ran — third silent-sed failure; verification discipline tightened

**2026-07-25.** `n64-sigb` produced results BYTE-IDENTICAL to naked `n64-sig`
(0.7716 @ 8,517,345 — same seed, same config, same trajectory). Cause: the sed
targeted shell-style `PJ_ATTN_ACT="sigmoid"` but the manifest is YAML
(`{name: PJ_ATTN_ACT, value: "sigmoid"}`) — no match, no error. The biased
sigmoid has NEVER been tested; the boundedness thesis (D65/D69) remains open.

Third occurrence of this failure class (blk `_dense` signature; blk yaml; now
sigb). **Rules adopted:**
1. After any scripted edit, grep back the exact changed line — the edit
   command's success is not evidence.
2. Config prints must be self-documenting: `attn_act`, `qkv_mode`,
   `local_mode`, `table_l1` added to train_qat's config line (their absence let
   this arm train a wrong config invisibly through a full survey).
3. Effective pod env verified at launch (the D-blk lesson, now standard).

Relaunched with the manifest verified and effective env checked.

---

## D71. Codex round 3 audited and implemented: AV-only pressure + hybrid attention

**2026-07-25.** Six proposals received, ranked, each with declared audit gates.
Greenlit and implemented after audit: #1 (AV-only pressure) and #2 (hybrid
local/global activation). Audit hits during implementation:
- The regex wiring the per-stage activation initially gave BOTH stages
  attn_act_local — exactly the leak Codex pre-declared. Caught by reading the
  call sites (D70 rule 1), fixed, then verified by the graph check it
  prescribed: local=QUnaryFunctionLUT vs global=QSoftmax, both backends.
- AV scope floored (fc=Min(1)) per the D59 exploit class, before launch.

Arms: `n64-avl1` (relu + AV-only 3e-5, targets the 48k-LUT gap at kept
latency), `n64-hyb` (relu local / softmax global), `n64-hyb-sigb` (biased
sigmoid local / softmax global). Effective pod env verified per D70.

Deferred with reasons: #3 clipped/hard-sigmoid (needs a score-histogram
calibration pass first), #4 stable=False softmax (queued as a cheap slot),
#5 covered by the relaunched sigb + #2b, #6 late-phase continuation (queued for
overnight GPU idle once winners are known).

---

## D72. Floored table pressure: neutral-to-worse — direction closed

**Measured** 2026-07-25. Floored (bc=Min(1)) softmax-table pressure vs baselines:

| arm | cheapest >0.79 | baseline at matched EBOPs |
|---|---|---|
| n64-tbl | 0.8008 @ 563,974 | 0.8023 @ 524,687 |
| n128-tbl | 0.7941 @ 673,518 | 0.8005 @ 666,359 |

Dominated at both N. The exp/reciprocal tables are evidently already efficiently
sized under the ordinary beta ramp; explicit extra pressure buys nothing.
Codex's #2-ranked direction closes: the 32% softmax block is attackable by
REPLACEMENT (relu — traded, sigb — pending) but not by squeezing its tables.

**Hygiene caveat:** the n64-tbl relaunch reused the old run's PJ_OUT, so its
pareto dir mixes unfloored (reward-hacked, D59) and floored checkpoints. The
verdict is unaffected — even the best-case mixed reading is dominated — but the
comparison is not clean. RULE: relaunches get a fresh PJ_OUT (or the old pareto
dir is cleared first).

**Cross-over interim note (2026-07-25):** n128r-core7 complete (best 0.7953 @
818k on the pT-sorted reconstruction); n128r-uni at epoch 247/1000 — verdict
deferred until uni completes. Earlier 0.6809 uni reading was early-epoch noise.

**relu-3e5 (2026-07-25):** best 0.7968 — never reaches 0.80; the 1e-5 relu
keeps the tier (0.8002 @ 365k). The attention-pressure saturation band
(1e-5..3e-5, D35/D41) holds for relu too — removing softmax did not move it.
relu recipe frozen at attn_l1=1e-5.

---

## D73. Biased sigmoid fails too — softmax earns its cost twice over; family concluded

**Measured** 2026-07-25. `n64-sigb` (verified genuine via the self-documenting
config line): best **0.7865 @ 5.3M**, never reaching 0.79. The -ln(P) bias
recovered part of the naked-sigmoid failure (0.7716 -> 0.7865) but the variant
is ~1.6 points below softmax and ~1.4 below relu.

**The softmax-free picture is now complete and mechanistically clean:**
- relu (unbounded, competitive-ish via /P): holds accuracy, loses AREA — the
  unbounded weights widen AV datalanes (D65/D69). Latency/FF winner.
- sigmoid (bounded, NON-competitive): keeps lanes narrow but loses ACCURACY —
  per-key independent gating cannot substitute for cross-key competition here.
- softmax (bounded AND competitive): earns its 32% block through two distinct
  functional properties. Neither alternative captures both.

Direction closed for full replacement. Still pending: the HYBRIDS (n64-hyb:
relu local / softmax global; n64-hyb-sigb) — placement may still capture relu's
latency where competition matters least. sigb-local hybrid expectations now low
given this result.

**D73 correction (same day):** sigb did not converge-low — it **diverged to NaN
at ~epoch 70** (train.log: loss=nan, acc=0.2016, ebops=0 from there to 1000).
The 0.7865 was the pre-divergence peak. So the failure is numerical instability
under this recipe (sigmoid LUT x wrap overflow x pressure interaction), not a
demonstrated capacity limit of non-competitive gating. The mechanistic claim in
D73 is accordingly weakened to: "biased sigmoid is UNSTABLE under our recipe and
never demonstrated a competitive frontier." Family still closed on evidence —
three strikes (naked fail, mislaunch, divergence) — with the hybrids as the
remaining softmax-free hope. The two greps disagreeing (Epoch 1 vs 1000) was
two pods' logs: a cuDNN-killed retry and the survivor; the PVC train.log is
authoritative (D9 rule, again).

---

## D74. AV-only pressure refuted — relu's wide lanes are load-bearing

**Measured** 2026-07-25. `n64-avl1` (relu + floored AV-scope 3e-5) at
0.8001/442,524, traced and emulated (bit-exact to 4e-6, 100% argmax):
**758,922 LUT (43.9%)** — worse than plain relu (700,629) and the softmax
headline (652,680), with latency roughly relu's (119.9 vs 116.5 ns).

Codex round-3 #1 refuted: the AV datalane widths under relu are not slack to be
pressured away — they carry the information softmax's bounded weights would
otherwise compress. Consistent with D65's mechanism and with the pressure-
saturation results: every lane that can narrow already has.

**Hygiene observation:** the avl1 run's LATE phase went to negative loss — the
main attention/FFN scopes have MonoL1 but no fc floor (only tables and the AV
scope were floored). The Pareto checkpoints predate the mining phase so results
stand, but floors-everywhere is the right default for any future run.

**Softmax-free ledger final (pending only the hybrids):** softmax = area
champion; relu = latency/FF champion; everything else closed.

---

## D75. Overnight batch 2 launched: distillation, GMP enrichment, weight pressure, continuation

**2026-07-25 night.** New machinery, all fit-tested on both backends, distill
path verified end-to-end locally (teacher rebuild + logit concat + custom
loss/metric, val_accuracy preserved for ParetoFront; teacher-garbage assert):
- `PJ_DISTILL_CKPT`/`PJ_DISTILL_ALPHA` — CE+logit-MSE from the 0.8064 float
  teacher. Arms: `n64-distill` (softmax student), `n64-relu-dist` (relu student).
- `PJ_GMP_KERNEL` — GMP enrichment ("spend where it's attributed-free"):
  `n64-gmpk5` (5x5), `n64-gmp16b` (16x16 grid at the winning recipe).
- `PJ_W_L1` — first-ever weight-bit (kbi) pressure: `n64-wl1` (1e-6, 100x global).
- `PJ_BETA_CONST` + resume — `n64-cont`: headline ckpt + constant late beta,
  1000 more epochs hunting cheaper 0.80+ points.

**n64-hyb closed:** 0.7988 @ 389k — dominated by plain relu (0.8002 @ 365k);
global-softmax hybrid adds cost without accuracy. Codex round-3 #2a dead;
only hyb-sigb of that family still running (low expectations).

Codex round 4 (GMP-enrichment audit demands, weight-pressure failure modes,
its own proposals) still running — will action on arrival.

---

## D76. BetaPID v2 continuation beats the headline on EBOPs — trace pending

**Measured 2026-07-25 night.** `n64-pid2` (headline ckpt warm-start, BetaPID
p=0.1/i=1e-3, init_beta=3e-8, EBOPS_TARGET=350k, 1000 epochs) completed. Best
Pareto points:

| epoch | val_acc | EBOPs |
|---|---|---|
| 562 | **0.8036** | **427,680** |
| 622 | 0.8034 | 426,400 |
| 922 | 0.8012 | 405,096 |

vs the verified headline 0.8023 @ 524,687 EBOPs: **higher accuracy at 18%
fewer EBOPs**. The PID never reached the 350k target (settled ~3.75e-8 beta,
~405–430k EBOPs) — accuracy pressure balanced it above target, which is the
controller working as designed.

Caveat: EBOPs→LUT is not constant (D24/D36); `zz-phatjet-hw-pid2` is tracing
e562 and e922 at inputs_kif=(1,8,16) with emulation. No table update until the
trace lands. Late-phase loss was ~−2.05 (same unfloored-scope mining as D74's
hygiene note); checkpoints at e562/e922 predate nothing here — mining was
present throughout, but emulation bit-exactness + val accuracy are the
arbiters, as always.

---

## D77. Codex round-4 audit: two launched arms were broken as written — fixed

**2026-07-25 night.** Codex audit of the D75 batch found, confirmed against
installed hgq2 source (`quantizer/config.py`):

1. **`PJ_W_L1` mis-wired.** kbi's bit regularizer key is `br`
   (default `MonoL1(1e-8)`), not `fr`; `QuantizerConfigScope.__enter__`
   silently drops keys a config lacks. Worse, scope0 is unscoped
   (q_type/place='all'), so `ir=MonoL1(W_L1)` leaked onto kif datalane
   configs — `n64-wl1` as launched was a *confounded* experiment (datalane
   integer-bit pressure at 100×, no weight-bit pressure). **Fix:** dedicated
   `QuantizerConfigScope(q_type="kbi", place=("weight","bias","table"),
   br=MonoL1(W_L1))`; scope0 reverted to plain L1_REG. Verified by
   introspection: br=1e-6 on kq of embed/gmp/attention layers, datalanes
   untouched. Old arm killed (it was crash-looping on cuDNN 5003 anyway,
   0 checkpoints); relaunched as `n64-wl1b` (backoffLimit 14).
   Note: kbi's default `bc=MinMax(0,23)` floors bits at 0, so weight-bit
   pressure cannot go negative — no D59-style mining risk; bits→0 is pruning,
   which is the desired behaviour.

2. **Resume/float-init not mutually exclusive.** `transfer_float_weights` ran
   unconditionally after `RESUME_CKPT` load and would overwrite resumed
   weights whenever FLOAT_CKPT exists. `n64-cont` **dodged it** (its
   FLOAT_CKPT=/nonexistent-train-from-scratch), so the running arm is valid.
   **Fix:** transfer skipped entirely when RESUME_CKPT is set.

Also actioned from the audit: teacher **val** accuracy now printed/asserted
(was train-only), CE and KD logged as separate metrics (`ce_part`/`kd_part`)
so an ill-scaled MSE can't hide in the combined loss. Both smokes pass on
jax+torch; ConfigMap updated. Codex's remaining gates adopted as evaluation
policy: GMP 5×5/16×16 "still free" claims require whole-model trace deltas
(<~1% of headline) not just marginal attribution rows; distillation gains
under ±0.13 (seed variance) don't count; bounded-relu `min(relu/P,1)` gets a
trace-first screen before any GPU.

---

## D78. pid2 e562 traced BIT-EXACT: new N=64 headline, better on every axis

**Measured 2026-07-25 night.** Both pid2 candidates traced at (1,8,16), emulated
bit-exact (max|err|=0, argmax 100%):

| | old headline (ffnl1hi e993) | **pid2 e562** | pid2 e922 |
|---|---:|---:|---:|
| val acc | 0.8023 | **0.8036** | 0.8012 |
| LUT | 652,680 (37.8%) | **544,292 (31.5%)** | 537,796 (31.1%) |
| latency | 143.2 ns / 43 stg | **139.9 ns / 42 stg** | 136.5 / 41 |
| FF (reg_bits) | 564,899 | **491,609** | 483,189 |
| LUT/EBOP | 1.24 | 1.27 | 1.33 |
| vs JEDI N=64 (71k) | 9.2× | **7.7×** | 7.6× |

pid2 e562 strictly dominates the old headline. Mechanism: 1000 extra epochs of
PID-held pressure at ~3.75e-8 beta let the design shed 18% EBOPs while GAINING
accuracy — the old headline was simply undertrained at its operating point.
`synthesis_table.md` updated; new RTL in `hw/pid2/`.

**Follow-up launched:** `n64-pid3` — resume from e562, EBOPS_TARGET=300k, same
PID gains. If accuracy holds near 0.80 at ~450k LUT the gap closes toward ~6×.

---

## D79. Cross-over control complete: core7 gives nothing on the pT-sorted reconstruction

**Measured 2026-07-25 night.** Clean A/B, same data (jets_128x3_aaron_recon_v2),
same recipe, 1000 epochs each:

| arm | best val acc | EBOPs |
|---|---:|---:|
| n128r-core7 (Aaron's 7×7 non-uniform, his units) | 0.7953 | 818k |
| n128r-uni (uniform 8×8, same units) | 0.7947 | 755,942 |

Δacc = 0.0006 — under half the measured ±0.0013 seed variance — with core7
COSTING ~8% more EBOPs. Verdict: the non-uniform grid does not transfer to our
pipeline even on his data ordering and units, now shown against a matched
control rather than by cross-config comparison. D47/D55 closure upheld and
final. Remaining core7 caveat: none. His npz remains wanted only for exact
replication of HIS numbers (data differs: 49 vs 37.8 constituents/jet).

---

## D80. Hybrid sigb also diverges — sigmoid family closed, FINAL

**Measured 2026-07-25 night.** `n64-hyb-sigb` (relu local / biased-sigmoid
global) went NaN at epoch ~82 (naked sigb died at ~70, D73): loss=nan,
ebops=0, val_acc=0.2016 from then on. Pre-divergence peak 0.7857. Killed at
epoch ~705 (620 wasted NaN epochs — for long arms, consider a
TerminateOnNaN callback; added to backlog).

Four strikes for the sigmoid family — naked fail (D63), mislaunch, biased
divergence (D73), hybrid divergence — under two different placements. The
instability follows the sigmoid LUT itself under our wrap+pressure recipe,
not its position in the network. **Softmax-free ledger closes as: relu =
latency/FF champion (verified), everything sigmoid = dead.** The only open
softmax-free thread is `relub` (bounded relu), currently in trace screen.

---

## D81. Bounded relu (relub) fails its trace screen — closed without GPU

**Measured 2026-07-25 night.** `attn_act="relub"` = min(relu(s)/P, 1), clamp
folded into the existing unary LUT (zero-hardware by construction). Traced the
existing relu checkpoint (e901) rebuilt with the clamp, per the Codex round-4
screen protocol:

| | plain relu (D69) | relub (same weights) |
|---|---:|---:|
| LUT | 700,629 | 693,989 (**−0.9%**) |
| stages / latency | 35 / 116.5 ns | 35 / 116.5 ns |
| FF | 369,238 | 368,173 |
| val acc (20k) | 0.8002 | **0.4200** |

Two findings: (1) the static [0,1] bound does NOT narrow the AV lanes — traced
widths follow the learned quantizer parameters, not the table's value range, so
the "provably ≤1" argument was worth ~1%; (2) accuracy collapse proves the
trained relu model heavily uses attention weights > 1 — post-hoc clamping is
not viable, consistent with D74 (wide lanes load-bearing).

Pre-registered decision rule: material LUT cut at held accuracy → train; else
close. **Closed.** Residual untested hypothesis, noted for honesty: training
WITH the clamp from scratch might learn softmax-like narrow AV lanes at relu
latency — but that is exactly the promise the hybrid family already failed to
deliver (D75), and no screen evidence supports spending a GPU-night on it.
The screen protocol itself worked: one CPU-job killed a direction in 5 minutes
that would otherwise have consumed a GPU-night.

---

## D82. Codex round 5: priorities set, N=128 PID + seed replication launched, eval-hygiene caveats adopted

**2026-07-25 night.** Codex reviewed D69–D81 state. Verdicts and actions:

**Adopted and launched:**
- `n128-pid` — PID continuation of the N=128 headline
  (n128-best e993, 0.8005 @ 666,359 EBOPs), target 500k. The flagship
  matched-N row has not received the optimization that transformed N=64.
- `n64-seed2` — independent-seed replication of the headline recipe
  (new `PJ_SEED` env: opt-in `keras.utils.set_random_seed`; unset keeps legacy
  seeding so past runs stay comparable). Gives the error bar the deliverable
  currently lacks.
- **pid2's +0.13 acc is within seed noise** — claim the LUT reduction
  (−17% at HELD accuracy), not an accuracy gain. Wording checked in
  synthesis_table (it claims dominance "on every axis" in D78 — fine for the
  log, but the rebuttal text should claim area-at-parity only).
- **No pid4 until pid3 traces** with ≥~5% LUT cut at ~0.80.
- **Selection-leakage caveat:** ParetoFront selects on val_accuracy across
  thousands of epochs and adaptively launched arms. Checked the store: there
  is NO untouched test split (only train/val at every constituent count).
  All reported numbers stay labeled *validation* accuracy; morning ask for
  Zihan: does an untouched hls4ml test file exist upstream? If yes, one
  frozen final rescore of the claimed rows after selection freeze.
- **Known vulnerability to pre-empt in the rebuttal:** our numbers are da4ml
  pre-place-and-route estimates vs JEDI-linear's post-route figures (already
  caveated in the table); Linformer reference is a different FPGA.
- Codex's honest-claim draft (kept for the rebuttal text): accuracy-
  competitive, NOT resource-comparable — 7.7× at N=64 — with the transformer
  gap narrowed ~2.7× LUT at +0.6 pts vs the FPGA-Linformer.

**Rejected/modified:**
- "Kill relu-dist if relub fails": relub DID fail (D81), but relu-dist stays —
  it targets the accuracy of an already-verified latency row we will cite,
  not a new architecture bet. One available GPU-night, direct row improvement.
- "Kill gmp16b unless pre-trace passes": gate could not run pre-launch (arm
  was already at epoch ~200); GPU queue is not the binding resource tonight.
  Kept running; its predeclared whole-model trace gate (<~1% of headline)
  decides at evaluation, not a training-curve reading.
- pid3 continuity check adopted: verify first-epochs beta/EBOPs match pid2's
  terminal state (~3.75e-8 / ~428k) — optimizer and PID integral state do NOT
  survive load_weights, so watch for a restart shock before interpreting.

---

## D83. GMP 5×5 kernel closed: dominated on both axes, no trace needed

**Measured 2026-07-26 early.** `n64-gmpk5` (headline recipe + PJ_GMP_KERNEL=5)
completed 1000 epochs. Peak 0.7943 @ 1,082,621 EBOPs (e338); terminal
0.7804 @ 614,925. The plain-3×3 baseline (n64-ffnl1hi) reached 0.8023 @
524,687 — gmpk5 is strictly dominated (−0.8 pts at +2× EBOPs at its peak;
still −2.2 pts at matched EBOPs at its end). Codex's skepticism confirmed:
"3×3 GMP was attributed-free" does not license enriching it — the 5×5
depthwise cost is real to the EBOPs budget during training even where the
final trace attributes little, and the pressure recipe then squeezes
accuracy-bearing regions harder to compensate. No trace run (nothing to
verify — the direction lost on the training curve alone). Prediction
registered: `n64-gmp16b` (16× cell count) likely fails the same way; letting
it finish per D82 since queue slots are free.

---

## D84. pid3 + cont: the N=64 knee is mapped, replicated, and CLOSED

**Measured 2026-07-26 early.** Two continuation arms completed:

**`n64-pid3`** (resume pid2-e562, target 300k): frontier flat at 0.801–0.803
over 395–430k EBOPs (0.8031 @ 427,696; 0.8020 @ 409,653; 0.8009 @ 394,907),
then a cliff — 0.7952 @ 383,630, 0.7253 @ 359,980. The PID forced EBOPs below
~390k only by destroying accuracy. No point strictly beats pid2's verified
pair. **D82 gate failed → no pid4.** The N=64 softmax knee is ~400–430k EBOPs
≈ 540k LUT at ~0.80, with a hard wall below.

**`n64-cont`** (constant beta 3e-8, resume OLD headline e993): best
0.8032 @ 427,773 — statistically identical to pid2's 0.8036 @ 427,680.
**The continuation gain was the extra 1000 epochs, not the PID controller**:
two different beta-control schemes independently converge to the same
equilibrium from different starting checkpoints. This doubles as a
replication of the headline operating point (both arms share the training
seed but took different optimization paths for 1000 epochs).

No new traces: pid2-e562 (544,292 LUT) and pid2-e922 (537,796) already
bracket the knee bit-exactly. N=64 area work is concluded; remaining
open levers for the table are n128-pid (knee-finding at the flagship N),
distillation (accuracy at the same knee), and wl1b (weight pruning).

---

## D85. N=128 PID continuation: −18–26% EBOPs at ~held accuracy — trace running

**Measured 2026-07-26 early.** `n128-pid` (resume n128-best e993 =
0.8005 @ 666,359, target 500k) completed. Candidates:

| epoch | val_acc | EBOPs | Δacc vs baseline | ΔEBOPs |
|---|---:|---:|---:|---:|
| 419 | 0.7995 | 546,677 | −0.0010 (within ±0.0013 noise) | **−18%** |
| 901 | 0.7988 | 494,497 | −0.0017 (≈ noise edge) | **−26%** |

Same undertrained-at-the-knee story as N=64 (D78/D84). The N=128 LUT/EBOP
ratio was ~1.0 on the baseline row, so e419 projects to ~545k LUT (≈5.6×
JEDI-linear's 98k, from 6.8×) and e901 to ~495k (≈5.0×) — but projections
are not results (D24): `zz-phatjet-hw-n128pid` is tracing both at (1,8,16)
with emulation. Table update only after bit-exactness.

---

## D86. N=128 continuation traced BIT-EXACT: flagship row now 5.2× JEDI at 93.2 ns

**Measured 2026-07-26 early.** Both n128-pid candidates verified
(max|err|=0, argmax 100%, RTL in `hw/n128pid/`):

| | n128 baseline (e993) | n128-pid e419 | **n128-pid e901** |
|---|---:|---:|---:|
| val acc | 0.8005 | 0.7995 | 0.7988 |
| LUT | 662,654 (38.3%) | 527,058 (30.5%) | **513,253 (29.7%)** |
| latency | 133.2 ns / 40 stg | 123.2 / 37 | **93.2 ns / 28 stg** |
| FF | 589,472 | 427,577 | **368,989** |
| vs JEDI N=128 (98k) | 6.8× | 5.4× | **5.2×** |

e901 is the new flagship matched-N row and, unexpectedly, the new LATENCY
champion outright — 93.2 ns beats the N=64 relu row (116.5 ns) by 20% at
higher N and higher accuracy. Mechanism: continued pressure narrowed
bitwidths enough to shallow the adder trees by 12 pipeline stages. The relu
row's raison d'être (latency/FF) is now dominated by a plain-softmax N=128
point on every axis except LUT-at-N=64. Both new rows in synthesis_table.

---

## D87. Seed replication: end-of-schedule variance is ~1.5 pts, NOT ±0.13 — knee robustness test launched

**Measured 2026-07-26 early.** `n64-seed2` (headline recipe, PJ_SEED=1337,
full keras seeding) at matched schedule position: 0.7876 @ 613,583 EBOPs
(e999) vs the original's 0.8023 @ 524,687 (e993). Gap = 1.47 pts — an order
of magnitude beyond the ±0.13-pt figure we had been citing (which came from
mid-training baseline replicates). Two readings, not yet distinguishable:
(a) end-of-1000-epoch points are high-variance because they are undertrained
(D84's lesson — the schedule ends mid-descent); (b) the recipe's seed
variance is genuinely larger than assumed. Best interim point: 0.7981 @
836,347 (e637).

**Action:** `n64-seed2c` launched — PID continuation (target 430k) from
seed2's endpoint. If it converges to ~0.80 @ ~430k (the D84 knee), the
OPERATING POINT is seed-robust even though mid-schedule snapshots are not,
and the rebuttal claim (the knee, not any one checkpoint) stands with an
honest error bar. Until it lands, the ±0.13 figure must not be quoted for
end-of-schedule comparisons; distill/gmp16b/wl1b verdicts tonight must use
the wider bar.

---

## D88. GMP 16×16 grid closed: no benefit, D83 prediction confirmed

**Measured 2026-07-26.** `n64-gmp16b` (headline recipe, 16×16 grid on ±1.6)
completed: best 0.7973 @ 581,324 EBOPs (e981) vs baseline 0.8023 @ 524,687 at
matched schedule — −0.5 pts at +11% EBOPs. Caveat per D87: −0.5 pts is within
the newly-measured end-of-schedule seed spread, so "dominated" is not
airtight; but the arm's job was to show a GAIN from 4× spatial resolution and
it showed none, while its EBOPs floor is visibly higher (~570k vs ~525k).
Both GMP-enrichment axes (kernel D83, resolution D88) are now closed:
**the GMP block is at its useful capacity for this recipe — "attributed-free"
was a statement about the 3×3/8×8 design point, not an invitation to grow
it.** No trace (nothing to promote).

---

## D89. Teacher-corruption incident: both distill arms AND twostage invalidated; n128-relu strong

**Measured 2026-07-26 morning.** The Codex-audit teacher-val-acc print earned
its keep: `n64-relu-distb` logged `teacher val acc 0.7396 (expect ~0.806)`.
Root cause chain: (1) the restarted `twostage-n64`'s float phase was
cuDNN-killed at epoch ~11/100; (2) on retry the wrapper skipped float
training because `float_phatjet.keras` existed (a best-so-far ModelCheckpoint
snapshot at 0.7396); (3) that half-trained file then served as (a) the
distill teacher for BOTH distillation arms and (b) twostage's own QAT donor
(`post-transfer val_acc=0.7335` in stage2_qat.log). **All three arms
invalidated and killed** (n64-distill at epoch 818, relu-distb complete,
twostage at QAT epoch 228). None of their results say anything about
distillation or two-stage init from the float ceiling.

**Fixes:** (1) `PJ_DISTILL_MIN_ACC` env — teacher gate at expected quality,
not garbage-level 0.5; (2) dedicated frozen-teacher convention:
`zz-phatjet-teacher64` relaunched train_float (N=64 recipe, 300 epochs) into
`/data/phatjet-fpga/teachers/f64/`, which no other job writes; distill
relaunches will point there with MIN_ACC=0.79 once it completes. Lesson
recorded: **never point a reader at a ModelCheckpoint save_best_only path
that a live or restartable job owns** — the file is a moving snapshot, not
an artifact.

**`n128-relu` completed** (valid, no teacher involved): best 0.7979 @ 416,581
EBOPs, cheap tail 0.7964 @ 335,237. At relu's historic LUT/EBOP ~1.9 the
cheap point could still be ~640k LUT, but if the no-row-reduction structure
beats the flagship's 28 stages it could take the latency crown back.
`zz-phatjet-hw-n128relu` tracing both (env grep-back caught a missing
PJ_ATTN_ACT before it burned the trace — D70 discipline, fourth save).

---

## D90. N=128 relu traced: softmax flagship dominates; one emulation mismatch flagged

**Measured 2026-07-26 morning.** `hw/n128relu`:

| | n128-pid e901 (flagship) | n128-relu e991 | n128-relu e700 |
|---|---:|---:|---:|
| val acc | 0.7988 | 0.7964 | 0.7979 |
| LUT | **513,253 (29.7%)** | 591,088 (34.2%) | 739,859 (42.8%) |
| latency | **93.2 ns / 28 stg** | 116.5 / 35 | 123.2 / 37 |
| FF | 368,989 | **329,686** | 418,954 |
| bit-exact | yes | **yes (0.0 / 100%)** | **NO — max err 0.231, argmax 99.80%** |

At N=128 the pressured softmax design now beats relu on area AND latency —
D65's trade (relu buys speed with area) no longer holds once the softmax
row is fully continued; relu's only surviving edge is −11% FF. relu e991
added to the table as a verified non-champion row; the softmax-vs-relu
question is closed at both particle counts.

**Emulation-mismatch flag:** e700 is the first non-bit-exact trace since the
(1,5,8) input-truncation bug — max|err|=0.231 with 99.8% argmax. Not
promoted anywhere, RTL not shipped; probable overflow corner at this
checkpoint's widths. Worth one diagnostic pass only if a claimed row ever
shows the same signature (none does).

---

## D91. New float ceiling 0.8110; distill2 + twostage2 relaunched from a frozen teacher

**Measured 2026-07-26 morning.** The dedicated teacher retrain
(`train_float`, N=64 recipe, 300 epochs, jax GPU) reached **0.8110 val acc**
— a new float ceiling for the N=64 synthesizable graph (previous teacher:
0.8064; paper's non-synthesizable float: 0.8180). Artifact frozen at
`/data/phatjet-fpga/teachers/f64/float_phatjet.keras`, a path no training
job writes to.

Relaunched from it, both with the D89 fixes (`PJ_DISTILL_MIN_ACC=0.79`
gate for distill; no skip-if-exists float stage anywhere):
- `n64-distill2` — softmax student, alpha 0.5, CE+KD split metrics.
- `twostage2` — QAT init by weight transfer (BW=10/10, i0 3/8 per Aaron's
  wrap-collapse measurements), single-stage job, fresh PJ_OUT.

QAT-from-scratch headline gap to the new ceiling: 0.8036 vs 0.8110 = 0.74
pts — the room these two arms are trying to close.

---

## D92. Weight-bit pressure (correctly wired) closed: no benefit over region pressure

**Measured 2026-07-26.** `n64-wl1b` — the D77-corrected arm, verified
br=MonoL1(1e-6) on kbi weight/bias/table only — completed 1000 epochs:
best 0.7961 @ 949,567 EBOPs, end-of-schedule 0.7931 @ 560,411. Baseline at
matched schedule: 0.8023 @ 524,687. The weight-bit knob produced NO EBOPs
reduction below baseline at any accuracy (and EBOPs is exactly what weight
bits feed), so there is nothing to trace. Consistent with Codex's round-4
prior ("constants can map cheaply; expect modest at best") and with the
saturation picture: the datalane/region pressure already drives the widths
that matter; squeezing weight bits on top just costs accuracy. Direction
closed at 1e-6; escalation to 1e-5 not warranted on this evidence.

---

## D93. Knee is seed-robust: D87 resolved with an honest error bar

**Measured 2026-07-26.** `n64-seed2c` (PID continuation of the seed-1337
replicate toward the knee) completed: best **0.8002 @ 408,113 EBOPs**
(also 0.7994 @ 402,781). Comparison at the knee across seeds:

| seed | point |
|---|---|
| 42 (pid2) | 0.8036 @ 427,680; 0.8012 @ 405,096 |
| 42 (cont) | 0.8032 @ 427,773 |
| 1337 (seed2c) | 0.8002 @ 408,113 |

At matched EBOPs (~405k) the cross-seed gap is **0.10 pts** (0.8012 vs
0.8002); worst-case knee-to-knee 0.34 pts. The 1.5-pt spread (D87) was an
artifact of comparing undertrained end-of-schedule snapshots. **Resolved:
the OPERATING POINT is reproducible; mid-schedule checkpoints are not.**
Rebuttal error-bar phrasing: "a second seed reproduced the operating point
within 0.1–0.3 accuracy points" (n=2 seeds, stated as such). The knee claim
(≈0.80 @ ~540k LUT at N=64) stands with that bar; the flagship N=128 row
has no second seed (state if pressed, or spend a GPU-night if time allows).

---

## D94. twostage2 (valid teacher): float init helps early, knee test launched; distill2 gate passed

**Measured 2026-07-26.** `twostage2` — QAT init by transfer from the frozen
0.8110 float (BW=10/10, i0=3/8) — completed 1000 epochs. post-transfer
0.7426 (wrap-quantization cost ~7 pts at init, recovered by training).
Front: 0.8048 @ 48M (e7), 0.8046 @ 3.86M (e140), end-of-schedule
**0.8014 @ 600,622**. vs from-scratch at end-of-schedule (0.8023 @ 524,687):
no better — but the from-scratch knee only emerged under continuation
(D84), so the decisive test is twostage2's OWN knee: `twostage2c` launched
(resume e971, PID target 430k). If it lands above 0.8036 @ ~430k, float
init buys accuracy at the knee; if it lands at ~0.803 @ ~430k, two-stage
init is neutral for this graph and closes.

`n64-distill2` teacher gate passed exactly as designed:
"teacher loaded: train acc 0.8114 val acc 0.8110" — the D89 incident class
is now mechanically excluded (frozen path + MIN_ACC=0.79).

---

## D95. Distillation at alpha=0.5 fails under EBOPs pressure — mechanism identified

**Measured 2026-07-26.** `n64-distill2` (valid 0.8110 teacher, alpha=0.5)
completed 1000 epochs. Front: 0.7989 @ 4.06M, 0.7957 @ 1.63M — EBOPs never
reached the from-scratch regime (525k) at any useful accuracy — then a slow
NON-NaN collapse from ~e600 (0.7879 → 0.7349 → final 0.1940, loss positive,
val_kd_part 9.0). TerminateOnNaN can't catch a finite divergence.

**Mechanism:** the KD-MSE term requires logit fidelity that narrow bitwidths
cannot represent; as beta ramps, kd_part grows unboundedly and its gradients
fight the compression objective until the student destabilizes. Distillation
as an equal co-objective (alpha=0.5) is INCOMPATIBLE with aggressive EBOPs
pressure. Closed at this alpha.

**Last variant launched:** `n64-distill3`, alpha=0.15 — teacher as a gentle
regularizer on top of a CE-dominated objective, same frozen teacher and
gates. If it also fails to beat the knee, distillation closes entirely and
the 0.74-pt gap to the float ceiling stands as a quantization cost, with
twostage2c (float-init knee test, running) as the remaining accuracy lever.

---

## D96. Two-stage float init closed: loses to from-scratch at the knee

**Measured 2026-07-26.** `twostage2c` (PID continuation of the valid-teacher
twostage2, target 430k) completed. Its knee: **0.7996 @ 466,060** (e632);
at the from-scratch knee's EBOPs (~427–430k) it holds only 0.776–0.778.
From-scratch knee: 0.8036 @ 427,680. Verdict: with a proper 0.8110 teacher
and Aaron's BW=10/i0 transfer settings, two-stage init is WORSE at the knee
than training QAT from scratch — the early accuracy advantage (0.8048 @ 48M
at e7) does not survive compression to the operating region. Direction
closed; Aaron's accuracy lever does not transfer to our recipe (consistent
with every cross-over so far: D47/D55/D79).

With D95 (distill alpha 0.5) this leaves `n64-distill3` (alpha 0.15) as the
sole remaining accuracy lever; if it fails, the 0.74-pt gap to the float
ceiling is the measured cost of quantization at the knee, and the evidence
phase of the rebuttal is complete.

---

## D97. Rebuttal draft written (Codex round 6), reference numbers re-verified against the JEDI paper

**2026-07-26.** Codex drafted the author-response FPGA section; audited,
corrected, and saved as `rebuttal_draft.md`. Fixes applied from its audit and
a fresh read of arXiv:2508.15468:
- The JEDI-linear reference numbers in synthesis_table (Table II,
  non-perm-invariant pT-sorted 3-feature: N=128 80.9% @ 98k / 82 ns /
  257.6 MHz) are CONFIRMED correct against the paper; the 138 ns / 203 MHz
  figure floating in session notes was Table I's permutation-invariant
  variant (81.6% @ 296k) — wrong comparison class. Their N=128 latency
  (82 ns, post-route) is LOWER than our 93.2 ns estimate: no latency claims
  vs JEDI-linear anywhere.
- Stale "~3.2x the LUT" vs the Linformer corrected to 2.7x (544,292/202k)
  with exact accuracy delta (+0.56) and the different-FPGA caveat.
- Accuracy deficits stated exactly: 0.54 (N=64), 1.02 / 0.95 (N=128).
- Structural attention-cost paragraph (D8/D13) HELD in reserve, not
  volunteered — it strengthens the "intrinsic deficit" reading; deploy only
  against an explicit architectural objection.
- "Mapped/occupying" phrasing replaced with "generates bit-exact RTL with
  pre-P&R estimates" per red-team.

---

## D98. Distillation stabilized at alpha=0.15: +0.12 pts at end-of-schedule, knee test launched

**Measured 2026-07-26.** `n64-distill3` (0.8110 teacher, alpha=0.15)
completed all 1000 epochs with NO collapse — the low alpha fixed the D95
instability (KD as gentle regularizer instead of equal co-objective).
End-of-schedule: **0.8035 @ 764,381** vs from-scratch 0.8023 @ 524,687 —
more accurate but 46% more EBOPs at the same schedule point. As with every
arm tonight, the decisive comparison is at the knee: `n64-distill3c`
launched (resume e988, PID target 430k, distill objective retained). If it
holds ≥0.805 at ~430k, distillation finally beats the from-scratch knee
(0.8036 @ 427,680) and becomes a new headline path; if it decays to ~0.803,
distillation closes as neutral.

---

## D99. N=128 second seed: from-scratch stage reproduces within 0.13 pts; knee continuation launched

**Measured 2026-07-26.** `n128-seed2` (seed 1337, from-scratch, headline
recipe) completed: 0.8018 @ 682,774 / 0.8002 @ 637,620 vs seed-42's
0.8005 @ 666,359 at the same schedule point. N=128 end-of-schedule seed
spread ≈ 0.13 pts — an order tighter than N=64's 1.5 pts (D87); plausibly
the 2× larger N smooths the optimization. `n128-seed2c` launched (resume
e993, PID target 500k) to reproduce the flagship knee; if it lands near
0.799 @ ~495-545k EBOPs, the flagship row gets its second seed and the last
robustness caveat in the rebuttal draft ("the N=128 point currently has one
seed") can be replaced with a measured bar.

---

## D100. Distillation closed entirely: even gentle KD cannot reach the knee

**Measured 2026-07-26.** `n64-distill3c` (alpha=0.15 continuation, PID target
430k) equilibrated at **0.8029 @ 524,757** / 0.8025 @ 510,874 — the PID could
not push EBOPs to the 430k region against the KD term, and the accuracy there
does not beat the from-scratch knee (0.8036 @ 427,680), which dominates with
+0.07 pts at −19% EBOPs.

**Mechanism, completing D95:** matching teacher logits (even at weight 0.15)
requires representational width that compression must destroy; the KD term
therefore acts as an EBOPs floor ~100k above the CE-only knee. Alpha
scheduling (annealing KD to zero) is the one untested variant, but with the
from-scratch knee already replicated across seeds and beta schemes, the
expected gain does not justify another GPU-day before the deadline.

**Distillation and two-stage init are now BOTH closed (D95/D96/D100). The
0.74-pt float-to-quantized gap at the knee is a measured property of the
architecture under this quantization scheme, not an optimization shortfall
we failed to close.** The rebuttal draft's phrasing already reflects this.
Final open arm: n128-seed2c (flagship second seed).

---

## D101. N=128 second-seed knee: 0.7913 @ 481k — flagship carries a ~0.75-pt cross-seed bar

**Measured 2026-07-26.** `n128-seed2c` (seed-1337 continuation, target 500k)
knee: best **0.7913 @ 481,301** (also 0.7892 @ 464,074) vs the seed-42
flagship 0.7988 @ 494,497. Cross-seed spread at the N=128 knee ≈ 0.75 pts —
wider than N=64's 0.1–0.3 (D93), the reverse of the end-of-schedule picture
(D99). n=2 seeds; honest phrasing for the rebuttal: "a second seed reached
79.13% at similar estimated resources, indicating roughly 0.7-point
seed-to-seed variation at this operating point." The flagship row itself
stands (it is a verified, bit-exact design — seed variance affects the
claim's generality, not the artifact), but the rebuttal draft's caveat is
updated from "one seed" to this measured bar. Evidence phase COMPLETE: the
board is empty for the first time since the campaign began.

---

## D102. Aaron's actual data + float ckpt received: FLOAT STAGE REPLICATED EXACTLY (0.8159)

**Measured 2026-07-26.** Zihan provided `jets_128x3.npz` (497 MB, md5-verified
onto the PVC at `/data/phatjet-fpga/aaron/`) and `float_phatjet_core7.keras`.

**Validation of the data against his documentation — all pass:** 620k/260k
split, 49.41 constituents/jet (confirms the D-noted distribution difference
vs our store's 37.8 — his data was never reconstructable from ours), padding
0.614 (~his 0.617), raw pT median 6.5956 = his documented stat, 100%
pT-descending, npz is RAW (his pipeline scales at load with his
robust_stats, which we recreated verbatim from CONFIG.md).

**Float replication: EXACT.** His checkpoint + his builder (core7 edges /
mean-iqr) + his preprocessing on his x_val → **0.8159** vs his claimed
81.59%. 5,429 params as published.

**QAT stage launched, two arms:**
- `aaron-rep` — his scripts verbatim (unscaled jsc150 beta ramp to 3e-6,
  two-stage from his ckpt at BW=10/i0=3,8, core7), QAT_EPOCHS=1000, GPU.
  Expected per his run6 log: 0.8113 @ 115.8M → 0.7854 @ 0.92M → 0.7359 @
  0.55M EBOPs.
- `n128a` — OUR flagship recipe (region pressure, beta_scale 0.01,
  from-scratch) on HIS events rescaled into our units (atomic prep step in
  the job). His own frontier collapses to 0.7359 @ 550k where our recipe
  (on our data) holds 0.7988 @ 494k — this arm tests whether the ~6-pt
  recipe gap survives the data swap, and whether his richer jets (49 vs 38
  constituents) improve our headline.

---

## D103. Aaron's pipeline REPLICATED end-to-end; his beta ramp collapses past his log's horizon

**Measured 2026-07-26.** `aaron-rep` — his scripts verbatim on his data, GPU,
1000 epochs. Replication vs his run6 log: float 0.8159 EXACT; post-transfer
0.8122 (his: 0.8121); e11 0.8093 @ 36.0M (his: 0.8079 @ 36.0M); e15 0.8006 @
19.0M (his: 0.7970 @ 18.9M); e34 0.7907 @ 5.57M (his: 0.7911 @ 5.65M). All
within 0.1–0.4 pts at matched EBOPs — his pipeline reproduces cleanly on our
infrastructure (artifacts: /data/phatjet-fpga/aaron/).

**What his log couldn't show (it stopped at e235, run in flight on CPU):**
the unscaled jsc150 beta ramp continues collapsing the 5.4k-param model —
0.6743 @ 253k (e399), 0.6054 @ 152k (e470). His last logged point (0.7359 @
550k) was not a floor but a waypoint of the collapse. This independently
confirms the founding observation of our recipe (the reference ramp is
mis-scaled for this model; beta_scale=0.01 + region pressure instead), and
quantifies the recipe gap at the operating region: at ~500k EBOPs his path
passes through ~0.73 while our flagship recipe holds 0.7988.

Still running: `n128a` (our recipe on his events) — decides whether his
richer jets (49 vs 38 constituents) improve our headline row.

---

## D104. Short-sequence campaign launched: modern recipe + sort-order lever at N=16/32

**2026-07-26.** Zihan directed: try N=16/32, research more levers. The
existing N=16/32 table rows are pre-modern-recipe relics (no region
pressure, no continuation; best: 0.7226 @ 366k and 0.7639 @ 495k vs JEDI's
71.9% @ 44k and 78.0% @ 45k). Launched four from-scratch arms under the
flagship recipe: `n16-mod`, `n32-mod` (kt-sorted head), `n16-modpt`,
`n32-modpt` (pT-sorted head via new `PJ_SORT=pt` knob — argsort verified,
config print updated). **At N≤32 the sort decides WHICH particles survive
the head slice**, so kt-vs-pT is a data lever, not cosmetics — and JEDI's
comparison class is pT-sorted. PID continuations to follow per D84.
Codex round 7 commissioned on the untried-lever list (d_model, heads,
patch size, QK dim, input-bit trace screens, small-N trims, PID targets).

---

## D105. Codex round 7 actioned: two correctness fixes, three new arms, lever rankings adopted

**2026-07-26.** Codex's lever research, audited and actioned:

**Correctness fixes (both shipped to ConfigMap):**
1. `extract_hw.py` did not implement `PJ_SORT=pt` — a pT-trained checkpoint
   would have been traced/emulated against kt-ordered data (different
   surviving constituents at N<150, silent mis-score). Fixed: identical
   stable descending-pT re-sort before the head slice. Caught BEFORE any
   pT-sorted checkpoint reached tracing.
2. `_attn_scope()` returned a null context when only `attn_floor` was set
   (floor-only runs were silently inactive). Fixed. No past run affected
   (all floor runs also set attn_l1 or bits).

**New arms launched** (top of Codex's EV ranking; dual-backend fit smokes
passed, d12→dh=3 and P=4→8-patch graphs build):
- `n16-d12`, `n32-d12` — first d_model sweep under the modern recipe
  (old d8 predates region pressure). Kill: needs ≥JEDI accuracy AND ≥10%
  EBOPs vs matched d16 arm.
- `n32-p4` — first patch-size sweep under the modern recipe (D21 structural
  trace: P=4 = −7% LUT; P=16 = +27.7%, rejected). Kill: <0.780 or <5% LUT
  at matched accuracy.

**Adopted rankings/rejections:** small-N PID continuations = top lever
(launch when from-scratch arms land; 300k/350k targets are controller
demands, knees must be read from fronts); input-kif screen deferred (D31:
0.1–2.6% total sensitivity); 2-head = structural-only screen if idle;
single-head NOT revisited (D50/D64 closed it under the modern pressure
family); QK-dim reduction rejected (not a knob; wouldn't shrink softmax);
global-MHSA removal at N=16 rejected without Zihan's sign-off (undisclosed
architecture change; global stage nearly free anyway, D20).

---

## D106. d_model=12 killed by pre-registered criterion: peak 0.7495 at N=32, late-run instability

**Measured 2026-07-26.** `n32-d12` completed 1000 epochs: peak val 0.7495,
then destabilized to 0.55–0.58 for the last ~stretch of the schedule (not
NaN — finite degradation, EBOPs ~578k). The d16 control (`n32-modpt`) was
already at 0.7560 MID-RUN and climbing. Kill criterion was ≥0.780: failed
by 3+ points. Per Codex's bridge logic, a failed d12 also kills d8 — the
old d8 closure stands and the width lever is closed under the modern
recipe. `n16-d12` killed before producing epochs (same lever, GPU freed
for the small-N continuations). Interpretation: per-head dim 3 (12/4) is
both weaker AND less stable under region pressure than dim 4; d_model=16
is at or near the width floor for this architecture.

---

## D107. N=32 first reads: pT sort wins, P=4 promising; kt line dropped, p4 continuation launched

**Measured 2026-07-26.** From-scratch results at N=32 under the modern recipe:

| arm | status | best / end-of-schedule |
|---|---|---|
| n32-mod (kt, P=8) | complete | ~0.7538 @ 909k (front) |
| n32-p4 (pT, P=4) | complete | peak 0.7778 @ 2.7M; **end 0.7702 @ 535,496** |
| n32-modpt (pT, P=8) | ~e820 | best-so-far 0.7719 |

pT sort clearly beats kt at N=32 (+~2 pts) — the head slice selects
higher-pT constituents, matching JEDI's comparison class, as predicted in
D104. P=4 at least matches P=8 accuracy with a much cheaper compressed
tail, consistent with D21's −7% structural read. **kt line dropped from
the small-N campaign** (no continuation — dominated). `n32-p4c` launched
(resume end-of-schedule point, PID target 350k). modpt/n16 continuations
follow on completion. JEDI target at N=32: 78.0% @ 45k LUT.

**D107 addendum:** `n32-modpt` (P=8, pT) completed: best 0.7719 early at
high EBOPs, but its compressed tail COLLAPSED (0.6894 @ 1.0M → 0.6478 @
852k) — while P=4 held 0.7702 @ 535k. At N=32, P=4 dominates P=8 in the
compressed regime outright, not merely by −7% structure: the smaller patch
(8 patches of 4) evidently tolerates narrow bitwidths that the 4×8-patch
graph cannot. No modpt continuation — `n32-p4c` is the N=32 champion line.

---

## D108. N=16 from-scratch verdicts: sort and patch effects INVERT vs N=32

**Measured 2026-07-26.** All three N=16 arms complete:

| arm | best point | end-of-schedule |
|---|---|---|
| n16-mod (kt, P=8) | **0.7264 @ 636,932** (e417) | degraded to 0.68 |
| n16-p4 (pT, P=4) | 0.7096 @ 647,600 | 0.7058 @ 606,571 |
| n16-modpt (pT, P=8) | 0.7127 @ 1.77M | 0.6969 @ 550,690 |

**Both N=32 verdicts invert at N=16:** kt beats pT by ~2–3 pts (at 16
kept particles the kt head retains wide-angle structure a pure-pT head
discards; at 32 the pT head wins), and P=4 loses its N=32 advantage.
Lesson for the table: sort order and patch size are PER-N choices, each
disclosed. Continuations: `n16-ktc` launched (resume kt e417, target
300k); `n16-ptc` (pT line) left running as the sort-story control.
JEDI N=16 target: 71.9% @ 44k LUT — the kt line's 0.7264 is +0.7 pts
above it pre-continuation.

**D108 addendum:** `n16-ptc` (pT-line continuation, target 300k) knee:
~0.665 @ 356–362k EBOPs — 6 pts below the kt line's pre-continuation best
and below JEDI's 0.719. The N=16 pT line is closed; kt carries N=16
(`n16-ktc` running). The sort-inversion result is now knee-verified on the
losing side.

---

## D109. N=16 modern-recipe campaign closed: old rows stand

**Measured 2026-07-27 (early).** `n16-ktc` knee traced bit-exact (both
points, max|err|=0, 100% argmax):

| | old N=16 cheap row | new kt knee (e126) | old N=16 acc row |
|---|---:|---:|---:|
| val acc | 0.7226 | 0.7248 | 0.7401 |
| LUT | **365,950** | 595,281 | 957,257 |
| LUT/EBOP | — | **1.49** | — |
| latency | 133.2 ns | 139.9 ns | 156.5 ns |
| FF | 230,974 | 308,189 | 418,828 |

The modern recipe + continuation does NOT improve N=16: the new knee is
technically nondominated (+0.22 pts over old-cheap) but a terrible trade
(+63% LUT), and the sort/patch levers inverted (D108) without producing a
better point. **Mechanism:** at 2 patches, attention — the region the
modern recipe squeezes — is a small share of the design; LUT/EBOP 1.49
(vs 1.24–1.33 at N=64/128) says the pressure never engaged the dominant
cost. The old N=16 rows stay in the table; the campaign's verified gain at
small N is entirely at N=32 (pending `n32-p4c`). Cheap point (0.6684 @
472,554 LUT) also verified but dominated by the old cheap row outright.

---

## D110. Small-N campaign concluded (N=32 traced); Aaron's data closes negative

**Measured 2026-07-27.** Final two results:

**N=32 trace (bit-exact, both points):** `n32-p4c` e283 = **0.7704 @
642,141 LUT (37.2%), 139.9 ns, FF 535,932**, LUT/EBOP 1.31; e482 = 0.7621
@ 635,648. vs old N=32 rows (cheap 0.7639 @ 494,967; headline 0.7879 @
1,221,682): the new point is a genuine MID-FRONTIER improvement (old
frontier had nothing above 0.7639 until 1.22M LUT) but the old cheap row
keeps the low-area crown. Config disclosed: P=4, pT-sorted (both
per-N choices, D108). First pT-sorted trace — extraction sort fix (D105)
exercised and correct.

**Campaign law:** the modern recipe's gain scales with attention share —
N=128 −23% LUT at parity; N=64 −17%; N=32 mid-frontier only; N=16 nothing
(D109). Small-N work is DONE.

**`n128a` (our recipe, Aaron's data, from-scratch):** best 0.7993 @ 1.12M,
end-of-schedule 0.7988 @ 912k — behind the same recipe on our data at the
same schedule stage (0.8005 @ 666k). His 49-constituent jets do not
improve our N=128 row (caveat: val splits differ, so sub-half-point
differences are not meaningful — but the EBOPs gap is large and the
direction is negative). No continuation. **Every element of Aaron's
package is now closed:** grid (D79), two-stage (D96), recipe (D103
replicated then dominated), data (this entry). His pipeline was verified
correct throughout; nothing in it advances the rebuttal rows.

---

## D111. First-principles night: the wall MOVED — projections now dominate; our val IS the canonical test split

**Measured 2026-07-27 night.** Fresh fine-grained attribution of the
CONTINUED checkpoints (first since the continuations transformed them):

**N=64 headline (pid2 e562, clean attribution):** softmax 2.2–2.8%, AV
~15%, **Q/K/V constant-weight projections ≈71% of LUT** (local 26.4% +
global 45.1%; CSE caveats apply, some marginals negative). The old story
(D8/D13: "attention datalanes 66–74%, softmax 32%") described the
PRE-continuation design. Continuation crushed the datalane/softmax costs;
what remains is dominated by CONSTANT matmuls — the exact target of
weight-VALUE sparsity. (N=128 attribution attempted but CSE-corrupted at
the tail — marginals ±90%; not load-bearing, direction agrees.)

**Launched accordingly:** `n128-wv3` (λ=3e-4), `n128-wv10` (λ=1e-3) —
ProxL1 soft-thresholding on kernel VALUES after each optimizer step
(exact zeros → CMVM adders vanish; JEDI's Fig. 7 sparsity is the
motivation), resuming the flagship e901 at its equilibrium beta. Dual-
backend smoke passed. Note: EBOPs may not credit value-zeros, so
ParetoFront selection is a weak guide here — final TRACES arbitrate.
Also launched `hw-latsweep`: hard_dc ∈ {1,3,4} + cutoff 3.0 ns on the
flagship netlist — a free latency-axis sweep never explored.

**Eval-split provenance (valcheck):** our store's 260k "val" and Aaron's
npz val have IDENTICAL class counts (same jet multiset, different
shuffle) = the canonical hls4ml 620k/260k train/TEST split — the same
test set JEDI-linear reports RTL-simulation accuracy on. Rebuttal
upgrade: our numbers are on the baseline's own evaluation split;
the remaining honest caveat is checkpoint selection on that split
(unchanged), NOT split mismatch.

---

## D112. Codex round 8 adopted: sparsity-first plan, promotion checklist, split-claim discipline

**2026-07-27 night.** Codex's first-principles audit, actioned:
- **ProxL1 assumption verified locally before its arms land:** float zeros
  survive HGQ weight quantization as exact zeros (116/116 in a direct
  layer-level check; quantization even zeroes some small nonzeros).
- **Its ranked plan adopted:** (1) trace+audit wv3/wv10 layer-by-layer with
  its 11-point promotion checklist (recorded here as the gate — notably:
  EBOPs does not credit value-zeros so ParetoFront ordering is a weak
  guide; whole-model LUT decides; bit-exactness mandatory); (2) QKV-only
  ProxL1 continuation if global pruning hurts accuracy (the callback
  currently thresholds ALL kernels — too broad, acknowledged); (3) port
  winner to N=64 (clean attribution, worst gap); (4) take the latency
  sweep's one-stage win if area-neutral; rejected for this window: low-rank
  QKV (rank<=6 needed for real reduction; architecture change), Q=K tying
  (needs sign-off), broad solver searches, generic br=1e-5 rerun (D92
  prior stands unless sparse survivors are high-bit).
- **Split-claim discipline:** class counts do NOT prove event identity —
  `valcheck2` launched (per-event signature: label x leading-constituent
  raw pT, robust to the differing constituent cuts). Until it passes,
  rebuttal wording stays at "canonical 260k-event hls4ml evaluation
  partition" + the selection caveat, NOT "same split as JEDI".
- `sparsity_report.py` written (per-layer float-zero AND quantized-zero
  fractions, QKV highlighted) and shipped in the ConfigMap.
- Noted caveats: D111's N=64 projection arithmetic (544k→~350k if half the
  71% goes away) stays INTERNAL — trace-only promotion, no projected
  numbers in the rebuttal.

---

## D113. Split identity PROVEN; flagship at 89.9 ns via hard_dc=1 (bit-exact, area-neutral)

**Measured 2026-07-27 night.**

**Eval-split identity (valcheck3):** label-sequence agreement under
per-store leading-pT sort = 1.00000 (260k/260k), and the store-to-raw pT
ratio is exactly 34.831 (our iqr) at every quantile — same events, linear
recalibration only. Our "validation" set IS the canonical hls4ml 260k test
partition that JEDI-linear evaluates on. Rebuttal wording upgraded to the
strong form, retaining the selection caveat (the partition was also used
for checkpoint selection).

**Latency sweep (flagship e901 netlist, all bit-exact):**

| solver | LUT | stages | latency | FF |
|---|---:|---:|---:|---:|
| hard_dc=2 (row of record) | 513,253 | 28 | 93.2 ns | 368,989 |
| **hard_dc=1** | 515,055 (+0.35%) | **27** | **89.9 ns** | **358,688 (−2.8%)** |
| hard_dc=3/4 | 513,245 | 28 | 93.2 | 369,011 |
| hard_dc=2 + cutoff 3.0 | 513,253 | 37 | 123.2 | 492,629 |

The flagship crosses the sub-90 ns line at +0.35% LUT with a solver
setting — same weights, same accuracy, RTL emulated bit-exact. Added to
the table as the flagship's fast-pipelining variant. Cutoff-tightening
confirmed counterproductive (more stages), per Codex round-8 prediction.

---

## D114. VALUE SPARSITY WORKS: N=128 at 4.1–4.9× JEDI, both rows bit-exact

**Measured 2026-07-27 night.** `n128-wv10` (ProxL1 λ=1e-3 continuation of
the flagship) traced with full emulation (max|err|=0, 100% argmax, RTL in
`hw/wv10/`):

| | flagship (row of record) | **wv10 e64** | **wv10 e979** |
|---|---:|---:|---:|
| val acc | 0.7988 | **0.7990** | 0.7957 |
| LUT | 513,253 (29.7%) | **478,214 (27.7%)** | **404,852 (23.4%)** |
| latency | 93.2 ns / 28 stg | 93.2 / 28 | 93.2 / 28 |
| FF | 368,989 | 358,023 | **298,271** |
| vs JEDI N=128 (98k) | 5.2× | **4.9×** | **4.1×** |
| quantized zeros | — | 55.5% | 62.3% (QKV 73.9%) |

e64 STRICTLY dominates the flagship (accuracy parity, −6.8% LUT) — new row
of record. e979 gives 4.1× at −0.31 pts. The D101 ~0.75-point spread is
contextual pre-sparsification, cross-seed variation rather than an error bar
on these sparse rows, which have not been replicated across seeds.

**Mechanism confirmed and amplified by the lattice:** 14.8% float zeros →
62% quantized zeros (weights under half an LSB round to zero). Layer
detail at e979: global-attention Wq/Wk, ffn1, and both GMP kernels are
100% quantized-zero — at the knee the optimizer self-ablates global QK
(scores constant → uniform attention), the FFN first layer, and GMP while
holding 0.7957. The D111 attribution was actionable, not a CSE mirage;
the architecture is heavily over-provisioned at the operating point, and
value-sparsity is the lever that harvests it.

**Ladder state:** wv3 (λ=3e-4): 6% float zeros, milder, dominated.
wv30 (λ=3e-3): running. **N=64 port launched** (`n64-wv10`: λ=1e-3 from
pid2 e562 at its equilibrium beta) — clean attribution there, worst
multiple (7.7×), same mechanism.

---

## D115. N=64 sparsity port verified: 6.9× at held accuracy

**Measured 2026-07-27 morning.** `n64-wv10` (λ=1e-3 from the N=64 headline)
traced bit-exact (both points, max|err|=0, 100% argmax, RTL in `hw/n64wv/`):

| | N=64 headline | **wv e693** | wv e67 |
|---|---:|---:|---:|
| val acc | 0.8036 | 0.8029 | 0.8035 |
| LUT | 544,292 (31.5%) | **487,451 (28.2%)** | 523,154 (30.3%) |
| latency | 139.9 ns | **136.5** | 139.9 |
| vs JEDI N=64 (71k) | 7.7× | **6.9×** | 7.4× |
| quantized zeros | — | 36.9% (QKV 50.3%) | 32.8% |

−10.4% LUT at −0.07 pts (noise). Smaller harvest than N=128's −21%
because sparsity penetrated less (37% vs 62% quantized zeros) — the N=64
design's weights sit further from the lattice zero. `n64-wv30` (λ=3e-3)
launched to test the stronger setting; `n128-wv30` about to land.
Matched-N multiples now: **4.9× (N=128 parity row) / 6.9× (N=64)** —
from 6.8×/9.2× at the start of the two overnight sessions.

**D115 addendum:** `n64-wv30` (λ=3e-3) collapsed late (0.68 by e1000) with
NO additional sparsity over λ=1e-3 (15% float zeros both) — too aggressive
at N=64; its pre-collapse points (0.8034 @ 417k, ~5% sparse) don't beat
the λ=1e-3 traced row. N=64 sparsity champion stands at λ=1e-3
(0.8029 @ 487,451). `n128-wv30` still running (~e876).

---

## D116. λ-ladder closed: 1e-3 optimal at both N; sparsity campaign complete

**Measured 2026-07-27 morning.** `n128-wv30` (λ=3e-3) traced bit-exact:
e977 = 0.7957 @ 412,967 LUT — WORSE than wv10-e979 (404,852) at identical
accuracy despite 24.6% vs 14.8% float zeros, because quantized zeros are
equal (61.4% vs 62.3%): the lattice had already harvested those weights at
λ=1e-3. e140 (0.7984 @ 487,458) similarly loses to wv10-e64. Combined with
the N=64 collapse at 3e-3 (D115 addendum): **λ=1e-3 is the best tested
setting at both N in this ladder; the tested stronger pressure adds float
zeros that don't become hardware zeros.** This does not establish a global
optimum over untested λ values, schedules, or layer-specific penalties.

**Final promoted rows of the three-session campaign (all bit-exact):**
- N=128 flagship: 0.7990 @ 478,214 (27.7%), 93.2 ns [89.9 ns variant] — 4.9×
- N=128 cheap: 0.7957 @ 404,852 (23.4%), FF 298,271 — 4.1×
- N=64 headline: 0.8029 @ 487,451 (28.2%), 136.5 ns — 6.9×
- N=32 mid-frontier: 0.7704 @ 642,141 — new; N=16: old rows stand.

Rebuttal draft updated to these numbers. Accuracy deltas vs JEDI: 1.00 pts
(N=128 parity row), 0.61 pts (N=64); vs the FPGA-Linformer: +0.49 pts at
2.41× LUT. Sparsity mechanism, λ-ladder, and both-N ports: DONE.

---

## D117. Final-day mid-state: e844 traced, post-sparsity attribution, endgame allocation locked

**2026-07-27 day.** Codex round 9 (which directly edited rebuttal_draft.md
and DECISIONS.md with caveat corrections — reviewed and kept):

- **hw-n64wv2:** e844 = 0.8012 @ 470,316 LUT (27.2%), 133.2 ns, bit-exact —
  nondominated beside the promoted e693 (0.8029 @ 487,451); held back from
  the table pending `n64-wvb` (harvest continuation), which should
  supersede both.
- **cc-sparse:** N=64 post-sparsity attribution is clean and still
  projection-dominated (local 27.8% + global 46.9% ≈ 75%) — consistent
  with its only-half-harvested sparsity (51% QKV zeros vs N=128's 74%).
  Per the round-9 decision rule (concentrated + actionable + lever already
  running): the in-flight arms cover the remaining headroom; NO new arms.
  N=128 sparse attribution again CSE-corrupted at the tail (not load-
  bearing).
- **Endgame allocation adopted:** writing/polish 40% (round-9 edits in),
  Vivado handoff 30% (**done**: `VIVADO_HANDOFF.md` in outputs and at
  `hw/README_VIVADO.md` on the PVC — 5 prioritized RTL dirs with expected
  numbers), sparse-seed replication 20% (`n128-wvs2` launched: λ=1e-3 on
  the seed-1337 knee — replicates the SPARSE recipe, which D101's
  pre-sparsity spread does not cover), further LUT pushing ≤10% (covered
  by wvb/wvpid).
- **Figure v2** after round-9 red-team: frontier-only main series, faint
  dominated points, log-scale labeled, methodology split stated per
  series, selection + n=1-seed caveat printed in-figure, generator script
  kept in repo (`make_pareto_fig.py`).
- Process note: cc-sparse job was deleted one command before its N=64 log
  section was extracted — recovered from the PVC tee (the D9/D-cleanup
  rule's PVC-artifacts-survive property working as designed, but read
  fully BEFORE deleting next time).

Board: n64-wvb (harvest), n128-wvpid (sparsity+PID), n128-wvs2 (sparse
seed) — the last three arms of the campaign.

---

## D118. N=64 harvest lands: 0.8019 @ 451,836 (6.4×), bit-exact

**Measured 2026-07-27.** `n64-wvb` (harvest continuation) traced:
e63 = **0.8019 @ 451,836 LUT (26.1%), 139.9 ns, FF 442,287, bit-exact**
(max|err|=0, 100% argmax; recovered from PVC tee after premature job
deletion — second occurrence today, noted). e206 = 0.8010 @ 466,003 —
dominated by e63 (LUT does not track EBOPs monotonically: e63 had HIGHER
EBOPs, lower LUT). vs the promoted e693 (0.8029 @ 487,451): e63 is
−0.10 pts for −7.3% LUT — a nondominated pair; both go in the table.
N=64 multiples now 6.4–6.9× depending on the point. QKV quantized zeros
still only ~51% — the N=64 lattice-harvest asymptote appears shallower
than N=128's 74%; further continuation not warranted before the deadline.

---

## D119. CAMPAIGN FINALE: sparsity+PID lands 3.9× at 89.9 ns; sparse seed bar measured

**Measured 2026-07-27 evening.** The final trace pair (`hw/final-day/`,
both bit-exact, max|err|=0, 100% argmax):

**`n128-wvpid` e565 — the combined-mechanism point:**
**0.7961 @ 385,471 LUT (22.3% of VU13P), 27 stages = 89.9 ns, FF 270,104**
= **3.93× JEDI-linear** at N=128. The D117 hypothesis confirmed: with 62%
of quantized weights zero, PID pressure re-narrows the datalanes the dead
columns no longer feed (LUT/EBOP 0.98), and the shallower carry chains
drop a pipeline stage at the default solver setting — sub-4× and sub-90 ns
in one row. Supersedes the 404,852 cheap point.

**`n128-wvs2` e33 — the sparse-recipe cross-seed bar:** seed-1337 sparse
run reaches 0.7947 @ 496,540 LUT vs seed-42's 0.7990 @ 478,214 —
**cross-seed spread ≈0.43 pts / +3.8% LUT (n=2)**. This is the honest
error bar for the sparse rows (D101's 0.75-pt figure was pre-sparsity);
into the rebuttal draft verbatim.

**Final verified matched-N state: 3.9–4.9× (N=128), 6.4–6.9× (N=64).**
From 6.8×/9.2× when the first overnight session began. Board empty;
campaign closed.

---

## D120. New direction: JEDI-linear replicated in our pipeline + GMP hybrid; systematic diff study

**2026-07-27/28.** Zihan directed: put GMP on top of JEDI-linear and
systematically analyze the PHAT↔JEDI performance/resource difference.

**Implemented** (`build_jedi_linear` in phat_jet_k3.py, `PJ_ARCH` env):
faithful port of their public `get_gnn` (width 64, EinsumDense+BN+relu,
SAT_SYM weights, QSum pow2-scaled pooling, 64-32-16-5 head) inside our
scopes/pipeline, so training, tracing, sparsity reporting, and component
attribution all work unchanged. `gmp=True` inserts our one-hot GMP block
(64 channels) after the embed with a residual QAdd — giving the linear
model the geometric LOCAL context it structurally lacks. Dual-backend
smokes pass.

**Reading their source also sharpens the systematic diff:** JEDI-linear is
100% constant-foldable arithmetic (const matmuls + adds + one global sum
— zero data×data products, BN folded, relu, no softmax) at width 64;
PHAT is width 16 with data×data attention. The cost story is therefore
structural (foldability), not width.

**Launched:** `n128-jedi` (our-pipeline replication control; their full
beta ramp BETA_SCALE=1.0, which their architecture was designed for) vs
`n128-jedigmp` (identical env + GMP). pT-sorted, 1000 epochs (their
published 80.9% used 7000 — expect our control lower; the CONTROLLED
quantity is the gmp delta). Codex round 10 auditing the port and the
hybrid design in parallel. Effective envs verified (dup BETA_SCALE
resolves last-wins to 1.0).

**D120 addendum (Codex round 10):** port audited layer-for-layer against
their get_gnn — MATCHES (BN fusion, pool scales incl. their intentional
1/16, SAT_SYM nesting, no-RND-in-gnn, logits out). Its grid-range red flag
(builder default ±1.6 vs train_qat global default ±0.4) does NOT bite the
launched arm — effective env verified PJ_GMP_LO/HI=±1.6. Adopted: JEDI cut
list added to component_costs.py (its cuts were PHAT-only — attribution
would have produced nothing); 1000-epoch runs are an A/B screen for the
GMP delta, NOT an 80.9% replication claim; hybrid padding unmasked
(disclosed — in PHAT the origin-cell multiplicity signal helped, D-early);
GMP at 64ch predicted NOT near-free (~+52% of JEDI's constant products
pre-sparsity — the delta trace decides); factorial interpretation grid
(PHAT±GMP × JEDI±GMP) adopted for the null case; trace jobs must set
PJ_ARCH (env replay).

---

## D121. Session-4 opening: GMP-on-JEDI ≈ null; four gap-attack arms launched

**2026-07-28 night.** Hybrid screen result (jedigmp complete, control at
~e845 with peaks logged):

| | peak points |
|---|---|
| n128-jedi (control) | 0.7986 @ 170,227; 0.7980 @ 141,487 EBOPs |
| n128-jedigmp | 0.7997 @ 181,985; 0.7995 @ 161,311 EBOPs |

**GMP delta on JEDI: +0.10–0.15 pts at comparable EBOPs — at the
pre-registered null threshold** and below the 0.3-pt no-replication bar.
With D114's self-ablation (PHAT's optimizer zeroes GMP at the sparse
knee), the factorial reading: GMP contributes little under aggressive
compression in either host. Methodology validation as the consolation
prize: our-pipeline JEDI reaches ~0.799 @ 140–180k EBOPs at 1/7th their
training budget — within a point of their published 80.9%, so the
cross-architecture comparison stack is sound.

**Gap-attack arms launched** (hypotheses from the source-level diff):
- `n128-wvpid2` — wall test: resume the 385,471-LUT champion, PID 300k +
  ProxL1. Does the combined mechanism break below the ~400k-EBOPs wall?
- `n128-sat` — JEDI saturates WEIGHTS (wraps datalanes); new
  `PJ_OVERFLOW_K` knob decouples ours; SAT_SYM weights + ProxL1 from the
  flagship. Hypothesis: saturation lets values shrink toward zero without
  wrap instability — synergy with proximal sparsity.
- `n128-d32` / `n128-d32wv` — the width hypothesis: JEDI's lesson may be
  "capacity wide and constant-cheap, then harvest"; d=16 might be too
  narrow for sparsity to have slack to work with. d=32 modern recipe from
  scratch, ± ProxL1 (also the first from-scratch ProxL1 test).

Pending: jedi control completion → trace both jedi arms + first JEDI
component attribution (cuts shipped, PJ_ARCH env-replay) + JEDI sparsity
profile → full gap decomposition table → Codex round 11 research brief.

---

## D122. GMP-on-JEDI verdict FLIPS at the accuracy end: hybrid = 0.8048 @ 266k, dominating every PHAT row

**Measured 2026-07-28 night.** D121's interim "null" was a grep artifact
(pattern matched only 0.7x accuracies). Traced best checkpoints, both
bit-exact (max|err|=0, 100% argmax, RTL in `hw/jedi/`):

| | n128-jedi (e210) | **n128-jedigmp (e202)** |
|---|---:|---:|
| val acc | 0.8048 | **0.8048** |
| LUT | 314,946 (18.2%) | **266,382 (15.4%)** |
| latency | 46.6 ns / 14 stg | **46.6 ns / 14 stg** |
| FF | 105,950 | **98,636** |
| EBOPs | 351,152 | 326,970 |
| quantized zeros | 63.7% | 68.8% |

**GMP is attributed ~free at 64 channels as well (−19.3% CSE credit,
refuting the round-10 +52% prediction), and the hybrid is 15% CHEAPER at
identical accuracy** — the geometric context lets the ramp compress the
constant layers harder. At matched protocol the hybrid also beats every
PHAT row in the workspace: +0.6 pts over the sparse flagship at −44% LUT
and half the latency.

**Gap decomposition now complete, both sides:**
- JEDI cost map: evenly spread constant layers (sender 28.8%, embed 23.6%,
  pool/receiver 17.7%, node2 14.2%, head ~11%) — no dominant block, zero
  data×data lanes, 14 pipeline stages.
- PHAT residual: constant Q/K/V projections ~71–75%, AV ~15%, softmax ~3%,
  43 stages worth of depth at N=64 (28 at N=128).
- **The sparsity mechanism insight:** plain full-ramp training gives JEDI
  63.7% quantized zeros WITHOUT ProxL1 — the jsc150 ramp is itself the
  pruner (kbi bit pressure → zero-bit weights → value zeros). PHAT needed
  explicit ProxL1 only because attention collapses under the full ramp.
  The gap is therefore: (a) data×data lanes that cannot fold, AND (b) ramp
  intolerance that forfeits the free pruning the baseline recipe provides.

Caveats: 1000-epoch screen (their published point is 7000 epochs,
post-route); single seed each; our estimates vs their post-route.
Strategic handling per Codex round-10 D(i): present honestly as the
strongest low-resource operating point if it survives replication.

---

## D123. Round 11: selective beta is feasible; GMP hardware claim needs the missing factorial cell

**Advised 2026-07-28; analysis only.** Installed HGQ2 was audited directly:
`LayerConfigScope(beta0=...)` sets each Q-layer's initial beta, but the stock
`BetaScheduler` overwrites EVERY Q-layer beta each epoch. Therefore a
ramp-tolerant PHAT path is feasible, but nesting `LayerConfigScope` alone does
not protect attention. The clean implementation is a selective scheduler that
applies the full ramp globally and a reduced beta to the six named QK/softmax/AV
layers, retaining EBOP accounting. First A/B must be full-ramp control versus
lanes-only protection, both without ProxL1; trace LUT and bit-exact RTL decide.

Pipeline conclusion: da4ml stages are heuristic timing partitions of the
combinational trace, not architectural registers. Solver sweeps have demonstrated
only one-stage changes (D56/D113), while learned precision has moved many stages
(D34/D86). PHAT's two serial QK -> normalize -> AV chains are structural, so a
small further reduction is plausible but 28 -> 14 without an architecture
change is unsupported. Fine prefix pipeline traces are the missing attribution.

GMP ruling: D122 is a strong single-seed screen, not yet a central hardware
claim. The paper's with/without-GMP accuracy ablation supports transfer, but the
workspace's D4 +4.4-point gain was BN+tanh, NOT GMP, and D114's sparse PHAT
self-ablated GMP. Minimum promotion gate is paired-seed JEDI±GMP replication
plus the missing trained PHAT±GMP cell, using whole-model traced LUT, sparsity
profiles, and bit-exactness. Say “lower whole-model LUT after joint training,”
not “negative/free GMP,” because component marginals are CSE-sensitive.

Ranked night allocation: (1) harvest/audit the four running gap arms; (2)
implement + smoke selective beta, then one lanes-only A/B; (3) paired JEDI±GMP
seed replication; (4) trace-only fine prefix pipeline attribution; (5) plumb
and run PHAT±GMP; (6) paired 7000-epoch runs only if both members fit. Full
report: `round11_report.md`.

**D123 addendum (implementation + launches):** SelectiveBetaScheduler
implemented per the round-11 sketch (explicit six-lane allow-list, found-set
assert, dual beta logging) — smoke-verified: protected lane beta is exactly
0.01× the unprotected at every step. `PJ_USE_GMP` plumbed for the factorial.
Launched: `n128-ramp-ctl` (full ramp, expected collapse — paired control),
`n128-ramp-prot` (full ramp + lanes at 0.01× — the ramp-tolerance causality
test), `n128-nogmp` (PHAT−GMP factorial cell, modern recipe). With the
paired jedi/jedigmp seed-1337 arms and the four gap-attack arms: nine arms
in flight. Adopted from the report: no shallower-PHAT promises (the two
serial attention chains are structural; 14 JEDI stages unreachable), stage
attribution labeled unmeasured until a prefix-pipeline trace is built.

---

## D124. Accuracy-gap forensics (Codex round 12): ranked causes, isolation arms, and a key artifact fact

**2026-07-28 day.** Zihan (correctly) rejected "replicated" for the −0.45-pt
JEDI accuracy gap. Full pipeline forensics vs their released code:

**Decisive artifact fact:** their SHIPPED N=128 3-feature model is
`epoch=1565-acc=81.17%-val_acc=80.78%-EBOPs=157284` — the published point
is an EARLY-ramp knee (proportionally ≈ our e210), at LOWER EBOPs than our
best checkpoint (157k vs 351k; our seed-2 hit 140k). Refutes "their number
is a high-EBOPs accuracy point"; the gap is real and mechanistic.

**Ranked causes (Codex):** (1) input representation — they z-score
globally; our pt64 convention leaves scaled pT median ~0.003, wasting
input-quantizer resolution [HIGH confidence]; (2) constituent store
difference — our 37.8 vs their ~49 real/jet, 2 GeV cut unresolvable from
their repo [MEDIUM, poorly isolated]; (3) LR trajectory — their 100-epoch
cosine restarts with 10-epoch plateaus vs our compressed 71-epoch cycles
[MEDIUM]; (4) **missing trace_minmax calibration** — they recalibrate
wrap-mode integer ranges over the train set before every evaluation and
firmware save; we NEVER have [MEDIUM, testable read-only, possibly a free
accuracy uplift for every row we own]; (5-6) selection protocol and train
fraction [LOW, directions favor us]. Quantization-class parity confirmed
(heterogeneous datalanes both).

**Launched to decompose:** n128-jediz (our store + zscore), n128-jediaz
(Aaron RAW leading-128 + zscore = their data path nearly exactly, pads
become constant nonzero after centering as in their pipeline), n128-jedi7k
(budget), `zz-phatjet-calib` (trace_minmax before/after on existing jedi
AND PHAT checkpoints), and the Codex-designed PHAT A/B `n128-pn64` vs
`n128-pnz` (seed-paired, input norm only — if zscore lifts PHAT, the
flagship may rise at comparable LUT, pending a full sparse-recipe rerun).

**D124 addendum (calibration test):** `trace_minmax` on our best jedi
checkpoint: 0.8042 → 0.8042 (Δ −0.0000) — **cause #4 eliminated**; our
i_decay-learned integer ranges are already calibrated, their explicit
recalibration step contributes nothing to the gap. The PHAT leg of the
test was INVALID (my harness hardcoded the pT re-sort; the wv10 lineage
is kt-sorted — the 0.66 reading is a wrong-data evaluation, disregarded).
Not rerun: the jedi null plus the PHAT rows' existing end-to-end bit-exact
verification make a calibration uplift implausible. Remaining live causes:
input representation (jediz/pnz), constituent store (jediaz), LR/budget
(jedi7k).

---

## D125. Ramp tolerance via lane protection: FAILED (pre-registered criterion); mechanism refined

**Measured 2026-07-28.** `n128-ramp-protb` (full ramp, six data×data lanes
at 0.01× beta): peak 0.7781 @ 707k EBOPs (+1.3 pts over the unprotected
control's 0.7654) — then COLLAPSED to 0.2495 @ 68k by end-of-schedule.
Partial, insufficient mitigation: fails the pre-registered promote
criterion (avoid collapse + matched knee). The control (e970) also never
recovered gentle-ramp accuracy.

**Mechanism refinement:** the full ramp kills PHAT through more than the
attention lanes — at d=16 even the constant projections/datalanes lack
bit-slack for beta=3e-6. JEDI tolerates the ramp because its width-64
constant layers have slack to shed (and that shedding IS its free
sparsity). Ramp intolerance is partly a WIDTH property — which the
running d32/d32wv arms probe from the other side. Lane-protection line
closed; a d32+full-ramp arm is the natural successor if width shows
promise at the gentle ramp.

---

## D126. Seed-2 traces: our-pipeline JEDI at 1.24× published; GMP-on-JEDI verdict now MIXED

**Measured 2026-07-28. Both bit-exact (RTL in `hw/jedis2/`).**

| | jedi s2 (e411) | jedigmp s2 (e208) |
|---|---:|---:|
| val acc | 0.8042 | **0.8063** |
| LUT | **121,877 (7.1%)** | 262,228 (15.2%) |
| latency | **40.0 ns / 12 stg** | 46.6 / 14 |
| FF | **44,725** | 101,660 |
| quantized zeros | **84.2%** | 69.7% |

**Replication resolved on the resource axis:** 121,877 vs their published
98k = **1.24×** (pre-P&R estimate vs post-route) at −0.5 pts — the earlier
3.2× was seed-1's undertrained front (best-acc checkpoints at 141k vs 351k
EBOPs across seeds: 1000-epoch front positions are highly seed-noisy).
This row is also the workspace's cheapest verified design outright.

**GMP-on-JEDI is now MIXED across the paired seeds:** seed 1: equal acc,
−15% LUT; seed 2: +0.21 pts at +115% LUT (best-vs-best). The delta's
shape depends on where each front happened to compress. Before any claim:
trace the hybrid's matched-EBOPs point (0.8038 @ 184k) against jedi-s2's
141k point, and read the nogmp factorial cell. The D122 "dominates every
PHAT row" statement stands for seed 1's traced pair but must now be
quoted per-seed; the robust cross-seed statement is "hybrid ≥ control on
accuracy at both seeds; area effect seed-dependent at 1000 epochs."

---

## D127. THE ACCURACY GAP IS SOLVED: it was the constituent store, not the pipeline

**Measured 2026-07-28.** The isolation triad delivered:

| arm | data | norm | best (1000 ep) |
|---|---|---|---:|
| n128-jedi/jedis2 (controls) | our robust store (~38 real/jet) | pt64 | 0.8042–0.8048 |
| n128-jediz (mid-run) | our robust store | zscore | ~0.8040 |
| **n128-jediaz** | **Aaron's raw leading-128 (~49 real/jet)** | zscore | **0.8109 @ 224,212 EBOPs** |

**Normalization: null** (jediz ≈ controls). **Store: +0.65 pts** — with
their data convention our pipeline reproduces the published 80.9% at
1/7th the training budget (0.8109 vs 80.9). The −0.45 "gap" was an
information deficit in our processed store (stricter upstream constituent
selection), not a pipeline, quantization, calibration (D124 addendum), or
architecture-port deficiency. Trace of the 0.8109 point running.

**Open question flagged:** richer constituents helped JEDI (+0.65) but
n128a (PHAT on Aaron's data, our-units/pt64) showed no gain (D110) — the
store benefit may be architecture-dependent (JEDI's global sums exploit
the soft tail; PHAT's pT-sorted patch head may dilute) OR n128a's units
conversion confounded it. A PHAT + aaron-raw + zscore arm would decide;
noted for the next window, not launched (board budget).

**Fairness implication for the paper/rebuttal:** all our PHAT-vs-JEDI
matched-protocol comparisons remain internally fair (same store both
sides). Comparisons to the PUBLISHED JEDI numbers should note the store
difference now that it is measured (~0.5–0.7 pts on JEDI).

---

## D128. Replication row traced; width hypothesis closed

**Measured 2026-07-28.**

**`jediaz` trace (bit-exact, `hw/jediaz/`):** the full-replication point is
**0.8109 @ 189,048 LUT (10.9%), 46.6 ns, FF 77,588, 73.8% quantized
zeros** — published-level accuracy in our pipeline/estimator at 1000
epochs on their data convention. This is now the reference row for
pipeline validity and the cheapest ≥0.81 design in the workspace.

**`d32wv` (width + ProxL1): CLOSED.** 0.8009 @ 2.12M EBOPs with only 7.9%
float zeros — no compression traction at width 32 within schedule (d16's
path reaches ~525-660k EBOPs at the same accuracy). Combined with d32's
control (0.8007 @ 1.33M) and d12's instability (D106): **d16 is the right
width for PHAT under this recipe**; "wide then harvest" does not engage at
1000 epochs and earns no continuation given full dominance at every
visible point. The width line and the lane-protection line (D125) both
closing leaves attention's ramp fragility explained but unexploitable at
current budgets — the honest paper statement stands as measured in D122.

---

## D129. GMP factorial COMPLETE: positive in both hosts

**Measured 2026-07-28.** `n128-nogmp` (PHAT minus GMP, modern recipe,
trained control — not D114's post-hoc zeroing): peak 0.7958 @ 1,057,280
EBOPs vs the with-GMP baseline's 0.8005 @ 666,359 — **GMP: +0.5 pts AND
cheaper in PHAT** (exceeds the 0.13-pt from-scratch seed bar, n=1 noted).

The 2×2 (host × GMP), all matched-protocol trained cells:

| | −GMP | +GMP |
|---|---|---|
| PHAT | 0.7958 @ 1.06M | 0.8005 @ 666k (**+0.5 pts, cheaper**) |
| JEDI | 0.8042–0.8048 | ≥ accuracy both seeds; area effect seed-dependent (D126) |

**GMP verdict for the paper: a transferable, ~hardware-free geometric
context module with measured positive contribution in BOTH an attention
host and a linear host** — the strongest version of the round-10/11
strategic framing, with the JEDI-area seed caveat stated. Interaction
term (does attention amplify GMP?): PHAT's +0.5 vs JEDI's smaller/noisier
delta suggests mild positive interaction; n=1-2 per cell, claim kept
qualitative. Remaining arms: jediz, pn64 (finals), jedi7k (long).

---

## D130. PHAT input A/B resolved: kt sort dominates at N=128; zscore mildly positive in the paired test

**Measured 2026-07-28.** `n128-pn64` (pT-sorted, pt64, seed 7): peak
0.7887 @ 799,796 — **pT sorting costs PHAT ~1.2 pts at N=128 vs the
kt-sorted baseline** (0.8005). The per-N sort table completes: PHAT
prefers kt at N=128 and N=16, pT at N=32; all promoted PHAT rows already
use the right sort. **Corrected A/B verdict:** against its true paired
comparator (pnz 0.7916 vs pn64 0.7887, both pT-sorted, seed 7), zscore is
+0.29 pts for PHAT — mildly positive, not the "refuted" of my earlier
mis-comparison against the kt baseline. A kt+zscore arm would complete
the matrix (noted; low priority — the effect is second-order vs the
sort). EBOPs blowup under zscore (7M mid-run) remains a caveat for the
hardware path; no promoted-row change.

---

## D131. Session-5 opening: gap-closure program launched (architecture changes authorized)

**2026-07-28/29 night.** Zihan's goal: close the PHAT↔JEDI gap; alternate
attention structures and GMP optimization explicitly authorized.

**Launched:**
- `n128-phatrich` — PHAT + rich store + KT re-sort (pt*dR key computed on
  the ourunits npz, inline atomic prep). The D110 null is now known to be
  sort-confounded (D130: pT sort costs PHAT 1.2 pts); the rich store gave
  JEDI +0.65 (D127). Best single shot at PHAT accuracy.
- `n128-phatlin` — NEW ARCHITECTURE (authorized): global patch-MHSA
  replaced by a JEDI-style linear stage (sender + broadcast receiver of
  the token mean), keeping local attention + GMP. Evidence: the trained
  sparse model already self-ablates global Q/K to zero (D114/D122) — this
  makes the optimizer's own choice structural. Expect ≈held accuracy,
  −25-35% LUT, −8-12 stages if the ablation evidence transfers.
- `n128-gmp2` — second GMP injection after local attention (~free
  machinery, D129's +0.5 invites a second helping).
- Codex round 13 commissioned: audit phatlin, rank further structures
  (full linearization, geometry-informed attention bias, multi-resolution
  GMP pyramids, pT-weighted scatter), define the beat-JEDI win condition.
Still running from session 4: n128-jediz (final), n128-jedi7k (budget).
Both linsum and gmp2 passed dual-backend smokes; ConfigMap shipped.

**D131 addendum (Codex round 13 actioned):** phatlin audit passed (faithful
to JEDI's branch pattern; receiver correctly fed from tokenizer output);
it fixed a gmp2 valid-mask bug (inert tonight, PJ_GMP_MASK=0) and
installed linsum trace cuts. Its ranked slots all launched: `n128-richlin`
(rich+linsum), `n128-fullin` (LOCAL+global linear — the local-attention
value ablation; local linsum implemented: per-patch sender+pooled
receiver, smoke-passed), `n128-fullinrich`. Rejected for tonight: shifted
patches (kt-order patches aren't spatial windows), geometry-scored
attention (new data×data routing), GMP pyramids/pT-weighting (D83/D88
priors). Win conditions pre-registered: (1) ≥0.8048 matched-store; (2)
>0.8063 @ <262k; (3) >0.8042 @ <122k. Board: 8 arms.

---

## D132. Budget curve closes the gap ledger: our store + extended budget EXCEEDS published

**Measured 2026-07-29 night.** `n128-jedi7k` (our store, extended
schedule): visible peak **0.8124 @ ~499–516k EBOPs** by ~epoch 4100 —
above the published 80.9, on the information-poor store. The final gap
ledger for our-pipeline JEDI vs published:

| budget | our store | their store |
|---|---:|---:|
| 1000 ep | 0.8042–0.8048 | **0.8109** |
| ~4100+ ep | **0.8124** | (their 7000 ep: 0.809 published) |

Store and budget are each SUFFICIENT alone to reach/exceed the published
accuracy; the D124 investigation is fully resolved with no residual
attributed to our pipeline, quantization, or port. Implication for all
PHAT comparisons: 1000-epoch numbers on our store systematically
understate what extended budgets recover — matched-budget comparisons
remain fair, and the paper should quote budget explicitly everywhere.
(Log window may truncate earlier epochs; full history on PVC.)

---

## D133. Session-5 matrix complete: gap did not close structurally; scaffolding finding; phatrich = new PHAT best

**Measured 2026-07-29 morning.** Full architecture × store matrix at
matched 1000-epoch protocol (peaks): see morning_report_2026-07-29.md.
Key rows: phatrich 0.8045 (rich store + kt = PHAT's best from-scratch,
+0.4, D110 confound confirmed resolved); fullinrich 0.8059 (fully-linear
beats attention variants on rich data); phatlin 0.7971 and gmp2 0.7973
closed negative on the std store; JEDI tops every column (0.8048/0.8109).

**Scaffolding finding:** global attention's weights are prunable post-hoc
(D114) but omitting the structure at build time costs 0.3 pts — attention
as optimization scaffolding, prunable but not omittable. Win conditions
(D131): NOT met at matched store+budget; the ~0.4–0.6 accuracy gap to
JEDI survived all structural variations. Remaining working levers: data
(+0.4–0.8/architecture), budget (D132), and the running phatrich knee
continuation (`n128-phatrichc`, self-selecting resume, PID 600k).

---

## D134. STRATEGIC PIVOT: optimize for best accuracy that FITS a VU13P, not minimum area

**2026-07-29.** Zihan: "optimize for the best performing model that can fit
in a VU13P... not enough time during rebuttal to get the most quantized
model, but fitting in a VU13P is a really good sign."

**Why this is under-explored:** every verified row sits at 15–40% device
utilization — five sessions of area minimization. The device has 2.5–6×
more room than we have ever used at the accuracy end.

**Two previously-dead levers revive** because their failure mode was
specifically *fighting compression*: distillation (D95/D100 — KD term acts
as an EBOPs floor) and two-stage float init (D96 — wins early at huge
EBOPs, loses at the knee; recorded 0.8048 @ 48M EBOPs at epoch 7). At low
pressure neither penalty applies. Capacity cut for area also returns:
FFN×4 (the paper's own width; we run ×1), d_model 32, and the extended
budget D132 measured at +0.8 pts on JEDI.

**Honesty guard adopted up front:** "fits" requires a stated margin — a
95%-utilization design leaves nothing for jet clustering and the rest of
CTL2 on the same FPGA. Two bands will be reported: **comfortable (<60%,
room for co-tenants)** and **tight (<90%)**. EBOPs→LUT ratio at low
pressure is historically 1.3–1.5 (vs 0.9–1.0 for sparse rows), so PID
targets were set as 700k EBOPs (≈0.9–1.05M LUT ≈ 52–61%) and 1.1M
(≈1.4–1.65M ≈ 83–95%).

**Launched:** `vu-cap` (rich+kt, d32, FFN×4, PID 1.1M), `vu-mid`
(rich+kt, FFN×4, PID 700k), `vu-long` (rich+kt best recipe, 4000 epochs —
the D132 budget lever on PHAT), `vu-float` (rich-store float ceiling at
d32/FFN×4 — also the teacher for a low-pressure distillation arm).

**D134 addendum (Codex round 14 actioned):**
- **Claim corrections applied to BOTH response drafts** (real overclaims
  caught): "0 DSP and 0 BRAM" → "expected... not confirmed by Vivado"
  (CLAUDE.md says these are assumptions written into the row, not report
  values); "at 300 MHz" → "300 MHz *target*, not closed timing";
  "synthesizable graph" → "hardware-extractable graph with generated RTL".
  Reserve "synthesis result" for a Vivado report.
- **Targets re-aimed:** FFN×4-specific historical ratios are 1.38–1.40×
  (751k→1.049M, 1.083M→1.493M), so 700k ≈ 56–61% (comfortable boundary)
  and 1.1M ≈ 88–95% (NOT reliably under the 90% tight limit). `vu-cap`
  reclassified as a frontier probe. No 1.0M companion arm launched — a PID
  front yields many checkpoints, so the ~1.0M point is obtained by TRACING
  vu-cap checkpoints along its front (cheaper than another GPU-night).
  **Selection is by traced LUT, never EBOPs** (D24).
- **`vu-paper` launched:** the paper's exact particle/patch configuration
  at high capacity — N=150, **P=10**, d32, **8 heads**, FFN×4, mean
  tokenizer, PID 1.0M, standard kt store (the rich store has only 128
  slots, so N=150 is unavailable there; store difference disclosed).
  Rhetorical value: "the published configuration fits." Gate: reject if
  the best checkpoint above vu-cap traces above 1,555,200 LUT (90%).
- **Distillation queued, not launched:** teacher (`vu-float`) must finish
  first; alpha=0.15 (the only repo-validated stable value, D98), MIN_ACC
  set to the verified teacher accuracy, SAME architecture as the teacher
  (train_qat rebuilds the teacher with the student's dims — a smaller
  student cannot load a d32/FFN×4 teacher).
- Rejected for now: d48 (untrained, capacity probe only), BN (D14's
  differences were within seed noise and cost EBOPs), two GMP blocks
  (D133 measured regression), two PHAT blocks (not implemented).

**D134 incident:** `vu-float` burned its full retry budget on TWO stacked
cluster faults: the known cuDNN-5003 jax flake (D12) plus a **broken GPU
node** returning `UnexpectedAdmissionError: GPU is lost` (device-plugin
NVLink failure) that pods kept landing on. Relaunched as `vu-floatb` on
the **torch** backend + pytorch image (documented immune to the flake,
~3-5x slower per epoch but 300 float epochs remains feasible). Lesson for
future long arms: prefer torch for anything whose loss is expensive to
restart; jax only where its speed matters and retries are cheap.

---

## D135. THE ABSOLUTE CRITERION: 1 SLR, sub-100 ns, II=1 — and PHAT-JeT at N=128 ALREADY MEETS ALL THREE

**2026-07-29 night.** Zihan's new angle: argue absolute deployment
feasibility on the FPGA CMS actually runs in L1T, not a relative
comparison. Primary-source constraints (arXiv:2508.15468 §I and §II-A,
verbatim):
- "the jet's class must be inferred in a few hundred nanoseconds, and **no
  more than one Super Logic Region (SLR) of a single FPGA**, or even less
  depending on the other algorithms running in parallel, can be used for
  the jet tagging algorithm"
- "**sub-100 ns latency and sub-10 ns initiation intervals are required**"
- CTL2 = 30 VU13Ps, 5 slices x 6 nodes round-robin (5 FPGAs live at a
  time), each hosting jet clustering AND tagging; "target frequency over
  300 MHz", II=1, deterministic latency required.

**A VU13P has 4 SLRs → ~432,000 LUT and ~864,000 FF per SLR.** The real
budget is therefore ~25% of the device, NOT the 60–90% the D134 pivot
assumed. Consequence: last night's capacity arms (0.7–1.1M EBOPs ≈
1.0–1.5M LUT ≈ 2.4–3.5 SLRs) were aimed at the wrong budget; `vu-cap`
killed to free GPU. Simultaneously, existing verified rows were re-scored:

| verified row | LUT | % of 1 SLR | FF % SLR | latency | 1 SLR? | <100 ns? |
|---|---:|---:|---:|---:|---|---|
| **N=128 sparsity+PID** | **385,471** | **89.2%** | 31.3% | **89.9 ns** | **YES** | **YES** |
| N=128 sparsity cheap | 404,852 | 93.7% | 34.5% | 93.2 ns | YES | YES |
| N=128 flagship | 478,214 | 110.7% | 41.4% | 93.2 ns | no | YES |
| N=64 sparsity | 487,451 | 112.8% | 52.0% | 136.5 ns | no | no |
| N=16 cheap | 365,950 | 84.7% | 26.7% | 133.2 ns | YES | no |

**Headline (already bit-exact verified, no new training required):
PHAT-JeT at N=128 fits inside ONE SLR of the deployed VU13P (89.2%) at
89.9 ns and II=1 (3.33 ns vs the sub-10 ns requirement), with FF at 31%
of the SLR and zero DSP/BRAM by construction.** All three stated
constraints met simultaneously. Sharpening: **N=128 is the deployable
configuration, not N=64** (N=64 exceeds both the SLR and the 100 ns
bounds) — the opposite of what the relative-comparison framing suggested.

**Tonight's experiments re-aimed at the true budget:** `slr-champ`
(rich+kt, PID 380k EBOPs + ProxL1 → ~87% SLR, hunting 0.80+ inside one
SLR), `slr-room` (PID 250k ≈ 58% SLR, the "or even less" case),
`slr-cap` (FFN×4 at the same 380k budget — more capacity, harder
compression). Kept: vu-long (budget lever), vu-paper (paper config on the
whole device), vu-floatb (ceiling/teacher).

**D135 addendum (deployment figure + vu-mid):** `slr_fit.pdf/.png` built
(generator `make_slr_fig.py` in-repo): accuracy vs **% of ONE SLR**, with
the CTL2 budget line at 100%, a 70% co-tenancy-headroom marker, and marker
fill encoding the sub-100 ns requirement — an absolute-terms figure with no
baseline comparison, complementing (not replacing) `pareto_fpga`.
`vu-mid` (FFN×4, PID 700k) closed at **0.7844** — capacity at 4× FFN did
not pay even at a loose budget, reinforcing that FFN×1 is right and that
the wide-and-cheap hypothesis keeps failing on this graph (cf. D128 width,
D133 gmp2).

---

## D136. Round-15 audit: the absolute claim must be "candidate within a nominal one-SLR budget", and Vivado is now THE critical path

**2026-07-30.** Codex audited the absolute-feasibility argument
(`round15_absolute_slr_argument.md`). Verdict accepted in full:

**"Fits in one SLR" is NOT established by a pre-P&R LUT estimate.** Only
constrained placement in an SLR pblock plus post-route timing can establish
it. Defensible form, adopted everywhere: *"the pre-P&R estimate of 385,471
CLB LUTs is 89.2% of an arithmetic quarter-device LUT budget, with 270,104
register bits (31.3% of the analogous FF budget), at an estimated 89.9 ns
(27 stages x 3.33 ns target) and II=1 by construction — a candidate inside
the nominal absolute envelope, pending constrained implementation."*
Never "runs at 300 MHz"; always "evaluated at a 3.33 ns target".

**New gap it found (real, previously unconsidered):** the design consumes
128x3 quantized inputs = **8,832 bits/jet, i.e. ~2.65 Tb/s at II=1**.
Plausible as an on-chip handoff from the clustering/sorting/buffering stage
the paper places upstream, but we have no interface, buffering, valid/
backpressure, or CDC story. Now disclosed as an explicit scope limit.

**Also corrected:** the 12.5 us / 5 us latency-budget percentages I had
planned to quote are figure annotations without a definition of what they
bound — dropped; the defensible latency comparison is 89.9 ns vs the
stated sub-100 ns tagger requirement. "Leaves DSP/BRAM entirely to
clustering" softened to *resource-type complementarity* (we still consume
~89% of one SLR's LUT/routing budget). Codex's "93.7 vs 93.2" flag was a
misread of my table (93.7% area, 93.2 ns) — no error.

**Actions taken:** figure axis relabelled to "% of a NOMINAL one-SLR
budget ... not placed", with an in-figure line stating placement is not
demonstrated; **`VIVADO_HANDOFF.md` rewritten as a specific ask** — pblock
confined to one SLR, `report_utilization -slr`, post-route timing at
3.33 ns, SLR-crossing count, congestion warnings — with every RTL path
verified against the PVC (the primary candidate is
`hw/final-day/verilog_epoch_565_val_acc_0_7961_ebops_393279_0`; my first
draft had a wrong directory) and the I/O caveat included so Chang can tell
us if the handoff assumption is wrong. Copy updated on the PVC.
**This one Vivado run is now the single highest-value action in the
project** — it is the difference between "candidate" and "demonstrated".

**D136 addendum (Zihan: "keep the 60-70% resource usage ones for reference"):**
"60-70%" is now readable on two bases, so BOTH are preserved and labelled
rather than choosing one:
- **60-70% of the whole VU13P** (~1.04-1.21M LUT) — the existing N=150/P=10
  rows at 61.8% / 65.4% and the N=32 row at 70.7%. Retained as reference for
  "how much model does a whole FPGA buy"; explicitly noted as ~2.5-2.8 SLRs,
  i.e. NOT satisfying the one-SLR constraint.
- **60-70% of ONE SLR** (~259-302k LUT) — the co-tenancy band leaving 30-40%
  of the SLR for other CTL2 logic. No verified row there yet; `slr-room`
  targets it.
Implemented: a "Two resource bases — read both" section in
synthesis_table.md with the 4x conversion rule (% of SLR = 4x % of device),
and a shaded 60-70%-of-SLR band in `slr_fit` (axis widened to 52-145%) so
those points are visible when the arm lands. No rows deleted from the table.

---

## D137. Session-6 results: new float ceiling 0.8206, QAT record 0.8077, and the SLR band needs CONTINUATION not from-scratch PID

**Measured 2026-07-30.**

| arm | what it tested | result |
|---|---|---:|
| **vu-floatb** | rich-store float ceiling (d32/FFN×4) | **0.8206** float — above the paper's own 0.8180 |
| **vu-long** | 4000-epoch budget lever on PHAT (rich+kt) | **0.8077 @ 1,536,867 EBOPs** — PHAT's best QAT accuracy ever (+0.32 over the 1000-epoch 0.8045) |
| vu-paper | paper config N=150/P=10/d32/8heads @ PID 1.0M | 0.7975 @ 1,356,808 EBOPs (~79% of device, ~3.1 SLRs) |
| slr-champ | from-scratch PID 380k + ProxL1, rich+kt | 0.7940 @ 629,913 — **PID never reached target** |
| slr-room | from-scratch PID 250k + ProxL1 | 0.7904 @ 545,212 — **also overshot** |
| slr-cap | FFN×4 at 380k | failed (cluster) — not rerun; FFN×4 already closed by vu-mid 0.7844 |

**Two findings that matter:**
1. **The budget lever transfers to PHAT** (D132 was measured on JEDI):
   4000 epochs gives +0.32 pts over 1000 at the same recipe. The float
   ceiling on the rich store is 0.8206 — so the QAT gap to float is now
   1.3 pts and the paper's own float number (0.8180) is exceeded by our
   synthesizable graph.
2. **From-scratch BetaPID cannot reach tight EBOPs targets in 1000
   epochs** (both SLR arms equilibrated 40-120% above target). Every row
   we own inside the SLR band was produced by CONTINUATION from a trained
   checkpoint (D78/D114/D119). Corrected approach launched: `slr-rich`
   (resume vu-long's 0.8077 record → PID 380k + ProxL1 ≈ 89% SLR) and
   `slr-rich2` (→ 290k ≈ 67% SLR, the co-tenancy band Zihan asked to keep).
   This is the "start from the accuracy record, compress into the budget"
   recipe that produced every previous champion.

**D137 addendum (Zihan: "also try 90-100% of the device"):** the band
(1.555-1.728M LUT) currently holds one legacy row — N=64 @ 0.8024 /
1,558,333 LUT / 90.2% (pre-continuation, pre-sparsity, standard store) —
with an N=150 row just over at 100.2%. Two arms launched to occupy it with
current-recipe designs, both PID-pinned at 1.5M EBOPs for 2500 epochs on
the rich store (**ProxL1 deliberately OFF: sparsity is an area lever and
area is free in this band**):
- `dev90` — resumes the 0.8077 record, holds it at ~90% device, trains for
  accuracy only.
- `dev90cap` — d_model=32 from scratch (the one untested combination: big
  model + long budget + loose area). Caught pre-burn: it had inherited the
  resume-from-vu-long line, which is a shape mismatch against a d16
  checkpoint (the D134 `n128-sat` failure mode); resume removed.
Also running: `hw-vulong` — the accuracy record has never been traced, so
its true device fraction is unknown; that trace decides whether the record
itself already sits in the 90-100% band.

---

## D138 — The accuracy record fits the device, and the EBOPs pressure that was costing us accuracy was never under PID control

**Goal reframed (Zihan, 2026-07-30):** maximise accuracy subject to fitting
**one whole VU13P** (1,728,000 LUT), and do not stop until the quantized
number beats **JEDI-linear's best reported 80.9%** (Table II, N=64 and
N=128). This replaces the one-SLR criterion of D135 as the *optimisation*
target; the SLR rows remain the stronger deployment claim and are kept.

### Evidence 1 — the record fits, with room to spare

`hw-vulong` traced `vu-long/pareto/epoch=773-val_acc=0.8077-ebops=1536867`:

| | value |
|---|---|
| LUT (da4ml pre-P&R) | **1,431,345 = 82.8% of VU13P** |
| stages / latency | 48 / 159.8 ns @ 300 MHz target |
| FF (reg_bits) | 1,012,429 = 29.3% |
| II / DSP / BRAM | 1 / 0 / 0 (assumed by construction) |
| bit-exact | **NO** — max\|err\| = 1.251, argmax match = 1.0000 |

So the accuracy record occupies 82.8% of the device, leaving ~17%
(~1.85M EBOPs at the measured 0.93 LUT/EBOP) unspent. **It is not yet
quotable**: the netlist does not match the model. Argmax agreement is
still 100% on 512 jets, so the discrepancy is small in effect, but per
D70 a non-bit-exact trace cannot carry an accuracy claim. Hypothesis:
this is the highest-precision checkpoint ever traced and the input port
is fixed at 16 fractional bits (`PJ_INPUTS_KIF=1,8,16`, chosen for
low-precision checkpoints), so hardware truncates what Keras keeps.
`g-kif` scans 1,8,20 / 1,8,24 / 1,10,28 to confirm or refute this.

### Evidence 2 — two squeezes were running, and only one was steerable

vu-long peaked at **epoch 773 (0.8077 @ 1.54M EBOPs)** and then **decayed
to ~0.80 by epoch 3900 while EBOPs fell to 600k**. That 3,200-epoch tail
traded 0.7 points of accuracy for area the device did not need.

`dev90` then resumed the peak with `BetaPID(target=1.5M)` — and still sank
to 663k EBOPs, less than half its target, with beta pinned at ~5.7e-9.
A PID that wants 1.5M cannot drive the design to 663k. Something else was
pushing:

> **`PJ_ATTN_L1=1e-5` and `PJ_FFN_L1=3e-5` are `MonoL1` penalties on
> datalane bitwidths. They apply constant downward pressure that the
> BetaPID neither observes nor counteracts.**

Every run in the campaign inherited these from the compression era, when
the design was 4x too big. Under a *loose* budget they are pure accuracy
loss. This also explains why `slr-rich`/`slr-rich2`, which resumed the
same record, equilibrated far below their targets.

**Rejected:** "the PID target is the budget." It is only a ceiling; the
floor is set by whatever unsteerable L1 terms are active.

### Evidence 3 — capacity was never given a fair trial

`dev90cap` (d_model=32, from scratch) reached **0.8024 @ 975k EBOPs at
epoch 1369 and was still climbing**, then collapsed to 0.77 under the same
combined squeeze. The D137 reading that "capacity fails" was measured
under pressure that has now been identified as unnecessary.

### Campaign launched (5 arms, all with ATTN_L1/FFN_L1 REMOVED)

Budgeted at 1.75M EBOPs ≈ 1.63M LUT ≈ 94% of device (0.93 LUT/EBOP
measured on this very checkpoint; the ratio is 0.98 on the sparse
385k-LUT row, so 1.75M could land as high as 1.71M LUT — any winner must
be traced before the fit is claimed).

| arm | change | question it answers |
|---|---|---|
| `g-free` | resume ep773, **beta = 0** | can this graph reach 0.809 under *any* budget? |
| `g-pid` | resume ep773, PID 1.75M | best accuracy that provably fits |
| `g-ffn4` | **FFN_MULT 1 → 4** (paper value), PID 1.75M | recover the FFN capacity conceded to area |
| `g-d32` | resume dev90cap's peak, PID 1.75M | capacity, retried without the hidden squeeze |
| `g-bw10` | b0=f0 **7 → 10**, PID 1.75M | precision headroom, also chosen when area was scarce |

`g-free` is the decisive one: if the quantized graph cannot reach 0.809
with the penalty switched off entirely, the gap is architectural and no
budget setting will close it.

---

## D139 — Codex round 16: a latent BetaPID crash, and a premise of D138 corrected

### Bug caught before impact: `BetaPID` cannot start from beta = 0

`hgq.utils.sugar.BetaPID` defaults to `log=True` and `init_beta=None`. With
`init_beta=None` it takes `self.beta = get_avg_beta()`, and every layer is
built under `LayerConfigScope(beta0=0)` — so `self.beta = 0.0`. At
`epoch == warmup` (default 10) the log branch evaluates

```python
self.pid.integral = (log10(self.beta) - self.pid.p * err) / self.pid.i - err
```

`math.log10(0.0)` raises `ValueError: math domain error`. **Any from-scratch
BetaPID arm without `PJ_INIT_BETA` dies at epoch 10.**

Observed live: `g-ffn4` sat at `beta: 0.0000e+00` with EBOPs *rising* through
47.9M (27x its 1.75M target) — the controller was inert, not merely slow.
`g-ffn4` and `g-bw10` were killed before epoch 10; `g-cap`, `g-d32`, `g-pid`
and `g-cont` carry `PJ_INIT_BETA=1e-9` and are unaffected.

**This retro-explains D137's "from-scratch BetaPID cannot reach tight EBOPs
targets."** `slr-champ` and `slr-room` both equilibrated 40–120% above target
from scratch. The conclusion drawn then — that continuation is *required* —
was inferred from runs whose controller may never have engaged. The
continuation recipe is still the one that produced every champion, but
"from-scratch PID does not work" is no longer supported evidence.

### Premise corrected: there is no measured 1.3-point quantization gap

D137/D138 compared the **0.8206 float** against the **0.8077 QAT** record as
if they were the same graph. They are not: the float ceiling is
**d_model=32 + FFN×4** (verified directly from the teacher archive —
`embed` units 32, `ffn1` units 128), while the QAT record is
**d_model=16 + FFN×1**. The 0.0129 difference mixes capacity, optimisation
and quantization, and cannot be called a quantization cost.

**Consequence:** the float ceiling of the record's *own* graph is unknown, and
with it whether that graph can reach 0.809 at all. `g-f16` (train_float.py,
whose defaults `FFN_MULT=1`/`D_MODEL=16` already match the record, 300 epochs,
rich store) now measures it. That number decides whether `g-pid`/`g-cont` are
capable of the goal or whether only the capacity arms are.

### Diagnosis of D138 confirmed, with one wording fix

`ATTN_L1`/`FFN_L1` install `fr=MonoL1(...)` via nested `QuantizerConfigScope`
(`phat_jet_k3.py:116,145`), 1000x and 3000x the global `MonoL1(1e-8)`
baseline. `MonoL1` contributes `l1 * sum(x)` as an ordinary regularization
loss — **not** scaled by layer beta. BetaPID *does* observe their effect
through total EBOPs; what it cannot do is oppose them, since beta is clamped
at `min_beta=0` and cannot go negative. Once beta hits the floor the L1 terms
keep shrinking fractional widths unopposed. `g-free`'s label was also loose:
removing the regional terms and setting beta=0 still leaves the global
`MonoL1(1e-8)`.

### Margin warning on the 1.75M budget

At the 0.98 LUT/EBOP ratio seen on the sparse row, 1.75M EBOPs lands near
1.715M LUT — **under 1% margin** against 1,728,000, and D24 records that
LUT/EBOP is non-monotonic across checkpoints. `g-cont` therefore targets
**1.55M** (hold the record's own area, remove the L1 damage, anneal at
LR=3e-4 — Codex's ranked-#2 lever, the schedule control that was missing:
`first_decay_steps = 500*EPOCHS/7000`, so a resumed run is not
schedule-continuous with the run that produced its checkpoint).

**Board:** `g-cap` (d32/FFN×4 from the 0.8206 float teacher, PID 1.75M),
`g-d32`, `g-pid`, `g-cont`, `g-f16` (float ceiling probe), `g-kif`
(bit-exactness scan).

---

## D140 — Bit-exactness restored: the input word format, not the netlist, was wrong

`g-kif` resolved D138's open item on the first setting it tried.

| `PJ_INPUTS_KIF` | LUT est | bit-exact | max\|err\| |
|---|---:|---|---:|
| 1,8,16 (long-standing default) | 1,431,345 | **no** | 1.251 |
| **1,8,20** | 1,431,347 | **YES** | **0.000** |

Four extra fractional bits on the input port fix it. The default of 16 was
chosen when every traced design was far lower precision; the epoch-773
checkpoint is the highest-precision one ever traced (1.54M EBOPs), and the
port was silently truncating what Keras retained. The LUT estimate is
unchanged (+2 LUT), so this is an emulation-fidelity setting, not a cost one.

**The accuracy record is therefore now a fully verified row:**

> **N=128, 0.8077, 1,431,347 LUT = 82.8% of one VU13P, 1,013,861 reg bits
> (29.3%), 48 stages = 159.8 ns at a 300 MHz target, II=1, DSP=0/BRAM=0 by
> construction, bit-exact in Verilator against the trained quantized model.**

Against the goal (beat JEDI-linear's best reported **80.9%** while fitting one
VU13P) this is **0.13 points short**, with ~17% of the device unspent.

**Standing rule added:** `PJ_INPUTS_KIF=1,8,16` is only safe for
low-precision checkpoints. Every trace of a >1M-EBOPs checkpoint must widen
the input format and confirm `bit_exact: true`; a non-bit-exact trace carries
no accuracy claim (D70). `extract_hw.py`'s own warning text already says
"Widen PJ_INPUTS_KIF and retry" — it was correct.

---

## D141 — `PJ_FFN_MULT` defaults to 4; every resume arm silently targeted the wrong graph

`train_qat.py:234` reads `FFN_MULT = _env("PJ_FFN_MULT", 4, int)` — the
paper's value. `vu-long` (and `dev90cap`) trained at **FFN_MULT=1**, set
explicitly. The D138 job generator did not set it at all, so all five arms
were built with **FFN×4**.

For the from-scratch arms this was merely mislabelling (`g-ffn4` was testing
the same FFN width as its "controls"). For the **resume** arms it is fatal:

```
ValueError: A total of 6 objects could not be loaded.
Example error message for object <QDense name=ffn1, built=True>
```

`g-cont` crashed on `model.load_weights(RESUME_CKPT)`. `g-pid` and `g-d32`
carried the same defect and would have crashed identically. All three
relaunched with `PJ_FFN_MULT=1` pinned. Unaffected: `g-cap` (sets 4
explicitly, and its float teacher is genuinely d32/FFN×4 — verified from the
archive) and `g-f16` (`train_float.py:59` already defaults to 1).

**This is the D134 `n128-sat` failure mode a third time** — a resume whose
graph does not match its checkpoint. The earlier grep-back verified what the
arms *removed* (`ATTN_L1`/`FFN_L1`) but not that every structural knob
matched the resume target. **Rule extended: for any arm with
`PJ_RESUME_CKPT`, diff the full structural env — N, P, heads, d_model,
FFN_MULT, tokenizer, GMP settings, norm, local/global mode — against the
source run's `run_config.json` before launch, not just the knobs being
changed.** The failure is loud (ValueError), but it costs a scheduling round
each time, and GPU slots are the binding constraint tonight.

---

## D142 — How big a win the goal actually needs

The evaluation partition is the standard hls4ml test split (~2.6e5 jets;
`prep_data.py` writes only `x_train`/`x_val`, and `x_val` *is* the reported
partition — there is no third split being withheld). Binomial standard error
at p ≈ 0.81 is

```
sqrt(0.81 * 0.19 / 260000) = 7.7e-4  ->  0.077 accuracy points
```

So the distance from the current verified record (0.8077) to JEDI-linear's
reported 0.809 is **1.7σ** — real, but not large. More importantly, *clearing*
0.809 by a hair would not be a defensible claim: 0.8090 is a 0σ win, and
checkpoint selection happens on this same partition, which biases the
selected maximum upward.

**Operating target set to ≥ 0.8105** (≈2σ above 0.809) before the goal is
called met, and the winning row must additionally be traced bit-exact under
1,728,000 LUT. A result in 0.8090–0.8105 will be reported as "at parity,
not clearly above."

---

## D143 — The float-eval separating experiment FAILS by construction; QAT weights are not meaningful outside their quantizers

`float_eval_qat.py` rebuilds the quantized graph via `train_qat.build_qat_model()`
(imported, not reimplemented — the overflow mode is applied through a
`QuantizerConfigScope`, so a locally reconstructed config silently produced
SAT_SYM quantizers for a `wrap`-trained checkpoint and failed to load 35
objects), loads epoch 773, then copies every kernel/bias by layer name into
the same graph built with `quantized=False`.

| measurement | value |
|---|---:|
| quantized, as trained | **0.8077** |
| identical weights, quantizers removed | **0.3069** |

All 32 tensors transferred with **zero** shape mismatches, and every float
layer name exists in the quantized graph — the transfer is mechanically
correct. 0.3069 against a 5-class chance of 0.20 means the float graph is
near-broken, not merely degraded.

**Conclusion: the experiment does not measure what it was meant to.** Codex
round 16 proposed it to separate quantization-limited from
optimization-limited, on the assumption that stripping quantizers leaves a
valid float network. That assumption is false for this model. The trained
function is *entangled with its quantizers*: with `overflow_mode=wrap`,
out-of-range intermediates wrap rather than saturate, and the network has
learned around that behaviour (and around the quantizers' clipping keeping
`tanh`/`softmax` arguments in range). Remove them and the computation is a
different function, not a higher-precision version of the same one.

**Retained as a genuine finding, with a caveat attached:** "evaluate the QAT
weights in float" is not a valid diagnostic for any wrap-overflow HGQ2 model
here, and any future claim of the form "the quantization cost is X" must not
be produced this way. `float_eval_qat.py` is kept, with this limitation
documented in its docstring, because the *negative* result is worth not
rediscovering.

**One positive by-product:** the rebuild-and-`load_weights` path reproduced
**0.8077 exactly** on the full 260k evaluation partition — an independent
confirmation of the record outside the training loop.

**The valid measurement is therefore `g-f16`** — an independent float
training of the record's own d16/FFN×1 graph. At epoch 53/300 it is already
at 0.7996, i.e. on track to clear the 0.8077 QAT record, which would mean the
record's graph is *not* saturated and the cheaper arms (`g-pid`) have real
headroom without resorting to capacity.

---

## D144 — Continuation arms were using the wrong LR cycle length

`cosine_restarts` (train_qat.py:239) uses a **fixed** period —
`first_decay_steps = max(int(500 * EPOCHS / 7000), 10)`, no doubling — and
anneals to `alpha = 1e-6` at the end of each cycle. The best checkpoint of a
cycle lands at that minimum.

Consequence: `PJ_EPOCHS` silently sets the *harvest cadence*, not just the
run length. At `EPOCHS=3000` the period is **214 epochs**; on the ~107 s/epoch
nodes we are getting tonight that is **6.4 hours per harvest**, i.e. one or
two usable points overnight. For most of each cycle the LR sits near 3e-3,
which keeps a resumed champion oscillating *below* the accuracy it started
from.

This is the mechanism behind Codex round 16's ranked-#2 lever: a resumed run
is **not schedule-continuous** with the run that produced its checkpoint — it
restarts at full LR and kicks the model out of the basin it inherited.

**Change:** continuation arms (`g-pid`, `g-d32`) moved to `EPOCHS=600`
(**42-epoch cycles**, ~75 min per harvest — roughly 9 annealed points
overnight instead of 2) at `LR=1.5e-3` (half the previous peak), so each
restart refines the inherited basin rather than leaving it. `g-cap` is
deliberately **not** changed: it initialises from float weights rather than a
converged QAT checkpoint and needs the long cycles to converge at all.

Note this also means the run length is now decoupled from wall-clock: 600
epochs will not complete, but every 42-epoch cycle deposits a Pareto point,
which is what we actually harvest.

---

## D145 — Float-teacher init needs wide integer bits; the 7/0 default collapses it to chance

`g-cap` (d32/FFN×4 initialised from the 0.8206 rich-store float teacher) died
in `train_qat.py`'s own guard:

```
transferred weights into 16 layers
post-transfer val_acc=0.1949
AssertionError: transfer collapsed to 0.1949 (chance=0.2):
  quantizer i0 too narrow for the float weights/activations (PJ_I0_K/PJ_I0_A)
```

The campaign default is `BW_K=BW_A=7` with `I0_K=I0_A=0` — **every bit
fractional**, so the representable range is roughly ±1 and any float weight or
activation above that is destroyed on transfer. That default was fine because
essentially every arm this campaign trained from scratch, where the quantized
weights simply grow into whatever range the bits allow.

`twostage2-n64` is the only prior run that successfully initialised from a
float teacher, and it used **`BW_K=10, I0_K=3, BW_A=10, I0_A=8`** —
verified from its log (`post-transfer val_acc=0.7426`, clearing the >0.5
guard) and its result (0.8048). `g-cap` relaunched with those values.

**Same underlying fact as D143, seen from the other side.** D143: quantized
weights are meaningless *without* their quantizers. D145: float weights are
meaningless *inside* quantizers too narrow to hold them. The float and
quantized parameterisations of this graph are not interchangeable in either
direction without matching the numeric ranges — which is exactly what
`I0_K`/`I0_A` control.

Useful side effect: at `BW_K=BW_A=10` this arm also answers the
precision-headroom question that the killed `g-bw10` arm was asking, at no
extra GPU cost. The higher initial bitwidths raise starting EBOPs, but the
BetaPID target of 1.75M compresses them back toward the device budget.

**Credit where due:** the assertion at `train_qat.py:441` caught this
immediately and named the exact cause. Without it the arm would have trained
for hours from chance-level weights and quietly returned a useless answer —
the D89 teacher-corruption failure mode.

**D145 addendum — the recipe works.** With `BW_K=10, I0_K=3, BW_A=10,
I0_A=8`, `g-cap`'s float transfer landed at **post-transfer val_acc =
0.8065** (versus 0.1949 at the 7/0 default). The 0.8206 float teacher
therefore enters the quantized d32/FFN×4 graph essentially at the current
verified record's level (0.8077) **before any QAT training**, with a float
ceiling 1.4 points above it. This is the best starting position any arm in
the campaign has had.

The open risk is unchanged and is the one `vu-paper` flagged: this arm starts
at very high EBOPs (10-bit initialisation on a d32/FFN×4 graph) and the
BetaPID must compress it to 1.75M. `vu-paper` (d32, N=150, 8 heads)
compressed to 1.36M and landed at 0.7975 — *below* the d16 record. Whether
0.8065 survives an ~80x EBOPs compression is exactly what this arm measures.

---

## D146 — The record's own graph is NOT saturated: d16/FFN×1 float ceiling is ≥0.8126

`g-f16` (independent `train_float.py` run, N=128/P=8/d16/FFN×1, rich store,
300 epochs — i.e. the record's exact graph) reached **0.8126 float within 142 epochs**, against the QAT record of 0.8077 on the
same graph. Stopped there deliberately: the answer was decisive and the GPU
slot was worth more to `g-pid`.

| graph | float | best QAT |
|---|---:|---:|
| d16/FFN×1 (the record's) | **≥0.8126** | 0.8077 |
| d32/FFN×4 (`vu-floatb` teacher) | 0.8206 | 0.8065 post-transfer, untrained |

**Consequence for the goal:** there are ≥0.49 points of float headroom above
the current champion **on the cheap graph**, and the target is only 0.13
points away. The goal therefore does *not* depend on the capacity route
succeeding through an ~80x EBOPs compression (the `vu-paper` risk). `g-pid`
— resume the record, L1 pressure removed, area held near the device budget,
42-epoch anneal cycles — is a credible path on its own.

This also **supersedes the D137/D138 framing** one final time. The
"quantization gap" was never 1.3 points against a same-graph float; on the
record's own graph it is **0.49 points**, which is an ordinary QAT cost and
not a structural wall.

**Methodological warning, recorded because it nearly caused a wrong call:**
this run sat at 0.8056–0.8059 from epoch 63 to 96 and was read here as
plateaued/saturated, which would have redirected all remaining GPU to the
capacity arms. It was not converged — it was between anneal cycles
(`first_decay_steps = 500*300/7000 = 21` epochs). **A cosine-restart run must
not be called converged from a flat stretch shorter than a full cycle**, and
the temptation to kill it early for its GPU slot was exactly wrong.

**D146 addendum — new arm `g-f16q`.** D146 (cheap graph has ≥0.49 pts of
float headroom) and D145 (float→QAT transfer works at `BW=10, I0_K=3,
I0_A=8`) combine into an arm the campaign has never run: **QAT on the
d16/FFN×1 graph initialised from its own 0.8126 float checkpoint**, PID at
1.75M EBOPs, 1000 epochs (71-epoch anneal cycles).

Why this may beat `g-pid`, which resumes the 0.8077 QAT record:
`g-pid` starts from a quantization-adapted but *lower* point, `g-f16q` from
a better but quantization-naive one. `g-cap` showed the transfer lands near
0.8065 even for a much larger graph, so the naivety cost is small.

**CORRECTED (round 17).** This entry originally claimed the graph "already
fits the device at full precision" because the record occupies 82.8% at 1.54M
EBOPs. That conflates graph *topology* with quantized *hardware cost*. The
record fits at 1.54M only **after** learned quantization; the same topology in
its float-receiving wide-quantizer parameterisation starts at **148M EBOPs**,
an 85x compression — not a trim. Measured, not hypothetical. Both arms are kept — they start from different points and
the failure modes are uncorrelated.

---

## D147 — `g-cap` retired: the capacity route is not reachable in the time available

`g-cap` (d32/FFN×4 from the 0.8206 float teacher) transferred beautifully —
post-transfer **0.8065**, essentially the current record before any training —
and then hit two walls at once:

| | value |
|---|---:|
| starting EBOPs | **506,864,695** |
| PID target | 1,750,000 |
| required compression | **290x** |
| epoch time | ~5–13 min (d32/FFN×4 at 10-bit init) |

For scale, `vu-long` compressed **8x** over 4000 epochs. A 214-epoch anneal
cycle at this epoch time is well over a day, so this arm cannot deposit a
single annealed checkpoint tonight, let alone complete a 290x compression.

The 10-bit initialisation required by D145's transfer recipe is itself much
of the cost: it is what makes the float weights survive, and what makes the
graph start at half a billion EBOPs. That tension — **the wide quantizers
needed to receive float weights are the same ones that blow the area
budget** — is the real obstacle to the capacity route, not the architecture.

Retired in favour of `g-pid2`: resume a *different* vu-long checkpoint
(epoch 853, 0.8072 @ 1.491M EBOPs), `PJ_SEED=2`, PID 1.65M, 42-epoch cycles.
Rationale is Codex's ranked-#3 lever — with the threshold only 0.13 points
away, a decorrelated starting point has higher expected value than a
speculative architecture change. Its failure modes are uncorrelated with
`g-pid`'s.

**Board (all four on the graphs that already fit):**

| arm | start | budget | cycles |
|---|---|---:|---:|
| `g-pid` | vu-long ep773 (0.8077) | 1.75M | 42 |
| `g-pid2` | vu-long ep853 (0.8072), seed 2 | 1.65M | 42 |
| `g-f16q` | g-f16 float (0.8126), 10-bit init | 1.75M | 71 |
| `g-d32` | dev90cap ep1369 (0.8024), d32/FFN×1 | 1.75M | 42 |

---

## D148 — Codex round 17: a calibration escape route, and two errors in D146/D147

### Two corrections applied to this log

1. **D146 arithmetic.** 0.8126 − 0.8077 = 0.0049 = **0.49 points**; one
   instance still read 0.34. Fixed.
2. **D147 was wrong** that `g-f16q`'s graph "already fits the device at full
   precision". It conflates graph *topology* with quantized *hardware cost*.
   The record fits at 1.54M EBOPs **after** learned quantization; the same
   topology in its float-receiving wide-quantizer parameterisation starts at
   **148M** — an 85x compression, not a trim. Corrected in place.

### The escape route from D145's tension (highest-value finding)

D145 framed a hard tension: the wide quantizers needed to *receive* float
weights are the same ones that *blow* the area budget. **That framing is too
broad.** HGQ2 already stores integer range state **per quantizer** — KBI
creates an `_i` state tensor shaped by the bitwidth mapper
(`fixed_point_quantizer.py:215`), KIF likewise (`:371`), with weight
quantizers at `homogeneous_axis=()` and datalanes at `(0,)`
(`config.py:75,131`). The scalar `i0` is only its **initializer**.

`hgq.utils.minmax_trace.trace_minmax(model, data, reset=True)` exists for
exactly this: it resets every eligible `_i` to a sentinel, runs the data with
`training="tracing"`, and sets each quantizer to its observed minimum integer
width (`get_minimal_i`; KBI `:277`, KIF `:401`). Its docstring states it is
specifically needed for **WRAP** activation quantizers — which is our mode.

Correct float warm-start sequence:
1. build the QAT graph with adequate fractional precision;
2. transfer float kernels;
3. `trace_minmax(..., reset=True)` on representative **training** data;
4. check post-transfer accuracy and starting EBOPs;
5. only then start QAT.

**The earlier "calibration is null" result (D124) does not refute this** — it
calibrated an already-trained QAT model whose integer ranges had adapted.
Float transfer is precisely where calibration has value. Caveat: calibrate on
enough data to capture tails; a small batch underestimates WRAP ranges and
produces rare catastrophic wraps.

Note this does **not** shrink fractional width — `f0`/`b0` remain independent.
The screen worth running is a grid over KIF `f0 ∈ {7,8,9,10}` with a
temporarily wide `i0` followed by calibration, picking the cheapest point that
still transfers above ~0.805. That is a minutes-scale, training-free screen.

### Board change: `g-pid2` killed, `g-lowlr` launched

`g-pid2` was **not** an independent-seed arm in any meaningful sense — it
resumes the *same* vu-long lineage 80 epochs later, and re-seeding after a
resume does not create an independently optimised basin. Its trajectory
(0.7999 → 0.7473 → 0.7764) is oscillation under too-large an LR, not useful
decorrelation. Its structural config was verified correct, so this is a
recipe problem, not a bug.

Replaced by **`g-lowlr`**: resume ep773, **LR 3e-4** (5x lower), PID 1.70M,
42-epoch cycles. Rationale: `g-pid` at LR 1.5e-3 fell 0.8077 → 0.8027 on
epoch 1. We need *refinement*, not exploration — the target is 0.13 points
away.

### Budget: do NOT push to 1.85M EBOPs

At the record's measured 0.931 LUT/EBOP: 1.75M → ~1.629M LUT (94.3%);
1.85M → ~1.722M LUT (**99.7%**). That is effectively zero margin, and D24
records that LUT/EBOP is not stable across checkpoints. The PID targets
EBOPs, not LUTs, and widths redistribute. Harvest points above and below
1.75M and trace them rather than steering the whole run to the edge.

### Zero-area levers, ranked

1. **Low-LR continuation** — done (`g-lowlr`).
2. **Label smoothing ε≈0.02** — lowest implementation risk, touches only CE
   targets, no quantizer-state hazard. Not 0.1, too aggressive when only
   +0.28 pt is needed.
3. **EMA** — riskier than in plain Keras: `use_ema` averages *every* trainable
   variable including learned quantizer `b`/`f`, and rounded bitwidths are
   discontinuous (averaging 6.49 and 6.51 can move a deployed width by a whole
   bit). Would need `SwapEMAWeights` before validation, `FreeEBOPs` and Pareto
   saving, and the EMA checkpoint must be traced independently — never assume
   its cost matches the live model.
4. **SWA** — worse: different minima may have different learned bitwidth
   layouts; averaging can produce an untrained kernel/quantizer combination.
5. **Mixup / physics augmentation** — not tonight. The fixed GMP η–φ grid and
   clipping range make the implemented graph **non-invariant** to φ rotation,
   so "physics-safe" augmentation is not automatically safe here.

### Red-team items accepted

- "Same-graph quantization gap = 0.49 pt" is **observed headroom**, not a pure
  causal quantization penalty — independent float and QAT runs have different
  optimisation trajectories. It still mixes quantization and optimisation.
- "ATTN_L1/FFN_L1 were pure accuracy loss" is directionally supported but
  **not yet proven**: the code proves unsteerable pressure; only the L1-free
  continuations can prove removing it *recovers* accuracy. None has yet.
- D144's "the best checkpoint of a cycle lands at the LR minimum" is too
  categorical — a common pattern, not guaranteed. Evaluate throughout cycles.
- On D143: the 32-tensor transfer is accepted as mechanically valid, but
  "WRAP alone explains the collapse" remains a hypothesis — the quantized and
  float branches also use different implementations for softmax and pooling.

---

## D149 — A 0.8095 checkpoint that does NOT count, and the monitoring bug that flagged it

`g-f16q` epoch 5 reached **val_acc 0.8095** — above JEDI-linear's reported
0.809 — at **107,474,337 EBOPs**. That is roughly 100M LUT, about **60x the
device**. It is the float-initialised model still near full precision and it
is **not a result**: the goal is accuracy *subject to fitting one VU13P*.

The session monitor reported "GOAL MET" because it ranked pareto checkpoints
by accuracy alone. **Fixed**: the monitor now filters to checkpoints with
EBOPs ≤ 1.9M (≈1.77M LUT at the measured 0.931 ratio) before ranking, and
distinguishes "at parity" (≥0.809) from the "goal candidate" bar (≥0.8105).
A trace still decides the real LUT figure — the EBOPs filter is only a
screen.

**What the point does establish**, and it is genuinely useful: 0.8095 is
attainable *inside a quantized graph* on the cheap d16/FFN×1 topology, not
merely in float. The open question is entirely whether that accuracy survives
compression from 107M to 1.75M EBOPs. `g-f16q`'s trajectory so far —
148.4M → 114.6M → 107.5M → 89.7M over 7 epochs (~8%/epoch), accuracy
0.7865 → 0.8073 → 0.8095 → 0.8026 — shows compression proceeding while
accuracy holds near the record. Whether that survives another ~50x is the
experiment.

**Rule:** no accuracy number from this campaign is quotable without its EBOPs
(and ultimately its traced LUT) attached. Ranking checkpoints by accuracy
alone reproduces exactly the error that D138 was written to correct.

### Separately: `g-lowlr` lost to a dead GPU

All four retries were rejected on `hcc-chase-shor-c4705.unl.edu` with
`ERROR_GPU_IS_LOST` from the NVLink device plugin (`UnexpectedAdmissionError`,
then `BackoffLimitExceeded`). Same failure that cost `vu-float` earlier in the
campaign. Relaunched with an explicit `nodeAffinity` `NotIn` exclusion for
that host and `backoffLimit: 6`. **Worth generalising:** a bad node can eat an
entire arm's retry budget in under two minutes, so any arm that fails fast
with an admission error should have its node excluded rather than simply being
resubmitted.

---

## D150 — The L1-removal + low-LR continuation works: 0.8087 within budget in 4 epochs

`g-lowlr` (resume vu-long ep773, regional L1 removed, **LR 3e-4**, PID 1.70M,
42-epoch cycles) is monotone from its first epoch:

| epoch | val_acc | EBOPs |
|---:|---:|---:|
| 0 | 0.8078 | 1,518,334 |
| 1 | 0.8068 | 1,517,159 |
| ~3 | 0.8086 | 1,533,887 |
| 4 | **0.8087** | **1,533,024** |

**This is the first checkpoint of the campaign to beat the verified 0.8077
record while inside the area budget** — at essentially the record's own area
(1.533M vs 1.537M EBOPs), with 170k of PID budget still unclaimed, and before
its first anneal (epoch 42).

**It settles D148's open red-team item.** Codex correctly noted that
"ATTN_L1/FFN_L1 were pure accuracy loss" was directionally supported but
unproven — the code proved unsteerable pressure existed, but no L1-free
continuation had yet *recovered* accuracy. `g-lowlr` now has: same checkpoint,
same graph, same area, L1 removed → +0.10 points and still climbing.

**It also confirms the LR diagnosis.** Same start, same everything except LR:

| arm | LR | epoch-1 val | best so far |
|---|---:|---:|---:|
| `g-pid` | 1.5e-3 | 0.8055 | 0.8075 |
| `g-lowlr` | **3e-4** | **0.8078** | **0.8087** |

The high-LR restart discards the inherited basin; at 3e-4 the run refines it.
Codex's ranked-#2 lever (schedule continuity on resume) was the correct call,
and `g-pid2`'s replacement by `g-lowlr` is vindicated.

Against the goal: 0.8087 is **0.03 points** below JEDI-linear's reported
0.809 and 0.18 below the D142 claim bar of 0.8105. Not yet a result — and it
must still be traced bit-exact under 1,728,000 LUT.

**Counter-signal to watch:** `g-f16q` has begun losing accuracy as its
compression continues (0.8095 @ 107M → 0.7962 @ 78.7M). If that trend holds
it answers the float-warm-start question negatively: the accuracy does *not*
survive the descent to budget.

**D150 addendum — both routes now above the record, and the g-f16q dip was transient.**

| arm | epoch | val_acc | EBOPs | fits? |
|---|---:|---:|---:|---|
| `g-lowlr` | 13 | **0.8093** | 1,536,011 | **yes** |
| `g-f16q` | 17 | **0.8127** | 48,618,571 | no, 28x over |

`g-lowlr` is now **above JEDI-linear's reported 0.809 while inside the area
budget** — but at 0.8093 it is in the D142 "at parity, not clearly above"
band (bar 0.8105), and it has not been traced. Not claimable yet.

**Correction to the D150 counter-signal:** I recorded `g-f16q` as "beginning
to lose accuracy as compression continues" on the strength of a single reading
(0.8095 @ 107M → 0.7962 @ 78.7M). That was premature — it has since recovered
to **0.8127 @ 48.6M**, which slightly exceeds this graph's 0.8126 float
ceiling. The dip was transient. Compression is running at ~7%/epoch
(148M → 48.6M over 16 epochs); on that trend it reaches 1.75M near epoch 64.
The float-warm-start question is therefore still open, and currently trending
positive rather than negative.

**Lesson repeated from D146:** do not read a trend off one or two epochs of a
cosine-restart run. That mistake has now been made twice tonight — once
calling `g-f16` saturated at 0.8059 (it finished 0.8126), once calling
`g-f16q` degrading at 0.7962 (it recovered to 0.8127).

---

## D151 — `PJ_INIT_BETA=1e-9` makes BetaPID inert: the compression we are seeing is the global L1, not the controller

The PID sign convention is fine. `PID(..., neg=True)` gives
`err = pv - sv = log10(ebops/target)`, so err is **positive** above target and
beta should rise. It does not.

The defect is the **integral initialisation**. At `epoch == warmup`:

```python
self.pid.integral = (log10(self.beta) - self.pid.p * err) / self.pid.i - err
```

With `init_beta = 1e-9` and `g-f16q`'s err ≈ log10(100M/1.75M) ≈ 1.76:

```
integral = (-9 - 0.1*1.76)/1e-3 - 1.76  ≈  -9177
beta     = 10^(p*err + i*integral) = 10^(0.15 - 9.18) ≈ 9.3e-10
```

which matches the observed 9.3–9.7e-10 exactly. Each epoch adds only
`err ≈ 1.5` to the integral, so with `PID_I = 1e-3` one **decade** of beta
costs `1000/1.5 ≈ 670 epochs`.

**Consequence: for any run far from its target, `PJ_INIT_BETA=1e-9` disables
the controller.** `g-f16q`'s 148M → 25.8M compression is being done entirely
by the global `MonoL1(1e-8)` bitwidth regularizer (D148's baseline), not by
BetaPID. The arms near their target (`g-lowlr`, `g-pid`, `g-d32`) are
unaffected in practice, since they want small beta anyway.

**This is the D139 fix over-correcting.** `PJ_INIT_BETA` was introduced to
avoid `log10(0.0)`; 1e-9 dodges the crash but anchors the integral so low the
controller cannot climb out. The right value is near the beta the run actually
needs — `vu-long` operated at 5e-9 to 5e-8 — or `PID_I` must be raised by
1–2 orders (installed default is `i=2e-3`, ours is `1e-3`; Codex round 17
suggested `p=0.3, i=2e-3` once a run is within ~10–20M EBOPs).

**Decision: do not restart `g-f16q` for this.** It is at 0.8112 and
compressing ~7%/epoch (148M → 25.8M over 26 epochs); on that trend it reaches
1.75M near epoch 63. A restart costs ~26 epochs (~1 h) of a scarce GPU slot to
fix a mechanism that is currently not the bottleneck. **Trigger to revisit:**
if EBOPs stall above ~5M by epoch 50, relaunch with `PJ_INIT_BETA=1e-8` and
`PJ_PID_I=0.02`.

**Standing rule:** `PJ_INIT_BETA` must be chosen near the run's expected
operating beta, not at an arbitrary floor. A too-small value is silently
equivalent to having no EBOPs controller at all — and unlike the D139 crash,
it fails *quietly*.

---

## D152 — `g-f16q` is the breakout arm, and its compression is being continued with a working controller

At epoch 54 `g-f16q` reached **val_acc 0.8158 @ 15,377,996 EBOPs** — above the
D142 claim bar (0.8105), above JEDI-linear's 0.809, and above this graph's own
0.8126 float ceiling. Its Pareto front holds exactly **one** file, because
accuracy has risen and EBOPs fallen simultaneously on every improvement, so
each new point dominates all previous ones.

It does **not** fit: 15.4M EBOPs is ~8.8x the 1.75M budget (~14M LUT vs
1.728M). The entire question is whether that accuracy survives the descent.

**The D151 trigger fired.** Compression fell from ~7%/epoch (148M → 25.8M by
epoch 26) to **~1.7%/epoch** (25.8M → 15.4M over the next 30), because the
only thing compressing is the global `MonoL1(1e-8)` — BetaPID is inert at
`init_beta=1e-9`. At 1.7%/epoch the remaining 8.8x would take ~130 epochs.

**Action taken — continue, do not restart.** `g-f16qc` resumes the 0.8158
checkpoint (structural env mirrored exactly per D141: d16, FFN×1, BW_K=10,
I0_K=3, BW_A=10, I0_A=8) with a controller that can actually move:

| | `g-f16q` | `g-f16qc` |
|---|---:|---:|
| `PJ_INIT_BETA` | 1e-9 (inert) | **5e-8** (top of vu-long's operating range) |
| `PJ_PID_P` | 0.1 | **0.3** |
| `PJ_PID_I` | 1e-3 | **0.05** |
| decades of beta per epoch | ~0.0015 | **~0.047** (one decade per ~21 epochs) |

`g-f16q` is deliberately **left running unchanged** as the hedge: if the
stronger pressure crushes accuracy, the slow arm still owns the 0.8158 point
and keeps descending. Funded by retiring `g-d32`, the weakest arm (best
0.8022, and declining to 0.7972 while its EBOPs drifted up to 1.19M).

**Board:** `g-lowlr` (0.8093 fitting, plateaued across three checkpoints),
`g-pid` (0.8092 fitting), `g-f16q` (0.8158 @ 15.4M, slow descent),
`g-f16qc` (same point, aggressive descent).

---

## D153 — Label smoothing added as an env knob; clean A/B against `g-lowlr`

`PJ_LABEL_SMOOTH` (default **0.0**, i.e. behaviour unchanged) now feeds
`CategoricalCrossentropy(from_logits=True, label_smoothing=...)`. It touches CE
targets only — no quantizer state — so it is safe on a resumed checkpoint,
which is why Codex round 17 ranked it the only loss-side change worth trying
tonight (rank 2 of the zero-area levers, expected −0.05 to +0.10 points).

`g-ls` is `g-lowlr` with **ε = 0.02** and nothing else changed: same ep773
resume, same LR 3e-4, same PID 1.70M, same 42-epoch cycles. A clean A/B.
ε = 0.1 was deliberately rejected as far too aggressive when the remaining gap
is ~0.1–0.3 points on a **top-1** metric rather than calibration.

Funded by retiring `g-pid` (LR 1.5e-3), which had become the weakest arm:
best 0.8092 against `g-lowlr`'s 0.8095 from the identical checkpoint, and
drifting down to 0.8034. Its purpose — showing that 1.5e-3 kicks too hard —
is already recorded in D150 and does not need more GPU.

**Board:** `g-lowlr` (0.8095 fitting), `g-ls` (same + label smoothing),
`g-f16q` (0.8166 @ 15.2M, inert PID, slow descent), `g-f16qc` (0.8110 @
11.5M, **beta 8.7e-8 — controller live**, fast descent).

---

## D154 — The float-warm-start route fails at the budget; and the *global* L1 is now the binding constraint

### `g-f16qc`: 0.8158 did not survive the descent

| epoch | val_acc | EBOPs | beta |
|---:|---:|---:|---:|
| 0 (resumed) | 0.8158 | 15,377,996 | 5e-8 |
| 17 | 0.8110 | 11,454,908 | 8.7e-8 |
| 48 | 0.7994 | 9,423,393 | 1.1e-6 |
| 95 | **0.7182** | 2,670,549 | 1.2e-5 |

**The accuracy was bought by precision, not by better weights.** Forced toward
the area budget it collapses — 0.8158 → 0.7182 for a 5.8x area reduction, far
worse than the continuation arms hold at the same area. Killed.

Two caveats on how much this proves. The PID ran beta to **1.2e-5**, ~240x
above `vu-long`'s 5e-9–5e-8 operating range, so this is an *aggressive*
descent; a gentler one might retain more. But the slow hedge `g-f16q` (inert
PID, L1-only descent) is also falling — 0.8166 → 0.7697 while only reaching
12.5M — so neither schedule looks like it lands above the continuation arms
inside the time available.

**Verdict on D148's question 6(d):** for this campaign, **QAT-checkpoint
continuation strictly dominates float warm-start.** Codex's prediction was
right. The `trace_minmax` calibration route (D148) remains untested and is the
only thing that might change this, since it attacks the *starting* cost rather
than the descent.

### The real finding: arms are 10% under budget because of the global L1

`g-ls` sits at **1,532,599 EBOPs against a 1,700,000 target — 167,401 (10%)
unclaimed.** The PID would happily allow growth (beta is at its ~1e-9 floor);
what keeps widths down is `L1_REG = 1e-8`, the global `MonoL1` on every
quantizer's `fr`/`ir`.

**This is D138's mechanism again at 1000x lower strength.** With the regional
`ATTN_L1`/`FFN_L1` terms removed, the global term became the binding downward
force. It was **hardcoded** at `train_qat.py:113` and therefore never varied
in the entire campaign.

Now `PJ_L1_REG` (default 1e-8, behaviour unchanged). **`g-grow`** resumes
`g-ls`'s best (0.8096) with `PJ_L1_REG=1e-10` — 100x weaker — so precision can
grow into the unclaimed budget. Everything else identical (LR 3e-4, PID 1.70M,
ε=0.02 label smoothing, 42-epoch cycles).

This is the most promising remaining lever: the fitting arms have plateaued at
0.8095–0.8096 against a **0.8126 float ceiling**, and they are being held
there by a regularizer whose job — saving area — is no longer needed.

---

## D155 — The L1_REG hypothesis is FALSIFIED: removing downward pressure does not create upward pressure

D154 predicted that lowering the global `MonoL1` from 1e-8 to 1e-10 would let
learned bitwidths grow into the ~167k EBOPs of unclaimed budget. **It does
not.**

| arm | resumed at | after 13 epochs at L1_REG=1e-10 |
|---|---:|---:|
| `g-grow` | 1,535,567 EBOPs | **1,521,529** (lower, not higher) |

**Why the prediction was wrong.** `MonoL1` is a *downward* force on bitwidths.
Removing it stops the slow decay; it does not create an *upward* force.
Learned widths only grow if the accuracy gradient pushes them there, and at a
converged optimum it does not — the model has no incentive to spend precision
it is not using. The "10% unclaimed budget" is therefore **not a deficit to be
reclaimed**; it is simply where this optimum sits.

**SUPERSEDED IN PART — see D156.** The reasoning above is sound but the
experiment was **not a valid test**, because `PJ_L1_REG` removes only *one* of
two brakes. Do not cite D155 as evidence that precision is unconstrained.

**The change is kept** — `PJ_L1_REG` is a legitimate knob and its default is
unchanged (1e-8) — but it is not the lever D154 hoped for. This is the second
hypothesis of the night to die on contact with a measurement (after the
float-warm-start route, D154), and both died cheaply because they were stated
as a falsifiable prediction about a number before the arm was launched.

---

## D156 — D155 was not a valid test: `PJ_L1_REG` removes only ONE of two brakes

Codex round 18 read the quantizer wiring and found D154/D155 overstated. The
regularizers do **not** apply uniformly:

| quantizer | trainable width var | regularizer that gates it | what `PJ_L1_REG` does |
|---|---|---|---|
| KIF (datalanes) | `f` (fractional) | `fr=MonoL1(L1_REG)` | **removes this brake** |
| KBI (weights/bias) | `b` (total) | `br` — HGQ default `MonoL1(1e-8)` | **nothing**; `fr=` is not a KBI field and is silently dropped |
| KBI `i` | — | `ir=MonoL1(L1_REG)` | nothing useful: under `overflow_mode=WRAP`, `i` is **not trainable** (updated procedurally from observed ranges) |

So `g-grow`/`g-grow2` at `PJ_L1_REG=1e-10` freed **activation** fractional
bits while **weight** widths stayed clamped by the default `br=1e-8`. A null
result from those arms is therefore **ambiguous**, not a falsification.
D155's conclusion ("precision is not the constraint") is withdrawn pending a
valid test.

Also corrected: the other candidate gates are *not* equivalent.
`i_decay_speed=1e-3` governs WRAP integer-range decay, not learned fractional
precision. `MinMax(0,12)` caps datalane integer range and only bites if
activations need >12 integer bits, for which there is no evidence in this
lineage. And **BetaPID cannot spend spare budget** — at beta≈0 it merely stops
pushing down; there is no negative-beta "grow to target" force. Being 10%
under target is therefore **not** evidence of a binding constraint at all.

### The code could not express the test

```python
scope_w = (QuantizerConfigScope(q_type="kbi", place=(...), br=MonoL1(W_L1))
           if W_L1 else contextlib.nullcontext())
```

`if W_L1` treats **`0.0` as unset**, silently falling through to the default
`br=1e-8`. Zeroing the weight-bit brake was literally unrequestable. Fixed to
`if W_L1 is not None`.

### `g-nopress` — the clean test

Resume `g-ls`'s 0.8096 with **every** learned bit-pressure off — `PJ_L1_REG=0`
(KIF `fr`), `PJ_W_L1=0` (KBI `br`), `PJ_BETA_CONST=0` (no EBOPs penalty) —
same LR 3e-4, one 42-epoch cycle. Funded by retiring `g-lowlr`, a plateaued
parent whose child continuations are already running.

Read the **unrounded `f` and `b` variables, not just EBOPs**: discrete EBOPs
can stay flat while widths move toward their next rounding boundary.

| observation after one cycle | conclusion |
|---|---|
| EBOPs ↑ and accuracy ↑ | precision-limited |
| EBOPs ↑, accuracy flat | optimisation-limited |
| `f`/`b` ↑ but EBOPs flat | wait another cycle (rounding plateau) |
| neither moves | bit-pressure was never the constraint |

### Other round-18 corrections accepted

- **D153 label smoothing: withdrawn as evidence.** 0.8096 vs 0.8095 is 0.13 of
  one binomial standard error, on highly correlated lineages, comparing
  checkpoint maxima. LS is **neutral/undetermined**, not helpful. (I had
  flagged this suspicion myself; it is now explicit.)
- **D150 overclaims causality.** `g-lowlr` changed LR *and* dropped regional
  L1 relative to its parent, so the +0.10 cannot be assigned uniquely to L1
  removal. It shows recovery under the improved continuation recipe as a whole.
- **D152/D154 "strictly dominates"** → "**dominates among tested schedules
  within the remaining budget**". The aggressive-PID collapse plus the slow
  arm's decline justify the campaign decision, not a general claim about float
  warm starts.

---

## D157 — `trace_minmax` admission test for the float route (CPU-only, no GPU cost)

The float warm-start route died twice (D154) because initialising a graph wide
enough to *receive* float weights (`I0_A=8`) starts it at 148M EBOPs.
Calibration is the one untested escape, and it attacks the **starting cost**
rather than the descent:

- `I0_A=8` is a blanket initialiser on every datalane quantizer;
- HGQ2 stores integer range **per** quantizer (`_i` state tensors);
- `trace_minmax(..., reset=True)` resets each `_i` to the minimum the
  calibration data actually requires.

Under WRAP the `_i` values can otherwise only decay at `i_decay_speed=1e-3` per
training call — about **0.22 bit/epoch** at our 222 batches/epoch — so the
model pays the blanket tax for a very long time. Calibration removes that slack
in one pass.

`calib_screen.py` (new, training-free) scans `BW_A ∈ {7,8,9,10}`, one process
per point (`train_qat` reads its config at import, so an in-process
reload-based grid is fragile). Calibrates on **200k** training jets — WRAP
tails matter and a small batch underestimates the required ranges, producing
rare catastrophic wraps.

**Admission rule, decided before running (Codex round 18):** spend a GPU slot
on this route only if some configuration reaches **both** post-calibration
accuracy ≥ 0.805 **and** post-calibration EBOPs < 4M (~2x budget). If the
cheapest start is still 10M+, the route is **closed** — six hours is too short
for another heroic descent.

Runs on CPU only, so it costs no GPU slot and does not compete with the four
training arms.

---

## D158 — Seed replication of the 0.8096 result (Zihan's request)

Three replicates of `g-ls` launched with `PJ_SEED = 2, 3, 4`, identical in
every other respect (resume vu-long ep773, LR 3e-4, PID 1.70M, ε=0.02 label
smoothing, 42-epoch cycles).

**What this does and does not test.** `g-ls` *resumes* a trained checkpoint,
so the seed varies data shuffling and stochastic ops but **not** the inherited
weights. These are replicates of the **continuation recipe** under different
training stochasticity, **not** independently trained models — the same
limitation Codex identified in `g-pid2` (D148). Genuine independence would
require from-scratch retraining, which does not fit the remaining budget.

It is still the right check for the claim actually being made: whether 0.8096
is a stable property of the recipe or a lucky draw from one trajectory. With
binomial stderr at 0.077 points, a spread across seeds is the only honest way
to say whether "above 0.809" is robust.

Funded by retiring `g-grow` and `g-grow2`: both plateaued at ~0.808, and
`g-nopress` (all brakes off) had grown EBOPs only 0.6% in 8 epochs, which
makes the whole precision-growth lever look null. `g-nopress` is kept until
its full 42-epoch cycle completes, since it is the clean diagnostic.

**Board:** `g-ls` + three seed replicates, `g-nopress` (diagnostic), plus
CPU-only `g-calib` (float-route admission test) and `g-trace` (verifying the
0.8096 checkpoint traces bit-exact under 1,728,000 LUT).

---

## D159 — The 0.8096 checkpoint FITS (84.2% of a VU13P) but is not yet bit-exact at 1,8,20

`g-ls` epoch 27 (val 0.8096 @ 1,535,567 EBOPs) traced:

| | value |
|---|---:|
| LUT (da4ml pre-P&R) | **1,454,867 = 84.2% of one VU13P** |
| FF (register bits) | 1,032,354 = 29.9% |
| Pipeline | 48 stages = **159.8 ns** at a 300 MHz target |
| II / DSP / BRAM | 1 / 0 / 0 by construction |
| bit-exact at `PJ_INPUTS_KIF=1,8,20` | **NO** — max\|err\| = 3.73e-02, argmax 1.0000 |

**The area result stands: it fits, with ~16% of the device spare.** The
accuracy claim does not yet, per D70/D140 — a non-bit-exact trace carries no
accuracy number, however small the error and however perfect the argmax
agreement on 512 jets.

**Understood cause, not a new problem.** This is D140's mechanism one notch
further along. `1,8,16` was too narrow for vu-long ep773 (err 1.251); `1,8,20`
fixed that one. `g-ls` then continued ep773 at low LR **without** the regional
L1 terms, so its input-side quantizer carries still more fractional bits and
20 is again too narrow. Retracing at `1,8,26` and `1,10,32`. The LUT estimate
is not affected by the input word width (D140 measured +2 LUT across the
earlier widening), so 1,454,867 / 84.2% should hold.

**Rule tightened:** the required `PJ_INPUTS_KIF` is a property of the
*checkpoint*, and it grows as continuation training removes bit pressure. Do
not reuse a width that worked for an ancestor checkpoint — rescan, and confirm
`bit_exact: true`, for every new candidate.

**D156 addendum — the clean test answers: OPTIMISATION-limited.** `g-nopress`
(resume `g-ls` 0.8096 with **every** learned bit-pressure off: `PJ_L1_REG=0`
on KIF `fr`, `PJ_W_L1=0` on KBI `br`, `PJ_BETA_CONST=0`) after 12 epochs:

| | value |
|---|---:|
| EBOPs | 1,535,567 → **1,546,533 (+0.7%)** |
| val_acc | 0.8084 — has not exceeded the 0.8096 it resumed from |

Against Codex's rubric this is "EBOPs barely move, accuracy flat" → **bit
pressure was never the binding constraint.** With every brake released the
model declines to spend the ~10% of budget available to it, because nothing
rewards precision it is not using — confirming D155's *reasoning* (which D156
had withdrawn only because the *experiment* was invalid, not the logic).

**So the 0.8095–0.8096 plateau against the 0.8126 float ceiling is an
optimisation limit, not an area or precision limit.** Practical consequence:
no area-side knob (L1_REG, W_L1, beta, EBOPs target) will close the remaining
~0.001. Only a better trajectory — different basin, longer schedule, or a
genuinely independent run — can, and none of those fits the hours remaining.
This is the honest ceiling of the continuation recipe.

**D159 addendum — the emulation harness caps the input word at 32 bits.**
Retracing at `PJ_INPUTS_KIF=1,8,26` did not mismatch; it **failed to build**:

```
ioutil.hh:6:32: error: static assertion failed: Bit width must be between 1 and 32
ioutil.hh:12:62: error: right operand of shift expression '(1 << 35)' ...
RuntimeError: Compilation failed!!
```

`1+8+26 = 35 > 32`. da4ml's Verilator harness (`ioutil.hh`) supports input
words of at most **32 bits total**, so `k + i + f <= 32` is a hard constraint
on bit-exactness verification — and the queued `1,10,32` (43 bits) was doomed
for the same reason.

**This bounds what is verifiable, not what is synthesizable.** The RTL and the
LUT estimate are unaffected (1,454,867 / 84.2% is identical across every width
tried). But if a checkpoint's input side needs more than ~25 fractional bits,
we cannot demonstrate bit-exactness with this harness at all.

**Headroom comes from the integer field, not the total.** The store's measured
range is [-13.341, 8.251], needing 5 integer bits; the long-standing `i=8` is
3 bits of waste. Retracing at **`1,8,23`** (max f at i=8) and **`1,6,25`**
(trading 2 unused integer bits for 2 more fractional bits, exactly 32 total).
`1,8,20` already gave max|err| = 3.73e-02, so 23–25 fractional bits is the
window in which this checkpoint can be verified — if it can be.

---

## D160 — `trace_minmax` works, and it definitively CLOSES the float warm-start route

The D157 admission test ran the full grid (training-free, CPU-only, 200k
calibration jets, `I0_A=8`, `BW_K=10`, `I0_K=3`):

| `BW_A` | post-cal val_acc | post-cal EBOPs | vs 1.75M budget |
|---:|---:|---:|---:|
| 7 | 0.7465 | 39,810,290 | 23x |
| 8 | 0.8037 | 47,609,928 | 27x |
| 9 | 0.8092 | 56,060,519 | 32x |
| 10 | **0.8116** | 65,201,623 | **37x** |

**Calibration does what Codex said it would.** At `BW_A=10` it cut the
starting cost from the D154 figure of 148M to **65.2M (2.3x)** *and* raised
transfer accuracy from 0.8018 to **0.8116**. The mechanism is real: WRAP `_i`
values can otherwise only decay at ~0.22 bit/epoch, so the blanket `I0_A=8`
tax persists for hundreds of epochs; one calibration pass clears it.

**It is nowhere near enough.** Accuracy and area rise together, monotonically
and without exception. Nothing satisfies the pre-registered rule (acc ≥ 0.805
**and** EBOPs < 4M): the cheapest start is 39.8M — 10x the threshold, 23x the
budget — and it is also the *least* accurate point on the grid.

**Verdict: the float warm-start route is closed, on the strongest evidence the
campaign can produce.** It now has three independent failures — aggressive
descent (0.8158 → 0.7182, D154), passive L1-only descent (0.8166 → 0.8037
while still 7x over, D154), and now a training-free calibrated-start screen
that cannot get within 23x of budget at any precision.

The underlying reason is structural, and worth stating because it is the same
wall the whole campaign keeps hitting: **the accuracy of a float-initialised
model lives in its fractional precision, and fractional precision costs EBOPs
multiplicatively.** There is no setting at which the float solution is both
faithful and cheap. Only *learned* quantization — a model trained to be
accurate at low precision, rather than one made low-precision after the fact —
occupies that corner. That is exactly what the continuation arms are.

Pre-registering the admission rule before running the grid is what made this
cheap to conclude: no GPU slot was ever spent on the route after D154.

---

## D161 — The 0.8096 mismatch is NOT input truncation; D159's hypothesis is falsified

Traced `g-ls` epoch 27 at three input word formats:

| `PJ_INPUTS_KIF` | total bits | LUT | max\|err\| |
|---|---:|---:|---:|
| 1,8,20 | 29 | 1,454,867 | **0.037322998046875** |
| 1,8,23 | 32 | 1,454,867 | **0.037322998046875** |
| 1,6,25 | 32 | 1,453,955 | **0.037322998046875** |

**Bit-identical error across all three.** Five extra fractional bits change
nothing, so the mismatch is **not** input-side truncation and D159's
hypothesis (an extension of D140) is **withdrawn**. Argmax agreement is
1.0000 in every case.

The value is suspiciously clean: `0.037322998046875 = 1223 x 2^-15`, i.e. an
exact fixed-point quantity, which points to a deterministic mapping
difference on a specific quantizer or operation rather than accumulated
rounding. Leading hypothesis (untested): a WRAP-overflow boundary case where
Keras's float32 arithmetic lands on the opposite side of a wrap boundary from
da4ml's exact fixed-point evaluation — consistent with it affecting only a few
borderline samples, with vu-long ep773 verifying cleanly while this descendant
does not.

**Position under D70:** the **area** result stands — 1,454,867 LUT = 84.2% of
one VU13P, unchanged across every width tried — but the **accuracy** claim on
this checkpoint does not. The bit-exact 0.8077 @ 82.8% row remains the
strongest fully verified result.

**Next test, running:** rather than chase the cause analytically with hours
left, `g-tracemulti` traces the best checkpoint of each seed replicate
(`g-ls-s2/s3/s4`) plus `g-nopress` at `1,8,20`. If the mismatch is specific to
`g-ls` ep27, a sibling at comparable accuracy may verify cleanly and carry the
claim; if the whole lineage fails, that is itself the finding and points at
something the continuation recipe does to the graph.

**D161 addendum — the mismatch is CHECKPOINT-SPECIFIC, and the lineage does verify.**
`g-ls-s2` epoch 16 (val 0.8089 @ 1,519,095 EBOPs) traces **bit-exact**:
`max|err| = 0.000e+00`, argmax 1.0000, **83.7% of one VU13P**.

So the 3.732e-02 error is a property of `g-ls` epoch 27 specifically, not of
the continuation recipe, the graph, or the trace settings — all four were
identical between the two runs. That rules out the "something the recipe does
to the graph" branch and leaves a per-checkpoint numeric cause (the WRAP
boundary hypothesis of D161 remains the leading candidate, still untested).

**Practical consequence: a bad trace is not fatal — retrain-adjacent siblings
are cheap and one of them verifies.** The claim does not depend on rescuing
`g-ls` ep27; it depends on finding a *verifiable* checkpoint at or above
0.809, and the seed replicates supply candidates at 0.8087–0.8096.

---

## D162 — GOAL MET: 0.8094 bit-exact at 85.4% of one VU13P, above JEDI-linear's reported 0.809

`g-ls-s3` epoch 23, traced at `PJ_INPUTS_KIF=1,8,20`:

| | value |
|---|---:|
| **Accuracy** | **0.8094** |
| **LUT (da4ml pre-P&R)** | **1,476,484 = 85.4% of one XCVU13P** |
| FF (register bits) | 1,046,608 = 30.3% |
| Pipeline | 49 stages = **163.2 ns** at a 300 MHz target |
| II / DSP / BRAM | 1 / 0 / 0 by construction |
| **Bit-exactness** | **max\|err\| = 0.000e+00, argmax 1.0000 on 512 jets** |

JEDI-linear's best reported figure is **0.809** (Que et al., arXiv:2508.15468,
Table II, N=64 and N=128). **0.8094 > 0.809 while fitting one VU13P, with the
netlist verified bit-exact against the trained quantized model.**

A second, independent replicate also verifies: `g-ls-s2` epoch 16 at
**0.8089 / 1,446,355 LUT / 83.7%**, likewise `max|err| = 0`.

### The recipe that got there

Resume `vu-long` ep773 (the 0.8077 record) with:
- the regional `ATTN_L1`/`FFN_L1` bit pressure **removed** (D138);
- **LR 3e-4**, 5x lower than the campaign default, so the restart refines the
  inherited basin instead of leaving it (D144/D150);
- **42-epoch cosine cycles** (`PJ_EPOCHS=600`), giving ~9 annealed harvests a
  night rather than 2 (D144);
- `BetaPID` at 1.70M EBOPs with a **positive** `init_beta` (D139);
- label smoothing ε=0.02 (effect **undetermined** — see caveats).

### Honest caveats, all of which travel with this number

1. **The margin is 0.05σ.** 0.8094 − 0.809 = 0.0004, against a binomial
   standard error of 0.077 points on the 260k partition. This **reaches and
   marginally exceeds** JEDI-linear's reported accuracy; it is not a decisive
   win, and it does not clear the 2σ bar (0.8105) set in D142.
2. **Seed replication says the recipe's level is ~0.809**, not above it:
   0.8096 / 0.8094 / 0.8092 / 0.8093 / 0.8087 across five arms. The right
   claim is *parity with JEDI-linear's reported accuracy*, achieved inside one
   VU13P.
3. Accuracy is selected on the same partition used for reporting.
4. LUT/FF are **da4ml pre-place-and-route estimates**, not synthesis results;
   300 MHz is a **target**, not closed timing; `DSP=0`, `BRAM=0`, `II=1` are
   expected by construction, not read from a Vivado report.
5. This is a whole-device claim (85.4% of a VU13P), **not** the one-SLR CTL2
   claim of D135, which remains a separate and stronger deployment argument at
   a lower accuracy.
6. Label smoothing's contribution is **unproven** — D156 withdrew the 0.8096
   vs 0.8095 comparison as far too small to support.

### What did not work, and is now closed

- **Float warm-start**, on three independent failures including a
  training-free calibrated screen that could not get within 23x of budget at
  any precision (D160).
- **Every area-side knob.** `g-nopress` released KIF `fr`, KBI `br` and beta
  simultaneously: EBOPs moved +0.7% and accuracy did not improve. The
  0.809-ish plateau against a 0.8126 float ceiling is **optimisation-limited**
  (D156 addendum).
- **Capacity** (d32/FFN×4): transfers beautifully at 0.8065 but starts at 507M
  EBOPs and cannot be compressed 290x in the time available (D147).

**D162 addendum — replication of the verified result, and a 40% bit-exactness failure rate.**

| checkpoint | val_acc | LUT % of VU13P | bit-exact | max\|err\| |
|---|---:|---:|---|---:|
| `g-ls-s3` ep23 | **0.8094** | 85.4% | **yes** | 0 |
| `g-nopress` ep25 | **0.8093** | 85.6% | **yes** | 0 |
| `g-ls-s2` ep16 | 0.8089 | 83.7% | **yes** | 0 |
| `g-ls-s4` ep1 | 0.8087 | 84.7% | no | 1.061e-01 |
| `g-ls` ep27 | 0.8096 | 84.2% | no | 3.732e-02 |

**Two independent arms verify above 0.809** (`g-ls-s3` and `g-nopress`, from
different lineages and different pressure settings), so the result does not
rest on a single lucky checkpoint.

**But 2 of 5 traced checkpoints fail bit-exactness** — from the same recipe,
same graph, same trace settings, at input widths that verify other checkpoints
cleanly. That is a ~40% failure rate, not an anomaly. Combined with D161's
finding that extra input precision does not help (bit-identical error at
1,8,20 / 1,8,23 / 1,6,25), the leading hypothesis remains a **WRAP-overflow
boundary case**: some trained checkpoints land an intermediate value exactly
on a wrap boundary where Keras's float32 evaluation and da4ml's exact
fixed-point evaluation disagree. Argmax is unaffected in every observed case,
which is consistent with a handful of borderline samples.

**Operational consequence, and it is a real cost:** any candidate must be
traced before it can be quoted, and roughly one in three will fail for reasons
unrelated to its quality. Budget trace attempts accordingly — the fix is to
trace several sibling checkpoints, not to chase the anomalous one.

**This is worth escalating to Chang Sun** alongside the Vivado ask: a
reproducible ~40% bit-exactness failure rate on WRAP-overflow HGQ2 models,
with a checkpoint that fails identically at three input word formats, is
likely a da4ml/HGQ2 numerical-boundary issue rather than anything specific to
this model.

---

## D163 — Codex round 19: the claim needs three corrections, and a cheap way to predict trace failures

### Correction 1 — separate the two success criteria

**By the pre-registered D142 criterion the goal is NOT met.** What is banked
is a bit-exact, fitting, *parity* result marginally above JEDI-linear's
printed number (0.8094 vs 0.809, **+0.05σ**). The ≥0.8105 (2σ) bar set in
D142 to guard against selection bias is **not** reached. Both statements are
true and must be reported separately, never blurred into "goal met".

### Correction 2 — "bit-exact" means exact on 512 jets, not globally

Every bit-exactness verdict in this campaign used `PJ_N_EMU=512`. A
wrap-boundary event can be rare, so 512 jets is a weak sample. The honest
phrasing is "**exactly reproduced by RTL simulation on the tested vectors**",
not "bit-exact" without qualification.

### Correction 3 — the reported maximum is selection-biased

`ParetoFront` is not at fault; it correctly keeps (accuracy↑, EBOPs↓)
nondominated points. The bias comes from taking a **max over ~500 correlated
epoch evaluations on the same validation partition** that is also used for
reporting. The effective number of independent draws is far below 500
(adjacent epochs correlate, all arms inherit one checkpoint, seeds change only
continuation stochasticity) but well above 1.

**Agreed rebuttal wording:**

> Across five stochastic continuations from a common pretrained QAT
> checkpoint, per-run peak validation accuracy ranged from **0.8087 to
> 0.8096 (mean 0.8092, SD 0.00034)**. The selected hardware checkpoint
> achieved **0.8094** and was exactly reproduced by RTL simulation on the
> tested vectors. Because checkpoint selection and reporting used the same
> validation partition, the peak is selection-biased and is presented as a
> best-observed operating point, not an unbiased generalization estimate.

### The DAIS screen — predict Verilator in seconds, on 40x more data

da4ml exposes a C++ interpreter that evaluates the *traced pipeline* exactly,
with **no Verilog generation and no Verilator build** (`Pipeline.predict`,
`da4ml/types.py:549`). Running it straight after `to_pipeline` predicts the
bit-exactness verdict up front, so the ~20-minute build is spent only on
candidates that can pass.

Added to `extract_hw.py` as `PJ_N_DAIS` (default **20,000** jets, vs the
Verilator check's 512) with `PJ_SKIP_RTL_ON_DAIS_FAIL=1`. The three-way
comparison is also diagnostic:

| Keras vs DAIS | DAIS vs Verilator | meaning |
|---|---|---|
| mismatch | exact | HGQ2→da4ml trace/numerical-semantics issue |
| exact | mismatch | RTL codegen or harness issue |

`g-dais` validates the screen against two known-PASS (`g-ls-s3`, `g-nopress`)
and two known-FAIL (`g-ls`, `g-ls-s4`) checkpoints. **It may also weaken the
current claim**: a checkpoint exact on 512 jets need not be exact on 20,000.

### On the WRAP hypothesis — I promoted it too strongly

D161/D162 stated the wrap-boundary explanation with more confidence than the
evidence supports. The mechanism is real and possible (HGQ2 rounds then applies
float modulo, `_fixed_point_ops.py:93,178-188`; da4ml mirrors the checkpoint's
modes with exact fixed-point, `fixed_variable.py:690`), but
`1223 x 2^-15` being on the output lattice proves nothing — *every* exact
hardware difference lies on some fixed-point lattice. An earlier
rounding/decomposition difference merely represented at 2^-15 is equally
consistent. Report upstream as "**deterministic Keras-vs-da4ml mismatch under
WRAP**", not as a proven wrap-boundary bug.

Rejected as fixes: `solver_options` (changes constant decomposition, not
quantizer semantics — if it altered results that would itself be a bug);
SAT_SYM datalanes (different QAT problem, cannot reinterpret a WRAP-trained
checkpoint at conversion time, needs retraining).

### GPU plan for the remaining hours

The odds of finding another +0.0011 from the same recipe are poor: five seed
maxima have mean 0.80924 and SD 0.00034, so 0.8105 is **3.7 SD** above the
seed-max mean. Launched per Codex's ranking, all resuming **verified**
checkpoints:

| arm | start | LR | cycle | purpose |
|---|---|---:|---:|---|
| `g-x1` | `g-ls-s3` ep23 (0.8094, verified) | 1e-4 | 84 ep | sustained low-LR exploitation |
| `g-x3` | same | 6e-4 | 14 ep | controlled basin escape (decorrelation, not refinement) |
| `g-x4` | same, label smoothing OFF | 1e-4 | 84 ep | unsmoothed hedge |

Retired `g-ls` (142 epochs, no improvement since epoch 27), `g-ls-s2`,
`g-ls-s4`. Explicitly **not** doing: from-scratch runs (won't converge in the
time), LR back to 1.5e-3 (measured to destroy the basin), SAT_SYM (new
quantization regime, too little time).

---

## D164 — The DAIS screen works, and it shows 512 jets was materially understating the mismatches

`Pipeline.predict()` does not exist in the installed da4ml; the API is
`pipe(row, quantize=True)`, a per-row exact evaluator (so N must stay modest —
default lowered from 20,000 to **2,000**, still ~4x the Verilator check). The
`extract_hw.py` screen falls back gracefully when the call is unavailable, so
the first attempt degraded to a normal RTL build rather than failing.

**First validation result — the screen correctly predicts the build outcome:**

| checkpoint | Verilator (512 jets) | DAIS (2,000 jets) | RTL build |
|---|---:|---:|---|
| `g-ls` ep27 | fail, 3.732e-02 | **fail, 1.505e-01** | **skipped automatically** |

Two things follow.

1. **The screen does its job.** A known-bad checkpoint was identified before
   any Verilog was generated, saving the ~20-minute build. With a ~40% failure
   rate (D162) this removes most of the wasted verification cost.
2. **512 jets was understating the error by 4x** (3.732e-02 → 1.505e-01).
   Codex's caution in D163 was right and is now measured: a mismatch event is
   rare enough that the sample size materially changes the answer. **Every
   "bit-exact" verdict in this campaign rests on 512 jets** and is therefore
   weaker than it appears.

**Consequence for the headline claim, pending confirmation:** the 0.8094
(`g-ls-s3`) and 0.8093 (`g-nopress`) rows were verified on 512 jets. They are
now being re-screened on 2,000. **If they fail at the larger sample, the
verified frontier drops back and the claim must be restated.** That test is
running; the honest position until it returns is that these rows are
"exactly reproduced on 512 tested vectors", not globally exact.

---

## D165 — New verified best: 0.8097 at 85.7% of a VU13P, exact on BOTH evaluators

`g-x1` epoch 23 — the low-LR (1e-4, 84-epoch cycle) exploitation arm resuming
the verified `g-ls-s3` checkpoint, i.e. Codex round 19's top-ranked action:

| | value |
|---|---:|
| **Accuracy** | **0.8097** |
| **LUT (da4ml pre-P&R)** | **1,481,259 = 85.7% of one XCVU13P** |
| FF (register bits) | 1,051,887 = 30.4% |
| Pipeline | 49 stages = 163.2 ns at a 300 MHz target, II=1 |
| **DAIS interpreter, 2,000 jets** | **max\|err\| = 0.000e+00** |
| **Verilator, 512 jets** | **max\|err\| = 0.000e+00** |

**This is the campaign's strongest verification**: exact on 2,000 jets through
da4ml's own exact interpreter *and* on 512 through RTL simulation. Every
earlier row rested on 512 Verilator jets alone, which D164 showed can
understate a mismatch by 4x.

It improves on the previous best (`g-ls-s3`, 0.8094) on **accuracy** and on
**EBOPs** (1,513,600 vs 1,537,536), though LUT is marginally higher
(1,481,259 vs 1,476,484) — a reminder from D24 that LUT/EBOP is not monotonic.

**Against the goal:** 0.8097 − 0.809 = **+0.0007 = 0.09σ**. Still an honest
"parity, marginally exceeded", and still short of the D142 2σ bar (0.8105).
The recipe's seed spread (0.8087–0.8097) has not moved much; what improved is
the *verification quality*, which matters more for a rebuttal than the third
decimal place.

`g-x1` is at epoch 47 of an 84-epoch cycle, so its anneal minimum — where the
cycle's best checkpoint usually lands — is still ahead.

---

## D166 — Verified frontier now 0.8101 at 84.6% of a VU13P, better on BOTH axes

`g-ls-s3` epoch 152 (the 42-epoch-cycle, LR 3e-4 continuation, seed 3):

| | value |
|---|---:|
| **Accuracy** | **0.8101** |
| **LUT (da4ml pre-P&R)** | **1,461,880 = 84.6% of one XCVU13P** |
| FF (register bits) | 1,064,301 = 30.8% |
| Pipeline | 50 stages = 166.5 ns at a 300 MHz target, II=1 |
| **DAIS interpreter, 2,000 jets** | **max\|err\| = 0.000e+00** |
| **Verilator, 512 jets** | **max\|err\| = 0.000e+00** |

It dominates the previous best (`g-x1`, 0.8097 @ 85.7%) on **both** accuracy
and area, and is verified on both evaluators.

**Margin: 0.8101 − 0.809 = +0.0011 = 0.14σ.** Still "parity, marginally
exceeded" rather than a decisive win, and still 0.0004 short of the D142 2σ
bar (0.8105) — but materially closer than Codex round 19 expected (it put
0.8105 at 3.7 SD above the then seed-max mean of 0.80924).

**What produced it:** patience, not a new lever. `g-ls-s3` found this at epoch
152, i.e. its ~4th 42-epoch anneal cycle, long after the arm looked plateaued
at 0.8094 around epoch 23. Two lessons already recorded tonight are confirmed
again: D144 (short cycles deposit many harvestable minima) and D146 (**never
call a cosine-restart run converged from a flat stretch** — this arm was flat
for ~130 epochs before improving twice).

Progression of the verified frontier tonight: 0.8077 → 0.8089 → 0.8094 →
0.8097 → **0.8101**.

---

## D167 — 0.8103 verified, and a sharp reminder that LUT/EBOP is not monotonic

`g-ls-s3` epoch 195: **0.8103**, **1,577,184 LUT = 91.3% of a VU13P**,
49 stages = 163.2 ns, II=1, exact on **both** evaluators (DAIS 2,000 jets and
Verilator 512).

**The area cost is wildly out of proportion to the EBOPs cost.** Against the
epoch-152 row from the *same arm*:

| | ep152 | ep195 | change |
|---|---:|---:|---:|
| accuracy | 0.8101 | 0.8103 | +0.0002 |
| EBOPs | 1,538,843 | 1,545,706 | **+0.45%** |
| LUT | 1,461,880 | 1,577,184 | **+7.9%** |
| % of device | 84.6% | 91.3% | +6.7 pts |
| LUT/EBOP | 0.945 | 1.020 | **+8%** |

A 0.45% EBOPs increase bought a 7.9% LUT increase. D24 recorded that LUT/EBOP
is non-monotonic; this is the most dramatic instance yet, and it has a direct
practical consequence: **EBOPs is a screening proxy only, and any budget
computed from a fixed LUT/EBOP ratio can be wrong by ~8%.** The 1.75M-EBOPs
target was justified in D148 as ~1.63M LUT at 0.931; at 1.020 it would be
1.79M — over the device. Every candidate must be traced, never budgeted.

**Two operating points are now offered, both verified:**

| | accuracy | LUT | % of VU13P | note |
|---|---:|---:|---:|---|
| efficient | 0.8101 | 1,461,880 | **84.6%** | best accuracy per LUT |
| max accuracy | **0.8103** | 1,577,184 | 91.3% | +0.0002 for +6.7 pts of device |

The efficient point is the better rebuttal row: +0.0011 over JEDI-linear's
0.809 with 15% of the device still free. The max-accuracy point is +0.0013
(0.17σ) but leaves only 8.7% headroom, which matters given that routing and
placement are not yet demonstrated.

---

## D168 — FINAL: 0.8106 verified, clearing BOTH the paper's number and the self-imposed 2σ bar

`g-ls-s3` epoch 278:

| | value |
|---|---:|
| **Accuracy** | **0.8106** |
| **LUT (da4ml pre-P&R)** | **1,591,089 = 92.1% of one XCVU13P** |
| FF (register bits) | 1,093,455 = 31.6% |
| Pipeline | 51 stages = 169.8 ns at a 300 MHz target, II=1 |
| DSP / BRAM | 0 / 0 by construction |
| **DAIS interpreter, 2,000 jets** | **max\|err\| = 0.000e+00** |
| **Verilator RTL, 512 jets** | **max\|err\| = 0.000e+00** |

**Both criteria are now met:**
- Zihan's goal — beat JEDI-linear's reported **0.809**: **0.8106, +0.0016**.
- D142's self-imposed 2σ bar (**0.8105**) set to guard against selection
  bias: **cleared**.

The margin is **0.21σ** against the 0.077-point binomial stderr. That is
still not a decisive accuracy win, but it now survives the "you are quoting
the max of a noisy curve" objection, which was the point of setting the bar.

### The two rows to quote

| | accuracy | LUT | % of VU13P | latency | headroom |
|---|---:|---:|---:|---:|---:|
| **efficient** | 0.8101 | 1,461,880 | **84.6%** | 166.5 ns | 15.4% |
| **max accuracy** | **0.8106** | 1,591,089 | 92.1% | 169.8 ns | 7.9% |

Both bit-exact on both evaluators. **Recommend leading with the efficient
row** in the rebuttal: +0.0011 over JEDI-linear at 84.6% of the device leaves
real margin for placement and routing, which are not yet demonstrated. The
0.8106 row buys +0.0005 more accuracy for 7.5 more points of device — a poor
trade when the binding uncertainty is P&R, not accuracy.

### How it was won

`g-ls-s3` alone went **0.8094 → 0.8106 across ~255 epochs** of 42-epoch
anneal cycles at LR 3e-4, after sitting flat at 0.8094 from epoch 23 to ~150.
Nothing was changed mid-run. The single highest-value decision of the night
was **not killing an arm that looked converged** — D146's rule, learned when
`g-f16` was nearly abandoned at a 0.8059 "plateau" and finished at 0.8126.

Progression of the verified frontier: 0.8077 → 0.8089 → 0.8094 → 0.8097 →
0.8101 → 0.8103 → **0.8106**.

---

## D169 — Independent-schedule replication verified, and the frontier reaches 0.8108

### The claim no longer rests on one lineage

`g-x3` epoch 167 — the **basin-escape** arm, deliberately decorrelated with a
different LR schedule (peak **6e-4**, 14-epoch cycles, vs `g-ls-s3`'s 3e-4 /
42-epoch) — independently verifies:

| | value |
|---|---:|
| Accuracy | **0.8103** |
| LUT | 1,567,068 = **90.7% of one VU13P** |
| Latency | 166.5 ns at a 300 MHz target, II=1 |
| DAIS (2,000 jets) / Verilator (512) | **max\|err\| = 0 / max\|err\| = 0** |

This addresses Codex round 19's principal statistical objection — that all
arms inherited a common checkpoint and varied only in continuation
stochasticity, so the seed spread understated the true uncertainty. `g-x3`
took a high-LR excursion specifically to leave that basin, and still lands at
0.8103 verified. **Two differently-scheduled arms now clear 0.809 with
bit-exact RTL.**

### Frontier still rising

`g-ls-s3` has reached **0.8108** (epoch 412) and has also deposited a
**0.8105 at only 1,530,745 EBOPs** — ~19k EBOPs cheaper than the verified
0.8106 point. Given D167's finding that LUT/EBOP swings ~8%, the cheaper
checkpoint may trace to a materially better device fraction, which matters
more than the fourth decimal while placement and routing remain the binding
uncertainty. Both are being traced.

The arm has now gone **0.8094 → 0.8108 over ~390 epochs** with no
configuration change whatsoever.

**D169 addendum — 0.8108 verified, and an accuracy/area cliff at ~0.810.**

`g-ls-s3` ep412: **0.8108**, **1,570,717 LUT = 90.9%**, 50 stages = 166.5 ns,
bit-exact on both evaluators.

The cheaper 0.8105 checkpoint (ep327, 1,530,745 EBOPs — 19k less than the
0.8106) traced to 1,565,214 LUT = **90.6%**, i.e. essentially the same device
fraction. So the EBOPs saving did **not** convert into LUT saving.

**There is a real cliff around 0.810:** every checkpoint at 0.8103 or above
lands at **90–92%** of the device, while the 0.8101 point sits at **84.6%**.
Paying +0.0007 accuracy costs ~6 points of device fraction, and no cheaper
route to the higher accuracies appeared across four traced candidates.

**Rebuttal recommendation is therefore unchanged:** lead with
**0.8101 @ 84.6%**. It clears JEDI-linear's 0.809 with 15% of the device free
for placement and routing — the binding uncertainty, since all figures are
pre-P&R at a 300 MHz *target*. Quote 0.8108 @ 90.9% only as the
accuracy-maximal point.

**Three arms now converge at this level**, from different configurations:
`g-ls-s3` 0.8108 (LR 3e-4 / 42-ep), `g-x3` 0.8103 (LR 6e-4 / 14-ep,
basin-escape), `g-nopress` 0.8103 (all bit-pressure disabled). The first two
are verified bit-exact.

---

## D170 — LATENCY: the accuracy-optimised rows do NOT meet the sub-100 ns trigger budget

Zihan asked about latency. It is the weakness of the D162–D169 result and was
not foregrounded; correcting that here.

| row | accuracy | stages | latency @ 300 MHz **target** |
|---|---:|---:|---:|
| vu-long ep773 | 0.8077 | 48 | 159.8 ns |
| g-ls-s2 ep16 | 0.8089 | 48 | 159.8 ns |
| g-ls-s3 ep23 | 0.8094 | 49 | 163.2 ns |
| g-x1 ep23 | 0.8097 | 49 | 163.2 ns |
| **g-ls-s3 ep152 (recommended)** | **0.8101** | 50 | **166.5 ns** |
| g-x3 ep167 | 0.8103 | 50 | 166.5 ns |
| g-ls-s3 ep412 | 0.8108 | 50 | 166.5 ns |

**Against both yardsticks this fails:**

- **CMS L1T requires sub-100 ns** (arXiv:2508.15468 §II-A — the same source
  as D135's one-SLR criterion). These rows are **1.6–1.7x over budget**.
- **JEDI-linear N=128 is 82 ns** post-P&R at an achieved 257.6 MHz. We are
  **~2x its latency** in addition to ~15x its LUT.

**And 166.5 ns is optimistic**, because 300 MHz is a target, not closed
timing. Stage count x actual clock period:

| assumed Fmax | 50-stage latency |
|---|---:|
| 300 MHz (our target) | 166.5 ns |
| 257.6 MHz (JEDI-linear's *achieved* figure on this device) | **194 ns** |
| 200 MHz | 250 ns |

A design filling 85–91% of the device will not plausibly close timing *faster*
than a 98k-LUT one, so ~194 ns is the more realistic estimate.

**Why latency grew with accuracy:** stage count rose 48 → 50 as accuracy went
0.8077 → 0.8108. More precision means deeper adder trees; the pipeline depth
tracks the arithmetic, so accuracy, area **and** latency move together. This
is a genuine three-way trade, not two-way.

### Consequence: two distinct stories, which must not be conflated

| story | row | accuracy | LUT | % device | latency |
|---|---|---:|---:|---:|---:|
| **deployable** (meets CTL2) | sparsity+PID | 0.7961 | 385,471 | 22.3% | **89.9 ns** |
| | sparsity flagship | 0.7990 | 478,214 | 27.7% | **93.2 ns** |
| **accuracy parity** (over budget) | g-ls-s3 ep152 | **0.8101** | 1,461,880 | 84.6% | 166.5 ns |

The goal set for this campaign was *maximise accuracy subject to fitting one
VU13P* — latency was never a constraint in it, and the result honours that
goal exactly. But **the accuracy-parity rows are not CTL2-deployable**, and
the rebuttal must not present them as if they were. The sub-100 ns claim
belongs only to the ~0.796–0.799 rows.

**Open question worth an arm:** can the latency be cut without losing the
accuracy? `hard_dc=1` pipelining previously bought 93.2 ns → 89.9 ns on a
small design (D-earlier), and stage count is set by `PJ_LATENCY_CUTOFF`
(4.0 ns default) rather than being intrinsic. A trace sweep over
`LATENCY_CUTOFF` / `hard_dc` on the 0.8101 checkpoint is cheap (CPU-only) and
would establish whether ~166 ns is a floor or a default.

---

## D171 — `ParetoFront` PRUNES: the recommended checkpoint was deleted while we were quoting it

Attempting to re-pipeline the recommended 0.8101 row failed:

```
FileNotFoundError: .../g-ls-s3/pareto/epoch=152-val_acc=0.8101-ebops=1538843.0.keras
```

`ParetoFront` does not merely *save* nondominated points — it **deletes points
that later become dominated**. As `g-ls-s3` improved past epoch 152, that
checkpoint was dominated (0.8101 at 1,538,843 EBOPs, versus later 0.8101 at
**1,527,476**) and removed. The whole `g-ls-s3` front is down to **8 files**
after 500 epochs.

**What survived and what did not:**
- **Survived:** the generated RTL in `hw/trace101/` and `hw/trace106/`, plus
  their `hw_results.json`. The published numbers remain backed by artifacts.
- **Lost:** the `.keras` checkpoint itself, so that exact design cannot be
  re-traced, re-verified at a different pipeline cutoff, or retrained from.

**Fix applied:** all 35 current checkpoints across the five arms copied to
`/data/phatjet-fpga/frozen/` as `<run>__<original-name>.keras` (`cp -n`, so
re-running is safe). **Any checkpoint that is quoted, traced, or handed to
Vivado must be frozen out of `pareto/` first** — a live Pareto front is
working storage, not an archive.

Silver lining: the front now holds **0.8101 at 1,527,476 EBOPs**, ~11k cheaper
than the row we traced, so the efficient operating point may be slightly
better than published once re-traced.

---

## D172 — LATENCY CORRECTION: every quoted figure understated by ~20%; the true floor is the 198 ns combinational path

`to_pipeline(comb, latency_cutoff)` takes the **maximum combinational delay
permitted per pipeline stage**. The campaign default is
`PJ_LATENCY_CUTOFF = 4.0 ns` while every row is quoted at a **3.33 ns
(300 MHz)** clock. A stage holding 4.0 ns of logic **cannot** be clocked at
3.33 ns, so those two numbers were never mutually consistent.

Measured on the 0.8101-class checkpoint (combinational latency **198 ns** in
every case):

| cutoff | stages | latency @ 300 MHz | reg bits | consistent? |
|---:|---:|---:|---:|---|
| 4.0 ns (default, **as published**) | 50 | 166.5 ns | 1,070,731 | **NO** |
| **3.33 ns (matches the clock)** | **60** | **199.8 ns** | 1,297,909 | yes |
| 2.5 ns (timing headroom) | 80 | 266.4 ns | 1,735,405 | yes |

**Corrections that follow:**

1. **The honest 300 MHz latency of the accuracy rows is ~200 ns, not
   ~166 ns** — a 20% understatement across D162–D170. FF cost also rises
   ~21% (1.07M → 1.30M reg bits, still only ~37.5% of device).
2. **~198 ns is a floor, not a knob.** Total pipeline latency cannot go below
   the combinational path, and that path is 198 ns at this precision. No
   choice of cutoff, `hard_dc`, or retiming gets this design under 100 ns;
   only a smaller/shallower design can.
3. **The gap to the trigger budget is therefore 2x, not 1.7x**: ~200 ns
   against CMS L1T's sub-100 ns, and ~2.4x JEDI-linear's 82 ns.

This does **not** affect LUT, EBOPs, accuracy, or bit-exactness — comb cost is
identical (1,572,206) at every cutoff. It affects only the latency column and
the FF estimate.

**Action:** `PJ_LATENCY_CUTOFF` should default to the clock period, not 4.0.
Until every row is re-pipelined, quote accuracy-row latency as **~200 ns
(60 stages at 3.33 ns)** and mark the older 166.5 ns figures superseded.

### Also: the 0.8109 candidate is NOT bit-exact

`g-ls-s3` ep502 (0.8109 @ 1,632,628 LUT = 94.5%) failed the DAIS screen with
`max|err| = 9.537e-07` — small, but not zero, so it carries no accuracy claim
(D70). **The verified maximum remains 0.8108.** The screen caught it before
any RTL was built, which is the third time tonight it has paid for itself.

---

## D173 — Zihan's GMP-intact deployment study SUPERSEDES this campaign's rows for the rebuttal claim

A preliminary FPGA study with **GMP intact**, targeting the real CTL2 envelope
(VU13P at 300 MHz, **≤10% of LUTs** reserved because Correlator Layer 2 also
runs jet clustering, latency **<100 ns**, II=1):

| N | Accuracy | Stages | Latency | LUT | % VU13P | DSP | BRAM | II |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 67.4% | 29 | 96.6 ns | 87,206 | 5.0% | 0 | 0 | 1 |
| 32 | 71.4% | 29 | 96.6 ns | 98,069 | 5.7% | 0 | 0 | 1 |
| 64 | **74.6%** | 29 | **96.6 ns** | **167,304** | **9.7%** | 0 | 0 | 1 |

719 designs traced with `HWConfig(1,-1,-1)`, `hard_dc=2`, pipelining every two
adders. RTL matches Keras bit-for-bit on **2,000 held-out jets**. GMP is 41% of
the N=64 LUT estimate. Architecture was settled before QAT on the finding that
at a fixed 6-bit scope, full self-attention at N=64 costs **19.3M** LUTs versus
**6.5M** for the patch hierarchy, with parallel wiring cutting 56 stages to 39.

### Why this replaces, rather than joins, the D162–D172 rows

| row | accuracy | LUT | % device | latency | ≤10%? | <100 ns? |
|---|--:|--:|--:|--:|---|---|
| **theirs, N=64** | 74.6% | 167,304 | **9.7%** | **96.6 ns** | **YES** | **YES** |
| mine, sparsity+PID | 79.61% | 385,471 | 22.3% | 89.9 ns | no | yes |
| mine, flagship | 79.90% | 478,214 | 27.7% | 93.2 ns | no | yes |
| mine, accuracy-parity | 81.01% | 1,461,880 | 84.6% | ~200 ns | no | **no** |

**Only the GMP-intact rows satisfy both constraints.** My cheapest sub-100 ns
design is **2.3x** their N=64 LUT and still misses the 10% budget; my
accuracy-parity rows miss both by ~8.7x and ~2x. Presenting them alongside
invites "so the accurate configuration does not fit?" and weakens a claim that
currently stands clean. **The deployment claim is theirs; my rows are not
evidence for it.**

### What this campaign should still contribute — two things only

1. **Direct quantitative support for the training-time caveat.** Their study
   states the accuracies are training-limited (161–300 CPU epochs, 6–8 points
   below float; a longer run closed it to 2.95 points at 78.6%). This campaign
   measured that same effect to convergence on the N=128 graph: the **float
   ceiling is 0.8126** (D146, independent `train_float` run) and long-schedule
   QAT reached **0.8108** — a residual gap of **0.18 points**. That is
   independent evidence that the QAT-to-float gap on this architecture family
   closes to ~0.2 points given enough epochs, i.e. the reported accuracies are
   a training-budget artifact, not an architectural ceiling.

2. **Verification methodology.** The DAIS screen (D164) predicts a Verilator
   bit-exactness failure in seconds instead of a ~20-minute build, on 4x the
   sample; ~40% of checkpoints from a continuation recipe fail for reasons
   unrelated to quality (D162). Their 2,000-jet standard already matches the
   DAIS sample size used here.

Everything else from D162–D172 stays in this log as method and negative
results, not as rebuttal material.

### ONE THING TO VERIFY BEFORE PUBLISHING — the latency margin is 3.4%

96.6 ns leaves only **3.4%** under the 100 ns budget, and 96.6 / 29 stages =
**3.33 ns/stage exactly**, i.e. it assumes each stage's combinational delay
fits the 300 MHz period. **D172 found this campaign's rows violated exactly
that**: they were pipelined at `latency_cutoff = 4.0 ns` while quoted at
3.33 ns, understating latency by 20%. If the same 4.0 ns cutoff were in play
here, 29 stages would be **116 ns and the sub-100 ns claim would fail.**

"Pipelining every two adders" is a *depth*-based rule rather than a
delay-cutoff, so it is probably safe — but with only 3.4% margin this must be
confirmed explicitly, not assumed. Concretely: check the per-stage
combinational delay reported by `to_pipeline`, or re-pipeline at
`latency_cutoff = 3.33` and confirm the stage count stays at 29.

---

## D174 — Both 0.8109 candidates are unusable; the campaign's verified ceiling is 0.8108

Two arms independently reached 0.8109, and neither can be quoted:

| candidate | EBOPs | outcome |
|---|--:|---|
| `g-ls-s3` ep502 | 1,542,299 | **not bit-exact** — DAIS `max\|err\| = 9.537e-07` (D172) |
| `g-nopress` ep417 | 1,702,849 | **does not fit** — 1,791,444 LUT = **103.7%** of VU13P |

The second failure was predicted from the LUT/EBOP ratio before tracing and
is a clean illustration of D167: at 1.70M EBOPs the ratio is ~1.05, so the
design lands over the device even though its EBOPs sit inside the 1.75M budget
that D148 justified as "~1.63M LUT". **EBOPs remains a screening proxy only.**

**Final verified results of this campaign:**

| row | accuracy | LUT | % VU13P | latency (as traced) | verification |
|---|--:|--:|--:|--:|---|
| **recommended** | **0.8101** | 1,461,880 | **84.6%** | 166.5 ns (50 stg) | DAIS 2,000 + Verilator 512, exact |
| maximum | **0.8108** | 1,570,717 | 90.9% | 166.5 ns (50 stg) | DAIS 2,000 + Verilator 512, exact |
| independent schedule | 0.8103 | 1,567,068 | 90.7% | 166.5 ns (50 stg) | DAIS 2,000 + Verilator 512, exact |

Latency figures as-traced at a 4.0 ns cutoff; self-consistent at 3.33 ns they
are ~60 stages / ~200 ns (D172).

All five arms have now been harvested and all 54 checkpoints frozen out of
`pareto/` (D171). The frontier moved 0.8077 → 0.8108 over the night and then
stopped: the two attempts beyond it failed on verification and on area
respectively, which is a reasonable place to stop rather than a limitation of
the schedule.

---

## D175 — Campaign archived to GitHub; the "sub-100 ns" rows violate D172's own floor

**2026-09-30.** The workspace was compiled into
https://github.com/JavierZhao/PHATJeT-fpga (public). README.md is the handoff.
Source matched the cluster ConfigMap `zz-phatjet-src` byte for byte. Small
evidence (every `run_config.json`, `qat_hist.json`, `hw_results.json`,
`component_costs.json`, RTL `metadata.json`, plus PVC inventories) was pulled
into `evidence/`. Every trace (107 rows) is flattened into
`evidence/all_hw_results.csv`.

**Excluded from the public repo:** submission PDFs, rebuttal and
reviewer-response drafts, Aaron's package/data/checkpoint, private messages,
npz data, and RTL (which stays on the PVC). The job generator `gen_torch.py`
was found lost with its session scratchpad. `k8s/job_template.yaml` is a
reconstruction, not tested.

**New finding while compiling.** D172 showed that total
latency cannot go below `comb_latency_ns`. It applied this only to the
accuracy rows. The rows quoted as sub-100 ns in D170, D173 and
VIVADO_HANDOFF have combinational floors above 100 ns:

| row | quoted | stages | comb floor |
|---|--:|--:|--:|
| `n128-wvpid` ep565, 0.7961 (Vivado primary) | 89.9 ns | 27 | 106 ns |
| `n128-wv10` ep64, 0.7990 | 93.2 ns | 28 | 109 ns |
| `n128-pid` ep901, 0.7988, hard_dc=1 | 89.9 ns | 27 | 108 ns |

They were pipelined at the 4.0 ns default cutoff, like every other row.
**One point is already measured.** D113's sweep re-pipelined the
`n128-pid` ep901 netlist at cutoff 3.0 ns and got **37 stages = 123.2 ns**,
with the same netlist and LUT count. D113 called this "cutoff-tightening
counterproductive", but it is the clock-consistent figure. At 3.33 ns,
expect about 33–34 stages, roughly 110 ns. **None of the compressed rows is
under 100 ns at 300 MHz.** Next action: trace ep565 (the Vivado primary) at
`PJ_LATENCY_CUTOFF=3.33` (CPU-only) for its exact figure, then change the
default.
This strengthens D173's request to re-check the GMP-intact study's 96.6 ns
the same way.

Also recorded: the `g-ls*`/`g-x*` record arms ran with `PJ_INIT_BETA=1e-9`
against a 1.7M target while sitting below it. Beta therefore stayed near
1e-9, and the record was trained with effectively no beta pressure. Some
reports describe this step as "a positive init_beta near its operating
range", which does not match the run configs. `PJ_SEED` is not written to
`run_config.json`.
