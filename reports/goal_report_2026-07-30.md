# Fitting PHAT-JeT on one VU13P at maximum accuracy — session report

**Goal:** maximise post-quantization accuracy subject to fitting **one whole
XCVU13P** (< 1,728,000 CLB LUTs), and do not stop until the quantized number
beats **JEDI-linear's best reported 80.9%** (arXiv:2508.15468 Table II, N=64
and N=128).

---

## 1. Headline: the accuracy record fits the device, and is now verified

`vu-long` epoch 773 — the best quantized checkpoint the project has produced —
had never been traced. It has now been, and it is bit-exact:

| | value |
|---|---|
| Accuracy | **0.8077** |
| LUT (da4ml pre-P&R) | **1,431,347 = 82.8% of one VU13P** |
| FF (register bits) | 1,013,861 = 29.3% |
| Pipeline | 48 stages = **159.8 ns** at a 300 MHz target |
| II / DSP / BRAM | 1 / 0 / 0 (by construction, not from a synthesis report) |
| RTL vs model | **bit-exact**, `max\|err\| = 0`, 100% argmax on 512 jets |

So PHAT-JeT at its best accuracy **fits a single VU13P with ~17% spare**.
Against the goal it is **0.13 points short** of 80.9%.

### The bit-exactness scare, and what it was

The first trace of this checkpoint was **not** bit-exact (`max|err| = 1.251`,
though argmax still agreed on all 512 jets). The cause was not the netlist:
`PJ_INPUTS_KIF` has defaulted to `1,8,16` since the era when every traced
design was low precision. This checkpoint is the highest-precision one ever
traced (1.54M EBOPs), and 16 fractional input bits were truncating what Keras
retained. At `1,8,20` the error is exactly zero and the LUT estimate is
unchanged (+2 LUT) — an emulation-fidelity setting, not a cost one.

**Standing rule (D140):** any trace of a >1M-EBOPs checkpoint must widen the
input word format and confirm `bit_exact: true`. A non-bit-exact trace carries
no accuracy claim.

---

## 2. Why the campaign had been leaving accuracy on the table

`vu-long` **peaked at epoch 773 (0.8077 @ 1.54M EBOPs) and then decayed to
~0.80 by epoch 3900** while EBOPs were squeezed to 600k. That tail traded
0.7 accuracy points for area the device did not need.

`dev90` then resumed the peak with `BetaPID(target = 1.5M)` — and still sank
to 663k EBOPs, **less than half its own target**, with beta pinned near zero.
A controller that wants 1.5M cannot drive a design to 663k. Something else
was pushing.

> **`PJ_ATTN_L1=1e-5` and `PJ_FFN_L1=3e-5` install `fr=MonoL1(...)` on
> datalane quantizers — 1000x and 3000x the global `MonoL1(1e-8)` baseline.
> `MonoL1` is an ordinary regularization loss, *not* scaled by layer beta.
> BetaPID sees their effect in total EBOPs but cannot oppose them: beta is
> clamped at `min_beta = 0` and cannot go negative.**

Both terms were inherited from the compression era, when the design was 4x
too big for the device. Under a loose area budget they are pure accuracy
loss. Every arm launched this session removes them.

---

## 3. Two bugs caught before they cost results

**(a) `BetaPID` cannot start from beta = 0.** With `init_beta=None` it takes
`self.beta = get_avg_beta()`, and all layers are built under
`LayerConfigScope(beta0=0)` — so beta is 0.0. At `epoch == warmup` (10) the
log branch evaluates `log10(0.0)` and raises `ValueError: math domain error`.
Observed live: an arm sat at `beta: 0.0e+00` with EBOPs *rising* through 47.9M
against a 1.75M target — the controller was inert, not slow. Any from-scratch
PID arm without `PJ_INIT_BETA` dies at epoch 10.

*This retro-explains D137's "from-scratch BetaPID cannot reach tight EBOPs
targets".* That conclusion came from runs whose controller may never have
engaged, so it is no longer supported evidence.

**(b) `PJ_FFN_MULT` defaults to 4; the record trained at 1.** The job
generator never set it, so every resume arm tried to load an FFN×1 checkpoint
into an FFN×4 graph and crashed in `load_weights`. This is the third
appearance of the resume-graph-mismatch failure mode (D134, D137, D141).
**Rule extended:** for any arm with `PJ_RESUME_CKPT`, diff the *full*
structural env against the source run's `run_config.json` — not just the
knobs being changed.

---

## 4. A premise corrected

D137/D138 treated the **0.8206 float ceiling** and the **0.8077 QAT record**
as the same graph and inferred "a 1.3-point quantization gap". They are not
the same graph: the float number is **d_model=32 + FFN×4** (verified directly
from the teacher archive: `embed` units 32, `ffn1` units 128), the QAT record
is **d_model=16 + FFN×1**. That difference mixes capacity, optimisation and
quantization, and is not a quantization cost.

Two measurements were run to separate them. **One of them failed, and the
failure is itself a result.**

### The float-eval diagnostic does not work on this model (D143)

`float_eval_qat.py` loads the record into the quantized graph, copies every
kernel/bias by layer name into the same graph built `quantized=False`, and
evaluates both.

| | value |
|---|---:|
| quantized, as trained | **0.8077** |
| identical weights, quantizers removed | **0.3069** |

All 32 tensors transferred with zero shape mismatches, and 0.3069 against a
5-class chance of 0.20 means the float graph is near-broken, not degraded.
The premise — that stripping quantizers leaves a valid, higher-precision
version of the same network — is **false here**. Under `overflow_mode=wrap`
out-of-range intermediates wrap rather than saturate, and the network has
learned around that behaviour (and around the quantizers' clipping keeping
`tanh`/`softmax` arguments in range). Remove them and you have a different
function.

So "the quantization cost is X" must never be produced this way for a
wrap-overflow HGQ2 model. The script is kept with the limitation in its
docstring, so the negative result is not rediscovered.

*Positive by-product:* the rebuild-and-`load_weights` path reproduced
**0.8077 exactly** on all 260k evaluation jets — an independent confirmation
of the record outside the training loop.

### The valid measurement

`g-f16` — an independent `train_float.py` run of the record's own d16/FFN×1
graph. It reached **0.8126 float within 142 epochs**, clearing both the 0.8077
QAT record and the 0.809 target.

Two things follow. First, **the record's graph is not saturated**: there is
≥0.49 points of float headroom above the current QAT champion on the *cheap*
d16/FFN×1 graph, so the goal does not depend on the riskier capacity route.
Second, a caution about reading these curves — the run sat at 0.8056–0.8059
between epochs 63 and 96 and looked converged. It was not; it was between
21-epoch anneal cycles. **Do not call a cosine-restart run converged from a
plateau shorter than a full cycle.**

---

## 5. How large a win the goal actually requires

The evaluation partition is the standard hls4ml test split (2.6e5 jets).
Binomial standard error at p ≈ 0.81:

```
sqrt(0.81 * 0.19 / 260000) = 7.7e-4  ->  0.077 accuracy points
```

The record sits **1.7σ** below 80.9%. More importantly, *clearing* 80.9% by a
hair would not be defensible: 0.8090 is a 0σ win, and checkpoint selection
happens on this same partition, which biases the selected maximum upward.

**Operating target: ≥ 0.8105** (≈2σ), plus a bit-exact trace under 1,728,000
LUT, before the goal is called met. A result in 0.8090–0.8105 will be
reported as "at parity, not clearly above".

---

## 6. Caveats that travel with every number here

- LUT/FF are **da4ml pre-place-and-route estimates**, not synthesis results.
- 300 MHz is a **target**, not closed timing; no bitstream exists.
- `DSP=0`, `BRAM=0`, `II=1` are expected **by construction** (constant-
  coefficient products become shift-add logic), not read from a Vivado report.
- Accuracies are operating-point estimates on the partition also used for
  checkpoint selection.
- The one-SLR deployment claim (D135) is a *separate*, stronger claim about
  CTL2 co-tenancy; this report is about filling one whole device.


---

## 7. Overnight campaign: where the goal actually landed

### The headline number

| | accuracy | LUT | % of VU13P | bit-exact |
|---|---:|---:|---:|---|
| Verified record (vu-long ep773) | 0.8077 | 1,431,347 | 82.8% | **yes** |
| Best fitting checkpoint (`g-ls` ep27) | **0.8096** | 1,454,867 | **84.2%** | **not yet** |
| JEDI-linear reported (their Table II) | 0.809 | 98k @ N=128 | — | post-P&R |

`g-ls` **exceeds JEDI-linear's reported accuracy while fitting one VU13P**,
which is the goal as stated. Two honest qualifications travel with it: the
netlist is not yet demonstrated bit-exact (below), and 0.8096 vs 0.809 is
0.08 of one binomial standard error on this partition — *at parity*, not
clearly above.

### Seed replication (requested)

Three replicates of the `g-ls` recipe (`PJ_SEED` 2/3/4, identical otherwise),
all still inside their first 42-epoch cycle so these are lower bounds:

| arm | best fitting |
|---|---:|
| `g-ls` | 0.8096 |
| `g-ls-s3` | 0.8094 |
| `g-ls-s2` | 0.8088 |
| `g-ls-s4` | 0.8087 |

The recipe reproducibly lands at **~0.809**. Note these resume a shared
checkpoint, so they replicate the *continuation recipe* under different
training stochasticity — not independently trained models.

### What the plateau is, definitively

`g-nopress` released **every** learned bit-pressure at once (KIF `fr`, KBI
`br`, and beta). EBOPs grew 0.7% and accuracy did not improve. Combined with
the measured 0.8126 float ceiling of this exact graph, the 0.8095–0.8096
plateau is **optimisation-limited, not area- or precision-limited**. No
area-side knob can close the rest; only a different trajectory could.

### The float warm-start route is closed

Three independent failures: aggressive descent (0.8158 -> 0.7182), passive
descent (0.8166 -> 0.8037 while still 7x over), and a training-free
calibrated-start screen that cannot get within 23x of budget at any precision
(D160). The structural reason: **a float-initialised model's accuracy lives in
its fractional precision, and fractional precision costs EBOPs
multiplicatively.** Only learned quantization occupies the accurate-and-cheap
corner.

### Open item: bit-exactness of the 0.8096 checkpoint

`max|err|` is **3.732e-02 and identical at `1,8,20` and `1,8,23`** — adding
fractional input bits changes nothing, so this is **not** input truncation
(correcting the initial D159 hypothesis). Argmax agreement is 100% on 512
jets. The LUT figure is unaffected (1,453,955-1,454,867 across widths).
Separately, da4ml's Verilator harness caps the input word at **32 bits total**
(`k+i+f <= 32`), which bounds what can be verified at all.

Until this resolves, the 84.2%-of-device **area** result stands and the
**accuracy** claim on that specific checkpoint does not (D70). The 0.8077
record remains the strongest fully verified row.


---

## 8. FINAL RESULT

**Both success criteria are met.**

| | accuracy | LUT | % of one VU13P | latency | verification |
|---|---:|---:|---:|---:|---|
| **efficient (recommended)** | **0.8101** | 1,461,880 | **84.6%** | 166.5 ns (50 stg) | exact on 2,000 + 512 jets |
| **max accuracy** | **0.8106** | 1,591,089 | 92.1% | 169.8 ns (51 stg) | exact on 2,000 + 512 jets |
| JEDI-linear reported | 0.809 | 98k @ N=128 | — | 82 ns | post-P&R |

- **Zihan's goal** (beat JEDI-linear's reported 0.809): met, **+0.0016**.
- **D142's 2σ bar** (0.8105, set to guard against selection bias): **cleared**.

Both rows are II = 1, DSP = 0, BRAM = 0 by construction, and reproduce the
trained quantized model exactly in both da4ml's DAIS interpreter (2,000 jets)
and Verilator RTL simulation (512 jets).

**Recommended framing for the rebuttal.** Lead with the *efficient* row.
0.8101 at 84.6% of the device leaves 15% margin for placement and routing —
which remain the binding uncertainty, since these are pre-P&R estimates at a
300 MHz *target*, not closed timing. The 0.8106 row buys +0.0005 accuracy for
7.5 more points of device, a poor trade against that uncertainty.

**Read the margin honestly.** 0.8106 − 0.809 = 0.0016 = **0.21σ** against the
0.077-point binomial standard error on the 260k partition. This establishes
**parity, comfortably**, not a decisive accuracy win. Seed replicates of the
recipe span 0.8087–0.8106. Checkpoint selection used the same partition as
reporting, so the peak is a best-observed operating point, not an unbiased
generalization estimate. And PHAT-JeT still costs **~15x** the LUT of
JEDI-linear at matched N — the resource story is unchanged.

**Verified frontier through the night:** 0.8077 → 0.8089 → 0.8094 → 0.8097 →
0.8101 → 0.8103 → **0.8106**.

### What actually produced the gain

`g-ls-s3` went 0.8094 → 0.8106 over ~255 epochs of *unchanged* 42-epoch
anneal cycles at LR 3e-4 — after sitting flat at 0.8094 from epoch 23 to
~150. No new lever. The decisive choice was **not killing an arm that looked
converged**, a rule learned earlier the same night when `g-f16` was nearly
abandoned at a 0.8059 "plateau" and finished at 0.8126.

The recipe, in full: resume the 0.8077 record; remove the regional
`ATTN_L1`/`FFN_L1` bit pressure (which BetaPID cannot counteract); drop LR 5x
to 3e-4 so restarts refine rather than abandon the basin; use 42-epoch cosine
cycles so each night yields ~9 harvestable minima instead of 2; give BetaPID a
positive `init_beta` near its operating range.


---

## 9. LATENCY — correction and honest position (D170/D172)

**The accuracy-optimised rows do not meet the trigger latency budget, and the
latency I originally quoted was 20% optimistic.**

| | value |
|---|---:|
| Quoted (pipelined at 4.0 ns cutoff, 300 MHz clock) | 166.5 ns — **inconsistent** |
| **Self-consistent (3.33 ns cutoff = the 300 MHz clock)** | **60 stages = ~200 ns** |
| Combinational path (a hard floor) | **198 ns** |
| CMS L1T requirement | **< 100 ns** |
| JEDI-linear N=128 (post-P&R, 257.6 MHz achieved) | 82 ns |

Three consequences:

1. The gap to the trigger budget is **~2x**, and to JEDI-linear **~2.4x**.
2. **~198 ns is a floor, not a tuning parameter.** Total pipeline latency
   cannot fall below the combinational path. Cutoff, `hard_dc` and retiming
   only trade stages against registers — they cannot beat it.
3. Accuracy, area **and** latency move together: stage count tracks arithmetic
   depth, which tracks precision.

**So the campaign produced two distinct results, which must not be merged:**

| | accuracy | LUT | % device | latency | meets sub-100 ns |
|---|---:|---:|---:|---:|---|
| deployable | 0.7961 | 385,471 | 22.3% | 89.9 ns | **yes** |
| deployable | 0.7990 | 478,214 | 27.7% | 93.2 ns | **yes** |
| accuracy parity | **0.8101** | 1,461,880 | 84.6% | ~200 ns | **no** |
| accuracy parity | 0.8108 | 1,570,717 | 90.9% | ~200 ns | **no** |

The goal set for this campaign — *maximise accuracy subject to fitting one
VU13P* — contained no latency constraint, and the result honours it exactly.
But "fits one VU13P" and "is deployable in CTL2" are different claims, and
only the former is established for the accuracy-parity rows.
