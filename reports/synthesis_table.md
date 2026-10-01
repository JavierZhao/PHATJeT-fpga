# PHAT-JeT FPGA resource/latency estimates

## HEADLINE — GMP-intact designs inside the CMS L1T envelope

**This is the rebuttal result.** Target: VU13P at 300 MHz, **≤10% of LUTs**
(Correlator Layer 2 also runs jet clustering), latency **<100 ns**, II = 1.

| N | Accuracy | Stages | Latency | LUT | % VU13P | DSP | BRAM | II |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 67.4% | 29 | 96.6 ns | 87,206 | 5.0% | 0 | 0 | 1 |
| 32 | 71.4% | 29 | 96.6 ns | 98,069 | 5.7% | 0 | 0 | 1 |
| **64** | **74.6%** | 29 | **96.6 ns** | **167,304** | **9.7%** | 0 | 0 | 1 |

719 designs traced (`HWConfig(1,-1,-1)`, `hard_dc=2`, pipelining every two
adders). RTL matches Keras **bit-for-bit on 2,000 held-out jets**. GMP is 41%
of the N=64 LUT estimate. All values are da4ml estimates; vendor
place-and-route validation is future work.

Architecture was settled before QAT: at a fixed 6-bit scope, full
self-attention at N=64 estimates **19.3M** LUTs versus **6.5M** for the patch
hierarchy, and parallel wiring cuts 56 pipeline stages to 39.

**Claim:** PHAT-JeT with GMP intact fits the CMS Level-1 trigger envelope at
II = 1 with no DSPs or BRAMs.

**Accuracies are training-limited, not architectural.** These QAT runs had
161–300 CPU epochs and sit 6–8 points below their float counterparts; a longer
run closed the gap to 2.95 points at 78.6%. Independent corroboration from the
long-schedule N=128 study below: float ceiling **0.8126**, long-schedule QAT
**0.8108** — a residual gap of **0.18 points** once training is run to
convergence.

---

> **Everything below is a separate, larger-budget study.** Those rows explore
> how accurate the architecture becomes when the area and latency budgets are
> relaxed far beyond the CTL2 envelope. They reach ~81% but occupy 22–91% of
> the device, and the highest-accuracy ones exceed 100 ns. **They are not
> evidence for the deployment claim above and should not be quoted alongside
> it.**


Generated 2026-07-25. Numbers from `extract_hw.py` / `component_costs.py`; see
`DECISIONS.md` for how the configuration was arrived at.

> **Particle counts: 128 and 64 are the standard comparison points.**
> JEDI-linear reports N = 16/32/64/128, so N = 150 is **not** a matched
> comparison against its N = 128 row. The N = 150 results below are retained as
> the only data at the paper's own particle count, but must not be presented as
> matched. N = 128 runs are in progress. See DECISIONS.md D40.

## Accuracy vs. resources at N = 150 (paper's count — NOT matched to JEDI-linear)

Frontier for the N = 150 configuration with attention bit pressure. (The N = 150
row also appears in the particle-count sweep below, as its cheapest point.)

Configuration: **N = 150 particles x 3 features** (the paper's particle count),
d_model = 16, 4 heads, patch size P = 10 (15 patches), single PHAT block, GMP
8x8 grid on [-1.6, 1.6], tanh pre-norm, mean patch tokenizer, FFN 16->16,
quantization-aware trained with HGQ2 and per-region attention bit pressure
(`attn_l1 = 1e-6`). Target **xcvu13p**, 300 MHz (3.33 ns), **II = 1**.

| Accuracy | LUT | % of VU13P | FF (reg bits) | Latency | II | DSP | BRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8004 | 1,731,475 | 100.2% | 1,515,451 | 153.2 ns | 1 | 0 | 0 |
| 0.8035 | 1,771,319 | 102.5% | 1,495,921 | 156.5 ns | 1 | 0 | 0 |
| 0.8051 | 1,956,650 | 113.2% | 1,648,366 | 159.8 ns | 1 | 0 | 0 |
| 0.8057 | 2,164,665 | 125.3% | 1,635,849 | 159.8 ns | 1 | 0 | 0 |

Particle-count sweep. JEDI-linear reports N = 16/32/64/128 rows itself, so these
are matched comparison points rather than tuning. Shared recipe: tanh-only
normalisation, mean tokenizer, FFN 1x, no pT mask, GMP 8x8 on [-1.6,1.6].

**Not a controlled sweep** — the `attn_l1` column varies. Per-region attention
bit pressure lowers LUT/EBOP from ~1.88 to ~1.38 (DECISIONS.md D24), so rows with
it are much cheaper at equal EBOPs. Compare within a particle count, not across.
The two levers compound: at N = 64, adding `attn_l1` takes 127.2% -> 90.1% of the
device while accuracy is unchanged within noise (0.8043 -> 0.8024).

*(N=150 rows marked \* are the paper's count, not matched to any JEDI-linear row.)*

| N | attn_l1 | Accuracy | LUT | % of VU13P | FF | Latency | Fits? | JEDI-linear same N |
|---:|---|---:|---:|---:|---:|---:|---|---|
| **16** | no | 0.7401 | 957,257 | 55.4% | 418,828 | 156.5 ns | **YES** | 71.9% @ 44k |
| **16** | no | 0.7363 | 876,805 | 50.7% | 391,615 | 149.8 ns | **YES** | 71.9% @ 44k |
| **16** | no | **0.7226** | **365,950** | **21.2%** | 230,974 | 133.2 ns | **YES** | 71.9% @ 44k |
| **32** | no | 0.7912 | 1,477,150 | 85.5% | 708,617 | 156.5 ns | **YES** | 78.0% @ 45k |
| **32** | no | 0.7890 | 1,432,983 | 82.9% | 711,306 | 153.2 ns | **YES** | 78.0% @ 45k |
| **32** | no | **0.7879** | **1,221,682** | **70.7%** | 661,885 | 149.8 ns | **YES** | 78.0% @ 45k |
| 32 | modern recipe, P=4, pT-sorted | 0.7704 | 642,141 | 37.2% | 535,932 | 139.9 ns | YES | 78.0% @ 45k |
| **32** | no | 0.7639 | 494,967 | **28.6%** | 363,073 | 133.2 ns | **YES** | 78.0% @ 45k |
| **128** | **+ value sparsity (flagship)** | **0.7990** | **478,214** | **27.7%** | 358,023 | **93.2 ns** | **YES** | 80.9% @ 98k |
| **128** | **+ sparsity + PID (cheap champion)** | 0.7961 | **385,471** | **22.3%** | **270,104** | **89.9 ns** | **YES** | 80.9% @ 98k |
| 128 | + value sparsity, cheap point | 0.7957 | 404,852 | 23.4% | 298,271 | 93.2 ns | YES | 80.9% @ 98k |
| 128 | + PID continuation (pre-sparsity) | 0.7988 | 513,253 | 29.7% | 368,989 | 93.2 ns | YES | 80.9% @ 98k |
| 128 | flagship, hard_dc=1 pipelining | 0.7988 | 515,055 | 29.8% | 358,688 | **89.9 ns** | YES | 80.9% @ 98k |
| 128 | + PID continuation, higher acc | 0.7995 | 527,058 | 30.5% | 427,577 | 123.2 ns | YES | 80.9% @ 98k |
| 128 | relu-attention variant | 0.7964 | 591,088 | 34.2% | **329,686** | 116.5 ns | YES | 80.9% @ 98k |
| **128** | **1e-5 + ffn_l1 3e-5** | **0.8005** | **662,654** | **38.3%** | 589,472 | **133.2 ns** | **YES** | 80.9% @ 98k |
| **64** | **+ value sparsity (headline)** | **0.8029** | **487,451** | **28.2%** | 449,527 | **136.5 ns** | **YES** | 80.9% @ 71k |
| 64 | + sparsity harvest, cheap point | 0.8019 | **451,836** | **26.1%** | 442,287 | 139.9 ns | YES | 80.9% @ 71k |
| 64 | + value sparsity, parity point | 0.8035 | 523,154 | 30.3% | 479,466 | 139.9 ns | YES | 80.9% @ 71k |
| 64 | + PID continuation (pre-sparsity) | 0.8036 | 544,292 | 31.5% | 491,609 | 139.9 ns | YES | 80.9% @ 71k |
| 64 | + PID continuation, cheaper pt | 0.8012 | 537,796 | 31.1% | 483,189 | 136.5 ns | YES | 80.9% @ 71k |
| **64** | **1e-5 + ffn_l1 3e-5** | **0.8023** | **652,680** | **37.8%** | 564,899 | 143.2 ns | **YES** | 80.9% @ 71k |
| 64 | relu-attention variant | 0.8002 | 700,629 | 40.5% | **369,238** | **116.5 ns** | YES | 80.9% @ 71k |
| **64** | + beta 0.1 (3 levers) | 0.7911 | **399,554** | **23.1%** | 392,396 | 136.5 ns | **YES** | 80.9% @ 71k |
| 64 | 1e-5 | 0.8022 | 929,722 | 53.8% | 701,166 | 149.8 ns | YES | 80.9% @ 71k |
| 64 | 1e-6 | 0.8024 | 1,558,333 | 90.2% | 1,093,269 | 153.2 ns | YES | 80.9% @ 71k |
| 64 | no | 0.8043 | 2,198,207 | 127.2% | 1,219,429 | 159.8 ns | no | 80.9% @ 71k |
| 150* | 1e-5 | 0.7898 | **1,130,657** | **65.4%** | 754,684 | 149.8 ns | **YES** | 80.9% @ 98k (N=128) |
| 150* | 1e-6, beta 0.03 | 0.7856 | 1,068,266 | 61.8% | 1,145,290 | 146.5 ns | YES | 80.9% @ 98k (N=128) |

### Above JEDI-linear's reported accuracy, inside one VU13P — bit-exact

> **LATENCY IN THIS SECTION IS SUPERSEDED — see D172.** These rows were
> pipelined at `latency_cutoff = 4.0 ns` but quoted at a 3.33 ns (300 MHz)
> clock; a 4.0 ns stage cannot run at 300 MHz. Re-pipelined consistently at
> 3.33 ns the same design needs **60 stages = ~200 ns** (not 50 = 166.5 ns),
> with ~21% more registers. The combinational path is **198 ns**, which is a
> floor no pipelining choice can beat. **These designs do NOT meet the CMS
> L1T sub-100 ns budget** — that claim belongs only to the compressed
> ~0.796–0.799 rows at 89.9–93.2 ns. LUT, EBOPs, accuracy and bit-exactness
> are unaffected.

| N | recipe | Accuracy | LUT | % of VU13P | FF | Latency | Fits? | JEDI-linear same N |
|---:|---|---:|---:|---:|---:|---:|---|---|
| **128** | same arm, ep412 — **max accuracy** (`g-ls-s3`) | **0.8108** | 1,570,717 | 90.9% | — | 166.5 ns (50 stg) | **YES** | 80.9% @ 98k |
| **128** | **independent schedule** (`g-x3`, LR 6e-4 / 14-ep cycles) | **0.8103** | 1,567,068 | 90.7% | — | 166.5 ns (50 stg) | **YES** | 80.9% @ 98k |
| 128 | same arm, ep278 (`g-ls-s3`) | 0.8106 | 1,591,089 | 92.1% | 1,093,455 | 169.8 ns (51 stg) | YES | 80.9% @ 98k |
| 128 | same arm, ep195 | 0.8103 | 1,577,184 | 91.3% | 1,066,585 | 163.2 ns (49 stg) | YES | 80.9% @ 98k |
| **128** | same arm, ep152 — **best accuracy/area** (`g-ls-s3`) | **0.8101** | **1,461,880** | **84.6%** | 1,064,301 | 166.5 ns (50 stg) | **YES** | 80.9% @ 98k |
| **128** | + LR 1e-4 / 84-ep cycles from a verified ckpt (`g-x1` ep23) | **0.8097** | **1,481,259** | **85.7%** | 1,051,887 | 163.2 ns (49 stg) | **YES** | 80.9% @ 98k |
| **128** | L1-free continuation, LR 3e-4, 42-ep cycles (`g-ls-s3` ep23) | **0.8094** | **1,476,484** | **85.4%** | 1,046,608 | 163.2 ns (49 stg) | **YES** | 80.9% @ 98k |
| **128** | same recipe, independent seed (`g-ls-s2` ep16) | **0.8089** | **1,446,355** | **83.7%** | 1,030,032 | 159.8 ns (48 stg) | **YES** | 80.9% @ 98k |

All rows exactly reproduce the trained model in simulation. The top two rows are
the strongest: `max|err| = 0` on **2,000** jets through da4ml's DAIS
interpreter **and** on 512 through Verilator. The others were checked on 512
jets only — a weaker statement than it looks, since one known-bad checkpoint's
error grew 4x (3.7e-02 -> 1.5e-01) when the sample went from 512 to 2,000.
The top row
exceeds JEDI-linear's best reported 80.9% (Que et al., Table II) while fitting
one device.

**Read the margin honestly.** 0.8108 − 0.809 = 0.0018 = **0.23σ** against the
0.077-point binomial standard error on this 260k partition, and seed
replicates of the recipe span 0.8087–0.8108. The defensible claim is
**parity with JEDI-linear's reported accuracy, achieved inside one VU13P** —
not a decisive win. PHAT-JeT still costs ~15x the LUT at matched N.

Checkpoint selection and reporting use the same validation partition, so the
quoted peak is selection-biased (a maximum over many correlated epochs) and
should be presented as a best-observed operating point, not an unbiased
generalization estimate.

### Accuracy record — bit-exact, fills most of one VU13P

| N | recipe | Accuracy | LUT | % of VU13P | FF | Latency | Fits? | JEDI-linear same N |
|---:|---|---:|---:|---:|---:|---:|---|---|
| **128** | 4000-epoch budget lever, rich store (`vu-long` ep773) | **0.8077** | **1,431,347** | **82.8%** | 1,013,861 | 159.8 ns (48 stg) | **YES** | 80.9% @ 98k |

The highest quantized accuracy the project has produced, and it fits the
device with ~17% spare. **Bit-exact** (`max|err| = 0`, 100% argmax on 512
jets) — but only at `PJ_INPUTS_KIF=1,8,20`. At the long-standing default of
`1,8,16` the same checkpoint traces to the same LUT count and is **not**
bit-exact (`max|err| = 1.251`): 16 fractional input bits truncate what this
checkpoint retains, since that default was chosen for far lower-precision
designs. Any future high-precision trace must widen the input word format
and re-verify; the LUT estimate is unchanged by the widening (1,431,345 vs
1,431,347), so this is purely an emulation-fidelity setting.
| 150* | no, beta 0.1 | 0.7758 | **829,046** | **48.0%** | 1,097,997 | 136.5 ns | **YES** | 80.9% @ 98k (N=128) |
| 150* | 3e-5 | 0.7811 | 890,038 | 51.5% | 481,302 | **106.6 ns** | **YES** | 80.9% @ 98k (N=128) |
| 150* | 1e-6 | 0.8004 | 1,731,475 | 100.2% | 1,515,451 | 153.2 ns | marginal | *not matched* |

### Two resource bases — read both (retained for reference)

The rows above quote **% of the whole VU13P** (1,728,000 CLB LUTs). The CMS
CTL2 deployment constraint quoted in the trigger literature is different and
tighter: *no more than one Super Logic Region* of the FPGA, "or even less
depending on the other algorithms running in parallel". A VU13P has 4 SLRs, so
a nominal one-SLR budget is **≈432,000 LUT / ≈864,000 FF** — about 25% of the
device. To convert: **% of one SLR = 4 × % of device**.

Both bands are kept deliberately:

- **60–70% of the whole device** (≈1.04–1.21M LUT): the N=150/P=10 rows at
  61.8% and 65.4%, and the N=32 row at 70.7%. These are the largest designs
  that still fit one FPGA with headroom, retained as *reference* points for the
  "how much model does a whole VU13P buy" question. They correspond to ~2.5–2.8
  SLRs and therefore do **not** satisfy the one-SLR deployment constraint.
- **60–70% of one SLR** (≈259–302k LUT): the co-tenancy band — a tagger there
  leaves 30–40% of its SLR for other CTL2 logic. No verified row sits here yet;
  the `slr-room` arm targets it.

Rows at ≤100% of one SLR (currently the N=128 sparsity rows at 89.2% and 93.7%)
are the ones relevant to the absolute deployment claim. Placement and routing
inside a single SLR is **not** yet demonstrated — see the limitations section.

**At the two particle counts where our design fits the device, it exceeds
JEDI-linear's accuracy**: +1.5 points at N = 16 (74.01 vs 71.9) and +1.1 at
N = 32 (79.12 vs 78.0), at roughly 20-30x the LUT cost. (Single seed; margins
exceed within-arm scatter but carry no error bars.) At N = 64 and above we
are both larger and slightly less accurate. The architecture is
accuracy-competitive at small N and not resource-competitive anywhere; both
halves of that sentence are supported by the measurements above.

**Prefer region-targeted pressure over global beta.** Measured three times
(DECISIONS.md D24/D36/D38): global EBOPs pressure lowers the EBOPs count but not
LUT proportionally, because EBOPs cannot see softmax tables or datalane routing.
`attn_l1`/`ffn_l1` lower both. A design at 14% fewer EBOPs cost *more* LUT and
0.73 fewer accuracy points than one using region-targeted pressure instead.

**The paper's N = 150 configuration spans 48.0% to 100.2% of the device** at
77.58% to 80.04% accuracy — a usable deployment curve at the published particle
count, not a single point. Global EBOPs pressure and per-region attention
pressure contribute independently: at N = 150 the cheapest area comes from strong
global beta (48.0%), while attention pressure gives the best LUT-per-EBOP ratio
(1.22 vs 1.45) and by far the lowest latency.

**Latency is also a function of attention pressure.** Every configuration sat
near 150 ns (45-48 pipeline stages) until `attn_l1 = 3e-5`, which cut the N = 150
design to **32 stages / 106.6 ns**. Narrower attention datalanes shorten the
critical path as well as shrinking it, so the area/accuracy knob is really an
area/latency/accuracy knob. Worth stating: the cheapest N = 150 point is also the
fastest.

**Headline operating point — the paper's configuration, fitting, bit-exact:**
N = 150, P = 10 with `attn_l1 = 1e-5` gives **78.98% accuracy at 1,130,657 LUT
(65.4% of a VU13P), 149.8 ns, II = 1, 0 DSP, 0 BRAM**, with the generated RTL
verified bit-exact against the trained model. This is the published particle count,
not a reduced-N variant, and it leaves ~36% of the device free.

**Latency:** the fastest verified design is now the N = 128 flagship itself —
**93.2 ns (28 stages)** at 0.7988 and 29.7% LUT: the PID continuation narrowed
bitwidths enough to shallow the adder trees by 12 pipeline stages relative to
the pre-continuation row. (An earlier ReLU-attention variant at N = 64 — one
LUT per score, no row reduction, 116.5 ns, 35% fewer FFs — is retained in the
table but is dominated on latency; softmax's bounded [0,1] output keeps the AV
datalanes narrow, so eliminating the row reduction traded area for a speed
advantage that pressure alone eventually exceeded, DECISIONS.md D65/D69/D86.)

**Best overall operating point:** N = 64 with `attn_l1 = 1e-5`, `ffn_l1 = 3e-5`
and a 1000-epoch PID-controlled continuation reaches **80.36% accuracy at
544,292 LUT (31.5% of a VU13P), 139.9 ns, II = 1** — ~7.7x JEDI-linear's N=64
LUT (71k) at −0.5 accuracy points, leaving ~68% of the device free. Adding FFN
pressure on top of attention pressure removed 30% of the area at identical
accuracy, and the continuation shed a further 17% while *gaining* 0.13 points
(within seed noise — the defensible claim is area reduction at held accuracy);
the earlier stopping point was simply undertrained. The operating point is
robust: two independent 1000-epoch continuations with different beta-control
schemes (PID and constant-beta) converged to the same knee (0.8036 @ 427,680
and 0.8032 @ 427,773 EBOPs), and a third run pushed below it only by
collapsing accuracy (0.73 @ 360k EBOPs) — a genuine accuracy/area wall at
~400k EBOPs (~540k LUT) for this architecture at N = 64.

The same continuation applied to the flagship **N = 128** row gives
**79.88% @ 513,253 LUT (29.7%), 93.2 ns** — 5.2x JEDI-linear at the matched
particle count (was 6.8x pre-continuation), or 79.95% @ 527,058 at 123.2 ns.
Across the two matched particle counts the verified resource multiple vs
JEDI-linear is now **5.2–7.7x**. Seed robustness: an independent-seed
replication reproduced the N = 64 operating point within 0.1–0.3 accuracy
points once continued to its knee (0.8002 @ 408k EBOPs vs 0.8012 @ 405k for
the original seed; n = 2 seeds). Mid-schedule checkpoints vary far more
(~1.5 pts), so single-run intermediate points should not be quoted without
that caveat. The N = 128 flagship row has one seed.

Note the same configuration at `attn_l1 = 1e-6` costs 1,558,333 LUT (90.2%) for
0.8024 accuracy. Raising attention pressure 10x removed **40% of the area for
0.02 accuracy points**, which is the clearest single demonstration that the
design was over-provisioned in exactly the region the component trace
identified.

Both particle counts give usable curves rather than single points, and the gap to
JEDI-linear **narrows sharply toward the cheap end**:

| Operating point | Accuracy | LUT | vs JEDI-linear same N |
|---|---:|---:|---|
| N=32 headline | 0.7879 | 1,219,048 | +0.8 pts, ~27x LUT |
| N=32 cheap | 0.7639 | 494,967 | -1.6 pts, ~11x LUT |
| N=16 headline | 0.7401 | 957,257 | +2.1 pts, ~22x LUT |
| **N=16 cheap** | **0.7226** | **365,950** | **+0.4 pts, ~8x LUT** |

The N=16 cheap point is the most favourable comparison available: still ahead of
the baseline on accuracy, at 21.2% device utilisation and ~8x its LUT rather than
the ~27x at our headline. The resource deficit is therefore not a fixed property
of the architecture — it is largest where we are pushing accuracy hardest.

Float reference for this exact synthesizable graph: **0.8088**. Published float
PHAT-JeT: 0.8180 (0.8145 with a max patch tokenizer, paper Table 16).

## Per-component attribution

At the 0.8004 operating point (cumulative-prefix differencing — an attribution,
not a partition; see `component_costs.py`):

| Component | Share of LUT |
|---|---:|
| Local patch attention | 40.7% |
| Patch pooling + global MHSA | 28.4% |
| FFN | 26.7% |
| Aggregation + classifier head | 4.1% |
| Embedding | 2.9% |
| Geometric Message Passing | ~0% |

GMP — the architecture's distinguishing component — shows **zero marginal cost
under cumulative-prefix attribution** (da4ml's CSE folds its one-hot routing into
neighbouring logic). This is strong evidence the scatter/gather is cheap — it is
routing, not arithmetic — but it is an attribution result, not a standalone
synthesis of the block. Attention is
~69% of the design while being ~9% of the arithmetic, because activation x
activation products cannot be folded into constant shift-add networks.

## Reference: JEDI-linear (Que et al., arXiv:2508.15468, Table II)

Non-permutation-invariant models, 3 input features, same VU13P part, verified
against the published PDF. This is the correct comparison class: PHAT-JeT builds
patches from contiguous slices of a sorted sequence and is not
permutation-invariant.

| Model | N | Accuracy | LUT | FF | Latency | II | Fmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| JEDI-linear | 16 | 71.9% | 44k | 22k | 54 ns | 1 | 354.4 MHz |
| JEDI-linear | 32 | 78.0% | 45k | 26k | 63 ns | 1 | 300.8 MHz |
| JEDI-linear | 64 | 80.9% | 71k | 38k | 61 ns | 1 | 328.4 MHz |
| JEDI-linear | 128 | 80.9% | 98k | 48k | 82 ns | 1 | 257.6 MHz |
| MLP-Mixer | 128 | 79.8% | 83k | 21k | 72 ns | 1 | 208.7 MHz |

**Transformer reference** (arXiv:2510.24784, Laatu, Sun, …, Que — the first FPGA
transformer on this benchmark, same HGQ + distributed-arithmetic toolchain,
Xilinx XCU250, EBOPs-budgeted training, single attention head):

| Model | N | Accuracy | LUT | Latency | II |
|---|---:|---:|---:|---:|---:|
| Linformer (their best) | 64 | 79.8% | 202k | 78 ns | 1 |

Against this — the closest architectural comparator — our N=64 row is **+0.56
accuracy points (80.36 vs 79.8) at 2.7x the LUT** (544,292 / 202k), and
additionally carries GMP at ~zero cost. This is a materially narrower gap than
against the linear-GNN baseline and is the fairer "transformer on FPGA"
comparison — though it uses a different FPGA, so it is an architectural
reference, not a controlled device comparison.

## Caption (ready to paste)

> Resource and latency estimates for PHAT-JeT on an AMD VU13P at 300 MHz.
> Figures are **da4ml estimates from generated RTL, pre-place-and-route**;
> timing closure is not claimed and no bitstream was produced. DSP = 0 and
> BRAM = 0 follow from the distributed-arithmetic mapping, in which
> constant-coefficient products become shift-add networks; II = 1 is a property
> of the fully-unrolled dataflow. FF is reported as pipeline register bits, not
> post-route flip-flop utilisation. The model is quantization-aware trained with
> HGQ2, the same quantization-plus-distributed-arithmetic methodology used for
> JEDI-linear, on the same device — but PHAT-JeT does **not** achieve comparable
> LUT efficiency: at accuracy parity it is roughly an order of magnitude larger.
> We report the measured accuracy/resource trade-off rather than claiming
> resource comparability. JEDI-linear figures are post-place-and-route as
> published and are therefore a favourable comparison for that design.

## Honest limitations

1. **Estimates, not synthesis.** No Vivado run. The single highest-value
   follow-up is to put the generated Verilog through Vivado *synthesis* — this
   would upgrade every row from estimate to synthesis result at low cost.
2. **Not resource-competitive.** At N = 32, 1,477k LUT against JEDI-linear's 45k
   for +1.1 accuracy points; at N = 150, 1,731k against 98k. Roughly 20-30x
   across the sweep. Stated plainly rather than framed away.
3. **Component split is an attribution**, not a partition; CSE across cut points
   can shift cost between adjacent rows.
4. **Bit-exactness verified on the headline operating points.** Emulating the generated netlist against the Keras
   model gives `max|err| = 0.0` and **100% argmax agreement** at
   `inputs_kif = (1,8,16)`, on eight checkpoints end-to-end: N=32 @ 70.5%,
   N=128 @ 29.7% and 30.5% (the flagship pair, `hw/n128pid/`), N=64 @ 31.5%
   (`hw/pid2/`), N=64 @ 31.1%, N=64 @ 37.8% (pre-continuation), N=64 @ 23.1%
   (three-lever), and the N=64 relu variant @ 40.5%. Each emulated
   netlist matches the Keras model exactly (max|err| = 0.0, 100% argmax, 512
   jets). Regenerated buildable RTL for these rows: `hw/FINAL/` on the PVC.
   Rows not in this list are traced but not individually emulated. No FPGA
   synthesis has been run. An earlier run at `(1,5,8)` failed (59.8% agreement) because that format
   truncates the inputs; re-tracing at the verified format changed LUT by only
   +0.1% to +2.6%. All rows below are quoted at `(1,8,16)`. The minimum
   bit-exact format is `(1,6,16)` (23 bits), so the reported areas are slightly
   conservative. Note also that **argmax agreement reaches 100% one fractional
   bit earlier**, at `(1,8,15)`, where logits still differ by up to 0.24 — for a
   classifier the deployed decision is the argmax, so we report the stricter
   criterion than deployment strictly requires. See DECISIONS.md D30/D31/D32.
5. **Single seed.** No replicated runs, so small accuracy differences between
   rows are not separated from noise. The +1.1 and +1.5 point margins over
   JEDI-linear are larger than the row-to-row scatter within an arm, but have not
   been established against seed variance.
6. **The sweep is not fully controlled** — see the note above the table: the
   N = 150 row carries `attn_l1` and the others do not.
