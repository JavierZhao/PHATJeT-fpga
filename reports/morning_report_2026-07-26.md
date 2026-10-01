# Morning report — overnight session 2 (2026-07-25 22:00 → 2026-07-26 ~10:00)

## Headline: both matched-N rows transformed, all verified bit-exact

| Row | Last night | This morning | vs JEDI-linear |
|---|---|---|---|
| **N=128 flagship** | 0.8005 @ 662,654 LUT, 133.2 ns | **0.7988 @ 513,253 LUT (29.7%), 93.2 ns** | **5.2×** (was 6.8×) |
| **N=64 headline** | 0.8023 @ 652,680 LUT, 143.2 ns | **0.8036 @ 544,292 LUT (31.5%), 139.9 ns** | **7.7×** (was 9.2×) |

Every number bit-exact-emulated at `inputs_kif=(1,8,16)` (max|err|=0, 100%
argmax); RTL on the PVC (`hw/pid2/`, `hw/n128pid/`). The matched-N resource
multiple vs JEDI-linear is now **5.2–7.7×**, and the N=128 flagship is also the
outright **latency champion at 93.2 ns / 28 stages** — faster than the N=64
relu variant (116.5 ns) at higher N and accuracy.

**Mechanism (D78/D84/D86):** the 1000-epoch runs were undertrained at their
operating points. A second 1000-epoch continuation (PID-held beta) sheds
~17–26% EBOPs/LUT at held accuracy. The knee is real and replicated: PID and
constant-beta continuations independently converged to the same N=64 point
(0.8036 @ 427,680 vs 0.8032 @ 427,773 EBOPs), and pushing below it collapses
accuracy (0.73 @ 360k) — a genuine accuracy/area wall at ~400k EBOPs.

## New float ceiling: 0.8110

A clean 300-epoch float retrain (D91) beat the old 0.8064 reference. Frozen at
`teachers/f64/float_phatjet.keras`. The QAT knee (0.8036) is now 0.74 pts below
the float ceiling — that headroom is what the two relaunched arms are chasing.

## Incident caught by audit tooling (D89)

Both distillation arms and the twostage arm trained against a **half-trained
teacher (0.74)**: twostage's float stage was cuDNN-killed at epoch 11, its
retry skipped float because the checkpoint file existed, and that best-so-far
snapshot silently fed three consumers. The teacher-val-acc print Codex demanded
in its round-4 audit is what exposed it. All three arms invalidated and killed;
relaunched with a frozen-teacher convention + `PJ_DISTILL_MIN_ACC=0.79` gate.
Lesson recorded: never point a reader at a `save_best_only` path a restartable
job owns.

## Directions closed tonight (all with evidence, D79–D92)

- **core7 grid** — clean A/B on identical pT-sorted data: Δacc 0.0006, +8% EBOPs. Final.
- **Sigmoid attention family** — 4th strike (hybrid diverged to NaN @ epoch 82). Dead.
- **Bounded relu (relub)** — 5-minute CPU trace screen: −0.9% LUT, accuracy 0.42. No GPU spent.
- **GMP enrichment** — both axes: 5×5 kernel (−0.8 pts, 2× EBOPs) and 16×16 grid (−0.5 pts, +11%). The block is at capacity; "attributed-free" was a property of the 3×3/8×8 point.
- **Weight-bit pressure** — correctly wired (`br` on kbi, after Codex caught the mis-wiring): no EBOPs reduction at any accuracy.
- **relu attention at N=128** — verified 0.7964 @ 591k / 116.5 ns: softmax now dominates on area AND latency at both N; relu's only edge is −11% FF.
- **pid4 hops** — knee mapped; gate failed; area work at N=64 concluded.

## Caveats now attached to the claims (D87, Codex round 5)

- **Seed variance, RESOLVED (D93):** mid-schedule checkpoints vary by ~1.5 pts
  across seeds, but the *operating point* reproduced within 0.1–0.3 pts once
  continued to the knee (seed-1337: 0.8002 @ 408k vs seed-42: 0.8012 @ 405k).
  The knee claim carries an honest n=2-seed bar; the N=128 flagship has one seed.
- **No untouched test split exists in the data store** — all numbers are
  validation accuracy, selected via val-based ParetoFront. **Question for you:**
  does an untouched hls4ml test file exist upstream? If yes, one frozen rescore
  of the claimed rows.
- Ours are da4ml pre-place-and-route estimates vs JEDI-linear's post-route
  figures (already caveated in the table).

## In flight (2 arms)

- `n64-distill2` — softmax distillation from the 0.8110 teacher (queued on GPU scarcity, will start when the cluster frees up).
- `twostage2` — QAT init by float transfer (BW=10, i0=3/8) from the 0.8110 teacher (same queue).

## Standing asks (unchanged)

- **Chang Sun:** Vivado synthesis of `hw/FINAL/` + `hw/pid2/` + `hw/n128pid/`
  RTL to upgrade estimates → synthesis results.
- **Aaron:** his `jets_128x3.npz` + `robust_stats.json` + float ckpt for exact
  replication (his store has 49 const/jet vs our 37.8 — not reconstructable).

Full log: `DECISIONS.md` D76–D92 (17 new entries tonight). Deliverables synced
to `~/outputs/phatjet/`.
