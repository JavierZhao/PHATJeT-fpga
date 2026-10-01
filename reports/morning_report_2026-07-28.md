# Morning report — overnight session 4 (the gap, from first principles)

## TLDR

The PHAT↔JEDI gap is now **measured, decomposed, and partially explained — and
the night's biggest result is that the JEDI+GMP hybrid beats every PHAT row at
matched protocol**: **0.8048 @ 266,382 LUT (15.4% of VU13P), 46.6 ns, bit-exact**
— GMP made the linear baseline 15% *cheaper* at identical accuracy, and the
seed replication running now is confirming it (interim: 0.8063 at seed 2).

## The completed two-sided decomposition (D122)

| | PHAT sparse flagship | our-pipeline JEDI | JEDI+GMP hybrid |
|---|---:|---:|---:|
| accuracy | 0.7990 | 0.8048 | **0.8048** |
| LUT (da4ml est.) | 478,214 | 314,946 | **266,382** |
| latency | 93.2 ns / 28 stg | 46.6 ns / 14 stg | **46.6 ns / 14 stg** |
| quantized zeros | 55.5% (needs ProxL1) | 63.7% (ramp alone) | **68.8%** |

All bit-exact, same data/split/estimator, 1000-epoch matched screens.

**The gap's three measured root causes:**
1. **Unfoldable arithmetic** — attention's data×data lanes can't become
   shift-add networks (attention: 9% of arithmetic, 66–74% of pre-compression
   area).
2. **Ramp intolerance** — the full jsc150 beta ramp gives JEDI 63.7% weight
   sparsity *for free* (bit pressure → zero-bit weights); PHAT collapses under
   it and needed explicit ProxL1 to reach similar sparsity. Attention forfeits
   the baseline recipe's built-in pruning.
3. **Serial depth** — QK→softmax→AV twice over gives 28–43 stages vs JEDI's 14;
   structural, not solver-fixable (solver sweeps move ±1 stage).

Also banked: **GMP attributes ~free at 64 channels too** (−19% CSE credit,
refuting the "not-free at width 64" prediction), and JEDI's cost map is evenly
spread constant layers (sender 29%, embed 24%, receiver 18%, node 14%) — no
attack surface, which is why it's cheap.

## Overnight arm verdicts

**Closed:**
- **Wall test (wvpid2):** the ~400k-EBOPs floor at N=128 held against
  sparsity+PID pushing (target 300k, equilibrated 400–408k @ 0.7965). Third
  independent confirmation — it's an information floor, not an optimization
  artifact. More float zeros (23.8%) may still buy a few % LUT; trace queued.
- **SAT_SYM weights (satb):** dead in our recipe — EBOPs stuck at 2M, 1.9%
  zeros, −1.6 pts. Saturating weights never *need* narrow bits, so the gentle
  ramp exerts no compression force (JEDI's SAT_SYM works only because their
  100× ramp supplies it). Also learned: overflow modes change the quantizer
  variable schema — wrap checkpoints can't resume into SAT_SYM graphs.

**In flight (interim reads, all healthy):**
- **Seed replication of the hybrid result:** jedis2 at 0.8042 @ 141k EBOPs
  (even better compression than seed 1); jedigmps2 at 0.8063 mid-run. The
  headline hybrid claim is replicating.
- **Ramp tolerance (the causality test):** control at full ramp peaked 0.7654
  (−3.5 pts, the expected damage); the protected arm (attention lanes at 0.01×
  beta) is at **0.7781 and ahead of the control at an earlier epoch** — early
  signal that lane protection recovers ramp tolerance. Verdict + sparsity
  comparison when both finish.
- **Width hypothesis:** d32wv holding 0.8009 under ProxL1 mid-compression —
  d=32 not collapsing (unlike d=12), harvest phase still running.
- **Factorial cell (nogmp):** 0.7919 at epoch ~400 — needs end-of-schedule for
  the PHAT−GMP comparison.

## Codex round 11 (its report: round11_report.md, in outputs)

Delivered the selective-beta implementation sketch (stock HGQ2 scheduler
overwrites per-layer beta0 — the callback with an explicit six-lane allow-list
is what we built and launched), the pipeline-depth verdict (don't promise
shallower PHAT; the serial chains are structural), the hybrid's evidence
requirements (paired seeds ✓ running; PHAT−GMP cell ✓ running; 7000-epoch pair
optional), and the strategic framing: **GMP as the paper's transferable
hardware contribution** — free geometric context that improves accuracy in
PHAT and cuts 15% LUT in JEDI — pending the factorial's completion.

## Incidents
- ramp-prot exhausted retries on the cuDNN flake (anti-affinity kept forcing it
  onto a bad node with 8 siblings holding the good ones) — relaunched without
  anti-affinity, now healthy.
- The D121 "GMP null" interim was a grep artifact (pattern missed 0.80+ rows);
  corrected in D122 within hours, before anything shipped.

## Next (as arms land)
1. Consolidated trace batch: hybrid seed-2 rows, ramp-prot winner, d32wv
   winner, nogmp end-state, wvpid2 candidate.
2. The factorial 2×2 verdict and, if ramp tolerance confirms, a
   ramp-tolerant + ProxL1 combined arm — the first genuinely new path below
   the wall.
3. Consolidation entry (D124) + updated gap analysis in the campaign report.

All deliverables synced to `~/outputs/phatjet/`. DECISIONS.md through D123.
