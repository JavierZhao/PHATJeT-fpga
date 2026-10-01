# Morning report — overnight session 5 (goal: close the PHAT↔JEDI gap)

## TLDR

Two of the three directives resolved decisively; the third produced an honest
negative with one live thread.

1. **Pipeline-vs-published gap: FULLY RESOLVED AND EXONERATED.** The −0.45 was
   the constituent store (+0.65 on richer data) and training budget (extended
   schedule reaches **0.8124 on our own store — above the published 80.9**);
   normalization and calibration proven null. Our pipeline reproduces or
   exceeds JEDI-linear's published accuracy given either their data or their
   budget. Nothing attributes to our port.
2. **PHAT accuracy: best-ever from-scratch result** — the rich store + kt sort
   gives **0.8045** (+0.4; D110's old null was purely a sort confound). Knee
   continuation running.
3. **Architecture exploration: the gap did not close structurally.** Every
   authorized variant completed; none beats matched-store JEDI at matched
   budget.

## The completed architecture × store matrix (peaks, 1000 epochs, N=128)

| | standard store | rich store (kt) |
|---|---:|---:|
| PHAT (full attention) | 0.8005 | **0.8045** |
| local attn + linear global (phatlin) | 0.7971 | 0.8025 |
| fully linear + patches + GMP (fullin) | 0.7982 | **0.8059** |
| JEDI-linear (our pipeline) | **0.8048** | **0.8109** |

**Findings the matrix establishes:**
- **Global attention is training scaffolding.** The sparse-trained model
  prunes global Q/K to zero (D114), but *building without it* costs 0.3 pts —
  you can prune it after training, not omit it before. A genuinely interesting
  result for the paper.
- **Attention's value shrinks and inverts with richer data**: on the rich
  store the fully-linear variant beats both attention variants.
- **Patches without attention underperform flat JEDI** — the hierarchy only
  earns its keep with local attention inside it.
- **gmp2 closed**: a second GMP block costs 0.3 pts — geometric context
  saturates after one injection (first block remains +0.5, D129).
- **Win conditions (pre-registered): not met** at matched store+budget. The
  matched-budget accuracy gap to JEDI (~0.4–0.6) survived every structural
  variation tried.

## Where the gap-closure stands honestly

At trigger scale and matched training, JEDI-linear is the stronger
accuracy-per-LUT architecture on this benchmark, and the sessions have now
*measured why* rather than assumed it: fully-foldable arithmetic, ramp-driven
free sparsity, half the pipeline depth, and no scaffolding cost. PHAT's
measured advantages are the multiplicity-binned selection power at paper scale
(your JetClass tables: uniformly best per-bin, margins growing with
multiplicity) and the GMP module (transferable, ~free, +0.5 in PHAT, positive
in JEDI). The remaining accuracy levers that *work* are data (store) and
budget — both now quantified — plus the running phatrich continuation.

## Codex's role (rounds 12–13)

Round 12's forensics found the shipped-artifact fact (their published point is
epoch 1565 @ 157k EBOPs — an early-ramp knee) that redirected the gap hunt to
data/budget; round 13 audited the new architectures (fixed a gmp2 mask bug
pre-burn), ranked the structural slots, killed weak ideas before GPU spend
(shifted patches, geometric attention bias, GMP pyramids), and pre-registered
the win conditions the matrix was scored against.

## In flight / next
- `n128-phatrichc` — knee continuation of the 0.8045 row (PID 600k): PHAT's
  best-accuracy path; trace + possible new table row when it lands.
- Candidate next levers (not launched): extended-budget PHAT (the D132 lesson
  applied to PHAT — a 4000-epoch run), and the D126 matched-EBOPs hybrid trace.
- All deliverables synced; DECISIONS.md through D132 + this session's entries.
