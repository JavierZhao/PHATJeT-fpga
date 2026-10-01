# Vivado handoff — PHAT-JeT rebuttal RTL (for Chang Sun)

**What changed vs the earlier version of this file:** the ask is now specific.
Our central claim has become an *absolute* deployment claim — that the tagger
fits the CTL2 envelope (one SLR, sub-100 ns, II=1) — and the one thing we
cannot establish ourselves is whether the design **places and routes inside a
single SLR at 3.33 ns**. That is the critical path; everything else is done.

All RTL below is da4ml-generated, fully unrolled, and **Verilator-emulated
bit-exact** against the trained quantized model (max|err| = 0, 100% argmax on
512 jets). Each directory has `build_vivado_prj.tcl`, `src/`, `sim/`,
`metadata.json`. Target: `xcvu13p-flga2577-2-e`, 3.33 ns.

## The ask, in priority order

**1. Constrained one-SLR implementation of the primary candidate.**
Directory: **`hw/final-day/verilog_epoch_565_val_acc_0_7961_ebops_393279_0`** (N=128, 0.7961).
da4ml estimate: **385,471 CLB LUTs, 270,104 register bits, 27 stages**.

Specifically, if possible:
- create a **pblock confined to a single SLR** and place the design inside it;
- `report_utilization -slr` (and pblock utilization) post-synth and post-route;
- **post-route timing at 3.33 ns** (or the achieved Fmax if it misses);
- SLR-crossing count if the tool spreads it despite the constraint;
- any congestion / routability warnings.

This single run converts our claim from "candidate within a nominal quarter-
device LUT budget" to "placed and routed in one SLR at N MHz" — or tells us
honestly that it does not fit, which we would also rather know now.

**2. Unconstrained synthesis of the same design** (if the pblock run is
awkward): post-synth LUT/FF/DSP/BRAM and estimated Fmax still upgrades every
number we quote from estimate to synthesis result, and confirms the
DSP=0/BRAM=0 expectation we currently label as an assumption.

**2b. The accuracy-maximising point, if you have appetite for a big run.**
**UPDATED 2026-07-31 — these now beat JEDI-linear's reported accuracy.**

| what | directory | accuracy | da4ml LUT | % of VU13P |
|---|---|---:|---:|---:|
| **preferred** | `hw/trace101/kif_1_8_20/verilog_epoch_152_val_acc_0_8101_ebops_1538843_0` | **0.8101** | **1,461,880** | **84.6%** |
| max accuracy | `hw/trace106/kif_1_8_20/verilog_epoch_278_val_acc_0_8106_ebops_1549720_0` | 0.8106 | 1,591,089 | 92.1% |

Both are N=128, II=1, 50–51 stages (166.5–169.8 ns at a 300 MHz target), and
reproduce the trained model exactly in both da4ml's DAIS interpreter (2,000
jets) and Verilator (512 jets). **Please use the preferred row** — 84.6%
leaves ~15% for placement/routing, whereas 92.1% may not route.

These are whole-device candidates, not one-SLR candidates; a one-SLR pblock
will not fit them and should not be attempted. Unconstrained synthesis is the
useful run: post-synth LUT/FF/DSP/BRAM plus estimated Fmax.

Superseded: `hw/kifscan/.../epoch_773_val_acc_0_8077...` (0.8077, 82.8%).

**3. Additional points, only if 1–2 are cheap:**

| Row | Directory | da4ml estimate |
|---|---|---|
| N=128 cheap | `hw/wv10/verilog_epoch_979_val_acc_0_7957_ebops_427029` | 404,852 LUT / 28 stg |
| N=128 accuracy | `hw/wv10/verilog_epoch_64_val_acc_0_7990_ebops_502148` | 478,214 LUT / 28 stg |
| N=128 fast pipelining | `hw/latsweep/verilog_epoch_901_val_acc_0_7988_ebops_494497_0` (hard_dc=1) | 515,055 LUT / 27 stg |
| N=64 | `hw/n64wv/verilog_epoch_693_val_acc_0_8029_ebops_385065` | 487,451 LUT / 41 stg |

## Notes

- Module names are sanitized (`[^A-Za-z0-9_] -> _`); RTL generated before
  2026-07-25 is not buildable — use only the directories listed here.
- The primary candidate has ~62% of its quantized weights exactly zero, so its
  adder graphs are much smaller than earlier drops; synthesis should be quick.
- We do **not** have an interface/integration story: the design consumes
  128×3 quantized inputs (8,832 bits/jet) per cycle, which we assume is an
  on-chip handoff from the clustering/sorting/buffering stage upstream. If that
  assumption is wrong in CTL2, please say so — it would change what we claim.
- Context: methodology and provenance for these specific rows are in
  `DECISIONS.md` D111–D135; contact Zihan for model/config questions.
