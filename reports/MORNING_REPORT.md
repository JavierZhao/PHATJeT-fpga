# Overnight report — 2026-07-24 into 2026-07-25

One-page summary. Details in `DECISIONS.md` (43 entries); numbers in
`synthesis_table.md`.

## Headline

**The paper's own N=150 configuration now fits a VU13P at 48% utilisation.**
It did not fit at all when the night started — the best design was 431% of the
device.

| | Start of night | End of night |
|---|---|---|
| Best N=150 design | 7,449,939 LUT (**431%** of VU13P) | 829,046 LUT (**48.0%**) |
| Accuracy there | 0.8101 | 0.7758 |
| Latency | 186.5 ns | 106.6 ns (at the 51.5% point) |
| RTL verified against model | not tested | **bit-exact on all 3 headline rows** |

**9x area reduction, at a cost of 3.4 accuracy points at that cheapest point**
(the best-accuracy fitting rows give 11.4x at 0.8 points — see the curve). Also produced a full deployment curve rather than a single
point.

## The rebuttal artifact

At the published N=150, a usable operating curve:

| Accuracy | LUT | % VU13P | Latency |
|---:|---:|---:|---:|
| 77.58% | 829,046 | 48.0% | 136.5 ns |
| 78.11% | 890,038 | 51.5% | **106.6 ns** |
| 78.98% | 1,130,657 | 65.4% | 149.8 ns |
| 80.04% | 1,731,475 | 100.2% | 153.2 ns |

Best overall across all configurations: **N=64 with attention AND FFN pressure,
80.23% at 652,680 LUT (37.8% of the device), 143.2 ns** — within 0.7 accuracy
points of JEDI-linear at the same particle count, leaving ~62% of the device free.
That is an **11.4x area reduction** from where the night started.

All rows: II=1, 0 DSP, 0 BRAM, xcvu13p at 300 MHz.

## What to claim, and what to concede

**Claim:** the architecture generates RTL whose emulation is verified bit-exact
against the trained model on all three headline operating points (no FPGA
synthesis has been run); fits the target device at the published particle count with ~half
the device free; is accuracy-competitive with the deployed baseline at matched N
(+1.5 at N=16, +1.1 at N=32, -0.7 at N=64); and exposes a tunable
area/latency/accuracy curve spanning 21-100% of the device.

**Concede plainly (already written into the caption):** not resource-competitive.
13-27x JEDI-linear's LUT depending on operating point. Do not use "comparable
resources", "competitive efficiency" or "modest overhead".

**Bonus finding worth putting in the paper:** GMP — the architecture's
distinguishing contribution — shows zero marginal LUT cost under component
attribution, consistent with its scatter/gather being one-hot routing rather
than arithmetic (attribution result, not standalone synthesis). Independent of the
resource story, this is a positive result about the design.

## What produced the win

Per-region quantization pressure on the QK/AV attention datalanes (`attn_l1`),
after a component trace showed attention was 66-74% of the design while being
only ~9% of the arithmetic. Raising it 10x removed 40% of the area for 0.02
accuracy points. Global EBOPs pressure is an independent second lever; the two
are still being combined.

The diagnosis came from Codex's first review. My first implementation of its
recommendation was the wrong mechanism (an initial-value knob rather than
pressure) and had to be redone.

## Honest accounting

Seven of my own conclusions were overturned and are recorded with evidence:
two cluster misdiagnoses, "BatchNorm is not free", a softmax/heads prediction
falsified by controlled trace (4.1% measured against 24% predicted), D17's
N-scaling claim, D6's beta guidance, and a bit-exactness failure that resolved to
my own harness bug.

Three real bugs found and fixed: missing N-slicing in `extract_hw.py`; illegal
Verilog module names (**every RTL project generated before the fix would also
have failed in Vivado**); and a hardcoded input format that silently produced a
netlist disagreeing with the model on 40% of predictions.

Method note (D21): every time a mechanism was inferred from a differenced or
correlated measurement without an independent test, it was wrong. Controlled
traces of untrained models settled in twenty minutes what a night of training
arms had muddled.

## Decisions needing you

1. **Which rows go in the rebuttal.** Suggest the N=150 curve (it is the
   published configuration) plus the N=64 best-accuracy row.
2. **Whether reduced-N rows are acceptable** as disclosed configuration changes.
   JEDI-linear reports N=16/32/64/128 itself, so they are matched comparison
   points, but it is your call.
3. **The Vivado ask.** Highest-value remaining action and not mine to take:
   da4ml emits `build_vivado_prj.tcl`, so running the generated RTL through
   Vivado *synthesis* is one command and upgrades every row from estimate to
   synthesis result. Chang Sun is the named contact.

## Still running

Seven arms: combined global-beta + attention-pressure (`n150-both`, `n64-both`),
FFN per-region pressure (untested lever, 20-27% of the design), and two sampling
the attention saturation band. Results land over the next few hours; none will
change the claims above, only potentially improve the numbers.
