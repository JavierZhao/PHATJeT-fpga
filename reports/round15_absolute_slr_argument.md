# A. SLR Budget Audit

The arithmetic is defensible **as a screening budget**: AMD specifies 1,728k CLB LUTs and 3,456k CLB flip-flops for VU13P, so an *even* four-way division gives 432k LUTs and 864k FFs; D135 records the four-SLR assumption (`DECISIONS.md:3619-3620`). Use **CLB LUTs**, not AMD's 3,780k “system logic cells,” which is not the resource reported by da4ml. The repo has no per-SLR floorplan proving all four are identical. **UNVERIFIED:** exact usable LUT/FF capacity of each physical SLR; obtain Vivado `report_utilization -slr`.

The strongest honest statement is therefore: “the pre-P&R estimate of 385,471 CLB LUTs is 89.2% of an arithmetic quarter-device LUT budget, with 270,104 register bits (31.3% of the analogous FF budget), while its estimated 89.9 ns and assumed II=1 meet the stated latency/II thresholds” (`synthesis_table.md:53`; `DECISIONS.md:3625-3627`; caveats at `synthesis_table.md:223-229`). Do **not** say that it has been placed in one SLR. No repo evidence supports a universal “20–30% placement slack” rule. **UNVERIFIED:** routability at 89.2%; only synthesis plus implementation inside a one-SLR pblock can establish it.

The 27-stage fully unrolled graph (`DECISIONS.md:3076-3079`) does not inherently require SLR crossings; if constrained wholly to one SLR, it has none internally. But density, high-fanout nets, control sets, carry packing, and congestion can prevent placement or 3.33 ns timing. Unconstrained placement may spread it across SLRs, invalidating the interpretation. The 404,852-LUT point is tighter at 93.7%, not 93.2% (`DECISIONS.md:3628`).

# B. What Else Must Be Shown

- **(i) Throughput/rate.** The paper says CTL2 has 5 slices × 6 round-robin nodes and five FPGAs available at once, and that tagging processes multiple jets independently/concurrently (arXiv:2508.15468 §IV-A; quoted in `DECISIONS.md:3615-3617`). At the 40 MHz collision cadence stated in the paper (§I), uniform six-node round-robin implies an **estimate** of 6.67 Mevents/s/node. **UNVERIFIED:** the paper gives no jets/event count, so 10–16 jets/event and hence 67–107 Mjets/s/node (not 80–130 M) are assumptions. An II=1 instance would nominally accept 300 Mjets/s at the 300 MHz target (`synthesis_table.md:21`), but this margin is conditional on timing closure, interface supply, and whether “concurrently” requires replicated units.

- **(ii) Deterministic latency.** A fixed 27-stage static pipeline strongly supports deterministic *block* latency: 27 × 3.33 ns = 89.9 ns (`DECISIONS.md:3076-3079`), matching the paper's deterministic-latency requirement (§IV-A). It still needs RTL/interface validation under reset, backpressure, and valid gaps; bit-exactness alone does not test system scheduling.

- **(iii) Latency context.** 89.9 ns is 1.80% of 5 μs and 0.72% of 12.5 μs by arithmetic. **UNVERIFIED:** neither 12.5 μs nor a 5 μs correlator allocation appears in the cloned paper text/repo; do not use these percentages without the primary CMS source. The direct, defensible comparison is 89.9 ns versus the paper's sub-100 ns tagger requirement (paper §II-A; `DECISIONS.md:3614`), not the end-to-end L1 budget.

- **(iv) Resource co-tenancy.** The mapping is expected to use no DSP/BRAM because constant products become shift-add logic, but these are assumptions, not synthesis results (`CLAUDE.md:76-79`; `synthesis_table.md:223-228`). AMD's device totals are 12,288 DSPs and 2,688 BRAM blocks, but “leaves all to clustering” overstates matters: PHAT-JeT still consumes an estimated 89.2% of one SLR's LUT fabric and routing, while clustering shares the FPGA (paper §IV-A). The point is useful as **resource-type complementarity**, not proof of easy co-tenancy.

- **(v) I/O bandwidth.** This is a gap. The verified input format is 23 bits × 128 × 3 = 8,832 bits/jet (`DECISIONS.md:870-877`); accepting one whole jet each 3.33 ns implies 2.65 Tb/s of aggregate internal payload. That can be a very wide on-chip interface from buffering/preprocessing, not necessarily package I/O, but the repo shows no interface, buffering, fanout, or routing integration. The paper places tagging downstream of clustering/sorting/buffering (§IV-A), which makes an on-chip handoff plausible but does not validate it.

- **(vi) Fmax risk.** Say “evaluated at a 3.33 ns/300 MHz target,” never “runs at 300 MHz.” The repo explicitly says timing is not closed and no bitstream exists (`CLAUDE.md:76-79`; `synthesis_table.md:223-225`). Absolute feasibility requires constrained post-route timing at the intended SLR and interface; failure at 300 MHz would simultaneously reduce throughput and make 27-cycle latency exceed 89.9 ns.

# C. Ranked Next Experiments

1. **Constrained Vivado synthesis and P&R of the existing 385,471-LUT RTL in one SLR.** This directly tests the claim's largest uncertainty: actual resources, routing, SLR confinement, and 300 MHz timing. It is more valuable than another QAT arm; D135 already supplies the candidate (`DECISIONS.md:3627`).
2. **Trace a deployment curve and prioritize a ≤70%-SLR N=128 point.** Several bit-exact accuracy/LUT points inside or near the envelope make the result robust rather than binary; ≤302,400 estimated LUT would provide co-tenancy/routing headroom. This complements the running `slr-room` ~58% target and `slr-champ` accuracy search (`DECISIONS.md:3641-3644`).
3. **Latency sweep within the one-SLR set, targeting <80 ns.** This adds timing/latency margin, but hard_dc has so far bottomed at 27 stages/89.9 ns (`DECISIONS.md:2917-2923`), so it is less likely to overturn the deployment case than P&R or area headroom.
4. **Package the traced points as a LUT-versus-accuracy figure** after results exist. The figure is high rhetorical value but low evidentiary value by itself; label accuracy as selected on the canonical evaluation partition (`DECISIONS.md:2912-2915`).

# D. Claim Drafts

**Paragraph.** For an N=128, three-feature, single-PHAT-block configuration, PHAT-JeT generates bit-exact RTL whose da4ml pre-place-and-route estimates are 385,471 CLB LUTs, 270,104 pipeline-register bits, and 27 cycles (89.9 ns at a 300 MHz target), with II=1 assumed (`synthesis_table.md:53`; `DECISIONS.md:3076-3079`). The LUT estimate is 89.2% of an arithmetic 432k-LUT quarter-device proxy for one XCVU13P SLR, while the paper requires no more than one SLR, sub-100 ns latency, and sub-10 ns II (arXiv:2508.15468 §§I–II-A; `DECISIONS.md:3610-3614`). This is evidence of a candidate inside the nominal absolute envelope, not proof of one-SLR implementation: Vivado synthesis, constrained placement/routing, and 300 MHz timing closure remain undone. DSP=0, BRAM=0, and II=1 are expected by construction rather than confirmed in a synthesis report, and the result covers only the tagging block, not clustering, interfaces, or system integration (`CLAUDE.md:76-79`; paper §IV-A). Accuracy was checkpoint-selected on the same canonical evaluation partition used for reporting, so a frozen independent rescore is still required (`DECISIONS.md:2912-2915`).

**One line.** PHAT-JeT has a bit-exact N=128 RTL candidate estimated pre-P&R at 385,471 CLB LUTs (89.2% of a nominal one-SLR proxy), 89.9 ns at a 300 MHz target, and assumed II=1—promising absolute feasibility pending constrained Vivado implementation and independent accuracy rescore.

# E. Red Team

- “You have not shown it fits”: no per-SLR resource report, pblock, routed utilization, congestion result, or SLR-crossing count.
- “You have not shown timing”: 300 MHz and therefore 89.9 ns/300 Mjets/s are targets derived from stage count, not achieved post-route metrics.
- “Your 89.2% LUT estimate leaves little physical margin,” and LUT count alone ignores routing, carry placement, control sets, clocks, and co-tenant floorplanning.
- “FF is only register-bit proxy; DSP/BRAM/II are asserted by construction,” not read from Vivado (`synthesis_table.md:223-229`).
- “Where is the integration contract?” No proof covers the 8,832-bit/jet handoff, buffering, valid/backpressure behavior, clock-domain crossings, jet scheduling, or synchronization.
- “One tagger may not meet the real workload”: jets/event, truncation policy, bursts, concurrency, and node allocation are unverified.
- “Clustering also lives here”: zero DSP/BRAM does not compensate for consuming nearly one SLR's LUT/routing budget (paper §IV-A).
- “This is only one ML block and selected accuracy”: no full firmware build, power/thermal analysis, reset/error behavior, independent test selection, or robustness study.
