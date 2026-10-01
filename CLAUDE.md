# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `DECISIONS.md` first.** It is the running record of what has been decided,
the evidence behind each decision, and what was rejected. This file describes the
machinery; that one describes the state of the investigation.

## What this working area is for

Not the PHAT-JeT model repo. It is a workspace producing **FPGA resource/latency
evidence** for the PHAT-JeT NeurIPS 2026 rebuttal (submission 28409). See
`claude_code_fpga_handoff.md` for the original brief — but note the toolchain
decision (D1) diverged from it deliberately.

> **Status 2026-07-31 (repo copy):** the "~76x" below is the D10-era figure
> and is stale. Current: PHAT 0.8108 @ 1.57M LUT vs JEDI-linear in our pipeline
> 0.8109 @ 189k (~8x), ~16x vs published. The wall is now the constant Q/K/V
> projections (D111), not attention datalanes. See README.md §1–2 and D173–D175.

**The headline finding, so you do not rediscover it:** at accuracy parity with
JEDI-linear (81.08% vs 80.9%) the design costs ~76x the LUTs (7,450k vs 98k).
The claim has been narrowed accordingly (D10). Attention is 66–74% of the design
while being 9% of the arithmetic (D8, D13), and that is the structural wall.

## Layout

| Path | Role |
|---|---|
| `phat_jet_k3.py` | The synthesizable model. Authoritative architecture. |
| `train_qat.py` | HGQ2 QAT. Everything is env-configurable (`PJ_*`). |
| `train_float.py` | Float reference for the same graph — the accuracy ceiling. |
| `prep_data.py` | Builds the kt-sorted npz; reverse-engineers the sort key. |
| `extract_hw.py` | da4ml trace → LUT/latency + Verilog + bit-exactness. |
| `component_costs.py` | Per-component LUT attribution by prefix differencing. |
| `analytical_estimates.py` | Op counts and analytical estimates (Track A). |
| `sweep_status.sh` | Job states + Pareto fronts across all runs. |
| `DECISIONS.md` | Decision log. Read this first. |
| `PHAT-JeT/` | Upstream clone. Read-only reference, not synthesizable. |
| `HGQ2-examples/` | Upstream clone. `jsc150/` is the reference recipe — consult it before inventing anything. |

## Environment

`.venv` exists (`python3 -m venv .venv`, then `pip install "keras>=3.11" hgq2
da4ml jax[cpu] numpy torch`). Use `.venv/bin/python`.

**The PyPI package is `hgq2`, not `hgq`.** `pip install hgq` silently gets you
HGQ **v1**, which pins `keras<3` and will wreck the environment.

## Running work: everything goes to the cluster

Training runs as Kubernetes Jobs in the NRP Nautilus `cms-ml` namespace.

- **Prefix all job names `zz-`.** `zh-` jobs belong to someone else.
- Source is shipped via a ConfigMap, not a git clone:
  ```
  kubectl create configmap zz-phatjet-src -n cms-ml \
    --from-file=phat_jet_k3.py --from-file=train_qat.py --from-file=prep_data.py \
    --from-file=train_float.py --from-file=extract_hw.py --from-file=component_costs.py \
    --dry-run=client -o yaml | kubectl apply -f -
  ```
- Data and all outputs live on the `j-jepa-vol` PVC under `/data/phatjet-fpga/`:
  `jets_150x3_kt.npz`, `runs/<tag>/{pareto/,train.log,run_config.json}`,
  `hw/<tag>/`.
- The original job generator (`gen_torch.py`) was lost with its session scratchpad;
  `k8s/job_template.yaml` is a reconstructed, untested replacement.
- `sh sweep_status.sh` for state + Pareto fronts.

### Cluster gotchas, all of which have bitten

- **Use `KERAS_BACKEND=torch`.** The jax/XLA path dies non-deterministically with
  `cudnn status 5003` on the depthwise conv (D12). Two earlier diagnoses (bad
  nodes, then GPU memory contention) were both wrong.
- **Use `command: ["/bin/bash", "-c"]`.** The PyTorch image's `/bin/sh` is dash
  and has no `set -o pipefail`; jobs die at line 2 with "Illegal option".
- **Always `set -o pipefail`.** `python ... | tee` reports success after a crash,
  so a crashed run shows as `Complete`.
- **Trust the epoch counter, not the k8s status column.**
- Set `PJ_DATA=""` for trace-only jobs — loading the 1.6 GB npz from
  cephfs-central stalled a job for 15+ min at 0% CPU.
- Generic `nvidia.com/gpu: 1` schedules far faster than `nvidia.com/a100`.

## The pipeline

```
phat_jet_k3.build_phat_jet_k3(quantized=False) → train_float.py → float ceiling
                             (quantized=True)  → train_qat.py   → pareto/*.keras
                                                       ↓
                              extract_hw.py / component_costs.py → LUT, latency, Verilog
```

`build_phat_jet_k3` emits float and quantized graphs from **one code path** with
identical layer names. `extract_hw.py` and `component_costs.py` rebuild the
architecture from `PJ_*` env and call `load_weights` — the GMP bin indicators are
closures and cannot be deserialized. **The env must match the checkpoint's
training config exactly** or weights load into the wrong graph; each run now
writes `run_config.json` next to its checkpoints for this reason.

## Constraints that shape the model code

The HGQ2 → da4ml flow requires static shapes, only registry ops, no dynamic
indexing, and ops applied **functionally** (custom `Layer` subclasses are
invisible to the tracer). That is why the file uses `_dense`/`_einsum`/`_softmax`
helpers switching on `quantized` rather than layer classes, and why the GMP
scatter/gather is re-expressed as one-hot indicators + einsums. Preserve all
four. Upstream `PHAT-JeT/models/PHAT_JeT.py` violates every one — do not port
from it verbatim.

## Standard particle counts: 128 and 64

All runs use **N = 128 and N = 64** (D40). JEDI-linear reports N = 16/32/64/128,
so N = 150 is not a matched comparison. N = 128 requires **P = 8**; the paper's
P = 10 does not divide 128. Existing N = 150 results are kept as context but are
not matched comparisons.

## Conventions

- Model outputs **logits**; train with `from_logits=True`.
- pT is scaled by `1/64` in `train_qat.py`, `train_float.py` and `extract_hw.py`.
  Change one, change all.
- Target `xcvu13p-flga2577-2-e` at 3.33 ns (300 MHz) — JEDI-linear's clock.
- `dsp=0, bram=0, ii=1` in `extract_hw.py` are **assumptions written into the
  row**, not values read from a synthesis report. Same for `reg_bits` as an FF
  proxy and 300 MHz as a target rather than closed timing. Caveat them.

## Deviations from the paper

Enumerated in `phat_jet_k3.py`'s module docstring. **Never silently change model
config to make synthesis easier** — flag it and add to that list. Note the
docstring's original claim that 99.9% of constituents fall within ±0.4 is FALSE
for the robust-normalised store (measured: 34.1%).

## Escalation

Chang Sun (`calad0i`) — HGQ2/da4ml, and the highest-leverage external ask: run
the generated Verilog through Vivado *synthesis* to upgrade every number from
estimate to synthesis result. Zihan — model config vs paper. Aaron Wang
(`aaronw5`) — upstream PHAT-JeT.
