"""
Per-component da4ml cost attribution for PHAT-JeT.

WHY
---
`extract_hw.py` emits one row for the whole model, but the rebuttal table wants
a row per component (GMP / local attention / global MHSA / FFN / head), and we
cannot currently say whether attention's data-dependent multiplies or the sheer
per-particle datapath dominates the LUT count.

METHOD (cumulative differencing)
--------------------------------
Components are not separable subgraphs -- residual streams mean a component's
inputs come from earlier ones. So instead of tracing components in isolation we
trace PREFIXES of the graph:

    cost(component k) := cost(prefix ending at k) - cost(prefix ending at k-1)

Each prefix is a keras.Model from the real input to a named cut-point tensor,
traced with the same HWConfig and input format as extract_hw.py.

LIMITS -- read before quoting these numbers:
  * Differences can be slightly negative or inflated where da4ml's common
    subexpression elimination spans a cut, because a shared term charged to
    prefix k-1 may be reused free in prefix k.
  * A prefix ending mid-block still carries all the logic feeding it, so the
    attribution is "marginal cost of extending the graph to here", not
    "cost of this block in isolation".
  * Prefixes exclude anything downstream, so the last row absorbs the head.
It is an attribution, not a partition. Label it as such.

Usage:  python component_costs.py <ckpt.keras>
        (PJ_* env must match the checkpoint, exactly as for extract_hw.py)
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import hgq.layers  # noqa: F401

from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace

VU13P_LUT = 1_728_000

# (label, cut-point layer name). Order must follow the dataflow.
# PJ_FINE=1 splits the attention blocks into Q/K/V projection, QK, softmax and
# AV. D19 found LUT/EBOP rising as the model gets cheaper, implying a fixed cost
# that is not multipliers -- most likely softmax tables and datalane routing.
# This is the cut that tells us whether that is true.
COARSE = [
    ("Embedding",                    "embed"),
    ("GMP (bins+scatter+conv+gather+pointwise)", "gmp_pointwise"),
    ("Local patch attention",        "local_attn_wo"),
    ("Patch pooling + global message stage", "patch_msg_proj"),
    ("FFN",                          "ffn2"),
    ("Aggregation + classifier head", "head_out"),
]

FINE = [
    ("Embedding",                    "embed"),
    ("GMP",                          "gmp_pointwise"),
    ("local: Q/K/V projections",     "local_attn_wv"),
    ("local: QK scores",             "local_attn_qk"),
    ("local: SOFTMAX",               "local_attn_softmax"),
    ("local: AV",                    "local_attn_av"),
    ("local: output proj",           "local_attn_wo"),
    ("linear-global: sender",         "patch_send"),
    ("linear-global: pool + receiver", "patch_recv"),
    ("global: Q/K/V projections",    "patch_attn_wv"),
    ("global: QK scores",            "patch_attn_qk"),
    ("global: SOFTMAX",              "patch_attn_softmax"),
    ("global: AV",                   "patch_attn_av"),
    ("global: merge/out + msg proj", "patch_msg_proj"),
    ("FFN",                          "ffn2"),
    ("Aggregation + head",           "head_out"),
]

# JEDI-linear architecture (PJ_ARCH=jedi/jedigmp) — its layers are jedi_*;
# GMP cut only present in the hybrid (missing cuts are skipped gracefully).
JEDI = [
    ("Embed (3->64)",               "jedi_embed"),
    ("GMP (hybrid only)",           "gmp_pointwise"),
    ("Sender (64->64)",             "jedi_send"),
    ("Pool + receiver + add",       "jedi_add"),
    ("Node2 (64->64)",              "jedi_node2"),
    ("Global pool",                 "jedi_pool2"),
    ("Head 64",                     "jedi_head1"),
    ("Head 32",                     "jedi_head2"),
    ("Head 16",                     "jedi_head3"),
    ("Output",                      "jedi_out"),
]

if os.environ.get("PJ_ARCH", "phat") in ("jedi", "jedigmp"):
    CUTS = JEDI
else:
    CUTS = FINE if os.environ.get("PJ_FINE") else COARSE


def main():
    from train_qat import build_qat_model

    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    in_kif = tuple(int(v) for v in os.environ.get("PJ_INPUTS_KIF", "1,5,8").split(","))
    outdir = os.environ.get("PJ_OUT", ".")

    model = build_qat_model()
    if ckpt:
        model.load_weights(ckpt)
        print(f"loaded {ckpt}")
    else:
        print("NO CHECKPOINT -- tracing an untrained model; bitwidths are at "
              "initialisation, so absolute costs are meaningless. Shape only.")

    present = {l.name for l in model.layers}
    cuts = [(lbl, n) for lbl, n in CUTS if n in present]
    missing = [n for _, n in CUTS if n not in present]
    if missing:
        print(f"cut points absent from this config, skipped: {missing}")

    rows, prev = [], 0.0
    for label, name in cuts:
        sub = keras.Model(model.input, model.get_layer(name).output)
        inps, outs = trace_model(sub, hwconf=HWConfig(1, -1, -1), inputs_kif=in_kif)
        comb = comb_trace(inps, outs)
        cum = float(comb.cost)
        rows.append({
            "component": label,
            "cut_layer": name,
            "cumulative_lut": cum,
            "marginal_lut": cum - prev,
            "cumulative_latency_ns": float(comb.latency[0]),
        })
        print(f"{label:<44} cum={cum:>12,.0f}  marginal={cum-prev:>12,.0f}  "
              f"lat={comb.latency[0]:>6.1f} ns")
        prev = cum

    total = rows[-1]["cumulative_lut"] if rows else 0.0
    print(f"\n{'TOTAL':<44} {total:>12,.0f} LUT-est  "
          f"({100*total/VU13P_LUT:.1f}% of VU13P)")
    print("\nshare of total:")
    for r in rows:
        r["pct_of_total"] = round(100 * r["marginal_lut"] / total, 1) if total else 0.0
        print(f"  {r['component']:<44} {r['pct_of_total']:>5.1f}%")

    os.makedirs(outdir, exist_ok=True)
    dst = os.path.join(outdir, "component_costs.json")
    json.dump({"ckpt": ckpt, "inputs_kif": list(in_kif), "total_lut": total,
               "vu13p_pct": round(100 * total / VU13P_LUT, 1), "rows": rows},
              open(dst, "w"), indent=2)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
