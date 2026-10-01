"""Flatten every hw_results.json pulled from the PVC into one CSV.

    python evidence/compile_hw_results.py  ->  evidence/all_hw_results.csv

Each row is one extract_hw.py trace. Caveats that apply to every row:
  * lut_est / ff_est_reg_bits are da4ml pre-place-and-route estimates.
  * dsp=0, bram=0, ii=1 are written in by construction, not read from Vivado.
  * latency_ns_at_clock = stages x 3.33 ns, but the pipeline cutoff used is NOT
    recorded in hw_results.json and defaulted to 4.0 ns (PJ_LATENCY_CUTOFF), so
    most latencies are ~20% optimistic (DECISIONS D172). comb_latency_ns is the
    honest floor: total latency cannot go below it.
  * A row whose bit_exact / dais_exact is not true carries no accuracy claim.
"""
import csv
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ["trace_dir", "run", "epoch", "ckpt_val_acc", "ebops", "lut_est",
        "lut_pct_vu13p", "ff_est_reg_bits", "stages", "latency_ns_at_clock",
        "comb_latency_ns", "solver_options", "bit_exact", "emu_max_abs_err",
        "emu_argmax_match", "dais_exact", "dais_n", "dais_max_abs_err",
        "verilog_error", "ckpt", "verilog_dir"]

rows = []
for f in sorted(glob.glob(os.path.join(HERE, "pvc/hw/**/hw_results.json"), recursive=True)):
    d = json.load(open(f))
    for r in (d if isinstance(d, list) else [d]):
        ckpt = r.get("ckpt", "")
        parts = ckpt.split("/")
        run = parts[parts.index("runs") + 1] if "runs" in parts else (parts[-2] if len(parts) > 1 else "")
        if "ckpt_val_acc" not in r and "val_acc" in r:
            r["ckpt_val_acc"] = r["val_acc"]
        row = {c: r.get(c, "") for c in COLS}
        row["trace_dir"] = os.path.relpath(os.path.dirname(f), os.path.join(HERE, "pvc"))
        row["run"] = run
        if isinstance(row["solver_options"], dict):
            row["solver_options"] = json.dumps(row["solver_options"])
        rows.append(row)

rows.sort(key=lambda r: (-(float(r["ckpt_val_acc"] or 0))))
out = os.path.join(HERE, "all_hw_results.csv")
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows -> {out}")
