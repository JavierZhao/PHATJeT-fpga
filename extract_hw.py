"""
Turn a trained HGQ2 PHAT-JeT checkpoint into hardware numbers + Verilog.

For each checkpoint given on the CLI (or the best-accuracy one in $PJ_CKPT_DIR):
  1. rebuild the architecture from env (the GMP bin indicators are closures and
     are not deserializable, so we rebuild in code and load weights only),
  2. optionally evaluate val accuracy,
  3. da4ml trace -> combinational cost (LUT estimate) + latency,
  4. pipeline at PJ_LATENCY_CUTOFF ns,
  5. write a Verilog project (same part as JEDI-linear: xcvu13p-flga2577-2-e),
  6. bit-exactness check: da4ml pipeline emulation vs keras forward,
  7. append a row to $PJ_OUT/hw_results.json.

IMPORTANT: the PJ_* env must match the checkpoint's training config exactly, or
`load_weights` will silently mismatch or the shapes will not line up. The config
is echoed at startup -- check it against the run's train.log.

Each stage is guarded independently: if pipelining or Verilog emission fails we
still record the combinational trace, because that is the number the rebuttal
table needs and it is the cheapest to obtain.

LUT numbers are da4ml estimates (no Vivado here) -- label them as estimates.

Usage:  python extract_hw.py [ckpt.keras ...]
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

import hgq.layers  # noqa: F401 — registers Q* classes

from da4ml.codegen import VerilogModel
from da4ml.converter import trace_model
from da4ml.trace import HWConfig, comb_trace, to_pipeline


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


CLOCK_NS = _env("PJ_CLOCK_NS", 3.33, float)        # 300 MHz — JEDI-linear's target
LATENCY_CUTOFF = _env("PJ_LATENCY_CUTOFF", 4.0, float)
PART = _env("PJ_PART", "xcvu13p-flga2577-2-e")
CKPT_DIR = _env("PJ_CKPT_DIR", "pareto")
OUTDIR = _env("PJ_OUT", ".")
DATA = _env("PJ_DATA", "")
N_EVAL = _env("PJ_N_EVAL", 50000, int)             # 0 disables accuracy eval
N_EMU = _env("PJ_N_EMU", 512, int)                 # 0 disables bit-exactness check
# DAIS interpreter screen (see below): more jets than the Verilator check and
# no RTL build. Evaluated row-by-row in Python, so keep N modest. 0 disables.
N_DAIS = _env("PJ_N_DAIS", 2000, int)
SKIP_RTL_ON_DAIS_FAIL = _env("PJ_SKIP_RTL_ON_DAIS_FAIL", "1") not in ("0", "", "false", "False")
# Input quantiser format (keep, integer, fractional) presented to the tracer.
#
# DEFAULT CHANGED 2026-07-25. This was (1,5,8), which silently truncates the
# inputs: the emitted netlist then disagreed with the Keras model on 40% of
# predictions while still producing plausible-looking LUT numbers. (1,8,16) is
# verified bit-exact (max|err|=0, 100% argmax agreement) and costs only +0.1% to
# +2.6% LUT versus the old default. See DECISIONS.md D30/D31.
#
# This is a hardcoded port width, NOT derived from the model's learned input
# quantizer. If you change the input preprocessing or the datalane config,
# RE-VERIFY with PJ_N_EMU>0 before trusting any number.
IN_KIF = tuple(int(v) for v in _env("PJ_INPUTS_KIF", "1,8,16").split(","))
# da4ml constant-matrix solver options (Codex research, D54): never previously
# set; jsc150 passes {"hard_dc": 2}. JSON dict, e.g. '{"hard_dc": 2}'.
import json as _json
# Default hard_dc=2 (jsc150's setting): measured ~equal LUT (-0.09%) but one
# fewer pipeline stage (143.2 -> 139.9 ns) on the headline checkpoint (D56).
SOLVER_OPTS = (_json.loads(os.environ["PJ_SOLVER_OPTS"])
               if os.environ.get("PJ_SOLVER_OPTS") else {"hard_dc": 2})

# xcvu13p capacity, for utilisation %. The deployability argument is about the
# fraction of the device consumed, not the raw count, so report both.
VU13P = {"lut": 1_728_000, "ff": 3_456_000, "dsp": 12_288, "bram36": 2_688}


def parse_ckpt_name(path):
    """pareto filenames encode epoch/val_acc/ebops -- keep them for the curve."""
    import re
    out = {}
    for k, pat in (("epoch", r"epoch=(\d+)"), ("ckpt_val_acc", r"val_acc=([0-9.]+)"),
                   ("ebops", r"ebops=(\d+)")):
        m = re.search(pat, os.path.basename(path))
        if m:
            out[k] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    return out


def pick_checkpoints(argv):
    if argv:
        return argv
    if not os.path.isdir(CKPT_DIR):
        return []
    cks = sorted(
        (f for f in os.listdir(CKPT_DIR) if f.endswith(".keras")),
        key=lambda f: float(f.split("val_acc=")[1].split("-")[0]),
    )
    return [os.path.join(CKPT_DIR, cks[-1])] if cks else []


def load_data(num_particles=None):
    if not DATA:
        return None, None
    d = np.load(DATA)
    x, y = d["x_val"].copy(), d["y_val"]
    # train_qat.py slices the stored 150-constituent array down to the model's
    # particle count (kt-sorted head). This MUST match or the arrays will not
    # feed the graph -- and a silent mismatch would mis-evaluate rather than
    # error if the shapes happened to be compatible.
    if os.environ.get("PJ_SORT", "") == "pt":
        # Mirror train_qat.py's PJ_SORT=pt exactly (stable descending-pT
        # re-sort BEFORE the head slice). A pT-trained checkpoint evaluated
        # on kt-ordered data selects different constituents at N<150 and
        # mis-scores silently (Codex round-7 catch).
        order = np.argsort(-x[:, :, 0], axis=1, kind="stable")
        x = np.take_along_axis(x, order[:, :, None], axis=1)
        print("re-sorted constituents by descending pT (PJ_SORT=pt)")
    if num_particles is not None and x.shape[1] > num_particles:
        print(f"slicing val {x.shape[1]} -> {num_particles} particles (sorted head)")
        x = x[:, :num_particles]
    x[:, :, 0] *= 1.0 / 64.0  # identical preprocessing to train_qat.py
    return x, y


def main():
    from train_qat import build_qat_model, NUM_PARTICLES, PATCH_SIZE, PATCH_TOKENIZER
    from train_qat import GMP_BINS, GMP_LO, GMP_HI, GMP_MASK_PADDING, USE_BN, PRE_NORM

    print(f"arch: N={NUM_PARTICLES} P={PATCH_SIZE} tok={PATCH_TOKENIZER} "
          f"grid={GMP_BINS}x{GMP_BINS} on [{GMP_LO},{GMP_HI}] mask={GMP_MASK_PADDING} "
          f"bn={USE_BN} pre_norm={PRE_NORM}")
    print(f"hw:   part={PART} clock={CLOCK_NS}ns cutoff={LATENCY_CUTOFF}ns inputs_kif={IN_KIF}")

    ckpts = pick_checkpoints(sys.argv[1:])
    assert ckpts, f"no checkpoints found in {CKPT_DIR}"
    print(f"checkpoints: {ckpts}")

    x_val, y_val = load_data(NUM_PARTICLES)
    if x_val is not None:
        lo, hi = float(x_val.min()), float(x_val.max())
        need_i = int(np.ceil(np.log2(max(abs(lo), abs(hi)) + 1e-9))) + 1
        print(f"data range [{lo:.3f}, {hi:.3f}] -> needs ~{need_i} integer bits; "
              f"inputs_kif integer bits = {IN_KIF[1]}"
              + ("  OK" if IN_KIF[1] >= need_i else "  *** TOO FEW — inputs will clip ***"))

    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for ck in ckpts:
        print(f"\n=== {ck}")
        row = {"ckpt": ck, "part": PART, "clock_ns": CLOCK_NS}
        row.update(parse_ckpt_name(ck))

        model = build_qat_model()
        model.load_weights(ck)

        if x_val is not None and N_EVAL:
            xe, ye = x_val[:N_EVAL], y_val[:N_EVAL]
            logits = model.predict(xe, batch_size=8192, verbose=0)
            row["val_acc"] = float((logits.argmax(1) == ye.argmax(1)).mean())
            row["n_eval"] = int(len(xe))
            print(f"val_acc {row['val_acc']:.4f} on {len(xe):,} jets")

        # --- combinational trace: the number the table needs ---------------
        _kw = {"solver_options": SOLVER_OPTS} if SOLVER_OPTS else {}
        inps, outs = trace_model(model, hwconf=HWConfig(1, -1, -1), inputs_kif=IN_KIF, **_kw)
        if SOLVER_OPTS:
            row["solver_options"] = SOLVER_OPTS
        comb = comb_trace(inps, outs)
        row["lut_est"] = float(comb.cost)
        row["comb_latency_ns"] = float(comb.latency[0])
        row["lut_pct_vu13p"] = round(100 * comb.cost / VU13P["lut"], 1)
        fit = "FITS" if row["lut_pct_vu13p"] <= 100 else "DOES NOT FIT"
        print(f"comb cost {comb.cost:,.0f} LUT-est ({row['lut_pct_vu13p']}% of VU13P -> {fit}), "
              f"comb latency {comb.latency[0]:.2f} ns")

        # --- pipelining ----------------------------------------------------
        pipe = None
        try:
            pipe = to_pipeline(comb, latency_cutoff=LATENCY_CUTOFF, verbose=False)
            row["stages"] = len(pipe.solutions)
            row["latency_ns_at_clock"] = round(row["stages"] * CLOCK_NS, 1)
            row["ff_est_reg_bits"] = int(pipe.reg_bits)
            row["ff_pct_vu13p"] = round(100 * pipe.reg_bits / VU13P["ff"], 1)
            row.update(dsp=0, bram=0, ii=1)
            print(f"pipelined: {row['stages']} stages = {row['latency_ns_at_clock']} ns "
                  f"@ {1000/CLOCK_NS:.0f} MHz, II=1, reg_bits={row['ff_est_reg_bits']:,}")
        except Exception as e:
            row["pipeline_error"] = f"{type(e).__name__}: {e}"
            print(f"PIPELINE FAILED: {row['pipeline_error']}")
            traceback.print_exc()

        # --- DAIS screen: predict the Verilator verdict in seconds ---------
        # ~40% of checkpoints from the continuation recipe fail bit-exactness
        # (D162) for reasons unrelated to their quality, and each Verilator
        # build costs ~20 min. da4ml's C++ DAIS interpreter evaluates the same
        # traced pipeline exactly, WITHOUT generating or compiling any RTL, so
        # it predicts the outcome up front -- and it is cheap enough to run on
        # far more than the 512 jets the Verilator check uses. "Bit-exact on
        # 512 jets" is a much weaker statement than it sounds; a wrap-boundary
        # event can be rare.
        if pipe is not None and x_val is not None and N_DAIS:
            try:
                xd = x_val[:N_DAIS]
                refd = model.predict(xd, batch_size=8192, verbose=0)
                # Installed da4ml's Pipeline exposes only __call__(row,
                # quantize=True) -- a per-row exact evaluator, no batch
                # predict(). Hence the loop, and a smaller default N.
                _flat = xd.reshape(len(xd), -1)
                daisd = np.asarray([pipe(r, quantize=True) for r in _flat]).reshape(refd.shape)
                row["dais_n"] = int(len(xd))
                row["dais_max_abs_err"] = float(np.abs(daisd - refd).max())
                row["dais_argmax_match"] = float((daisd.argmax(1) == refd.argmax(1)).mean())
                row["dais_exact"] = row["dais_max_abs_err"] == 0.0
                print(f"DAIS screen ({len(xd)} jets): max|err|={row['dais_max_abs_err']:.3e} "
                      f"argmax={row['dais_argmax_match']:.4f} exact={row['dais_exact']}")
                if not row["dais_exact"] and SKIP_RTL_ON_DAIS_FAIL:
                    print("*** DAIS mismatch -> skipping the Verilog/Verilator build.")
                    print("*** This checkpoint cannot carry an accuracy claim (D70).")
                    pipe = None
            except Exception as e:
                row["dais_error"] = f"{type(e).__name__}: {e}"
                print(f"DAIS screen unavailable ({row['dais_error']}); continuing to RTL")

        # --- Verilog + bit-exactness ---------------------------------------
        if pipe is not None:
            # Module/prefix names reach Verilog and C++ identifiers, so strip
            # everything that is not [A-Za-z0-9_]. Previously only "=" was
            # replaced, leaving "-" and "." from the checkpoint filename, which
            # made verilator reject the prefix and would equally break a Vivado
            # build. Any RTL generated before this fix has an illegal module name.
            import re as _re
            tag = _re.sub(r"[^A-Za-z0-9_]", "_",
                          os.path.splitext(os.path.basename(ck))[0])
            outdir = os.path.join(OUTDIR, f"verilog_{tag[:40]}")
            try:
                vm = VerilogModel(pipe, f"phat_jet_{tag[:40]}", outdir,
                                  part_name=PART, clock_period=CLOCK_NS)
                vm.write()
                row["verilog_dir"] = outdir
                print(f"verilog -> {outdir}")
                if x_val is not None and N_EMU:
                    xb = x_val[:N_EMU]
                    ref = model.predict(xb, batch_size=N_EMU, verbose=0)
                    vm.compile()
                    emu = np.asarray(vm.predict(xb.reshape(len(xb), -1))).reshape(ref.shape)
                    row["emu_max_abs_err"] = float(np.abs(emu - ref).max())
                    row["emu_argmax_match"] = float((emu.argmax(1) == ref.argmax(1)).mean())
                    print(f"bit-exactness: max|err|={row['emu_max_abs_err']:.3e} "
                          f"argmax match={row['emu_argmax_match']:.4f}")
                    if row["emu_max_abs_err"] > 1e-6:
                        print("*** WARNING: NETLIST DOES NOT MATCH THE MODEL. ***")
                        print("*** The accuracy of this checkpoint is NOT a property of")
                        print("*** the generated hardware. Widen PJ_INPUTS_KIF and retry")
                        print("*** before quoting accuracy alongside these resources.")
                        row["bit_exact"] = False
                    else:
                        row["bit_exact"] = True
            except Exception as e:
                row["verilog_error"] = f"{type(e).__name__}: {e}"
                print(f"VERILOG/EMULATION FAILED: {row['verilog_error']}")
                traceback.print_exc()

        rows.append(row)
        print(json.dumps(row, indent=2))

    dst = os.path.join(OUTDIR, "hw_results.json")
    existing = json.load(open(dst)) if os.path.exists(dst) else []
    json.dump(existing + rows, open(dst, "w"), indent=2)
    print(f"\nwrote {len(rows)} rows to {dst}")


if __name__ == "__main__":
    main()
