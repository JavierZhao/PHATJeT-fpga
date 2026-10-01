"""Admission test for the float->QAT warm-start route (Codex round 18, D157).

The route died twice (D154): initialising a QAT graph wide enough to RECEIVE
float weights (`I0_A=8`) starts it at 148M EBOPs, and neither an aggressive
nor a gentle descent to 1.75M preserved the accuracy. `trace_minmax` is the
one untested escape, because it attacks the STARTING cost rather than the
descent:

  * `I0_A=8` is a blanket initialiser applied to every datalane quantizer;
  * HGQ2 stores integer range PER quantizer (`_i` state tensors);
  * `trace_minmax(..., reset=True)` resets each `_i` and sets it to the
    minimum width the calibration data actually requires.

Under `overflow_mode=WRAP` the `_i` values can otherwise only decay at
`i_decay_speed=1e-3` per training call — roughly 0.22 bit/epoch at our batch
count — so the model pays the blanket tax for a long time. Calibration
removes that slack immediately.

This is training-free and CPU-only, so it costs no GPU slot.

DECISION RULE (Codex round 18): only spend a training slot on this route if a
configuration reaches BOTH
  * post-calibration val accuracy >= 0.805, and
  * post-calibration EBOPs within ~2-3x of the 1.75M budget (< 4M preferred).
If it still starts at 10M+, the route is closed.

WRAP tails matter, so calibrate on a large sample: a small batch underestimates
the required ranges and produces rare catastrophic wraps. Default here is
200k training jets; PJ_CAL_N overrides.

Usage:  PJ_BW_A=<n> python calib_screen.py   # one grid point per process
Env:    the PJ_* set describing the graph, plus PJ_FLOAT_CKPT and PJ_DATA.
"""
import os

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras  # noqa: E402,F401
import hgq.layers  # noqa: E402,F401

CAL_N = int(os.environ.get("PJ_CAL_N") or 200_000)
# One BW_A per process: train_qat reads its config at import time, so a
# reload-based in-process grid is fragile. The job loops in bash instead.


def main():
    from hgq.utils import trace_minmax
    import train_qat

    model = train_qat.build_qat_model()
    d = np.load(train_qat.DATA)
    N = train_qat.NUM_PARTICLES
    x_tr = d["x_train"][:CAL_N, :N, :3].astype(np.float32).copy()
    x_va = d["x_val"][:, :N, :3].astype(np.float32).copy()
    y_va = d["y_val"]
    x_tr[..., 0] /= 64.0
    x_va[..., 0] /= 64.0

    train_qat.transfer_float_weights(model, train_qat.FLOAT_CKPT)
    pre, pre_e = _acc(model, x_va, y_va), _ebops(model)

    trace_minmax(model, x_tr, reset=True, batch_size=16384, verbose=0)
    post, post_e = _acc(model, x_va, y_va), _ebops(model)

    print(f"RESULT BW_A={train_qat.BW_A} I0_A={train_qat.I0_A} "
          f"cal_n={len(x_tr)}  "
          f"pre-cal acc={pre:.4f} ebops={pre_e:,.0f}  ->  "
          f"post-cal acc={post:.4f} ebops={post_e:,.0f}")
    # Codex round-18 admission rule.
    verdict = "OPEN" if (post >= 0.805 and post_e < 4e6) else "closed"
    print(f"VERDICT {verdict} (needs acc>=0.805 AND ebops<4,000,000)")


def _ebops(model):
    t = 0.0
    for l in model.layers:
        if hasattr(l, "ebops"):
            t += float(keras.ops.convert_to_numpy(l.ebops))
    return t


def _acc(model, x, y):
    p = model.predict(x, batch_size=16384, verbose=0)
    return float((p.argmax(1) == y.argmax(1)).mean())


if __name__ == "__main__":
    main()
