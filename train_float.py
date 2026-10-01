"""
Float reference for the SYNTHESIZABLE PHAT-JeT architecture.

WHY THIS MATTERS
----------------
The QAT runs are ~5 points below the paper's 81.8%, but that comparison is not
attributable: the paper's number is for the paper's model, and ours carries
several documented hardware deviations (3x3 depthwise kernel instead of 8x8, no
LayerNorm, static clamped GMP grid, max tokenizer). Without a float number for
*this exact graph* we cannot tell how much of the gap is quantization and how
much is architecture.

This script trains `build_phat_jet_k3(quantized=False)` with the identical data
pipeline, and answers exactly that:

  float here ~= 81.8%  -> the deviations are fine, the gap is quantization
                          (fix: beta schedule, bitwidths, longer schedule)
  float here <  81.8%  -> the deviations cost accuracy, and no amount of QAT
                          tuning will recover it (fix: kernel size, normalisation)

Also produces float_phatjet.keras, which train_qat.py can optionally warm-start
from (PJ_FLOAT_CKPT).

Env: same PJ_* knobs as train_qat.py where they apply, plus PJ_EPOCHS/PJ_BATCH.
"""

import json
import os
import random
from math import cos, pi

os.environ.setdefault("KERAS_BACKEND", "jax")

import keras
import numpy as np

np.random.seed(42)
random.seed(42)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3


def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


NUM_PARTICLES = _env("PJ_PARTICLES", 150, int)
PATCH_SIZE = _env("PJ_PATCH", 10, int)
PATCH_TOKENIZER = _env("PJ_TOKENIZER", "max")
GMP_BINS = _env("PJ_GMP_BINS", 8, int)
GMP_LO = _env("PJ_GMP_LO", -1.6, float)
GMP_HI = _env("PJ_GMP_HI", 1.6, float)
GMP_MASK_PADDING = _env("PJ_GMP_MASK", "1") not in ("0", "", "false", "False")
USE_BN = _env("PJ_USE_BN", "0") not in ("0", "", "false", "False")
PRE_NORM = _env("PJ_PRE_NORM", "") or None
USE_LN = _env("PJ_USE_LN", "0") not in ("0", "", "false", "False")
FFN_MULT = _env("PJ_FFN_MULT", 1, int)
D_MODEL = _env("PJ_D_MODEL", 16, int)
NUM_HEADS = _env("PJ_HEADS", 4, int)
DATA = _env("PJ_DATA", "/data/phatjet-fpga/jets_150x3_kt.npz")
OUTDIR = _env("PJ_OUT", ".")
EPOCHS = _env("PJ_EPOCHS", 300, int)
BATCH = _env("PJ_BATCH", 2790, int)
LR = _env("PJ_LR", 3e-3, float)


def cosine_restarts(lr0, first_decay_steps, alpha=1e-6):
    def schedule(epoch):
        step, cyc = epoch, first_decay_steps
        while step >= cyc:
            step -= cyc
        return alpha + 0.5 * (lr0 - alpha) * (1 + cos(pi * min(step / cyc, 1.0)))

    return schedule


def main():
    print(f"FLOAT reference: N={NUM_PARTICLES} P={PATCH_SIZE} tok={PATCH_TOKENIZER} "
          f"grid={GMP_BINS}x{GMP_BINS} on [{GMP_LO},{GMP_HI}] mask_pad={GMP_MASK_PADDING} "
          f"bn={USE_BN} pre_norm={PRE_NORM} ln={USE_LN} epochs={EPOCHS}")
    d = np.load(DATA)
    x_train, y_train = d["x_train"], d["y_train"]
    x_val, y_val = d["x_val"], d["y_val"]
    if x_train.shape[1] > NUM_PARTICLES:
        print(f"slicing {x_train.shape[1]} -> {NUM_PARTICLES} particles (kt-sorted head)")
        x_train, x_val = x_train[:, :NUM_PARTICLES], x_val[:, :NUM_PARTICLES]
    x_train, x_val = x_train.copy(), x_val.copy()
    for x in (x_train, x_val):
        x[:, :, 0] *= 1.0 / 64.0  # identical preprocessing to train_qat.py

    model = build_phat_jet_k3(
        num_particles=NUM_PARTICLES,
        patch_size=PATCH_SIZE,
        patch_tokenizer=PATCH_TOKENIZER,
        ffn_mult=FFN_MULT,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        gmp_bins=GMP_BINS,
        gmp_lo=GMP_LO,
        gmp_hi=GMP_HI,
        gmp_mask_padding=GMP_MASK_PADDING,
        use_bn=USE_BN,
        pre_norm=PRE_NORM,
        use_ln=USE_LN,
        quantized=False,
    )
    print(f"params: {model.count_params():,}")

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    os.makedirs(OUTDIR, exist_ok=True)
    ckpt = keras.callbacks.ModelCheckpoint(
        os.path.join(OUTDIR, "float_phatjet.keras"),
        monitor="val_accuracy", mode="max", save_best_only=True,
    )
    lr_cb = keras.callbacks.LearningRateScheduler(
        cosine_restarts(LR, max(int(500 * EPOCHS / 7000), 10)))

    hist = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH, epochs=EPOCHS, verbose=2,
        callbacks=[ckpt, lr_cb],
    )
    best = max(hist.history["val_accuracy"])
    json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()},
              open(os.path.join(OUTDIR, "float_hist.json"), "w"))
    print(f"BEST FLOAT val_accuracy: {best:.4f}   (paper float PHAT-JeT: 0.8180)")
    print(f"artifacts in {OUTDIR}")


if __name__ == "__main__":
    main()
