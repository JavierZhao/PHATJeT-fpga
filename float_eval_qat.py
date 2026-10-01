"""Evaluate a QAT checkpoint's learned weights in the FLOAT graph.

Separates the two explanations for why the quantized champion stalls at
0.8077 (Codex round 16's proposed cheap separating experiment, D139):

  * **quantization / bit-allocation is binding** -> the same trained weights,
    evaluated with the quantizers removed, score materially above 0.8077.
    More bits (or a better precision allocation) would then pay.
  * **optimization is binding** -> float evaluation lands at ~0.8077 too.
    More bits cannot help; only a better-trained or larger model can.

This is evaluation-only: no training, no GPU. It holds architecture AND
weights fixed, which neither a fresh float run nor a wider-bitwidth QAT run
does.

*** IT DOES NOT WORK ON THIS MODEL -- SEE D143. *** Measured on the epoch-773
record: 0.8077 quantized, 0.3069 with the quantizers removed, from a transfer
with zero shape mismatches. The trained function is entangled with its
quantizers: under `overflow_mode=wrap` out-of-range intermediates wrap rather
than saturate, and the network has learned around that (and around the
quantizers' clipping keeping tanh/softmax arguments in range). Stripping them
yields a different function, not a higher-precision one. Do NOT quote a
"quantization cost" from this script for a wrap-overflow model. Kept because
the negative result is worth not rediscovering; the valid way to get a float
ceiling is an independent `train_float.py` run of the same graph.

The quantized graph is built by `train_qat.build_qat_model()` -- imported,
not reimplemented, exactly as `extract_hw.py` does. That matters: the
quantizer variable schema depends on the overflow mode, which is applied
through a `QuantizerConfigScope`, not through a `build_phat_jet_k3` argument.
A locally reconstructed config silently produced SAT_SYM quantizers for a
`wrap`-trained checkpoint and failed to load 35 objects. The float graph is
built from the same module-level constants, so the two cannot drift apart.

Usage:  python float_eval_qat.py <ckpt.keras> [<ckpt.keras> ...]
Env:    the full PJ_* set that produced the checkpoint, plus PJ_DATA.
"""
import os
import sys

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras  # noqa: E402,F401
import hgq.layers  # noqa: E402,F401 — registers Q* classes

from phat_jet_k3 import build_phat_jet_k3  # noqa: E402
from train_qat import (  # noqa: E402
    build_qat_model, NUM_PARTICLES, PATCH_SIZE, PATCH_TOKENIZER, D_MODEL,
    NUM_HEADS, FFN_MULT, GMP_BINS, GMP_LO, GMP_HI, GMP_KERNEL, GMP_EDGES,
    GMP_MASK_PADDING, USE_BN, PRE_NORM, ATTN_ACT, LOCAL_MODE, GLOBAL_MODE,
    QKV_MODE, USE_GMP, GMP2, LIN_K, DATA, SORT,
)

N_EVAL = int(os.environ.get("PJ_N_EVAL") or 0)  # 0 = all


def build_float_model():
    """Same graph, no quantizers. Layer names are identical by construction."""
    return build_phat_jet_k3(
        num_particles=NUM_PARTICLES, patch_size=PATCH_SIZE,
        patch_tokenizer=PATCH_TOKENIZER, d_model=D_MODEL,
        num_heads=NUM_HEADS, ffn_mult=FFN_MULT, gmp_bins=GMP_BINS,
        gmp_lo=GMP_LO, gmp_hi=GMP_HI, gmp_kernel=GMP_KERNEL,
        gmp_edges=GMP_EDGES, gmp_mask_padding=GMP_MASK_PADDING,
        use_bn=USE_BN, pre_norm=PRE_NORM, attn_act=ATTN_ACT,
        local_mode=LOCAL_MODE, global_mode=GLOBAL_MODE, qkv_mode=QKV_MODE,
        use_gmp=USE_GMP, gmp2=GMP2, lin_k=LIN_K, quantized=False,
    )


def load_val():
    d = np.load(DATA)
    x, y = d["x_val"], d["y_val"]
    x = x[:, :NUM_PARTICLES, :3].astype(np.float32).copy()
    if SORT == "pt":  # mirror train_qat / extract_hw (D-sort discipline)
        order = np.argsort(-x[..., 0], axis=1, kind="stable")
        x = np.take_along_axis(x, order[..., None], axis=1)
    x[..., 0] /= 64.0  # pT scaling -- same convention everywhere
    if N_EVAL:
        x, y = x[:N_EVAL], y[:N_EVAL]
    return x, y


def transfer(src, dst):
    """Copy trainable tensors by layer name. Returns (copied, skipped)."""
    by_name = {l.name: l for l in dst.layers}
    copied, skipped = [], []
    for sl in src.layers:
        dl = by_name.get(sl.name)
        if dl is None:
            continue
        sw = {v.path.split("/")[-1]: v for v in sl.weights}
        dw = {v.path.split("/")[-1]: v for v in dl.weights}
        for k in ("kernel", "bias", "gamma", "beta", "moving_mean",
                  "moving_variance", "depthwise_kernel"):
            if k in sw and k in dw:
                if tuple(sw[k].shape) == tuple(dw[k].shape):
                    dw[k].assign(sw[k].value)
                    copied.append(f"{sl.name}.{k}")
                else:
                    skipped.append(f"{sl.name}.{k} "
                                   f"{tuple(sw[k].shape)}!={tuple(dw[k].shape)}")
    return copied, skipped


def acc(model, x, y):
    p = model.predict(x, batch_size=4096, verbose=0)
    t = y.argmax(1) if y.ndim > 1 else y.astype(int)
    return float((p.argmax(1) == t).mean())


def main(ckpts):
    x, y = load_val()
    print(f"eval set: {x.shape}  N={NUM_PARTICLES} P={PATCH_SIZE} "
          f"d_model={D_MODEL} heads={NUM_HEADS} ffn_mult={FFN_MULT} "
          f"tok={PATCH_TOKENIZER} pre_norm={PRE_NORM} sort={SORT or 'store'}")

    qmodel = build_qat_model()
    fmodel = build_float_model()
    print(f"quantized params {qmodel.count_params():,}  "
          f"float params {fmodel.count_params():,}")

    for ck in ckpts:
        qmodel.load_weights(ck)
        q = acc(qmodel, x, y)
        copied, skipped = transfer(qmodel, fmodel)
        f = acc(fmodel, x, y)
        print(f"\n=== {os.path.basename(ck)}")
        print(f"  transferred {len(copied)} tensors"
              + (f", SKIPPED {len(skipped)}: {skipped}" if skipped else ""))
        print(f"  quantized (as trained) : {q:.4f}")
        print(f"  same weights, no quant : {f:.4f}")
        print(f"  quantization cost      : {f - q:+.4f}")
        if skipped:
            print("  *** shape mismatches above -- the float graph does NOT")
            print("  *** match the checkpoint; this number is meaningless.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: float_eval_qat.py <ckpt.keras> [...]")
    main(sys.argv[1:])
