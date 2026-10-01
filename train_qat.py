"""
HGQ2 quantization-aware training of PHAT-JeT (trigger-scale, 128p x 3f).

RECIPE PROVENANCE (for humans and other agents)
-----------------------------------------------
Quantizer config + callbacks copied from HGQ2-examples/jsc150/run_train.py +
model.py `get_model` — the same flow the JEDI-Linear paper used:
  * weights/bias:  q_type='kbi', b0=7, i0=0, overflow_mode='wrap',
                   fr=ir=MonoL1(l1_reg), i_decay_speed=1e-3
  * datalane:      q_type='kif', f0=7, ic=MinMax(0,12), overflow_mode='wrap',
                   fr=MonoL1(l1_reg)
  * LayerConfigScope(beta0=0) + BetaScheduler ramping the EBOPs penalty beta
    during training -> single run traces the accuracy-vs-EBOPs Pareto frontier.
  * ParetoFront callback checkpoints every (val_accuracy, ebops)-nondominated
    epoch to pareto/ as .keras files.

DEVIATIONS from the reference recipe (documented honestly):
  * 7000 epochs on GPU -> PJ_EPOCHS (default 200) on CPU; the beta schedule is
    compressed proportionally, so the shape is preserved at any budget. On GPU,
    set PJ_EPOCHS near 7000 to match the reference and retire this deviation.
  * Optional init from the trained float model's weights, which compensated for
    the shortened CPU schedule. NOT needed when running the full schedule --
    jsc150 trains from scratch, so omitting float init is closer to the
    reference, not further from it. Absence is handled gracefully.
  * batch defaults to jsc150's 2790; PJ_BATCH overrides.

Everything is configured by environment variable so the same file runs locally
and as a cluster job:
  PJ_PARTICLES PJ_PATCH PJ_TOKENIZER PJ_DATA PJ_OUT PJ_EPOCHS PJ_BATCH PJ_LR
  PJ_FLOAT_CKPT

Run:  python train_qat.py
Outputs: $PJ_OUT/pareto/*.keras checkpoints, qat_hist.json, qat_final.keras
"""

import contextlib
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import random
from math import cos, pi

import keras
import numpy as np

np.random.seed(42)
random.seed(42)
# PJ_SEED: opt-in full determinism knob for seed-replication arms. Unset keeps
# the legacy behaviour above (np/random at 42, keras RNG unseeded) so past
# runs stay comparable.
if os.environ.get("PJ_SEED"):
    _seed = int(os.environ["PJ_SEED"])
    np.random.seed(_seed)
    random.seed(_seed)
    keras.utils.set_random_seed(_seed)
    print(f"PJ_SEED={_seed}: full keras/np/random seeding")

from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import MinMax
from hgq.regularizers import MonoL1
from hgq.utils.sugar import BetaScheduler, FreeEBOPs, ParetoFront, PieceWiseSchedule

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from phat_jet_k3 import build_phat_jet_k3

def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))


# Run configuration. Defaults reproduce the original 128p/P=8 CPU run; the
# cluster job overrides via env so nothing here has to be edited per-run.
NUM_PARTICLES = _env("PJ_PARTICLES", 128, int)
PATCH_SIZE = _env("PJ_PATCH", 8, int)
PATCH_TOKENIZER = _env("PJ_TOKENIZER", "mean")     # "mean" | "max"
GMP_BINS = _env("PJ_GMP_BINS", 8, int)
GMP_LO = _env("PJ_GMP_LO", -0.4, float)
GMP_HI = _env("PJ_GMP_HI", 0.4, float)
GMP_MASK_PADDING = _env("PJ_GMP_MASK", "0") not in ("0", "", "false", "False")
# Aaron's core7: comma-separated explicit edges, overrides bins/lo/hi (D43).
GMP_EDGES = ([float(v) for v in os.environ["PJ_GMP_EDGES"].split(",")]
             if os.environ.get("PJ_GMP_EDGES") else None)
GMP_KERNEL = _env("PJ_GMP_KERNEL", 3, int)
USE_BN = _env("PJ_USE_BN", "0") not in ("0", "", "false", "False")
PRE_NORM = _env("PJ_PRE_NORM", "") or None          # "" | "tanh"

# jsc150/get_model sets overflow_mode='wrap' globally, but get_transformer then
# overrides weights/bias to 'SAT_SYM' for its own scope. We inherited only the
# outer scope. With no normalisation an out-of-range activation wraps sign
# instead of saturating, so this is worth testing.
OVERFLOW = _env("PJ_OVERFLOW", "wrap")              # "wrap" | "SAT_SYM"
# Weights-only overflow override (JEDI-linear saturates WEIGHTS but wraps
# datalanes -- our single knob coupled both; D121 gap study decouples them).
OVERFLOW_K = _env("PJ_OVERFLOW_K", "") or OVERFLOW

# The reference beta ramp (2e-8 -> 3e-7 -> 3e-6) was tuned for jsc150's models
# over 7000 epochs. On this 5.4k-param model it drives accuracy to chance late
# in training. Scale it, or set 0 for no EBOPs pressure at all (accuracy ceiling).
BETA_SCALE = _env("PJ_BETA_SCALE", 1.0, float)
BETA_CONST = float(os.environ["PJ_BETA_CONST"]) if os.environ.get("PJ_BETA_CONST") else None
DATA = _env("PJ_DATA", "data/jets_128x3.npz")
OUTDIR = _env("PJ_OUT", ".")
FLOAT_CKPT = _env("PJ_FLOAT_CKPT", "float_phatjet.keras")

# jsc150 uses 7000 epochs on GPU. 200 was the CPU-feasible budget; on GPU set
# PJ_EPOCHS closer to 7000 to match the reference recipe (see docstring).
EPOCHS = _env("PJ_EPOCHS", 200, int)
TRAIN_SUBSET = None  # full train set (620k), as in the paper recipe
BATCH = _env("PJ_BATCH", 2790, int)  # jsc150 default --batch-size
LR = _env("PJ_LR", 3e-3, float)      # jsc150 default --learning-rate
# Global bitwidth pressure on every quantizer (kbi fr/ir, kif fr). With the
# regional ATTN_L1/FFN_L1 terms removed (D138) this becomes the binding
# downward force: arms sit ~10% BELOW their PID budget because MonoL1 keeps
# shrinking widths the controller would happily allow. Lower it to let
# precision grow into an unclaimed budget. Default unchanged.
L1_REG = _env("PJ_L1_REG", 1e-8, float)
W_L1 = float(os.environ["PJ_W_L1"]) if os.environ.get("PJ_W_L1") else None  # weight-bit pressure (kbi scope)
# jsc150/get_transformer initialises at b0=f0=4, not 7. Lower initial widths on
# BOTH operands cut multiplier EBOPs multiplicatively.
BW_K = _env("PJ_BW_K", 7, int)   # initial weight TOTAL bits (kbi b0)
BW_A = _env("PJ_BW_A", 7, int)   # initial datalane fractional bits (kif f0)
# Integer bits. Aaron's two-stage finding (D43): float-init transfer needs wide
# i0 (his: i0_w=3, i0_a=8) or activations wrap and the model silently collapses
# to chance. From-scratch training is fine at 0 (ranges grow on demand).
I0_K = _env("PJ_I0_K", 0, int)
I0_A = _env("PJ_I0_A", 0, int)
D_MODEL = _env("PJ_D_MODEL", 16, int)
NUM_HEADS = _env("PJ_HEADS", 4, int)
# None = inherit the global BW_A; an int squeezes QK/AV datalanes only.
ATTN_BITS = int(os.environ["PJ_ATTN_BITS"]) if os.environ.get("PJ_ATTN_BITS") else None
# Per-region bit pressure on QK/AV only. f0 alone does not hold bits down (D18).
ATTN_L1 = float(os.environ["PJ_ATTN_L1"]) if os.environ.get("PJ_ATTN_L1") else None
ATTN_FLOOR = int(os.environ["PJ_ATTN_FLOOR"]) if os.environ.get("PJ_ATTN_FLOOR") else None
ATTN_ACT = _env("PJ_ATTN_ACT", "softmax")   # softmax | sigmoid | sigmoidb | relu
TABLE_L1 = float(os.environ["PJ_TABLE_L1"]) if os.environ.get("PJ_TABLE_L1") else None
QKV_MODE = _env("PJ_QKV_MODE", "dense")     # dense | block
ATTN_ACT_LOCAL = _env("PJ_ATTN_ACT_LOCAL", "") or None
ATTN_ACT_GLOBAL = _env("PJ_ATTN_ACT_GLOBAL", "") or None
AV_L1 = float(os.environ["PJ_AV_L1"]) if os.environ.get("PJ_AV_L1") else None
# Axis 3 (2510.24784): fixed EBOPs budget via HGQ2's own BetaPID controller,
# replacing the beta ramp entirely when set.
EBOPS_TARGET = float(os.environ["PJ_EBOPS_TARGET"]) if os.environ.get("PJ_EBOPS_TARGET") else None
# Codex D66: the earlier gains (1e-9/3e-10) are inert against HGQ2's log-scale
# controller (library defaults p=1, i=2e-3); and beta must START positive.
PID_P = _env("PJ_PID_P", 0.1, float)
PID_I = _env("PJ_PID_I", 1e-3, float)
INIT_BETA = float(os.environ["PJ_INIT_BETA"]) if os.environ.get("PJ_INIT_BETA") else None
# Warm-start from an existing QAT checkpoint (same graph, direct load).
RESUME_CKPT = os.environ.get("PJ_RESUME_CKPT") or None
SORT = os.environ.get("PJ_SORT", "")  # "" = keep store order (kt); "pt" = re-sort by descending pT
WV_L1 = float(os.environ["PJ_WV_L1"]) if os.environ.get("PJ_WV_L1") else None  # weight-VALUE sparsity (proximal L1)
ARCH = os.environ.get("PJ_ARCH", "phat")  # phat | jedi | jedigmp (JEDI-linear replication / +GMP hybrid)
INPUT_NORM = os.environ.get("PJ_INPUT_NORM", "pt64")  # pt64 | zscore (JEDI-linear convention)
GLOBAL_MODE = os.environ.get("PJ_GLOBAL_MODE", "mha")  # mha | linsum (JEDI-style linear global stage)
GMP2 = _env("PJ_GMP2", "0") not in ("0", "", "false", "False")
USE_GMP = _env("PJ_USE_GMP", "1") not in ("0", "", "false", "False")  # PHAT no-GMP factorial cell
# Ramp-tolerant attention (Codex round 11 / D123): scale the EBOP beta DOWN on
# the data-x-data attention lanes only, so the full jsc150 ramp can prune the
# constant layers (JEDI's free sparsity mechanism, D122) without collapsing
# attention. Explicit allow-list; stock BetaScheduler overwrites per-layer
# beta0 so this needs its own callback.
ATTN_BETA_SCALE = float(os.environ["PJ_ATTN_BETA_SCALE"]) if os.environ.get("PJ_ATTN_BETA_SCALE") else None
ATTN_LANE_NAMES = ("local_attn_qk", "local_attn_softmax", "local_attn_av",
                   "patch_attn_qk", "patch_attn_softmax", "patch_attn_av")


class SelectiveBetaScheduler(keras.callbacks.Callback):
    """BetaScheduler variant: protected layers get beta * scale."""

    def __init__(self, schedule, protected, scale):
        super().__init__()
        self.schedule = schedule
        self.protected = set(protected)
        self.scale = float(scale)
        self._checked = False

    def on_epoch_begin(self, epoch, logs=None):
        beta = float(self.schedule(epoch))
        found = set()
        for layer in self.model._flatten_layers():
            b = getattr(layer, "_beta", None)
            if b is None:
                continue
            if layer.name in self.protected:
                b.assign(beta * self.scale)
                found.add(layer.name)
            else:
                b.assign(beta)
        if not self._checked:
            missing = self.protected - found
            assert not missing, f"SelectiveBeta: protected layers not found: {missing}"
            print(f"SelectiveBeta: {len(found)} protected lanes at {self.scale}x beta")
            self._checked = True
        if epoch % 50 == 0:
            print(f"  beta={beta:.3e} attn_beta={beta * self.scale:.3e}")


class ProxL1(keras.callbacks.Callback):
    """Proximal L1 (soft-threshold) on kernel VALUES after each optimizer step.

    w <- sign(w) * max(|w| - lam*lr, 0). Unlike gradient L1 this produces
    EXACT zeros, and a weight that quantizes to zero vanishes from the
    distributed-arithmetic netlist entirely. JEDI-linear's own bitwidth
    histogram (their Fig. 7) shows most of their weights are zero/<=3 bits —
    value sparsity, which our bit-pressure (D92) never targeted.
    """

    def __init__(self, lam):
        super().__init__()
        self.lam = lam
        self._vars = None

    def on_train_begin(self, logs=None):
        self._vars = [v for v in self.model.trainable_variables if "kernel" in v.path]
        print(f"ProxL1: soft-thresholding {len(self._vars)} kernel tensors, lam={self.lam}")

    def on_train_batch_end(self, batch, logs=None):
        lr = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        t = self.lam * lr
        for v in self._vars:
            w = v.value
            v.assign(keras.ops.sign(w) * keras.ops.maximum(keras.ops.abs(w) - t, 0.0))

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 25 == 0:
            nz = tot = 0
            for v in self._vars:
                w = keras.ops.convert_to_numpy(v.value)
                nz += int((w == 0.0).sum())
                tot += w.size
            print(f"  ProxL1 sparsity @ep{epoch}: {nz}/{tot} = {nz/tot:.3f} exact zeros")
# Distillation from the float teacher (D75): logit-matching MSE mixed with CE.
DISTILL_CKPT = os.environ.get("PJ_DISTILL_CKPT") or None
DISTILL_ALPHA = _env("PJ_DISTILL_ALPHA", 0.5, float)
# Same per-region mechanism aimed at the FFN (D35 lead).
FFN_L1 = float(os.environ["PJ_FFN_L1"]) if os.environ.get("PJ_FFN_L1") else None
FFN_MULT = _env("PJ_FFN_MULT", 4, int)
LOCAL_MODE = _env("PJ_LOCAL_MODE", "patch")   # "patch" | "linformer" (2510.24784)
LIN_K = _env("PJ_LIN_K", 4, int)
LABEL_SMOOTH = _env("PJ_LABEL_SMOOTH", 0.0, float)  # 0 = off (unchanged behaviour)


def cosine_restarts(lr0, first_decay_steps, alpha=1e-6):
    def schedule(epoch):
        cycle_step, cycle_len = epoch, first_decay_steps
        while cycle_step >= cycle_len:
            cycle_step -= cycle_len
        t = min(cycle_step / cycle_len, 1.0)
        return alpha + 0.5 * (lr0 - alpha) * (1 + cos(pi * t))

    return schedule


def build_qat_model():
    scope0 = QuantizerConfigScope(
        default_q_type="kbi",
        b0=BW_K,
        overflow_mode=OVERFLOW_K,
        i0=I0_K,
        fr=MonoL1(L1_REG),
        ir=MonoL1(L1_REG),
        i_decay_speed=1e-3,
    )
    # Weight-bit pressure. kbi's bit regularizer key is `br` (NOT `fr`/`ir`:
    # QuantizerConfigScope silently drops keys a config doesn't have, and an
    # unscoped `ir` leaks onto kif datalanes). Own scope, kbi places only.
    scope_w = (
        QuantizerConfigScope(q_type="kbi", place=("weight", "bias", "table"), br=MonoL1(W_L1))
        # `is not None`, NOT truthiness: W_L1=0.0 is a MEANINGFUL request (zero
        # the weight-bit brake), and `if W_L1` silently fell through to HGQ's
        # default br=MonoL1(1e-8). Without this, PJ_L1_REG=0 removes only the
        # KIF activation brake while KBI weight widths stay clamped, and a
        # null result is uninterpretable (D155/D156).
        if W_L1 is not None
        else contextlib.nullcontext()
    )
    scope1 = QuantizerConfigScope(
        default_q_type="kif",
        place="datalane",
        i0=I0_A,
        # Was hard-coded to "wrap", which silently made PJ_OVERFLOW a
        # weights-only switch: a run labelled SAT_SYM still wrapped its
        # activations. Default is still "wrap", so behaviour is unchanged
        # unless PJ_OVERFLOW is set.
        overflow_mode=OVERFLOW,
        f0=BW_A,
        fr=MonoL1(L1_REG),
        ic=MinMax(0, 12),
    )
    scope2 = LayerConfigScope(beta0=0)
    with scope0, scope1, scope_w, scope2:
        if ARCH in ("jedi", "jedigmp"):
            from phat_jet_k3 import build_jedi_linear
            return build_jedi_linear(
                num_particles=NUM_PARTICLES,
                gmp=(ARCH == "jedigmp"),
                gmp_bins=GMP_BINS, gmp_lo=GMP_LO, gmp_hi=GMP_HI,
                gmp_kernel=GMP_KERNEL, gmp_edges=GMP_EDGES,
                quantized=True,
            )
        return build_phat_jet_k3(
            num_particles=NUM_PARTICLES,
            use_gmp=USE_GMP,
            global_mode=GLOBAL_MODE,
            gmp2=GMP2,
            patch_size=PATCH_SIZE,
            patch_tokenizer=PATCH_TOKENIZER,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            attn_bits=ATTN_BITS,
            attn_l1=ATTN_L1,
            attn_floor=ATTN_FLOOR,
            attn_act=ATTN_ACT,
            table_l1=TABLE_L1,
            attn_act_local=ATTN_ACT_LOCAL,
            attn_act_global=ATTN_ACT_GLOBAL,
            av_l1=AV_L1,
            qkv_mode=QKV_MODE,
            ffn_l1=FFN_L1,
            ffn_mult=FFN_MULT,
            local_mode=LOCAL_MODE,
            lin_k=LIN_K,
            gmp_bins=GMP_BINS,
            gmp_lo=GMP_LO,
            gmp_hi=GMP_HI,
            gmp_mask_padding=GMP_MASK_PADDING,
            gmp_edges=GMP_EDGES,
            gmp_kernel=GMP_KERNEL,
            use_bn=USE_BN,
            pre_norm=PRE_NORM,
            quantized=True,
            use_ln=False,
        )


def transfer_float_weights(qmodel, float_path=None):
    """Copy weights from the float model into same-named Q layers.

    The donor is REBUILT in code and load_weights'd — never deserialized.
    load_model on this graph fails twice over: Keras 3 safe-mode blocks the
    GMP Lambda layers, and even with safe_mode=False a Lambda saved under one
    backend raises NotImplementedError when called under another (D60). The
    architecture must match the QAT config exactly; both come from the same
    env, mirroring Aaron's float_pretrain approach.
    """
    float_path = float_path or FLOAT_CKPT
    fmodel = build_phat_jet_k3(
        num_particles=NUM_PARTICLES, patch_size=PATCH_SIZE,
        patch_tokenizer=PATCH_TOKENIZER, d_model=D_MODEL, num_heads=NUM_HEADS,
        ffn_mult=FFN_MULT, local_mode=LOCAL_MODE, lin_k=LIN_K,
        gmp_bins=GMP_BINS, gmp_lo=GMP_LO, gmp_hi=GMP_HI, gmp_edges=GMP_EDGES,
        gmp_mask_padding=GMP_MASK_PADDING, use_bn=USE_BN, pre_norm=PRE_NORM,
        quantized=False, use_ln=False,
    )
    fmodel.load_weights(float_path)
    fw = {l.name: l.get_weights() for l in fmodel.layers if l.get_weights()}
    n_copied = 0
    for layer in qmodel.layers:
        if layer.name not in fw:
            continue
        src = fw[layer.name]
        # Q layers hold extra quantizer variables; match by shape prefix
        dst = layer.get_weights()
        replaced, si = list(dst), 0
        for di, w in enumerate(dst):
            if si < len(src) and src[si].shape == w.shape:
                replaced[di] = src[si]
                si += 1
        if si == len(src):
            layer.set_weights(replaced)
            n_copied += 1
    print(f"transferred weights into {n_copied} layers")
    return n_copied


def main():
    print(f"config: arch={ARCH} N={NUM_PARTICLES} P={PATCH_SIZE} tokenizer={PATCH_TOKENIZER} "
          f"grid={GMP_BINS}x{GMP_BINS} on [{GMP_LO},{GMP_HI}] mask_pad={GMP_MASK_PADDING} "
          f"gmp_edges={GMP_EDGES} "
          f"bn={USE_BN} pre_norm={PRE_NORM} overflow={OVERFLOW}/k={OVERFLOW_K} beta_scale={BETA_SCALE} "
          f"d_model={D_MODEL} heads={NUM_HEADS} ffn_mult={FFN_MULT} bw_k={BW_K} bw_a={BW_A} "
          f"attn_bits={ATTN_BITS} attn_l1={ATTN_L1} ffn_l1={FFN_L1} "
          f"attn_act={ATTN_ACT} act_local={ATTN_ACT_LOCAL} act_global={ATTN_ACT_GLOBAL} av_l1={AV_L1} "
          f"qkv_mode={QKV_MODE} local_mode={LOCAL_MODE} table_l1={TABLE_L1} "
          f"gmp_kernel={GMP_KERNEL} w_l1={W_L1} beta_const={BETA_CONST} distill={DISTILL_CKPT is not None}/{DISTILL_ALPHA} "
          f"sort={SORT or 'store(kt)'} wv_l1={WV_L1} use_gmp={USE_GMP} attn_beta={ATTN_BETA_SCALE} input_norm={INPUT_NORM} global_mode={GLOBAL_MODE} gmp2={GMP2} epochs={EPOCHS} batch={BATCH} lr={LR}")
    print(f"data:   {DATA}")
    d = np.load(DATA)
    if "provenance" in d:
        print(f"        provenance: {d['provenance']}")
    x_train, y_train = d["x_train"], d["y_train"]
    x_val, y_val = d["x_val"], d["y_val"]
    if SORT == "pt":
        # Re-sort constituents by descending pT before the head slice. At
        # N<=32 the sort decides WHICH particles survive the slice (kt and pT
        # heads select different constituents), so this is a data lever, not
        # cosmetics. pT column is monotone under the robust scaling, so
        # sorting the scaled column equals a raw-pT sort. Zero-pad rows have
        # pT=0 and sort to the tail, preserving the padding convention.
        def _ptsort(x):
            order = np.argsort(-x[:, :, 0], axis=1, kind="stable")
            return np.take_along_axis(x, order[:, :, None], axis=1)
        x_train, x_val = _ptsort(x_train), _ptsort(x_val)
        print("re-sorted constituents by descending pT (PJ_SORT=pt)")
    if x_train.shape[1] > NUM_PARTICLES:
        # Data is kt-sorted descending, so the first NUM_PARTICLES are the
        # highest-kt constituents. This is a DISCLOSED config change, not the
        # paper's N=150.
        print(f"slicing {x_train.shape[1]} -> {NUM_PARTICLES} particles (kt-sorted head)")
        x_train, x_val = x_train[:, :NUM_PARTICLES], x_val[:, :NUM_PARTICLES]
    assert x_train.shape[1] == NUM_PARTICLES, (
        f"data has {x_train.shape[1]} particles, model expects {NUM_PARTICLES}")
    x_train, x_val = x_train.copy(), x_val.copy()  # npz members are read-only views
    if INPUT_NORM == "zscore":
        # JEDI-linear's convention (their dataloader): global per-feature
        # z-score over (jets, particles). Our store is already affine-scaled,
        # so z-scoring it equals z-scoring raw. The default pt64 convention
        # leaves scaled pT with median ~0.003 in [0,0.7] — poorly conditioned
        # for 7-bit input quantizers (candidate cause of the JEDI accuracy
        # gap, D124 investigation).
        mu = x_train.mean(axis=(0, 1), keepdims=True)
        sd = x_train.std(axis=(0, 1), keepdims=True)
        x_train = (x_train - mu) / sd
        x_val = (x_val - mu) / sd
        print(f"input zscore: mu={mu.ravel().round(4).tolist()} sd={sd.ravel().round(4).tolist()}")
    else:
        for x in (x_train, x_val):
            x[:, :, 0] *= 1.0 / 64.0  # same pT scaling as float training
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x_train))[:TRAIN_SUBSET]
    x_train, y_train = x_train[idx], y_train[idx]

    model = build_qat_model()
    if RESUME_CKPT:
        model.load_weights(RESUME_CKPT)
        print(f"resumed QAT weights from {RESUME_CKPT}")
    # Resume and float init are mutually exclusive: transfer_float_weights
    # after a resume would silently overwrite the resumed weights whenever
    # FLOAT_CKPT happens to exist (Codex round-4 audit).
    want_init = os.path.exists(FLOAT_CKPT) and not RESUME_CKPT
    try:
        n_copied = transfer_float_weights(model) if not RESUME_CKPT else 0
        if want_init and n_copied:
            # Aaron's run-3 guard (D43): silent wrap-collapse does NOT throw --
            # it just predicts garbage. A healthy transfer lands near the float
            # accuracy; chance is 0.20. Widen PJ_I0_K/PJ_I0_A if this fires.
            _lg = model.predict(x_val[:20000], batch_size=8192, verbose=0)
            _acc = float((_lg.argmax(1) == y_val[:20000].argmax(1)).mean())
            print(f"post-transfer val_acc={_acc:.4f}")
            assert _acc > 0.5, (
                f"transfer collapsed to {_acc:.4f} (chance=0.2): quantizer i0 too "
                f"narrow for the float weights/activations (PJ_I0_K/PJ_I0_A)")
    except Exception as e:
        if want_init:
            # The checkpoint exists, so the user asked for two-stage init.
            # Falling back to from-scratch would silently change the experiment
            # (Codex review; also Aaron's run-3 postmortem).
            raise RuntimeError(f"float init FAILED for existing ckpt {FLOAT_CKPT}: {e}")
        print(f"float init skipped ({e}); training QAT from scratch")

    # Beta schedule: compressed version of jsc150's
    # [(0, 2e-8, linear), (2000/7000, 3e-7, log), (7000/7000, 3e-6, const)]
    # jsc150: [(0, 2e-8, linear), (2000, 3e-7, log), (7000, 3e-6, const)] over 7000
    # epochs. Same values + shapes, breakpoints scaled by 200/7000 (2000 -> 57).
    if BETA_CONST is not None:
        beta_sched = BetaScheduler(PieceWiseSchedule([(0, BETA_CONST, "constant"),
                                                      (EPOCHS, BETA_CONST, "constant")]))
        print(f"constant-beta mode: {BETA_CONST}")
    elif EBOPS_TARGET:
        from hgq.utils.sugar import BetaPID
        kw = dict(p=PID_P, i=PID_I)
        if INIT_BETA is not None:
            try:
                beta_sched = BetaPID(EBOPS_TARGET, init_beta=INIT_BETA, **kw)
            except TypeError:
                beta_sched = BetaPID(EBOPS_TARGET, **kw)
        else:
            beta_sched = BetaPID(EBOPS_TARGET, **kw)
        print(f"EBOPs-budget mode: BetaPID target={EBOPS_TARGET:,.0f} p={PID_P} i={PID_I} init_beta={INIT_BETA}")
    else:
        _sched = PieceWiseSchedule([
            (0, 2e-8 * BETA_SCALE, "linear"),
            (int(EPOCHS * 2000 / 7000), 3e-7 * BETA_SCALE, "log" if BETA_SCALE else "linear"),
            (EPOCHS, 3e-6 * BETA_SCALE, "constant"),
        ])
        if ATTN_BETA_SCALE is not None:
            beta_sched = SelectiveBetaScheduler(_sched, ATTN_LANE_NAMES, ATTN_BETA_SCALE)
            print(f"selective-beta mode: attention lanes at {ATTN_BETA_SCALE}x the ramp")
        else:
            beta_sched = BetaScheduler(_sched)
    json.dump({k: v for k, v in sorted(globals().items())
               if k.isupper() and isinstance(v, (int, float, str, bool, list, tuple, type(None)))},
              open(os.path.join(OUTDIR, "run_config.json"), "w"), indent=2)
    ebops = FreeEBOPs()
    pareto_dir = os.path.join(OUTDIR, "pareto")
    os.makedirs(pareto_dir, exist_ok=True)
    pareto = ParetoFront(
        pareto_dir,
        ["val_accuracy", "ebops"],
        [1, -1],
        fname_format="epoch={epoch}-val_acc={val_accuracy:.4f}-ebops={ebops}.keras",
        enable_if=lambda x: x["val_accuracy"] > 0.6,
    )
    # jsc150: cosine_decay_restarts(lr, first_decay_steps=500) over 7000 epochs
    # -> scaled 500 * 200/7000 ≈ 14 epochs per restart cycle.
    lr_cb = keras.callbacks.LearningRateScheduler(cosine_restarts(LR, max(int(500 * EPOCHS / 7000), 10)))

    if DISTILL_CKPT:
        # Teacher logits precomputed once; targets become [onehot(5) | logits(5)].
        tmodel = build_phat_jet_k3(
            num_particles=NUM_PARTICLES, patch_size=PATCH_SIZE,
            patch_tokenizer=PATCH_TOKENIZER, d_model=D_MODEL, num_heads=NUM_HEADS,
            ffn_mult=FFN_MULT, local_mode=LOCAL_MODE, lin_k=LIN_K,
            gmp_bins=GMP_BINS, gmp_lo=GMP_LO, gmp_hi=GMP_HI, gmp_edges=GMP_EDGES,
            gmp_kernel=GMP_KERNEL, gmp_mask_padding=GMP_MASK_PADDING,
            use_bn=USE_BN, pre_norm=PRE_NORM, quantized=False, use_ln=False)
        tmodel.load_weights(DISTILL_CKPT)
        t_tr = tmodel.predict(x_train, batch_size=8192, verbose=0)
        t_va = tmodel.predict(x_val, batch_size=8192, verbose=0)
        t_acc = float((t_tr.argmax(1) == y_train.argmax(1)).mean())
        t_vacc = float((t_va.argmax(1) == y_val.argmax(1)).mean())
        print(f"teacher loaded: train acc {t_acc:.4f} val acc {t_vacc:.4f} (expect ~0.806)")
        # >0.5 only catches wrong-arch garbage. A HALF-TRAINED teacher (0.74,
        # from reading a best-so-far ckpt mid-write) passes it and silently
        # invalidates the whole arm — happened to both distill arms AND
        # twostage on 2026-07-26. Gate on expected quality instead.
        min_acc = float(os.environ.get("PJ_DISTILL_MIN_ACC", "0.5"))
        assert t_vacc > min_acc, (
            f"teacher val acc {t_vacc:.4f} <= PJ_DISTILL_MIN_ACC={min_acc} — "
            "half-trained or wrong teacher; refusing to distill")
        y_train = np.concatenate([y_train, t_tr], axis=1)
        y_val = np.concatenate([y_val, t_va], axis=1)
        del tmodel

        a = DISTILL_ALPHA
        def distill_loss(y_true, y_pred):
            ce = keras.losses.categorical_crossentropy(y_true[:, :5], y_pred, from_logits=True)
            kd = keras.ops.mean(keras.ops.square(y_pred - y_true[:, 5:]), axis=-1)
            return (1.0 - a) * ce + a * kd

        def accuracy(y_true, y_pred):
            return keras.ops.cast(
                keras.ops.argmax(y_true[:, :5], axis=-1) == keras.ops.argmax(y_pred, axis=-1),
                "float32")

        # CE and KD logged separately so an ill-scaled MSE term can't hide
        # inside the combined loss (Codex round-4 audit gate).
        def ce_part(y_true, y_pred):
            return keras.losses.categorical_crossentropy(y_true[:, :5], y_pred, from_logits=True)

        def kd_part(y_true, y_pred):
            return keras.ops.mean(keras.ops.square(y_pred - y_true[:, 5:]), axis=-1)

        _loss, _metrics = distill_loss, [accuracy, ce_part, kd_part]
    else:
        # PJ_LABEL_SMOOTH: zero-area accuracy lever (Codex round 17 ranked it
        # the only loss-side change worth trying tonight). Touches CE targets
        # only — no quantizer state, so it is safe on a resumed checkpoint.
        # Keep it small: 0.1 is far too aggressive when the gap to close is
        # ~0.3 points and the metric is top-1, not calibration.
        _loss = keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=LABEL_SMOOTH)
        _metrics = ["accuracy"]
        if LABEL_SMOOTH:
            print(f"label smoothing: {LABEL_SMOOTH}")

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=_loss,
        metrics=_metrics,
        # The torch backend rejects anything but 1; jax accepts jsc150's 4.
        steps_per_execution=1 if keras.backend.backend() == "torch" else 4,
    )
    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=BATCH,
        epochs=EPOCHS,
        verbose=2,
        # TerminateOnNaN: hyb-sigb burned 620 epochs of GPU on NaN (D80).
        callbacks=[ebops, lr_cb, beta_sched, pareto, keras.callbacks.TerminateOnNaN()]
        + ([ProxL1(WV_L1)] if WV_L1 else []),
    )
    json.dump(
        {k: [float(v) for v in vs] for k, vs in hist.history.items()},
        open(os.path.join(OUTDIR, "qat_hist.json"), "w"),
    )
    model.save(os.path.join(OUTDIR, "qat_final.keras"))
    print(f"done; artifacts in {OUTDIR}")


if __name__ == "__main__":
    main()
