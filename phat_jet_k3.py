"""
PHAT-JeT ported to Keras 3 (backend-agnostic), in a hardware-synthesizable form.

WHY THIS FILE EXISTS (for humans and other agents)
--------------------------------------------------
The public PHAT-JeT repo (github.com/aaronw5/PHAT-JeT) is TF/Keras-2 with dynamic
shapes, tf.scatter_nd / tf.gather_nd and per-jet dynamic grid extents. None of that
can go through the HGQ2 -> da4ml FPGA flow, which requires:
  * static shapes everywhere,
  * only ops in the da4ml trace registry (Einsum, Conv2D, Dense, Softmax, LUTs,
    GetItem, Sum, Repeat, Reshape, Concatenate, scalar arithmetic, ...),
  * no dynamic indexing (scatter/gather must be re-expressed),
  * ops applied FUNCTIONALLY on Keras tensors (custom Layer subclasses are
    invisible to the tracer).

Deliberate, documented deviations from the paper's float model (all shared
between the float reference and the QAT model so weights transfer 1:1):

1. GMP grid FIXED to a static [lo,hi]^2 box with n_bins^2 cells (gmp_lo/gmp_hi/
   gmp_bins). scatter_nd/gather_nd -> per-bin indicator LUTs, outer product,
   einsum scatter-add, depthwise conv, einsum gather-back. Mathematically
   identical to dynamic scatter/gather, but statically shaped.
   WARNING: the original default [-0.4, 0.4] came with a claim that 99.9% of
   constituents fall inside it. That is NOT true of the robust-normalised
   processed store (x_*_robust_150const_ptetaphi): measured there, only 34.1%
   of real (pt>0) constituents satisfy both |eta|<0.4 and |phi|<0.4 -- the rest
   are clamped into edge bins. Coverage on that data is 93.3% at +/-1.6 and
   99.96% at +/-3.2. Pick the range to match the normalisation of the data you
   are actually training on.
   SEPARATE ISSUE: 74.8% of the 150 slots are zero padding, and every padded
   slot sits at (eta,phi)=(0,0), so they all accumulate into the single centre
   cell (~112 copies of embed's relu(bias) per jet). gmp_mask_padding=True
   gates the scatter/gather on a pt>0 validity bit, which removes them. Costs
   one AND per grid cell on-chip. Off by default to preserve existing behaviour.
2. Depthwise kernel 3x3 on the 8x8 grid (paper: 8x8 kernel on delta=0.05 grid;
   comparable physical receptive field ~0.3 vs ~0.4).
3. Patch size and particle count are configurable. P=8 on N=128 (16 patches) was
   the original hardware config; P=10 on N=150 (15 patches) matches the paper
   exactly and is preferred when the data supports it. Requires P | N.
   NOTE: the mean patch tokenizer is a free power-of-two shift only when P is a
   power of two. At P=10 the 1/P scale is a genuine constant multiply, which is
   part of why patch_tokenizer="max" is attractive there (see deviation 6).
4. No LayerNorm in the hardware variant (use_ln=False): runtime division/sqrt is
   hostile to a fully unrolled II=1 design; JEDI-Linear's FPGA models likewise
   have no runtime normalization. use_ln=True restores the paper's float model.
5. Attention scale 1/sqrt(d_head) is reparameterized into W_q (a Dense
   reparameterization, not an approximation). Model outputs LOGITS; train with
   from_logits=True. Argmax on-chip needs no softmax.
6. Patch tokenizer is selectable: "mean" (paper default) or "max". Max is a
   comparator tree -- no multiply, no scale, and no power-of-two constraint on
   P -- so it is strictly cheaper in hardware. Suggested by Aaron Wang; whether
   it costs accuracy is an empirical question, hence the flag rather than a
   silent swap.
"""

import keras
from keras import layers, ops


def _dense(units, quantized, activation=None, name=None, use_bn=False):
    """Dense, optionally fused with BatchNorm.

    BatchNorm is free on-chip: at inference it is an affine transform that folds
    into the preceding kernel/bias, so it costs nothing in the traced netlist.
    Every Dense in HGQ2-examples/jsc150/model.py is a QEinsumDenseBatchnorm --
    the reference recipe normalises everywhere, we originally normalised nowhere.
    """
    if quantized:
        if use_bn:
            from hgq.layers import QBatchNormDense
            return QBatchNormDense(units, activation=activation, name=name)
        from hgq.layers import QDense
        return QDense(units, activation=activation, name=name)
    if use_bn:
        seq = [layers.Dense(units), layers.BatchNormalization()]
        if activation:
            seq.append(layers.Activation(activation))
        return keras.Sequential(seq, name=name)
    return layers.Dense(units, activation=activation, name=name)


def _bounded_act(x, quantized, name):
    """Bounded squashing LUT used where LayerNorm would go.

    jsc150's transformer puts QAffinedUnaryFunctionLUT('tanh') before attention
    and before the FFN. With overflow_mode='wrap' an out-of-range activation
    wraps to the opposite sign instead of saturating, so bounding the datalane
    ahead of attention matters more in fixed point than it does in float.
    """
    if quantized:
        from hgq.layers import QAffinedUnaryFunctionLUT
        return QAffinedUnaryFunctionLUT("tanh", name=name)(x)
    return layers.Activation("tanh", name=name)(x)


def _attn_scope(attn_bits, quantized, attn_l1=None, attn_floor=None):
    """Tighter datalane precision for the QK/AV einsums only.

    The per-component trace puts the two attention stages at 66.2% of LUT while
    they are only 9.4% of arithmetic -- activation x activation products cannot
    be constant-folded into shift-add networks. BW_A is otherwise a single global
    scope, so squeezing attention alone requires a nested scope at the QK/AV
    construction sites. Embedding + head are ~2.3% of LUT and are left alone.
    """
    if not quantized or (attn_bits is None and attn_l1 is None and attn_floor is None):
        # attn_floor included: a floor-only run used to fall through to the
        # null context and silently do nothing (Codex round-7 audit).
        from contextlib import nullcontext
        return nullcontext()
    from hgq.config import QuantizerConfigScope
    kw = {}
    if attn_bits is not None:
        # NOTE: f0 is the INITIAL fractional-bit count, not an enforced width --
        # training moves it. On its own this barely dents cost (see D18).
        kw["f0"] = int(attn_bits)
    if attn_l1 is not None:
        # This is the knob that actually holds attention narrow: stronger
        # fractional-bit pressure on the QK/AV datalanes only, i.e. a per-region
        # beta. Inherited keys (overflow_mode, ic) come from the enclosing scope.
        from hgq.regularizers import MonoL1
        kw["fr"] = MonoL1(float(attn_l1))
    if attn_floor is not None:
        # arXiv:2510.24784 floors attention bitwidths at 1 so hard pressure
        # cannot prune the path entirely (D35 saturation was ACCURACY collapse
        # under over-pressure; a floor changes where that sets in).
        from hgq.constraints import Min
        kw["fc"] = Min(int(attn_floor))
    return QuantizerConfigScope(default_q_type="kif", place="datalane", **kw)


def _region_scope(l1, quantized, f0=None, floor=None):
    """Per-region datalane pressure, same mechanism as _attn_scope.

    `attn_l1` on the QK/AV datalanes removed ~40% of the design's area for ~0.02
    accuracy points (D22/D33). The FFN is the largest remaining block that has
    never had targeted pressure -- 20-27% of the design depending on config.
    """
    if not quantized or (l1 is None and f0 is None):
        from contextlib import nullcontext
        return nullcontext()
    from hgq.config import QuantizerConfigScope
    kw = {}
    if f0 is not None:
        kw["f0"] = int(f0)
    if l1 is not None:
        from hgq.regularizers import MonoL1
        kw["fr"] = MonoL1(float(l1))
    if floor is not None:
        from hgq.constraints import Min
        kw["fc"] = Min(int(floor))   # D59: pressure without a floor gets exploited
    return QuantizerConfigScope(default_q_type="kif", place="datalane", **kw)


def _einsum(eq, xs, quantized, name=None):
    if quantized:
        from hgq.layers import QEinsum
        return QEinsum(eq, name=name)(xs)
    return ops.einsum(eq, *xs)


def _softmax(x, quantized, name=None, mode="softmax", norm_n=None, table_l1=None):
    """Attention score normalisation. mode:
      softmax  — exact (exp LUT + sum tree + reciprocal + rescale). The single
                 largest block in the traced design (~32%, D20/fine trace).
      sigmoid  — elementwise sigmoid(score): ONE unary LUT per element, no row
                 reduction at all (Ramapuram et al. 2024 show parity at scale).
      relu     — relu(score)/norm_n: unary LUT with the 1/n scale folded into
                 the table; at n=8 the scale is a shift anyway (Wortsman et al.
                 2023, ViT-22B recipe).
    The AV einsum is unchanged in all modes."""
    if mode == "softmax":
        if quantized:
            from hgq.layers import QSoftmax
            if table_l1:
                from hgq.config import QuantizerConfigScope
                from hgq.regularizers import MonoL1
                from hgq.constraints import Min
                # bc=Min(1): without a floor the optimizer drives table bits
                # NEGATIVE and mines the regularizer for negative loss instead
                # of fitting (observed: total loss -1.83, D59).
                with QuantizerConfigScope(place="table", br=MonoL1(float(table_l1)),
                                          bc=Min(1)):
                    return QSoftmax(axis=-1, name=name)(x)
            return QSoftmax(axis=-1, name=name)(x)
        return layers.Softmax(axis=-1, name=name)(x)
    if mode == "sigmoid":
        f = lambda t: 1.0 / (1.0 + ops.exp(-t))
    elif mode == "sigmoidb":
        # Ramapuram et al.'s stabilisation: logit bias b = -ln(L). Naked sigmoid
        # measured 0.7716 max (D63) — the predicted output-norm failure. The
        # bias is a constant, so it folds into the same single LUT.
        import math
        b = -math.log(float(norm_n or 1.0))
        f = lambda t: 1.0 / (1.0 + ops.exp(-(t + b)))
    elif mode == "relu":
        n = float(norm_n or 1.0)
        f = lambda t: ops.relu(t) * (1.0 / n)
    elif mode == "relub":
        # min(relu(s)/n, 1): relu's row-reduction-free latency, but bounded
        # [0,1] like softmax so AV datalanes can stay narrow (D65/D74: relu's
        # wide AV lanes are its area penalty). The clamp folds into the same
        # single unary LUT — zero extra hardware by construction.
        n = float(norm_n or 1.0)
        f = lambda t: ops.minimum(ops.relu(t) * (1.0 / n), 1.0)
    else:
        raise ValueError(mode)
    return _unary_lut(f, x, quantized, name)


def _unary_lut(f, x, quantized, name):
    """Elementwise scalar function as a traceable LUT."""
    if quantized:
        from hgq.layers import QUnaryFunctionLUT
        return QUnaryFunctionLUT(f, name=name)(x)
    return layers.Lambda(f, name=name)(x)


def _bin_indicator(lo, hi, first, last):
    if first:
        return lambda x: ops.cast(x < hi, x.dtype)
    if last:
        return lambda x: ops.cast(x >= lo, x.dtype)
    return lambda x: ops.cast((x >= lo) & (x < hi), x.dtype)


def onehot_bins(coord, n_bins, lo, hi, quantized, tag, edges=None):
    """[B,N] -> [B,N,n_bins] one-hot bin indicators with edge clamping.

    `edges`: optional explicit ascending edge list (overrides n_bins/lo/hi).
    Non-uniform edges are FREE in hardware -- each bin indicator is one
    threshold-comparison LUT regardless of spacing (Aaron's core7 study).
    An odd bin count centres a cell on the origin, so the jet core is not
    split across a 4-cell corner as it is in any even symmetric grid."""
    if edges is not None:
        edges = list(edges)
    else:
        width = (hi - lo) / n_bins
        edges = [lo + g * width for g in range(n_bins + 1)]
    nb = len(edges) - 1
    n = coord.shape[1]
    cols = []
    for g in range(nb):
        f = _bin_indicator(edges[g], edges[g + 1], g == 0, g == nb - 1)
        c = _unary_lut(f, coord, quantized, f"oh_{tag}_{g}")
        cols.append(ops.reshape(c, (-1, n, 1)))
    return layers.Concatenate(axis=-1, name=f"oh_{tag}")(cols)


def gmp_block(x, eta, phi, channels, quantized, n_bins=8, lo=-0.4, hi=0.4,
              valid=None, use_bn=False, edges=None, kernel=3, name="gmp"):
    """Geometric Message Passing (static form): scatter-add onto (eta,phi) grid,
    3x3 depthwise conv, gather back. Returns message [B,N,C] (added residually).

    `valid` is an optional [B,N] 0/1 mask (see build_phat_jet_k3's
    gmp_mask_padding). Masking `cell` covers both directions at once: a masked
    particle contributes nothing to the scatter AND gathers back zero, because
    `cell` is the only tensor linking particles to grid positions.
    """
    eta_oh = onehot_bins(eta, n_bins, lo, hi, quantized, f"{name}_eta", edges=edges)
    phi_oh = onehot_bins(phi, n_bins, lo, hi, quantized, f"{name}_phi", edges=edges)
    cell = _einsum("bne,bnp->bnep", [eta_oh, phi_oh], quantized, name=f"{name}_cell")
    if valid is not None:
        # Elementwise broadcast multiply expressed as an einsum so it stays in
        # the da4ml trace registry. cell is one-hot, so in hardware this is one
        # extra AND per cell against the validity bit.
        cell = _einsum("bn,bnep->bnep", [valid, cell], quantized, name=f"{name}_maskpad")
    grid = _einsum("bnep,bnc->bepc", [cell, x], quantized, name=f"{name}_scatter")
    if quantized:
        from hgq.layers import QConv2D
        conv = QConv2D(channels, kernel, padding="same", groups=channels, name=f"{name}_dwconv")(grid)
    else:
        conv = layers.Conv2D(channels, kernel, padding="same", groups=channels, name=f"{name}_dwconv")(grid)
    msg = _einsum("bnep,bepc->bnc", [cell, conv], quantized, name=f"{name}_gather")
    msg = _dense(channels, quantized, name=f"{name}_pointwise", use_bn=use_bn)(msg)
    return msg


def patch_tokenize(x4, patch_size, quantized, mode, name="patch_tokens"):
    """[B,NP,P,D] -> [B,NP,D]. Reduce each patch to one token.

    mode="mean": exact mean. Free (a shift) only when P is a power of two.
                 HGQ2's QSum asserts scale is a power of 2, so for other P we
                 emit a plain SUM instead. That is not an approximation: the
                 next op is always a Dense (patch_attn_wq/wk/wv), and
                 Dense_W(s/P) == Dense_{W/P}(s), so the 1/P folds exactly into
                 the learned weights. It only widens the tokenizer's integer
                 range by log2(P) bits.
                 WARNING: this equivalence only holds for weights LEARNED under
                 this graph. transfer_float_weights() copies by layer name and
                 will happily move Q/K/V kernels between a true-mean model and a
                 sum model, which differ by a factor of P. Retrain rather than
                 transfer across that boundary.
    mode="max":  comparator tree; no multiply, no scale, any P. See deviation 6.
    """
    if mode == "mean":
        pow2 = float(patch_size).is_integer() and (patch_size & (patch_size - 1)) == 0
        if quantized:
            from hgq.layers import QSum
            return QSum(axes=2, scale=(1.0 / patch_size) if pow2 else 1.0, name=name)(x4)
        # Float path mirrors the quantized one exactly so weights transfer 1:1.
        s = ops.sum(x4, axis=2)
        return s * (1.0 / patch_size) if pow2 else s
    if mode == "max":
        # Treat [B,NP,P,D] as an image (H=NP, W=P, C=D) and pool the whole W axis.
        if quantized:
            from hgq.layers import QMaxPooling2D
            pooled = QMaxPooling2D(pool_size=(1, patch_size), strides=(1, patch_size), name=name)(x4)
        else:
            pooled = layers.MaxPooling2D(pool_size=(1, patch_size), strides=(1, patch_size), name=name)(x4)
        return ops.reshape(pooled, (-1, x4.shape[1], x4.shape[3]))
    raise ValueError(f"patch_tokenizer must be 'mean' or 'max', got {mode!r}")


def _qkv(x, d_model, num_heads, quantized, name, use_bn, mode="dense"):
    """Q/K/V projection. mode="dense": full d x d (default).
    mode="block": block-diagonal per head — reshape to [.., H, dh] and project
    each head's dh channels independently (QEinsumDense with a head axis).
    Arithmetic drops from d^2 to d^2/H; W_o still mixes heads (D54 #3)."""
    if mode == "dense":
        return _dense(d_model, quantized, name=name, use_bn=use_bn)(x)
    dh = d_model // num_heads
    shp = tuple(x.shape[1:-1])
    xh = ops.reshape(x, (-1,) + shp + (num_heads, dh))
    eq_in = "abchd" if len(shp) == 2 else "abhd"
    eq = f"{eq_in},hde->{eq_in[:-1]}e"
    if quantized:
        from hgq.layers import QEinsumDense
        yh = QEinsumDense(eq, shp + (num_heads, dh), bias_axes="e", name=name)(xh)
    else:
        yh = layers.EinsumDense(eq, shp + (num_heads, dh), bias_axes="e", name=name)(xh)
    return ops.reshape(yh, (-1,) + shp + (d_model,))


def patch_mha(x4, d_model, num_heads, quantized, name, use_bn=False, attn_bits=None, attn_l1=None, attn_floor=None, attn_act="softmax", table_l1=None, qkv_mode="dense", av_l1=None):
    """Exact MHA within each patch. x4: [B,NP,P,D] -> [B,NP,P,D].
    No 1/sqrt(dk) op: absorbed in W_q (see module docstring, deviation 5)."""
    NP, P = x4.shape[1], x4.shape[2]
    dh = d_model // num_heads
    q = _qkv(x4, d_model, num_heads, quantized, f"{name}_wq", use_bn, qkv_mode)
    k = _qkv(x4, d_model, num_heads, quantized, f"{name}_wk", use_bn, qkv_mode)
    v = _qkv(x4, d_model, num_heads, quantized, f"{name}_wv", use_bn, qkv_mode)
    q = ops.reshape(q, (-1, NP, P, num_heads, dh))
    k = ops.reshape(k, (-1, NP, P, num_heads, dh))
    v = ops.reshape(v, (-1, NP, P, num_heads, dh))
    with _attn_scope(attn_bits, quantized, attn_l1, attn_floor):
        scores = _einsum("bnphd,bnqhd->bnhpq", [q, k], quantized, name=f"{name}_qk")
        w = _softmax(scores, quantized, name=f"{name}_softmax", mode=attn_act, norm_n=P, table_l1=table_l1)
        with _region_scope(av_l1, quantized, floor=1 if av_l1 else None):
            o = _einsum("bnhpq,bnqhd->bnphd", [w, v], quantized, name=f"{name}_av")
    o = ops.reshape(o, (-1, NP, P, d_model))
    return _dense(d_model, quantized, name=f"{name}_wo", use_bn=use_bn)(o)


def token_mha(x, d_model, num_heads, quantized, name, use_bn=False, attn_bits=None, attn_l1=None, attn_floor=None, attn_act="softmax", table_l1=None, qkv_mode="dense", av_l1=None):
    """Exact MHA over patch tokens. x: [B,NP,D] -> [B,NP,D]."""
    T = x.shape[1]
    dh = d_model // num_heads
    q = _qkv(x, d_model, num_heads, quantized, f"{name}_wq", use_bn, qkv_mode)
    k = _qkv(x, d_model, num_heads, quantized, f"{name}_wk", use_bn, qkv_mode)
    v = _qkv(x, d_model, num_heads, quantized, f"{name}_wv", use_bn, qkv_mode)
    q = ops.reshape(q, (-1, T, num_heads, dh))
    k = ops.reshape(k, (-1, T, num_heads, dh))
    v = ops.reshape(v, (-1, T, num_heads, dh))
    with _attn_scope(attn_bits, quantized, attn_l1, attn_floor):
        scores = _einsum("bthd,bshd->bhts", [q, k], quantized, name=f"{name}_qk")
        w = _softmax(scores, quantized, name=f"{name}_softmax", mode=attn_act, norm_n=T, table_l1=table_l1)
        with _region_scope(av_l1, quantized, floor=1 if av_l1 else None):
            o = _einsum("bhts,bshd->bthd", [w, v], quantized, name=f"{name}_av")
    o = ops.reshape(o, (-1, T, d_model))
    return _dense(d_model, quantized, name=f"{name}_wo", use_bn=use_bn)(o)


def build_phat_jet_k3(
    num_particles=128,
    num_feats=3,
    d_model=16,
    num_heads=4,
    ffn_mult=4,
    patch_size=8,
    n_classes=5,
    use_gmp=True,
    use_ln=False,
    gmp_bins=8,
    gmp_lo=-0.4,
    gmp_hi=0.4,
    gmp_mask_padding=False,
    gmp_edges=None,
    gmp_kernel=3,
    attn_bits=None,
    attn_l1=None,
    attn_floor=None,
    attn_act="softmax",
    attn_act_local=None,
    attn_act_global=None,
    av_l1=None,
    table_l1=None,
    qkv_mode="dense",
    ffn_l1=None,
    use_bn=False,
    pre_norm=None,
    patch_tokenizer="mean",
    local_mode="patch",
    global_mode="mha",
    gmp2=False,
    lin_k=4,
    quantized=False,
):
    """Trigger-scale PHAT-JeT (1 PHAT block). Outputs LOGITS."""
    NP = num_particles // patch_size
    assert NP * patch_size == num_particles

    feats = keras.Input((num_particles, num_feats), name="features")
    pt = feats[..., 0]    # GetItem — traceable
    eta = feats[..., 1]
    phi = feats[..., 2]

    x = _dense(d_model, quantized, activation="relu", name="embed", use_bn=use_bn)(feats)

    if use_gmp:
        # Zero-padded slots carry (pt,eta,phi)=(0,0,0) and so all land in whichever
        # cell contains the origin. With ~75% of 150 slots padded, that one cell
        # accumulates ~112 copies of embed's relu(bias) and swamps the real signal
        # there. A pt>0 validity bit removes them; on-chip it is one AND per cell.
        valid = _unary_lut(lambda t: ops.cast(t > 0, t.dtype), pt, quantized,
                           "gmp_valid") if gmp_mask_padding else None
        msg = gmp_block(x, eta, phi, d_model, quantized, n_bins=gmp_bins,
                        lo=gmp_lo, hi=gmp_hi, valid=valid, use_bn=use_bn,
                        edges=gmp_edges, kernel=gmp_kernel)
        x = x + msg

    def maybe_ln(t, name):
        # Three normalisation points: before local attention, before the patch
        # message, before the FFN. use_ln restores the paper's LayerNorm (not
        # synthesizable); pre_norm="tanh" is the hardware-friendly stand-in.
        if use_ln:
            return layers.LayerNormalization(epsilon=1e-6, name=name)(t)
        if pre_norm == "tanh":
            return _bounded_act(t, quantized, f"{name}_tanh")
        return t

    # local patched attention (residual) — or a Linformer over the full
    # sequence (local_mode="linformer"), per arXiv:2510.24784. The Linformer
    # projects K/V to lin_k dims: softmax count drops from N*H*P to N*H*lin_k
    # elements' worth, attacking the D20 softmax block directly.
    h = maybe_ln(x, "ln1")
    if local_mode == "linformer":
        if quantized:
            from hgq.layers import QLinformerAttention
            with _attn_scope(attn_bits, quantized, attn_l1, attn_floor):
                a = QLinformerAttention(num_heads, lin_k, d_model // num_heads,
                                        name="local_attn")(h, h)
        else:
            from keras.layers import MultiHeadAttention
            a = MultiHeadAttention(num_heads, d_model // num_heads,
                                   name="local_attn")(h, h)  # float ref: full MHA
        x = x + a
    else:
        pass  # fall through to patch attention below
    if local_mode == "linsum":
        # Per-patch linear stage (full-linearization ablation, D131/Codex r13):
        # sender(particle) + broadcast(receiver(patch mean)) within each patch.
        # Tests whether local attention adds value beyond structured pooling.
        h4 = ops.reshape(h, (-1, NP, patch_size, d_model))
        snd = _dense(d_model, quantized, activation="relu",
                     name="local_send", use_bn=use_bn)(h4)
        pooled = ops.sum(h4, axis=2, keepdims=True) * (1.0 / patch_size)
        rcv = _dense(d_model, quantized, activation="relu",
                     name="local_recv", use_bn=use_bn)(pooled)
        a = snd + rcv
        a = _dense(d_model, quantized, name="local_attn_wo", use_bn=use_bn)(a)
        a = ops.reshape(a, (-1, num_particles, d_model))
        x = x + a
    elif local_mode != "linformer":
        h4 = ops.reshape(h, (-1, NP, patch_size, d_model))
        a = patch_mha(h4, d_model, num_heads, quantized, name="local_attn", use_bn=use_bn,
                      attn_bits=attn_bits, attn_l1=attn_l1, attn_floor=attn_floor,
                  attn_act=(attn_act_local or attn_act),
                  table_l1=table_l1, qkv_mode=qkv_mode, av_l1=av_l1)
        a = ops.reshape(a, (-1, num_particles, d_model))
        x = x + a

    if gmp2 and use_gmp:
        # second geometric-context injection after local attention (one-hot
        # scatter/gather machinery is ~free in hardware; D129 measured +0.5
        # pts for the first block — this tests a second helping).
        msg2 = gmp_block(x, eta, phi, d_model, quantized, n_bins=gmp_bins,
                         lo=gmp_lo, hi=gmp_hi, valid=valid, edges=gmp_edges,
                         kernel=gmp_kernel, name="gmp2")
        x = x + msg2

    # patch-message broadcast (residual): mean tokenizer -> token MHA -> proj -> repeat
    h = maybe_ln(x, "ln_msg")
    h4 = ops.reshape(h, (-1, NP, patch_size, d_model))
    ptok = patch_tokenize(h4, patch_size, quantized, patch_tokenizer)
    if global_mode == "linsum":
        # JEDI-style linear global stage (D122/D129 evidence: at the sparse
        # knee the trained model self-ablates global Q/K to zero, i.e. it
        # already uses ~uniform global mixing — make it structural). Removes
        # one serial attention chain, the global QKV projections, and a
        # softmax: sender(ptok) + broadcast(receiver(sum(ptok))).
        sN = int(ptok.shape[1])
        snd = _dense(d_model, quantized, activation="relu",
                     name="patch_send", use_bn=use_bn)(ptok)
        pooled = ops.sum(ptok, axis=1, keepdims=True) * (1.0 / sN)
        rcv = _dense(d_model, quantized, activation="relu",
                     name="patch_recv", use_bn=use_bn)(pooled)
        ptok = snd + rcv
    else:
        ptok = token_mha(ptok, d_model, num_heads, quantized, name="patch_attn", use_bn=use_bn,
                         attn_bits=attn_bits, attn_l1=attn_l1, attn_floor=attn_floor,
                         attn_act=(attn_act_global or attn_act),
                         table_l1=table_l1, qkv_mode=qkv_mode, av_l1=av_l1)
    ptok = _dense(d_model, quantized, name="patch_msg_proj", use_bn=use_bn)(ptok)
    msg = ops.repeat(ptok, patch_size, axis=1)
    x = x + msg

    # FFN (residual)
    h = maybe_ln(x, "ln2")
    with _region_scope(ffn_l1, quantized):
        h = _dense(ffn_mult * d_model, quantized, activation="relu", name="ffn1", use_bn=use_bn)(h)
        h = _dense(d_model, quantized, name="ffn2", use_bn=use_bn)(h)
    x = x + h

    # aggregate + head
    if quantized:
        from hgq.layers import QGlobalMaxPooling1D
        pooled = QGlobalMaxPooling1D(name="agg_max")(x)
    else:
        pooled = layers.GlobalMaxPooling1D(name="agg_max")(x)
    hd = _dense(d_model, quantized, activation="relu", name="head1", use_bn=use_bn)(pooled)
    logits = _dense(n_classes, quantized, name="head_out")(hd)

    return keras.Model(feats, logits, name="phat_jet_k3" + ("_q" if quantized else ""))


def build_jedi_linear(num_particles=128, num_feats=3, gmp=False, gmp_bins=8,
                      gmp_lo=-1.6, gmp_hi=1.6, gmp_kernel=3, gmp_edges=None,
                      quantized=True):
    """JEDI-linear GNN, faithful to the reference (Que et al., src/model.py
    get_gnn): width 64, QEinsumDenseBatchnorm+relu throughout, SAT_SYM weights,
    QSum pooling with power-of-two scales, MLP head 64-32-16-5.

    gmp=True inserts OUR GMP block after the embed (the hybrid experiment:
    does near-free geometric local context lift a linear GNN?). GMP operates
    at the JEDI width (64 channels) with the same one-hot scatter/gather
    mechanics as in build_phat_jet_k3.

    QAT-only: the reference has no float variant, and this builder exists for
    controlled comparison inside our pipeline (same scopes as build_qat_model;
    the SAT_SYM weight scope here matches their get_gnn's inner scope).
    """
    assert quantized, "build_jedi_linear is QAT-only"
    from math import log2
    from hgq.config import QuantizerConfigScope
    from hgq.layers import QAdd, QEinsumDenseBatchnorm, QSum

    N, n = num_particles, num_feats
    with QuantizerConfigScope(place=("weight", "bias"), overflow_mode="SAT_SYM"):
        inp = keras.Input((N, n), name="features")
        pool_scale = 2.0 ** -round(log2(N))
        x = QEinsumDenseBatchnorm("bnc,cC->bnC", (N, 64), bias_axes="C",
                                  activation="relu", name="jedi_embed")(inp)
        if gmp:
            eta = inp[..., 1]
            phi = inp[..., 2]
            msg = gmp_block(x, eta, phi, 64, quantized, n_bins=gmp_bins,
                            lo=gmp_lo, hi=gmp_hi, edges=gmp_edges,
                            kernel=gmp_kernel, name="gmp")
            x = QAdd(name="jedi_gmp_add")([x, msg])
        s = QEinsumDenseBatchnorm("bnc,cC->bnC", (N, 64), bias_axes="C",
                                  activation="relu", name="jedi_send")(x)
        d = QEinsumDenseBatchnorm("bnc,cC->bnC", (1, 64), bias_axes="C",
                                  activation="relu", name="jedi_recv")(
            QSum(axes=1, scale=pool_scale, keepdims=True, name="jedi_pool1")(x))
        x = QAdd(name="jedi_add")([s, d])
        x = QEinsumDenseBatchnorm("bnc,cC->bnC", (N, 64), bias_axes="C",
                                  activation="relu", name="jedi_node2")(x)
        x = QSum(axes=1, scale=1 / 16, keepdims=False, name="jedi_pool2")(x)
        x = QEinsumDenseBatchnorm("bc,cC->bC", 64, bias_axes="C",
                                  activation="relu", name="jedi_head1")(x)
        x = QEinsumDenseBatchnorm("bc,cC->bC", 32, bias_axes="C",
                                  activation="relu", name="jedi_head2")(x)
        x = QEinsumDenseBatchnorm("bc,cC->bC", 16, bias_axes="C",
                                  activation="relu", name="jedi_head3")(x)
        out = QEinsumDenseBatchnorm("bc,cC->bC", 5, bias_axes="C",
                                    name="jedi_out")(x)
    return keras.Model(inp, out)
