"""
Track A: analytical per-component resource/latency estimates for PHAT-JeT.

This is the handoff's documented fallback path -- no synthesis toolchain, no
trained weights, no dataset required. Everything below is derived from the
topology in `phat_jet_k3.py` and is regenerated whenever that changes.

METHOD
------
Operations are split into three classes, because they cost wildly different
amounts on an FPGA and lumping them into one "FLOPs" number is exactly the
thing the reviewers objected to:

  const-MAC : weight x activation. The weight is a compile-time constant, so
              da4ml turns each one into a fixed shift-add network (no DSP).
              A generic HLS mapping would instead spend ~1 DSP per MAC at II=1.
  data-MAC  : activation x activation (attention QK^T and A.V only). Cannot be
              constant-folded -- these need real multipliers on any backend.
  select    : one-hot AND / masked accumulate. The GMP scatter and gather are
              *entirely* this class: `cell` is one-hot by construction, so
              `einsum('bnep,bnc->bepc')` is a routed accumulate, not a multiply.
              This is the main reason GMP is cheap in hardware despite looking
              like ~260k MACs on paper.

Latency is reported as *logical depth in elementary op-levels* (one multiply,
one adder-tree level, or one LUT evaluation = 1 level). Converting levels to
clock cycles needs a fabric-timing assumption; see LEVELS_PER_CYCLE. That
conversion is the softest number here and is flagged as such in the output.

Usage:  KERAS_BACKEND=torch python analytical_estimates.py
"""

import os
import sys
from math import ceil, log2

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- config
N = int(os.environ.get("PJ_PARTICLES", 150))   # particles
F = 3            # input features (pT, deta, dphi)
D = 16           # d_model
H = 4            # heads
DH = D // H      # head dim = 4
P = int(os.environ.get("PJ_PATCH", 10))        # patch size
NP = N // P      # patches
TOKENIZER = os.environ.get("PJ_TOKENIZER", "max")  # "mean" | "max"
G = 8            # GMP grid is GxG
K = 3            # depthwise kernel
FFN = int(os.environ.get("PJ_FFN_MULT", 4)) * D
C = 5            # classes
BITS = 16        # ap_fixed<16,6>-equivalent datapath for the analytical pass

# Fabric-timing assumption for levels -> cycles. A 16-bit LUT-level op in
# UltraScale+ fabric is ~1.0-1.5 ns; at the 3.33 ns clock used by extract_hw.py
# that is 2-4 levels per cycle. Reported as a range.
LEVELS_PER_CYCLE = (2, 4)
CLOCK_NS = 3.33  # 300 MHz, JEDI-Linear's target (matches extract_hw.py)


def tree(k):
    """Adder/comparator tree depth over k terms."""
    return int(ceil(log2(k))) if k > 1 else 0


def dense(n_apply, k_in, m_out, relu=False):
    """(const-MACs, depth) for a Dense applied n_apply times."""
    return n_apply * k_in * m_out, 1 + tree(k_in) + (1 if relu else 0)


def softmax_depth(k):
    """exp LUT -> sum tree -> reciprocal LUT -> rescale multiply."""
    return 1 + tree(k) + 1 + 1


# ---------------------------------------------------------------- components
# Each: (name, const_macs, data_macs, selects, params, depth, note)
COMPONENTS = []

# 1. Embedding -------------------------------------------------------------
m, d = dense(N, F, D, relu=True)
COMPONENTS.append(("Embedding Dense(3->16)+ReLU", m, 0, 0, F * D + D, d, ""))

# 2. GMP -------------------------------------------------------------------
# indicators: 2 coords x G bins x N particles, each a compare (LUT)
gmp_sel = 2 * G * N                       # bin indicators
gmp_sel += N * G * G                      # cell = outer product (AND)
gmp_sel += N * G * G * D                  # scatter: masked accumulate
gmp_sel += N * G * G * D                  # gather: masked select
dw_macs = G * G * D * K * K
pw_macs, pw_depth = dense(N, D, D)
gmp_depth = (
    1                                     # indicator LUT
    + 1                                   # cell AND
    + 1 + tree(N)                         # scatter select + accumulate tree
    + 1 + tree(K * K)                     # depthwise conv
    + 1                                   # gather select
    + pw_depth                            # pointwise dense
    + 1                                   # residual add
)
COMPONENTS.append((
    "GMP (indicators+scatter+DWConv+gather+pointwise)",
    dw_macs + pw_macs, 0, gmp_sel,
    (D * K * K + D) + (D * D + D), gmp_depth,
    "scatter/gather are one-hot routing, not multiplies",
))

# 3. Local patch attention -------------------------------------------------
qkv_macs, qkv_depth = dense(N, D, D)
qkv_macs *= 3
qk = NP * H * P * P * DH
av = NP * H * P * P * DH
wo_macs, wo_depth = dense(N, D, D)
local_depth = qkv_depth + (1 + tree(DH)) + softmax_depth(P) + (1 + tree(P)) + wo_depth + 1
local_params = 4 * (D * D + D)
COMPONENTS.append((
    f"Local patch attention (all {NP} patches, parallel)",
    qkv_macs + wo_macs, qk + av, 0, local_params, local_depth,
    f"{NP} independent patches, {P}x{P} attention each",
))

# 4. Patch tokens + global MHSA -------------------------------------------
tok_qkv, tok_qkv_depth = dense(NP, D, D)
tok_qkv *= 3
tqk = H * NP * NP * DH
tav = H * NP * NP * DH
tok_wo, tok_wo_depth = dense(NP, D, D)
proj_macs, proj_depth = dense(NP, D, D)
# Tokenizer depth: "mean" is a sum tree; "max" is a comparator tree. Same depth,
# but max needs no scale -- which matters at P=10, where 1/P is not a shift and
# HGQ2's QSum rejects it outright (see phat_jet_k3.patch_tokenize).
global_depth = (
    tree(P)                               # sum tree (mean) or comparator tree (max)
    + tok_qkv_depth + (1 + tree(DH)) + softmax_depth(NP) + (1 + tree(NP)) + tok_wo_depth
    + proj_depth
    + 1                                   # broadcast residual add
)
COMPONENTS.append((
    f"Patch pooling ({TOKENIZER}) + global MHSA ({NP} tokens) + broadcast",
    tok_qkv + tok_wo + proj_macs, tqk + tav, 0,
    5 * (D * D + D), global_depth,
    "mean pool is a shift; Repeat is wiring",
))

# 5. FFN -------------------------------------------------------------------
f1, f1d = dense(N, D, FFN, relu=True)
f2, f2d = dense(N, FFN, D)
COMPONENTS.append((
    f"FFN Dense({D}->{FFN})+ReLU+Dense({FFN}->{D})",
    f1 + f2, 0, 0, (D * FFN + FFN) + (FFN * D + D), f1d + f2d + 1, "",
))

# 6. Head ------------------------------------------------------------------
h1, h1d = dense(1, D, D, relu=True)
h2, h2d = dense(1, D, C)
COMPONENTS.append((
    f"GlobalMaxPool + head Dense({D}->{D})+Dense({D}->{C})",
    h1 + h2, 0, N * D, (D * D + D) + (D * C + C), tree(N) + h1d + h2d,
    f"max pool over {N} particles = comparator tree",
))


def main():
    tot_c = sum(c[1] for c in COMPONENTS)
    tot_d = sum(c[2] for c in COMPONENTS)
    tot_s = sum(c[3] for c in COMPONENTS)
    tot_p = sum(c[4] for c in COMPONENTS)
    tot_depth = sum(c[5] for c in COMPONENTS)  # components are sequential

    lo = int(ceil(tot_depth / LEVELS_PER_CYCLE[1]))
    hi = int(ceil(tot_depth / LEVELS_PER_CYCLE[0]))

    # storage: at II=1 fully unrolled everything lives in registers/LUTs.
    # For a serialized variant, the buffers that would go to BRAM:
    grid_bits = G * G * D * BITS
    act_bits = N * D * BITS
    w_bits = tot_p * BITS

    L = []
    L.append("# PHAT-JeT — analytical per-component estimates (Track A)\n")
    L.append("**These are op-count arguments for ONE configuration, not the measured")
    L.append("results.** The rebuttal numbers live in `synthesis_table.md`, which reports")
    L.append("traced da4ml costs across the full accuracy/area curve. This document exists")
    L.append("as the handoff's analytical fallback and as a sanity check on those traces.")
    L.append("Its most durable finding -- that GMP's scatter/gather is one-hot routing")
    L.append("rather than arithmetic -- was later confirmed by trace: GMP costs ~0 LUT.\n")
    L.append("Generated by `analytical_estimates.py`. **No synthesis was run.** These are")
    L.append("counting arguments derived from the topology in `phat_jet_k3.py`, intended as")
    L.append("the handoff's documented fallback and as a sanity baseline for the da4ml")
    L.append("numbers from `extract_hw.py`.\n")
    L.append(f"Config: N={N} particles, {F} feats, d_model={D}, {H} heads, patch={P} "
             f"({NP} patches), GMP {G}x{G} grid, {K}x{K} depthwise, FFN {FFN}, {C} classes.")
    L.append(f"Datapath {BITS}-bit. Clock {CLOCK_NS} ns (300 MHz).\n")

    L.append("## Operation counts\n")
    L.append("| Component | const-MAC | data-MAC | select | Params | Depth (levels) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for n, c, d_, s, p, dep, _ in COMPONENTS:
        L.append(f"| {n} | {c:,} | {d_:,} | {s:,} | {p:,} | {dep} |")
    L.append(f"| **Total** | **{tot_c:,}** | **{tot_d:,}** | **{tot_s:,}** | "
             f"**{tot_p:,}** | **{tot_depth}** |")
    L.append("")
    L.append(f"Real multiplies = const-MAC + data-MAC = **{tot_c + tot_d:,}** "
             f"(~{2 * (tot_c + tot_d) / 1e6:.2f} MFLOPs at 2 FLOP/MAC).")
    L.append(f"Of these, only **{tot_d:,}** ({100 * tot_d / (tot_c + tot_d):.1f}%) are "
             "data-dependent; the rest have constant coefficients.\n")

    L.append("## Notes per component\n")
    for n, _, _, _, _, _, note in COMPONENTS:
        if note:
            L.append(f"- **{n}** — {note}")
    L.append("")

    L.append("## Latency\n")
    L.append(f"Critical path is sequential across components: **{tot_depth} op-levels**.")
    L.append(f"At {LEVELS_PER_CYCLE[0]}-{LEVELS_PER_CYCLE[1]} levels per {CLOCK_NS} ns cycle: "
             f"**{lo}-{hi} cycles = {lo * CLOCK_NS:.0f}-{hi * CLOCK_NS:.0f} ns**.")
    L.append("II = 1 (one jet per clock) for the fully-unrolled mapping.\n")
    L.append("The levels-to-cycles conversion is the softest assumption in this document; "
             "the op-level depth itself is exact given the topology.\n")

    L.append("## Storage\n")
    L.append(f"- Weights: {tot_p:,} x {BITS} b = {w_bits:,} b ({w_bits / 8 / 1024:.1f} KiB)")
    L.append(f"- GMP grid buffer: {G}x{G}x{D} x {BITS} b = {grid_bits:,} b")
    L.append(f"- Particle activations: {N}x{D} x {BITS} b = {act_bits:,} b")
    L.append(f"- Total ~{(w_bits + grid_bits + act_bits) / 1024:.0f} Kib, i.e. "
             f"~{ceil((w_bits + grid_bits + act_bits) / 18432)} BRAM18 *if* buffered.")
    L.append("  At II=1 fully unrolled these live in registers/LUTs and BRAM = 0, which is "
             "what `extract_hw.py` reports.\n")

    L.append("## DSP mapping\n")
    L.append("Per the handoff's fallback spec, DSP = MACs / II:\n")
    L.append("| Mapping | II | DSP | Comment |")
    L.append("|---|---:|---:|---|")
    L.append(f"| Generic HLS, fully parallel | 1 | {tot_c + tot_d:,} | "
             "infeasible on VU13P (12,288 DSP) — motivates the da4ml mapping |")
    L.append(f"| Generic HLS, folded | 64 | {ceil((tot_c + tot_d) / 64):,} | fits comfortably |")
    L.append(f"| da4ml constant shift-add | 1 | {tot_d:,} | "
             "only data-MACs need multipliers; const-MACs become LUT logic |")
    L.append("")
    L.append("The third row is the architecturally interesting one: constant-folding removes "
             f"{100 * tot_c / (tot_c + tot_d):.0f}% of the multipliers, leaving {tot_d:,} "
             "genuine multiplies, all inside attention.\n")

    L.append("## Reference row — JEDI-Linear\n")
    L.append("> **TODO — not filled in.** Numbers must be transcribed from Que et al., "
             "arXiv:2508.15468. Deliberately left blank rather than estimated.\n")

    L.append("## Caption text\n")
    L.append("> Analytical estimates at the stated unroll/II configuration, derived from "
             f"layer shapes; no place-and-route and no C-synthesis. {BITS}-bit fixed-point "
             f"datapath, target xcvu13p (UltraScale+), {CLOCK_NS} ns clock (300 MHz). "
             "Operation counts are exact; the resource mapping depends on the stated "
             "backend assumption.\n")

    open("analytical_estimates.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
