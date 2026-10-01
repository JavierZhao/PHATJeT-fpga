"""Absolute deployment-feasibility figure: accuracy vs CTL2 resource budget.

No baseline comparison — the x-axis is the deployment constraint itself
(one SLR of the VU13P used in CMS CTL2), and marker fill encodes whether a
design also meets the stated sub-100 ns latency requirement.
Reproduces slr_fit.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLR_LUT = 1_728_000 / 4  # VU13P has 4 SLRs

# (label, N, accuracy, LUT, latency_ns) — RTL-emulated bit-exact rows only
ROWS = [
    ("sparsity+PID",       128, 0.7961, 385_471, 89.9),
    ("sparsity cheap",     128, 0.7957, 404_852, 93.2),
    ("sparsity flagship",  128, 0.7990, 478_214, 93.2),
    ("PID continuation",   128, 0.7988, 513_253, 93.2),
    ("relu variant",       128, 0.7964, 591_088, 116.5),
    ("sparsity",            64, 0.8029, 487_451, 136.5),
    ("sparsity cheap",      64, 0.8019, 451_836, 139.9),
    ("three-lever",         64, 0.7911, 399_554, 136.5),
]

fig, ax = plt.subplots(figsize=(7.4, 4.8))
colors = {128: "#1f77b4", 64: "#d62728"}
seen_fast, seen_slow = set(), set()
for label, n, acc, lut, lat in ROWS:
    frac = 100 * lut / SLR_LUT
    fast = lat < 100
    kw = dict(s=80, color=colors[n], zorder=3)
    if fast:
        lbl = f"N={n}, meets <100 ns" if n not in seen_fast else None
        seen_fast.add(n)
        ax.scatter(frac, acc * 100, marker="o", label=lbl, **kw)
    else:
        lbl = f"N={n}, >100 ns" if n not in seen_slow else None
        seen_slow.add(n)
        ax.scatter(frac, acc * 100, marker="s", facecolors="none",
                   edgecolors=colors[n], s=80, zorder=3, label=lbl)

ax.axvspan(0, 100, color="#2ca02c", alpha=0.08, zorder=0)
ax.axvline(100, color="#2ca02c", lw=1.6, ls="--", zorder=1)
ax.text(99, 78.75, "nominal one-SLR budget\n(CTL2 constraint)", ha="right", va="bottom",
        fontsize=8, color="#2ca02c")
ax.axvspan(60, 70, color="#1f77b4", alpha=0.07, zorder=0)
ax.axvline(70, color="0.5", lw=1, ls=":", zorder=1)
ax.text(60.5, 79.30, "60–70% of one SLR:\nco-tenancy band (kept\nfor reference)", ha="left", va="bottom",
        fontsize=7.5, color="0.40")

ax.set_xlabel("LUT usage as % of a NOMINAL one-SLR budget\n"
              "(quarter of VU13P CLB LUTs ≈ 432k; da4ml pre-P&R estimate, not placed)")
ax.set_ylabel("Accuracy [%] — hls4ml 260k evaluation partition")
ax.set_xlim(52, 145)
ax.set_ylim(78.6, 80.8)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right", title="filled = also meets the\nsub-100 ns L1T requirement",
          title_fontsize=7.5)
ax.set_title("Absolute deployment feasibility: PHAT-JeT against the CMS CTL2 budget\n"
             "(every point: II = 1 assumed, 3.33 ns target, DSP/BRAM expected by construction)",
             fontsize=10)
ax.annotate("da4ml estimates, not synthesis results; 300 MHz is a target, not closed timing. Accuracies are\n"
            "operating-point estimates (partition also used for checkpoint selection). Placement/routing in one SLR is NOT yet demonstrated.",
            xy=(0.012, 0.02), xycoords="axes fraction", fontsize=6.8, color="0.35")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"/home/zihanzhao/projects/phatjet/slr_fit.{ext}", dpi=200)
print("wrote slr_fit.pdf/.png")
