"""Accuracy vs LUT figure for the rebuttal — v2 after Codex round-9 red-team.

Main markers: each N's non-dominated verified frontier (RTL-emulated bit-exact
only). Dominated verified points shown faint for context. Log scale labeled;
methodology differences scoped in the annotation, seed caveat included.
Kept in-repo for provenance (reproduces pareto_fpga.{pdf,png}).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (accuracy, LUT) — RTL-emulated bit-exact rows only
n128 = [(0.7990, 478_214), (0.7961, 385_471), (0.7957, 404_852),
        (0.7988, 513_253), (0.7995, 527_058), (0.7964, 591_088)]
n64 = [(0.8029, 487_451), (0.8035, 523_154), (0.8036, 544_292),
       (0.8023, 652_680), (0.8002, 700_629), (0.7911, 399_554)]
jedi = {"N=128": (0.809, 98_000, "#2ca02c"), "N=64": (0.809, 71_000, "#9467bd")}
linf = (0.798, 202_000)


def frontier(pts):
    front = []
    for a, l in pts:
        if not any(a2 >= a and l2 <= l and (a2, l2) != (a, l) for a2, l2 in pts):
            front.append((a, l))
    return front


fig, ax = plt.subplots(figsize=(7.2, 4.8))
for pts, color, label in ((n128, "#1f77b4", "PHAT-JeT N=128"),
                          (n64, "#d62728", "PHAT-JeT N=64")):
    fr = sorted(frontier(pts), key=lambda p: p[1])
    dom = [p for p in pts if p not in fr]
    ax.plot([l / 1000 for _, l in fr], [a * 100 for a, _ in fr],
            "o-", color=color, ms=7, lw=1.2, zorder=3,
            label=f"{label} frontier (RTL bit-exact)")
    if dom:
        ax.scatter([l / 1000 for _, l in dom], [a * 100 for a, _ in dom],
                   s=22, color=color, alpha=0.30, zorder=2)

for name, (a, l, c) in jedi.items():
    ax.scatter(l / 1000, a * 100, marker="*", s=230, color=c, zorder=4,
               label=f"JEDI-linear {name} (post-route)")
ax.scatter(linf[1] / 1000, linf[0] * 100, marker="^", s=90, color="#8c564b",
           zorder=4, label="FPGA Linformer 64p (different FPGA)")

ax.set_xscale("log")
ax.set_xlabel("LUT, thousands (log scale). PHAT-JeT: da4ml pre-P&R estimates; JEDI: post-route")
ax.set_ylabel("Accuracy [%], canonical hls4ml 260k test partition")
ax.set_xlim(40, 900)
ax.set_ylim(78.5, 81.5)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Jet-tagging accuracy vs FPGA area at matched particle counts\n"
             "(PHAT-JeT rows: VU13P, II=1, 0 DSP, 0 BRAM; faint = dominated verified points)",
             fontsize=10)
ax.annotate("PHAT-JeT accuracies: partition also used for checkpoint selection; sparse recipe cross-seed spread ~0.4 pts (n=2)",
            xy=(0.01, 0.015), xycoords="axes fraction", fontsize=7, color="0.35")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"/home/zihanzhao/projects/phatjet/pareto_fpga.{ext}", dpi=200)
print("wrote pareto_fpga.pdf/.png (v2)")
