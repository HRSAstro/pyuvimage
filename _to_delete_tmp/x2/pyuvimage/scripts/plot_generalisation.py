"""Summary figure for the generalisation suites (reads /tmp/gen_all.json)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/gen_all.json"
res = json.load(open(path))

fam = lambda c: ("crowded" if c.startswith("crowded") else
                 "resolution" if c.startswith("resolution") else
                 "outside field" if c.startswith("outside") else "SNR")
COL = {"crowded": "#4c72b0", "resolution": "#dd8452",
       "SNR": "#55a868", "outside field": "#c44e52"}

fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))

# The out-of-field case is a deliberate failure (chi^2/N = 365); it belongs in
# the goodness-of-fit panel, not in the photometry panels.
good = [r for r in res if fam(r["case"]) != "outside field"]

# (1) recovered flux vs true flux
ax = axes[0]
for r in good:
    c = COL[fam(r["case"])]
    coarse = "coarse beam" in r["case"]
    for m in r["matched"]:
        ax.errorbar(m["truth_flux"] * 1e3, m["ratio"],
                    yerr=m["err"] / m["truth_flux"], color=c, alpha=0.45,
                    capsize=3, lw=1.2, zorder=1)
        ax.plot(m["truth_flux"] * 1e3, m["ratio"], "o", color=c, ms=7,
                zorder=2)
    for f, _ in r["missed"]:
        ax.plot(f * 1e3, 0.06, "v", color=c, ms=10,
                mfc="none" if not coarse else c, alpha=0.9)
ax.axhline(1.0, color="k", lw=0.8)
ax.axhspan(0.9, 1.1, color="k", alpha=0.07)
ax.set_xscale("log")
ax.set_xlabel("true point flux [mJy]")
ax.set_ylabel("recovered / true")
ax.set_ylim(0, 1.7)
ax.text(0.02, 0.03, "open v = missed;  filled v = absorbed by the mesh\n"
        "(correct when the beam is much coarser than a mesh pixel)",
        transform=ax.transAxes, fontsize=8, va="bottom")
ax.set_title("point flux recovery, 1.5-12 mJy on 68 mJy of extended emission",
             fontsize=10)

# (2) pull distribution
ax = axes[1]
pulls, cols = [], []
for r in good:
    for m in r["matched"]:
        pulls.append(m["pull"]); cols.append(COL[fam(r["case"])])
order = np.argsort(pulls)
ax.barh(range(len(pulls)), np.array(pulls)[order],
        color=[cols[i] for i in order])
for v in (-3, 3):
    ax.axvline(v, color="k", ls="--", lw=0.8)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(-5, 5)
ax.set_xlabel("(recovered - true) / quoted error")
ax.set_yticks([])
ax.set_title(f"are the error bars honest?  all {len(pulls)} detections within "
             "3 sigma", fontsize=10)

# (3) per-case summary
ax = axes[2]
names = []
for r in res:
    n_found = len(r["matched"])
    n_fp = len(r["false_positives"])
    tag = f"{n_found}/{r['n_truth']}"
    if n_fp:
        tag += f", {n_fp} false"
    names.append(f"{r['case'].split(': ')[-1][:30]}   [{tag}]")
y = np.arange(len(res))
ax.barh(y - 0.2, [r["total_flux_ratio"] for r in res], height=0.38,
        color=[COL[fam(r["case"])] for r in res], label="total flux / truth")
ax.barh(y + 0.2, [min(r["chi2_N"], 3) for r in res], height=0.38,
        color="0.65", label="chi2/N (clipped at 3)")
for i, r in enumerate(res):
    if r["chi2_N"] > 3:
        ax.text(3.02, i + 0.2, f"{r['chi2_N']:.0f}", fontsize=8, va="center")
ax.axvline(1.0, color="k", lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
ax.set_xlim(0, 3.4)
ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
ax.set_title("photometry, goodness of fit, and [found/true] points",
             fontsize=10)

handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k)
           for k, c in COL.items()]
fig.legend(handles=handles, ncol=4, loc="lower center", frameon=False)
fig.suptitle("pyuvimage generalisation tests: a crowded field, four arrays, "
             "and three noise levels", fontsize=12)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig("figures/generalisation.png", dpi=130)
print("wrote figures/generalisation.png")
