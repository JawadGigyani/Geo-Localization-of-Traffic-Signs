# Report figures and the numbers derived from the committed results.
#
#   python scripts/13_report_figures.py
#
# Reads only committed files (results/model_metrics.json, data/dataset_summary.json,
# results/batch_evaluation_stage1.json, results/batch_evaluation_stage2.json,
# results/sequences_manifest.json). No dataset, no weights, no GPU.
#
# Writes:
#   results/figures/lookalike.png      (a) detector test AP50 against training
#                                      instances per class, (b) accuracy per
#                                      look-alike family before and after the
#                                      crop classifier
#   results/figures/per_sequence.png   deduplication score and recall per sequence
#   results/derived_numbers.json       percent changes, shares, and a
#                                      sequence-level bootstrap of recall and
#                                      deduplication score
#
# Env:
#   RESULTS_DIR=results

import os
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIG = RESULTS_DIR / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


metrics = load("results/model_metrics.json")
summary = load("data/dataset_summary.json")
s2 = load("results/batch_evaluation_stage2.json")
s1 = load("results/batch_evaluation_stage1.json")
manifest = load("results/sequences_manifest.json")

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def family_of(name):
    # same rule as pipeline_core.family_of: 'R2-140' -> 'R2'
    return name.split("-")[0]


# ------------------------------------------------------------ lookalike.png
# (a) Detector AP50 on the test split against training instances per class.
# (b) Classification accuracy per look-alike family, detector label vs. final label.
classes = summary["classes"]
fam_size = {}
for c in classes:
    fam_size[family_of(c)] = fam_size.get(family_of(c), 0) + 1
train_counts = summary["per_split_class_counts"]["train"]
per_class = metrics["detector_test"]["per_class"]
by_family = s2["classification"]["by_family"]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.7, 2.55),
                             gridspec_kw={"width_ratios": [1.0, 1.1], "wspace": 0.28})
multi = [c for c in classes if fam_size[family_of(c)] > 1]
single = [c for c in classes if fam_size[family_of(c)] == 1]
ax.scatter([train_counts[c] for c in single], [per_class[c]["ap50"] for c in single],
           s=18, color=BLUE, edgecolor="white", linewidth=0.5,
           label=f"Single-class family ({len(single)})", zorder=3)
ax.scatter([train_counts[c] for c in multi], [per_class[c]["ap50"] for c in multi],
           s=18, color=ORANGE, edgecolor="white", linewidth=0.5, marker="D",
           label=f"Multi-member family ({len(multi)})", zorder=3)
offsets = {"M3-1": (-8, -12), "M3-2": (5, -5), "M3-3": (5, -9), "M3-4": (5, 2),
           "W1-8": (5, -2)}
for c, off in offsets.items():
    ax.annotate(c, (train_counts[c], per_class[c]["ap50"]), textcoords="offset points",
                xytext=off, fontsize=6.5, color=INK2)
# two well-detected rare classes, labelled in the empty band above 1.0
for c, tx in (("VR-132", 104), ("E5-1", 165)):
    ax.annotate(c, (train_counts[c], per_class[c]["ap50"]), xytext=(tx, 1.075),
                textcoords="data", fontsize=6.5, color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5,
                                shrinkA=1, shrinkB=2))
ax.set_xscale("log")
ticks = [100, 200, 500, 1000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_xlabel("Training instances (log scale)")
ax.set_ylabel("Test AP50")
ax.set_ylim(0, 1.12)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="lower right", frameon=False, fontsize=7, handletextpad=0.2, borderaxespad=0.1)
ax.set_title("(a)", loc="left")

fams = [f for f, v in by_family.items() if v["confusable"]]
fams.sort(key=lambda f: -by_family[f]["n"])
x = list(range(len(fams)))
w = 0.4
a = [100 * by_family[f]["stage1_acc"] for f in fams]
b = [100 * by_family[f]["stage2_acc"] for f in fams]
bx.bar([i - w / 2 - 0.01 for i in x], a, width=w, color=BLUE, label="Detector label")
bx.bar([i + w / 2 + 0.01 for i in x], b, width=w, color=ORANGE, label="After crop classifier")
bx.set_xticks(x)
bx.set_xticklabels([f"{f}\n{by_family[f]['n']}" for f in fams], fontsize=7)
bx.set_ylabel("Accuracy (%)")
bx.set_ylim(0, 112)
bx.set_yticks([0, 20, 40, 60, 80, 100])
bx.grid(axis="x", visible=False)
bx.legend(loc="upper center", bbox_to_anchor=(0.55, 1.13), ncol=2, frameon=False, fontsize=7,
          handlelength=1.2, columnspacing=1.0)
bx.set_title("(b)", loc="left")
fig.savefig(FIG / "lookalike.png")
plt.close(fig)

# --------------------------------------------------------- per_sequence.png
# Per-sequence deduplication score and recall against sequence size.
per_seq = s2["per_sequence"]
names = list(per_seq)
posts = [per_seq[n]["posts"] for n in names]
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.05), sharex=True)
for ax, key, label in ((axes[0], "dedupe", "Deduplication score"),
                       (axes[1], "recall", "Signposts found (recall)")):
    vals = [per_seq[n][key] for n in names]
    ax.scatter(posts, vals, s=24, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    pooled = s2["pooled"][key]
    ax.axhline(pooled, color=INK2, linewidth=1, linestyle="--", zorder=2)
    ax.annotate(f"pooled {pooled:.3f}", (2.6, pooled), textcoords="offset points",
                xytext=(0, -10), ha="left", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ticks = [3, 10, 30, 100]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(2.4, 140)
    ax.set_xlabel("Signposts in the sequence (log scale)")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1.08)
    big = max(names, key=lambda n: per_seq[n]["posts"])
    ax.annotate("largest\nsequence", (per_seq[big]["posts"], per_seq[big][key]),
                textcoords="offset points", xytext=(-46, -30), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
axes[0].set_title("(a)", loc="left")
axes[1].set_title("(b)", loc="left")
fig.tight_layout()
fig.savefig(FIG / "per_sequence.png")
plt.close(fig)


# ---------------------------------------------------------- derived numbers
def pct_change(old, new):
    return round(100.0 * (new - old) / old, 1)


p1, p2 = s1["pooled"], s2["pooled"]
c2 = s2["classification"]
conf = c2["confusable"]
derived = {
    "fragmented_change_pct": pct_change(p1["fragmented"], p2["fragmented"]),
    "merged_change_pct": pct_change(p1["merged"], p2["merged"]),
    "rows_stage1": p1["rows"],
    "lookalike_error_stage1_pct": round(100 * (1 - conf["stage1_acc"]), 1),
    "lookalike_error_stage2_pct": round(100 * (1 - conf["stage2_acc"]), 1),
    "lookalike_error_change_pct": pct_change(1 - conf["stage1_acc"], 1 - conf["stage2_acc"]),
    "all_error_change_pct": pct_change(1 - c2["stage1_acc"], 1 - c2["stage2_acc"]),
    "geo_improvement_pct": round(100 * s2["geo_error_m"]["improvement"], 1),
    "never_found": p2["posts"] - p2["found"],
    "rows_scored_for_error": s2["geo_error_m"]["n"],
    "calibration_frames": sum(s["frames"] for s in manifest["sequences"] if s["role"] == "calibration"),
    "evaluation_frames": sum(s["frames"] for s in manifest["sequences"] if s["role"] == "evaluation"),
    "largest_sequence": max(names, key=lambda n: per_seq[n]["posts"]),
}
big = per_seq[derived["largest_sequence"]]
derived["largest_sequence_posts"] = big["posts"]
derived["largest_sequence_posts_share_pct"] = round(100 * big["posts"] / p2["posts"], 1)
derived["largest_sequence_merged"] = big["merged"]
derived["largest_sequence_merged_share_pct"] = round(100 * big["merged"] / p2["merged"], 1)
derived["largest_sequence_dedupe"] = big["dedupe"]
derived["dedupe_min"] = min(v["dedupe"] for v in per_seq.values())
derived["dedupe_max"] = max(v["dedupe"] for v in per_seq.values())


# Sequence-level bootstrap: resample the evaluation sequences with replacement
# and recompute the pooled rates. Only rates whose per-sequence counts are saved
# in batch_evaluation_stage*.json can be resampled.
def bootstrap(stage, n_boot=10000, seed=0):
    seqs = list(stage["per_sequence"].values())
    rng = random.Random(seed)
    rec, ded = [], []
    for _ in range(n_boot):
        sample = [seqs[rng.randrange(len(seqs))] for _ in seqs]
        posts_ = sum(s["posts"] for s in sample)
        found_ = sum(s["found"] for s in sample)
        bad = sum(s["fragmented"] + s["merged"] for s in sample)
        rec.append(found_ / posts_)
        ded.append((found_ - bad) / found_)
    rec.sort()
    ded.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {"recall_ci": [round(rec[lo], 3), round(rec[hi], 3)],
            "dedupe_ci": [round(ded[lo], 3), round(ded[hi], 3)]}


derived["bootstrap_stage2"] = bootstrap(s2)
derived["bootstrap_stage1"] = bootstrap(s1)
derived["bootstrap_settings"] = {"resamples": 10000, "seed": 0, "unit": "sequence"}

out = RESULTS_DIR / "derived_numbers.json"
out.write_text(json.dumps(derived, indent=2) + "\n", encoding="utf-8")
print(json.dumps(derived, indent=2))
print(f"wrote {out} and {FIG}")
