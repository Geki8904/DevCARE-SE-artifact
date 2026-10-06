from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "DevCarePaper/figures"
EXP = ROOT / "outputs/expanded-experiment"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
})


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def funnel():
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    stages = [
        ("Collection\naudit", "1,365 PRs / 17 repos"),
        ("Uncensored\n30-day outcome", "1,004 PRs"),
        ("Source-eligible\nPRs", "731 PRs"),
        ("Locked modeling\nartifact", "730 PRs / 14 repos"),
    ]
    colors = ["#E8EEF7", "#DCE9F2", "#CFE5DF", "#B9D9CC"]
    xs = [0.15, 2.7, 5.25, 7.8]
    for i, ((title, value), x, color) in enumerate(zip(stages, xs, colors)):
        box = FancyBboxPatch(
            (x, 0.55), 2.05, 1.05,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=0.8, edgecolor="#38556B", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + 1.025, 1.26, title, ha="center", va="center", weight="bold", fontsize=7.5, linespacing=1.0)
        ax.text(x + 1.025, 0.82, value, ha="center", va="center", fontsize=8.1)
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.08, 1.075), xytext=(x + 2.12, 1.075),
                        arrowprops=dict(arrowstyle="->", lw=1.0, color="#38556B"))
    ax.text(2.45, 0.23, "361 right-\ncensored", ha="center", va="center", color="#7A3E3E", fontsize=6.8, linespacing=0.95)
    ax.text(5.0, 0.23, "273 without\nanalyzable source", ha="center", va="center", color="#7A3E3E", fontsize=6.8, linespacing=0.95)
    ax.text(7.55, 0.23, "1 row excluded by\npre-specified repo rule", ha="center", va="center", color="#7A3E3E", fontsize=6.8, linespacing=0.95)
    save(fig, "dataset-funnel")


def comparison():
    combined = pd.read_csv(EXP / "hybrid/hybrid_model_comparison.csv").set_index("method")
    legacy = pd.read_csv(EXP / "cohort-legacy/hybrid_model_comparison.csv").set_index("method")
    expansion = pd.read_csv(EXP / "cohort-expansion/hybrid_model_comparison.csv").set_index("method")
    methods = [
        "C2 Complexity",
        "Expert DevCARE-SE",
        "Hybrid Expert + Logistic",
        "Logistic Context",
        "Random Forest Context",
        "C4 Unfamiliarity",
    ]
    labels = ["C2", "Expert", "Hybrid Log.", "Logistic Ctx.", "RF Ctx.", "C4"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15), gridspec_kw={"wspace": 0.28})

    x = np.arange(len(methods))
    width = 0.36
    axes[0].bar(x - width / 2, [combined.loc[m, "pr_auc"] for m in methods], width,
                label="PR-AUC", color="#4378BF")
    axes[0].bar(x + width / 2, [combined.loc[m, "precision_at_10"] for m in methods], width,
                label="Precision@10%", color="#E67E2E")
    axes[0].axhline(0.4, color="#555555", ls="--", lw=0.9, label="Test prevalence")
    axes[0].set_ylim(0, 0.75)
    axes[0].set_xticks(x, labels, rotation=32, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("(a) Combined temporal test (225 PRs)")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, ncol=1, loc="upper right")

    cohort_methods = ["C2 Complexity", "Expert DevCARE-SE", "Hybrid Expert + Logistic", "Random Forest Context"]
    cohort_labels = ["C2", "Expert", "Hybrid Log.", "RF Ctx."]
    cx = np.arange(len(cohort_methods))
    w = 0.25
    axes[1].bar(cx - w, [legacy.loc[m, "pr_auc"] for m in cohort_methods], w,
                label="Legacy (325)", color="#6C8EBF")
    axes[1].bar(cx, [expansion.loc[m, "pr_auc"] for m in cohort_methods], w,
                label="Expansion (405)", color="#82B366")
    axes[1].bar(cx + w, [combined.loc[m, "pr_auc"] for m in cohort_methods], w,
                label="Combined (730)", color="#D79B00")
    axes[1].set_ylim(0.3, 0.75)
    axes[1].set_xticks(cx, cohort_labels, rotation=28, ha="right")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("(b) Cohort sensitivity")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, loc="upper right")
    save(fig, "expert-guided-model-comparison")


if __name__ == "__main__":
    funnel()
    comparison()
    print("Generated expanded paper figures in", FIG)
