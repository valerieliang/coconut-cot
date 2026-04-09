# scripts/08b_plot_steering_snr.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
import os

sns.set_theme(style="white", context="talk")

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    "verbal": "#4C78A8",
    "coconut": "#F58518",
    "random": "#B0B0B0"
}


def plot_snr(results_path="results/steering/enhanced/full_results.json"):
    with open(results_path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, model in enumerate(["coconut", "verbal"]):
        ax = axes[idx]

        agg = defaultdict(lambda: {"true": [], "random": []})

        for res in results:
            if res["model"] != model:
                continue
            if "5.0" not in res["result"]["effects"]:
                continue

            rel_step = res["relative_step"]
            dir_type = res["direction_type"]
            effect = res["result"]["effects"]["5.0"]["raw_change"]

            if dir_type in ["true", "random"]:
                agg[rel_step][dir_type].append(effect)

        steps = sorted([
            s for s in agg.keys()
            if len(agg[s]["true"]) >= 3 and len(agg[s]["random"]) >= 3
        ])

        true_means, true_sems = [], []
        rand_means, rand_sems = [], []
        sigs = []

        for step in steps:
            true_vals = agg[step]["true"]
            rand_vals = agg[step]["random"]

            true_means.append(np.mean(true_vals))
            true_sems.append(np.std(true_vals) / np.sqrt(len(true_vals)))

            rand_means.append(np.mean(rand_vals))
            rand_sems.append(np.std(rand_vals) / np.sqrt(len(rand_vals)))

            if len(true_vals) >= 3 and len(rand_vals) >= 3:
                _, p_val = stats.ttest_ind(true_vals, rand_vals, equal_var=False)
                sigs.append(p_val < 0.05)
            else:
                sigs.append(False)

        x = np.arange(len(steps))
        width = 0.35

        ax.bar(x - width/2, true_means, width,
               yerr=true_sems,
               color=COLORS[model],
               alpha=0.9,
               capsize=3,
               linewidth=0,
               label="True")

        ax.bar(x + width/2, rand_means, width,
               yerr=rand_sems,
               color=COLORS["random"],
               alpha=0.6,
               capsize=3,
               linewidth=0,
               label="Random")

        # Significance stars
        if len(true_means):
            y_range = max(true_means + rand_means) - min(true_means + rand_means)
            offset = 0.05 * (y_range + 1e-6)

            for i, sig in enumerate(sigs):
                if sig:
                    max_h = max(true_means[i] + true_sems[i],
                                rand_means[i] + rand_sems[i])
                    ax.text(i, max_h + offset, "★",
                            ha='center', va='bottom', fontsize=11)

        ax.axhline(0, linestyle='--', linewidth=1, alpha=0.4)

        ax.set_xlabel("Relative Step")
        if idx == 0:
            ax.set_ylabel("Effect Size (Δ Logit)")

        # Titles
        if model == "coconut":
            ax.set_title("Coconut: Stable Causal Signal", fontsize=12, pad=10)
        else:
            ax.set_title("Verbal CoT: High Variance", fontsize=12, pad=10)


        ax.set_xticks(x)
        ax.set_xticklabels(steps)

        if idx == 1:
            ax.legend(frameon=False,
                      fontsize=10,
                      loc='upper left',
                      bbox_to_anchor=(1.02, 1))

    plt.suptitle("True vs Random Steering Effects by Step", fontsize=13)

    # Leave space on right for legend + top for title
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/steering_true_vs_random.png", bbox_inches='tight')
    plt.savefig("figures/steering_true_vs_random.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_snr()