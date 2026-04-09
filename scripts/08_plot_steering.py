# scripts/08_plot_steering.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Clean, minimal styling
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

def plot_steering_results(summary_path="results/steering/summary.json"):
    with open(summary_path) as f:
        summary = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for idx, model in enumerate(["verbal", "coconut"]):
        ax = axes[idx]

        rel_steps_true, effects_true, errors_true, sig_true = [], [], [], []
        rel_steps_random, effects_random, errors_random = [], [], []

        for stats in summary.values():
            if stats["model"] != model:
                continue

            rel_step = stats["relative_step"]

            if stats["direction_type"] == "true" and stats["n"] >= 3:
                rel_steps_true.append(rel_step)
                effects_true.append(stats["mean_effect"])
                errors_true.append(stats["sem"])
                sig_true.append(stats["significant"])

            elif stats["direction_type"] == "random" and stats["n"] >= 3:
                rel_steps_random.append(rel_step)
                effects_random.append(stats["mean_effect"])
                errors_random.append(stats["sem"])

        # Convert and sort
        def sort_data(steps, effects, errors, extra=None):
            if not steps:
                return [], [], [], []
            order = np.argsort(steps)
            steps = np.array(steps)[order]
            effects = np.array(effects)[order]
            errors = np.array(errors)[order]
            if extra is not None:
                extra = np.array(extra)[order]
                return steps, effects, errors, extra
            return steps, effects, errors, None

        rel_steps_true, effects_true, errors_true, sig_true = sort_data(
            rel_steps_true, effects_true, errors_true, sig_true
        )
        rel_steps_random, effects_random, errors_random, _ = sort_data(
            rel_steps_random, effects_random, errors_random
        )

        width = 0.35

        # Plot true
        if len(rel_steps_true):
            ax.bar(rel_steps_true - width/2, effects_true,
                   width=width,
                   yerr=errors_true,
                   color=COLORS[model],
                   alpha=0.9,
                   capsize=3,
                   label="True",
                   linewidth=0)

            # significance markers (clean placement)
            y_offset = 0.03 * (np.max(effects_true) - np.min(effects_true) + 1e-6)
            for step, effect, err, sig in zip(rel_steps_true, effects_true, errors_true, sig_true):
                if sig:
                    ax.text(step - width/2,
                            effect + err + y_offset,
                            "★",
                            ha='center',
                            va='bottom',
                            fontsize=10)

        # Plot random
        if len(rel_steps_random):
            ax.bar(rel_steps_random + width/2, effects_random,
                   width=width,
                   yerr=errors_random,
                   color=COLORS["random"],
                   alpha=0.6,
                   capsize=3,
                   label="Random",
                   linewidth=0)

        # Reference line
        ax.axhline(0, linestyle='--', linewidth=1, alpha=0.4)

        # Labels
        ax.set_title(f"{model.capitalize()} CoT", fontsize=12, pad=10)
        ax.set_xlabel("Relative Step")
        if idx == 0:
            ax.set_ylabel("Effect Size (Δ Logit)")

        # Ticks
        all_steps = sorted(set(rel_steps_true.tolist() if len(rel_steps_true) else []) |
                           set(rel_steps_random.tolist() if len(rel_steps_random) else []))
        if all_steps:
            ax.set_xticks(all_steps)

        # Legend (only once)
        if idx == 1:
            ax.legend(frameon=False, fontsize=10)

    plt.suptitle("Causal Steering Effects Across Steps", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/steering_relative_step.png", bbox_inches='tight')
    plt.savefig("figures/steering_relative_step.pdf", bbox_inches='tight')
    plt.show()

    # --- Focused Coconut Plot ---
    fig, ax = plt.subplots(figsize=(7, 4.5))

    rel_steps, effects, errors, sigs = [], [], [], []

    for stats in summary.values():
        if (
            stats["model"] == "coconut"
            and stats["direction_type"] == "true"
            and stats["n"] >= 3
        ):
            rel_steps.append(stats["relative_step"])
            effects.append(stats["mean_effect"])
            errors.append(stats["sem"])
            sigs.append(stats["significant"])

    if rel_steps:
        order = np.argsort(rel_steps)
        rel_steps = np.array(rel_steps)[order]
        effects = np.array(effects)[order]
        errors = np.array(errors)[order]
        sigs = np.array(sigs)[order]

        ax.bar(rel_steps,
               effects,
               yerr=errors,
               color=COLORS["coconut"],
               alpha=0.9,
               capsize=4,
               linewidth=0)

        y_offset = 0.03 * (np.max(effects) - np.min(effects) + 1e-6)
        for step, effect, err, sig in zip(rel_steps, effects, errors, sigs):
            if sig:
                ax.text(step,
                        effect + err + y_offset,
                        "★",
                        ha='center',
                        va='bottom',
                        fontsize=11)

        ax.axhline(0, linestyle='--', linewidth=1, alpha=0.4)

        ax.set_title("Coconut: Step-Invariant Steering Signal", fontsize=12, pad=10)
        ax.set_xlabel("Relative Step")
        ax.set_ylabel("Effect Size (Δ Logit)")
        ax.set_xticks(rel_steps)

    plt.tight_layout()
    plt.savefig("figures/coconut_steering_focused.png", bbox_inches='tight')
    plt.savefig("figures/coconut_steering_focused.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_steering_results()