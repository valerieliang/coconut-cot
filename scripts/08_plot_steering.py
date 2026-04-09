# scripts/08_plot_steering.py

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_steering_results(results_path="results/steering/aggregated_results.json"):
    with open(results_path) as f:
        data = json.load(f)

    alphas = [0.0, 1.0, 2.0, 3.0]

    fig, ax = plt.subplots(figsize=(6, 5))

    for model_name, color in [("verbal", "#4C78A8"), ("coconut", "#F58518")]:
        logit_means = []
        logit_sems = []

        for alpha in alphas:
            vals = []
            for res in data[model_name]:
                if res and str(alpha) in res:
                    val = res[str(alpha)].get("logit_diff")
                    if val is not None:
                        vals.append(val)
            if vals:
                logit_means.append(np.mean(vals))
                logit_sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                logit_means.append(np.nan)
                logit_sems.append(np.nan)

        ax.errorbar(alphas, logit_means, yerr=logit_sems,
                   marker='o', capsize=5, label=model_name, color=color)

    ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel("Steering Strength (alpha)")
    ax.set_ylabel("Logit Difference (Concept A - Concept B)")
    ax.set_title("Causal Steering Effect")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/steering_dose_response.png", dpi=150)
    plt.savefig("figures/steering_dose_response.pdf")
    plt.show()


if __name__ == "__main__":
    plot_steering_results()