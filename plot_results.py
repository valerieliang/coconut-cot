"""
plot_results.py
Reads results.csv and produces two figures:
  effect_by_position.png  -- accuracy delta per injection position
  alpha_sweep.png         -- accuracy vs alpha per position
Run: python steering/plot_results.py --config steering/steering_config.yaml
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot(cfg):
    out_dir = Path(cfg["output_dir"])
    df = pd.read_csv(out_dir / "results.csv")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_acc = df[df["alpha"] == 0].groupby("sample_idx")["correct"].first()

    # --- figure 1: effect by position (averaged across alpha) ---
    steered = df[df["alpha"] > 0].copy()
    steered["baseline"] = steered["sample_idx"].map(baseline_acc)
    steered["delta"] = steered["correct"] - steered["baseline"]
    effect = steered.groupby("position")["delta"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#D85A30" if abs(v) > 0.05 else "#888780" for v in effect.values]
    ax.bar(effect.index.astype(str), effect.values, color=colors, width=0.55)
    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Latent position injected")
    ax.set_ylabel("Accuracy delta (steered − baseline)")
    ax.set_title("Steering effect by latent position")
    plt.tight_layout()
    plt.savefig(fig_dir / "effect_by_position.png", dpi=150)
    plt.close()
    print(f"Saved effect_by_position.png")

    # --- figure 2: accuracy vs alpha per position ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for pos in sorted(steered["position"].unique()):
        sub = steered[steered["position"] == pos].groupby("alpha")["correct"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"L{pos}")
    base_mean = baseline_acc.mean()
    ax.axhline(base_mean, color="#444441", linewidth=0.8, linestyle="--", label="baseline")
    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. steering strength per position")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_dir / "alpha_sweep.png", dpi=150)
    plt.close()
    print(f"Saved alpha_sweep.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plot(load_config(args.config))