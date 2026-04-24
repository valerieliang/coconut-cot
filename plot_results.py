"""
plot_results.py
Reads steering/outputs/results.csv and produces two figures:
  effect_by_position.png  -- accuracy delta per injection position
  alpha_sweep.png         -- accuracy vs alpha per position

Run from ~/coconut-cot:
  python plot_results.py --config steering/steering_config.yaml
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def plot(cfg):
    out_dir = Path(os.path.expanduser(cfg["output_dir"]))
    df = pd.read_csv(out_dir / "results.csv")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline = df[df["alpha"] == 0].set_index("sample_idx")["correct"]
    steered  = df[df["alpha"] > 0].copy()
    steered["baseline"] = steered["sample_idx"].map(baseline)
    steered["delta"]    = steered["correct"] - steered["baseline"]

    # ── figure 1: effect by position (averaged across alpha) ─────────────────
    effect = steered.groupby("position")["delta"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#D85A30" if abs(v) > 0.05 else "#888780" for v in effect.values]
    ax.bar(effect.index.astype(str), effect.values, color=colors, width=0.55)
    ax.axhline(0, color="#444441", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Latent position injected")
    ax.set_ylabel("Accuracy delta (steered − baseline)")
    ax.set_title("Steering effect by latent position")
    plt.tight_layout()
    p1 = fig_dir / "effect_by_position.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"Saved {p1}")

    # ── figure 2: accuracy vs alpha per position ──────────────────────────────
    base_mean = baseline.mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for pos in sorted(steered["position"].unique()):
        sub = steered[steered["position"] == pos].groupby("alpha")["correct"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=f"L{pos}")
    ax.axhline(base_mean, color="#444441", linewidth=0.8,
               linestyle="--", label="baseline")
    ax.set_xlabel("Alpha (steering strength)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. steering strength per position")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p2 = fig_dir / "alpha_sweep.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"Saved {p2}")

    # ── quick summary to terminal ─────────────────────────────────────────────
    print(f"\nBaseline accuracy: {base_mean:.3f}")
    print("\nMean accuracy delta by position (averaged across all alphas):")
    for pos, delta in effect.items():
        marker = " ← strong effect" if abs(delta) > 0.05 else ""
        print(f"  L{pos}: {delta:+.3f}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    plot(load_config(args.config))