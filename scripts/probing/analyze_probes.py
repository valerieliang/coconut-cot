import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("ANALYZING RESULTS")
print("=" * 60)

# Style settings
sns.set_theme(style="white", context="talk")

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
})

COLORS = {
    "verbal": "#4C78A8",
    "coconut": "#F58518",
    "grid": "#E5E5E5",
    "text": "#333333"
}

def clean_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='y', color=COLORS["grid"], linewidth=0.8)
    ax.grid(False, axis='x')


# Data loading
print("\n[1/5] Loading data...")
def load_all_results():
    with open("results/verbal_cot_binary_probe.json") as f:
        verbal_binary = json.load(f)
    with open("results/coconut_binary_probe.json") as f:
        coconut_binary = json.load(f)

    with open("results/verbal_cot_progress_probe.json") as f:
        verbal_progress = json.load(f)
    with open("results/coconut_progress_probe.json") as f:
        coconut_progress = json.load(f)

    verbal_matrix = np.load("results/verbal_cot_binary_matrix.npy")
    coconut_matrix = np.load("results/coconut_binary_matrix.npy")
    verbal_pvals = np.load("results/verbal_cot_binary_pvalues.npy")
    coconut_pvals = np.load("results/coconut_binary_pvalues.npy")

    with open("results/statistical_comparison.json") as f:
        stats = json.load(f)

    return {
        'verbal_binary': verbal_binary,
        'coconut_binary': coconut_binary,
        'verbal_progress': verbal_progress,
        'coconut_progress': coconut_progress,
        'verbal_matrix': verbal_matrix,
        'coconut_matrix': coconut_matrix,
        'verbal_pvals': verbal_pvals,
        'coconut_pvals': coconut_pvals,
        'stats': stats
    }


data = load_all_results()
os.makedirs("figures", exist_ok=True)

# Print summary statistics
stats = data['stats']
print(f"\n--- SUMMARY STATISTICS ---")
print(f"Diagonal Accuracy (mean +- std):")
print(f"  Verbal CoT: {stats['verbal_mean']:.3f} +- {stats['verbal_std']:.3f}")
print(f"  Coconut:    {stats['coconut_mean']:.3f} +- {stats['coconut_std']:.3f}")
print(f"  Difference: {stats['difference']:+.3f}")
print(f"\nStatistical tests (n={stats['n_pairs']} pairs):")
print(f"  Paired t-test: t={stats['ttest_statistic']:.3f}, p={stats['ttest_p']:.4f}")
print(f"  Wilcoxon:      W={stats['wilcoxon_statistic']:.1f}, p={stats['wilcoxon_p']:.4f}")

print(f"\n--- PROGRESS STAGE ACCURACY ---")
for stage in ['early', 'middle', 'late']:
    v = data['verbal_progress'][stage]
    c = data['coconut_progress'][stage]
    v_mean = v.get('mean', float('nan'))
    c_mean = c.get('mean', float('nan'))
    if not np.isnan(v_mean) and not np.isnan(c_mean):
        diff = v_mean - c_mean
        print(f"  {stage:8s}: Verbal={v_mean:.3f}, Coconut={c_mean:.3f} (diff={diff:+.3f})")
    else:
        print(f"  {stage:8s}: insufficient data")

# Figure 1: Heatmaps
print("\n[2/5] Generating heatmaps...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for idx, (model_name, matrix, pvals) in enumerate([
    ('Verbal CoT', data['verbal_matrix'], data['verbal_pvals']),
    ('Coconut', data['coconut_matrix'], data['coconut_pvals'])
]):
    ax = axes[idx]
    mask = np.isnan(matrix)

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        vmin=0.4,
        vmax=1.0,
        cbar=idx == 1,
        square=True,
        linewidths=0.5,
        linecolor="white",
        mask=mask,
        annot=False
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                continue

            val = matrix[i, j]

            if i == j or val > 0.75:
                text = f"{val:.2f}"
                if not np.isnan(pvals[i, j]) and pvals[i, j] < 0.05:
                    text += "*"

                ax.text(
                    j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    fontsize=8,
                    color='white' if val > 0.65 else COLORS["text"]
                )

    ax.set_title(model_name, fontweight='semibold')
    ax.set_xlabel("Label Step")
    ax.set_ylabel("Representation Step")

    ax.set_xticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(range(matrix.shape[0]), rotation=0)
    ax.set_yticklabels(range(matrix.shape[0]), rotation=0)

plt.suptitle("Cross-Step Faithfulness Matrices", fontweight='semibold')
plt.savefig("figures/cross_step_matrices.png", bbox_inches='tight')
plt.savefig("figures/cross_step_matrices.pdf", bbox_inches='tight')
plt.close()
print("  -> Saved figures/cross_step_matrices.png/pdf")


# Figure 2: Trajectories
print("\n[3/5] Generating trajectory plots...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)

for idx, n_hops in enumerate([3, 4, 5]):
    ax = axes[idx]

    for model_name, binary_data, color in [
        ('Verbal CoT', data['verbal_binary'], COLORS["verbal"]),
        ('Coconut', data['coconut_binary'], COLORS["coconut"])
    ]:
        steps, accs, errs_low, errs_high = [], [], [], []

        for r in binary_data:
            if r.get('n_hops') == n_hops and r.get('step', 0) < n_hops:
                acc = r.get('acc_at_1') or r.get('mean')

                if acc is not None and not np.isnan(acc):
                    steps.append(r['step'])
                    accs.append(acc)

                    n = r.get('n', 10)
                    se = np.sqrt(acc * (1 - acc) / max(n, 1))
                    errs_low.append(acc - 1.96 * se)
                    errs_high.append(acc + 1.96 * se)

        if steps:
            order = np.argsort(steps)
            steps = np.array(steps)[order]
            accs = np.array(accs)[order]
            errs_low = np.array(errs_low)[order]
            errs_high = np.array(errs_high)[order]

            ax.plot(
                steps, accs,
                marker='o',
                markersize=6,
                linewidth=2,
                color=color,
                label=model_name
            )

            ax.fill_between(steps, errs_low, errs_high, alpha=0.15, color=color)

    ax.axhline(0.091, linestyle='--', color='gray', alpha=0.5, linewidth=1)
    ax.set_title(f"{n_hops}-Hop Problems", fontweight='semibold')
    ax.set_xlabel("Reasoning Step")
    ax.set_xticks(range(n_hops))
    ax.set_ylim(0.4, 1.05)
    clean_axis(ax)

axes[0].set_ylabel("Accuracy@1")

fig.legend(loc="lower center", ncol=2, frameon=False)
plt.suptitle("Step-Level Faithfulness Trajectories", fontweight='semibold')
plt.savefig("figures/faithfulness_trajectories.png", bbox_inches='tight')
plt.savefig("figures/faithfulness_trajectories.pdf", bbox_inches='tight')
plt.close()
print("  -> Saved figures/faithfulness_trajectories.png/pdf")


# Figure 3: Progress aggregation
print("\n[4/5] Generating progress aggregation plot...")
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

stages = ['early', 'middle', 'late']
x = np.arange(len(stages))
width = 0.35

verbal_means, verbal_errors = [], []
coconut_means, coconut_errors = [], []

for stage in stages:
    v = data['verbal_progress'][stage]
    c = data['coconut_progress'][stage]

    v_mean = v.get('mean', 0)
    c_mean = c.get('mean', 0)

    v_ci = v.get('ci_high', v_mean) - v_mean
    c_ci = c.get('ci_high', c_mean) - c_mean

    verbal_means.append(v_mean)
    verbal_errors.append(v_ci)
    coconut_means.append(c_mean)
    coconut_errors.append(c_ci)

bars1 = ax.bar(x - width/2, verbal_means, width,
               color=COLORS["verbal"], alpha=0.8,
               label='Verbal CoT',
               yerr=verbal_errors, capsize=4)

bars2 = ax.bar(x + width/2, coconut_means, width,
               color=COLORS["coconut"], alpha=0.8,
               label='Coconut',
               yerr=coconut_errors, capsize=4)

for bar, val in zip(bars1, verbal_means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f'{val:.2f}', ha='center', fontsize=8)

for bar, val in zip(bars2, coconut_means):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
            f'{val:.2f}', ha='center', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([
    'Early\n(<0.33)',
    'Middle\n(0.33-0.66)',
    'Late\n(>0.66)'
])

ax.set_ylabel("Balanced Accuracy")
ax.set_title("Progress-Based Aggregation", fontweight='semibold')
ax.set_ylim(0, 0.4)

clean_axis(ax)
ax.legend(frameon=False)

plt.savefig("figures/progress_aggregation.png", bbox_inches='tight')
plt.savefig("figures/progress_aggregation.pdf", bbox_inches='tight')
plt.close()
print("  -> Saved figures/progress_aggregation.png/pdf")


# Figure 4: Diagonal vs off-diagonal
print("\n[5/5] Generating diagonal vs off-diagonal box plots...")
fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

for idx, (model_name, matrix, color) in enumerate([
    ('Verbal CoT', data['verbal_matrix'], COLORS["verbal"]),
    ('Coconut', data['coconut_matrix'], COLORS["coconut"])
]):
    ax = axes[idx]

    diag_vals, off_vals = [], []
    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                if i == j:
                    diag_vals.append(matrix[i, j])
                else:
                    off_vals.append(matrix[i, j])

    bp = ax.boxplot([diag_vals, off_vals],
                    patch_artist=True,
                    widths=0.5)

    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Diagonal", "Off-diagonal"])
    ax.set_title(model_name, fontweight='semibold')

    if diag_vals:
        ax.text(1, 1.02, f"mu={np.mean(diag_vals):.2f}", ha='center', fontsize=9)
    if off_vals:
        ax.text(2, 1.02, f"mu={np.mean(off_vals):.2f}", ha='center', fontsize=9)

    ax.set_ylim(0.3, 1.1)
    clean_axis(ax)

plt.suptitle("Diagonal vs Off-Diagonal Accuracy", fontweight='semibold')
plt.savefig("figures/diag_vs_offdiag.png", bbox_inches='tight')
plt.savefig("figures/diag_vs_offdiag.pdf", bbox_inches='tight')
plt.close()
print("  -> Saved figures/diag_vs_offdiag.png/pdf")