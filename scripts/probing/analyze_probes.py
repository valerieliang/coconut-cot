# scripts/probing/analyze_probes.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Use results/probing/ directory for data
RESULTS_DIR = "results/probing"
# Save figures to figures/probing/
FIGURES_DIR = "figures/probing"

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_all_results():
    """Load all probe results from results/probing/"""
    data = {}
    
    for model in ["verbal_cot", "coconut"]:
        probe_file = Path(RESULTS_DIR) / f"{model}_binary_probe.json"
        matrix_file = Path(RESULTS_DIR) / f"{model}_binary_matrix.npy"
        pvalues_file = Path(RESULTS_DIR) / f"{model}_binary_pvalues.npy"
        progress_file = Path(RESULTS_DIR) / f"{model}_progress_probe.json"
        
        if probe_file.exists():
            with open(probe_file) as f:
                data[f"{model}_binary"] = json.load(f)
            print(f"  Loaded {probe_file}")
        
        if matrix_file.exists():
            data[f"{model}_matrix"] = np.load(matrix_file)
            data[f"{model}_pvalues"] = np.load(pvalues_file)
            print(f"  Loaded {matrix_file}")
        
        if progress_file.exists():
            with open(progress_file) as f:
                data[f"{model}_progress"] = json.load(f)
            print(f"  Loaded {progress_file}")
    
    # Load statistical comparison
    stats_file = Path(RESULTS_DIR) / "statistical_comparison.json"
    if stats_file.exists():
        with open(stats_file) as f:
            data["stats"] = json.load(f)
        print(f"  Loaded {stats_file}")
    
    return data


def plot_binary_matrices(data):
    """Plot side-by-side heatmaps of binary probe matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, model in enumerate(["verbal_cot", "coconut"]):
        matrix = data.get(f"{model}_matrix")
        pvalues = data.get(f"{model}_pvalues")
        
        if matrix is None:
            print(f"  Warning: No matrix for {model}")
            continue
        
        # Create mask for non-significant cells
        mask = (pvalues > 0.05) if pvalues is not None else None
        
        # Handle NaN values
        matrix_clean = np.nan_to_num(matrix, nan=0.5)
        
        sns.heatmap(
            matrix_clean, 
            ax=axes[idx],
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            vmin=0.5, 
            vmax=1.0,
            mask=mask,
            cbar_kws={'label': 'Accuracy'}
        )
        axes[idx].set_title(f'{model.replace("_", " ").title()}\nBinary Probe Accuracy')
        axes[idx].set_xlabel('Target Step')
        axes[idx].set_ylabel('Source Step')
    
    plt.tight_layout()
    output_path = f"{FIGURES_DIR}/binary_probe_heatmaps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved heatmap to {output_path}")
    plt.close()


def plot_progress_comparison(data):
    """Plot progress probe accuracy comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_labels = []
    verbal_accs = []
    verbal_cis = []
    coconut_accs = []
    coconut_cis = []
    
    for model, color in [("verbal_cot", "blue"), ("coconut", "orange")]:
        progress = data.get(f"{model}_progress")
        if progress:
            for stage in sorted(progress.keys()):
                if stage not in x_labels:
                    x_labels.append(stage)
                
                res = progress[stage]
                acc = res.get('mean', float('nan'))
                ci_high = res.get('ci_high', float('nan'))
                
                # Calculate error bar
                error = ci_high - acc if not np.isnan(ci_high) else 0
                
                if model == "verbal_cot":
                    verbal_accs.append(acc)
                    verbal_cis.append(error)
                else:
                    coconut_accs.append(acc)
                    coconut_cis.append(error)
    
    if not x_labels:
        print("  Warning: No progress data to plot")
        return
    
    x = np.arange(len(x_labels))
    width = 0.35
    
    ax.bar(x - width/2, verbal_accs, width, label='Verbal CoT', 
           color='blue', alpha=0.7, yerr=verbal_cis, capsize=5)
    ax.bar(x + width/2, coconut_accs, width, label='Coconut', 
           color='orange', alpha=0.7, yerr=coconut_cis, capsize=5)
    
    ax.set_ylabel('Probe Accuracy')
    ax.set_xlabel('Progress Stage')
    ax.set_title('Progress Probe: Verbal CoT vs Coconut')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance (0.5)')
    ax.legend()
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = f"{FIGURES_DIR}/progress_probe_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison to {output_path}")
    plt.close()


def plot_diagonal_comparison(data):
    """Plot diagonal accuracies comparison using box plots."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models_data = {}
    labels = []
    data_list = []
    
    for model in ["verbal_cot", "coconut"]:
        matrix = data.get(f"{model}_matrix")
        if matrix is not None:
            diag = np.diag(matrix)
            diag_clean = diag[~np.isnan(diag)]
            if len(diag_clean) > 0:
                models_data[model] = diag_clean
                labels.append(model.replace("_", " ").title())
                data_list.append(diag_clean)
    
    if not data_list:
        print("  Warning: No diagonal data to plot")
        return
    
    # Boxplot
    bp = ax.boxplot(
        data_list,
        labels=labels,
        patch_artist=True,
        showfliers=False
    )
    
    # Color the boxes
    colors = ["#4C78A8", "#F58518"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Overlay jittered points
    for i, accs in enumerate(data_list):
        x_jitter = np.random.normal(i + 1, 0.04, len(accs))
        ax.scatter(x_jitter, accs, alpha=0.5, s=30, color='black')
    
    ax.set_ylabel('Diagonal Probe Accuracy')
    ax.set_title('Step-Specific Probe Accuracy\n(Step i representation → Step i conclusion)')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Chance')
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    output_path = f"{FIGURES_DIR}/diagonal_accuracy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved diagonal comparison to {output_path}")
    plt.close()


def plot_cross_step_curves(data):
    """Plot how probe accuracy decays when probing step j from step i."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, model in enumerate(["verbal_cot", "coconut"]):
        matrix = data.get(f"{model}_matrix")
        
        if matrix is None:
            continue
        
        n_steps = matrix.shape[0]
        
        for source_step in range(min(n_steps, 5)):
            row = matrix[source_step, :]
            mask = ~np.isnan(row)
            if np.any(mask):
                x = np.arange(len(row))[mask]
                y = row[mask]
                axes[idx].plot(x, y, marker='o', label=f'Source step {source_step}')
        
        axes[idx].set_xlabel('Target Step')
        axes[idx].set_ylabel('Probe Accuracy')
        axes[idx].set_title(f'{model.replace("_", " ").title()}\nCross-Step Probing')
        axes[idx].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[idx].legend()
        axes[idx].set_ylim(0.4, 1.05)
    
    plt.tight_layout()
    output_path = f"{FIGURES_DIR}/cross_step_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved cross-step curves to {output_path}")
    plt.close()


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("PROBE ANALYSIS SUMMARY")
    print("="*60)
    
    print("\nDiagonal Accuracies (Step i -> Step i):")
    for model in ["verbal_cot", "coconut"]:
        matrix = data.get(f"{model}_matrix")
        if matrix is not None:
            diag = np.diag(matrix)
            diag_clean = diag[~np.isnan(diag)]
            if len(diag_clean) > 0:
                print(f"  {model.replace('_', ' ').title()}:")
                print(f"    Values: {[f'{x:.3f}' for x in diag_clean]}")
                print(f"    Mean: {np.mean(diag_clean):.3f} +/- {np.std(diag_clean):.3f}")
    
    stats = data.get("stats")
    if stats:
        print(f"\nStatistical Comparison (n={stats.get('n_pairs', 0)} pairs):")
        print(f"  Verbal CoT: {stats.get('verbal_mean', 0):.3f} +/- {stats.get('verbal_std', 0):.3f}")
        print(f"  Coconut:    {stats.get('coconut_mean', 0):.3f} +/- {stats.get('coconut_std', 0):.3f}")
        print(f"  Difference: {stats.get('difference', 0):+.3f}")
        print(f"  Paired t-test: t={stats.get('ttest_statistic', 0):.3f}, p={stats.get('ttest_p', 1):.4f}")
        print(f"  Wilcoxon: W={stats.get('wilcoxon_statistic', 0):.1f}, p={stats.get('wilcoxon_p', 1):.4f}")
        
        print("\nInterpretation:")
        if stats.get('ttest_p', 1) < 0.05:
            if stats.get('difference', 0) > 0:
                print(f"  [SIG] Coconut significantly MORE faithful (p={stats['ttest_p']:.4f})")
            else:
                print(f"  [SIG] Verbal CoT significantly MORE faithful (p={stats['ttest_p']:.4f})")
        else:
            print(f"  [NS] No significant difference in faithfulness (p={stats['ttest_p']:.4f})")


if __name__ == "__main__":
    print("="*60)
    print("ANALYZING PROBE RESULTS")
    print("="*60)
    
    if not Path("results/probing").exists():
        print("\nERROR: No probe results found!")
        print("Run train_probes.py first:")
        print("  python scripts/probing/train_probes.py")
        sys.exit(1)
    
    print(f"\n[1/5] Loading data from {RESULTS_DIR}/...")
    data = load_all_results()
    
    if not data:
        print("No data loaded!")
        sys.exit(1)
    
    print(f"\n  Loaded {len(data)} result files")
    
    print(f"\n[2/5] Creating binary probe heatmaps...")
    plot_binary_matrices(data)
    
    print(f"\n[3/5] Creating progress comparison plot...")
    plot_progress_comparison(data)
    
    print(f"\n[4/5] Creating diagonal and cross-step plots...")
    plot_diagonal_comparison(data)
    plot_cross_step_curves(data)
    
    print(f"\n[5/5] Generating summary...")
    print_summary(data)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nFigures saved to {FIGURES_DIR}/")
    print(f"  - binary_probe_heatmaps.png")
    print(f"  - progress_probe_comparison.png")
    print(f"  - diagonal_accuracy_comparison.png")
    print(f"  - cross_step_curves.png")