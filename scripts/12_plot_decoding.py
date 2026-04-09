# scripts/12_plot_decoding.py
"""
Plot decoding analysis results from 11_decoding_analysis.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    "highlight": "#E8A838",
}

def safe_get(d, *keys):
    """Safely retrieve nested dictionary values; return np.nan if missing"""
    try:
        for key in keys:
            d = d[key]
        if d is None or (isinstance(d, float) and np.isnan(d)):
            return np.nan
        return d
    except KeyError:
        return np.nan

def plot_decoding_results(results_path="results/decoding/full_results.json"):
    with open(results_path) as f:
        data = json.load(f)
    
    summary = data['summary']
    coconut_results = data.get('coconut_results', [])
    verbal_results = data.get('verbal_results', [])
    
    os.makedirs("figures/decoding", exist_ok=True)
    
    # Extract summary metrics with safe defaults
    unique_coco = safe_get(summary, "unique_candidates", "coconut", "mean")
    unique_verb = safe_get(summary, "unique_candidates", "verbal", "mean")
    overlap_coco = safe_get(summary, "candidate_overlap", "coconut", "mean")
    overlap_verb = safe_get(summary, "candidate_overlap", "verbal", "mean")
    adj_coco = safe_get(summary, "adjacent_similarity", "coconut", "mean")
    adj_verb = safe_get(summary, "adjacent_similarity", "verbal", "mean")
    
    # ============================================================
    # Print summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Coconut unique (top-10 retrieved): {unique_coco}")
    print(f"Verbal unique  (top-5 retrieved): {unique_verb}")
    print(f"Coconut overlap: {overlap_coco}")
    print(f"Verbal  overlap: {overlap_verb}")
    print(f"Coconut adjacent similarity: {adj_coco}")
    print(f"Verbal  adjacent similarity: {adj_verb}")
    print("="*60)
    
    # ============================================================
    # Figure 1: Unique Candidates Comparison
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot comparing unique candidates
    ax = axes[0]
    models = ['Coconut', 'Verbal CoT']
    unique_means = [
        summary['unique_candidates']['coconut']['mean'],
        summary['unique_candidates']['verbal']['mean']
    ]
    colors = [COLORS['coconut'], COLORS['verbal']]

    bars = ax.bar(range(2), unique_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_ylabel('Unique Candidates in Top-10')
    ax.set_ylim(0, 12)
    ax.set_xticks(range(2))
    ax.set_xticklabels(models)
    ax.set_title('Unique Candidates per Step', fontsize=11, fontweight='semibold')

    # Add value labels above bars
    for bar, val in zip(bars, unique_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    # Candidate overlap (Jaccard similarity)
    ax = axes[1]
    overlap_means = [
        summary['candidate_overlap']['coconut']['mean'],
        summary['candidate_overlap']['verbal']['mean']
    ]

    bars = ax.bar(range(2), overlap_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_ylabel('Jaccard Similarity')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(2))
    ax.set_xticklabels(models)
    ax.set_title('Candidate Overlap (Consecutive Steps)', fontsize=11, fontweight='semibold')

    for bar, val in zip(bars, overlap_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    # Candidate overlap (Jaccard similarity)
    ax = axes[1]
    overlap_means = [
        summary['candidate_overlap']['coconut']['mean'],
        summary['candidate_overlap']['verbal']['mean']
    ]

    bars = ax.bar(range(2), overlap_means, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_ylabel('Jaccard Similarity')
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(2))
    ax.set_xticklabels(models)
    ax.set_title('Candidate Overlap (Consecutive Steps)', fontsize=11, fontweight='semibold')  # smaller fontsize

    for bar, val in zip(bars, overlap_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Superposition Evidence: Multiple Candidates in Continuous Thought', 
                fontweight='semibold', fontsize=13)  # keep suptitle larger
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("figures/decoding/unique_candidates.png", bbox_inches='tight')
    plt.show()
    
    # ============================================================
    # Figure 2: Adjacent Step Similarity
    # ============================================================
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(2), [adj_coco, adj_verb], color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_ylabel('Cosine Similarity')
    # Adjusted y-axis for realistic proportion
    min_y = min(adj_coco, adj_verb) - 0.0015
    ax.set_ylim(min_y, 1.0)
    ax.set_xticks(range(2))
    ax.set_xticklabels(models)
    ax.set_title('Adjacent Step Cosine Similarity', fontweight='semibold')
    for bar, val in zip(bars, [adj_coco, adj_verb]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, f'{val:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("figures/decoding/adjacent_similarity.png", bbox_inches='tight')
    plt.savefig("figures/decoding/adjacent_similarity.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_decoding_results()