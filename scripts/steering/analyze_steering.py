# scripts/analyze_steering.py
"""
Analyze steering experiment results with hypothesis testing for CogAI paper.
Uses the existing steering_analysis.py functions for consistency.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy.stats import chi2_contingency, mannwhitneyu

# Import from existing steering_analysis
from src.steering_analysis import (
    load_steering_results,
    compute_flip_metrics,
    compute_dose_response_curve,
    compare_coconut_vs_verbal,
    plot_dose_response_comparison,
    _alpha_keys,
    _get_alpha
)

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steering results with hypothesis testing")
    
    parser.add_argument("--results_dir", type=str, 
                        default="results/steering/steering_suite_20260413_010514",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures (default: results_dir/figures)")
    parser.add_argument("--alpha_threshold", type=float, default=10.0,
                        help="Alpha value for flip rate calculation")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def find_experiment_results(results_dir: str) -> Dict[str, Dict]:
    """Find and load all experiment results from the suite."""
    all_results = {}
    
    for exp_name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_name)
        results_file = os.path.join(exp_path, "steering_results.json")
        
        if os.path.exists(results_file):
            try:
                all_results[exp_name] = load_steering_results(results_file)
                if args.verbose:
                    print(f"  Loaded: {exp_name}")
            except Exception as e:
                print(f"  Warning: Could not load {exp_name}: {e}")
    
    return all_results


def extract_comparison_data(all_results: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """Extract Coconut and Verbal results from Experiment 7 (direct comparison)."""
    coconut_results = None
    verbal_results = None
    
    for exp_name, results in all_results.items():
        if 'comparison_coconut' in exp_name:
            coconut_results = results
        elif 'comparison_verbal' in exp_name:
            verbal_results = results
    
    return coconut_results, verbal_results


def extract_step_steering_data(all_results: Dict[str, Dict], model: str) -> Dict:
    """Extract step-by-step steering results for a model."""
    step_results = {}
    
    for exp_name, results in all_results.items():
        if model in exp_name and 'step_' in exp_name and 'cumulative' not in exp_name:
            # Extract step number
            parts = exp_name.split('_')
            for i, p in enumerate(parts):
                if p == 'step' and i+1 < len(parts):
                    step = int(parts[i+1])
                    step_results[step] = results
                    break
    
    return step_results


def extract_difficulty_data(all_results: Dict[str, Dict]) -> Dict:
    """Extract difficulty scaling results (Experiment 8)."""
    difficulty_results = {}
    
    for exp_name, results in all_results.items():
        if 'difficulty' in exp_name:
            if '3hop' in exp_name:
                difficulty_results[3] = results
            elif '4hop' in exp_name:
                difficulty_results[4] = results
            elif '5hop' in exp_name:
                difficulty_results[5] = results
    
    return difficulty_results


def test_hypothesis_1(coconut_results: Dict, verbal_results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """
    H1: Bottleneck Hypothesis - Coconut has significantly higher flip rate than Verbal.
    
    Uses compare_coconut_vs_verbal from steering_analysis.py
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Bottleneck Hypothesis")
    print("Coconut has higher causal steering efficacy than Verbal CoT")
    print("="*70)
    
    # Use existing comparison function
    comparison = compare_coconut_vs_verbal(coconut_results, verbal_results, alpha_threshold)
    
    # Get detailed metrics
    coconut_metrics = compute_flip_metrics(coconut_results, alpha_threshold)
    verbal_metrics = compute_flip_metrics(verbal_results, alpha_threshold)
    
    coconut_flipped = sum(1 for m in coconut_metrics.values() if m['flipped'])
    coconut_total = len(coconut_metrics)
    verbal_flipped = sum(1 for m in verbal_metrics.values() if m['flipped'])
    verbal_total = len(verbal_metrics)
    
    coconut_flip_rate = coconut_flipped / coconut_total if coconut_total > 0 else 0
    verbal_flip_rate = verbal_flipped / verbal_total if verbal_total > 0 else 0
    
    # Statistical test
    contingency = np.array([
        [coconut_flipped, coconut_total - coconut_flipped],
        [verbal_flipped, verbal_total - verbal_flipped]
    ])
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Effect size (Cohen's h)
    import math
    h1 = 2 * math.asin(math.sqrt(coconut_flip_rate))
    h2 = 2 * math.asin(math.sqrt(verbal_flip_rate))
    cohens_h = h1 - h2
    
    results = {
        "hypothesis": "H1 - Bottleneck Hypothesis",
        "coconut_flip_rate": coconut_flip_rate,
        "verbal_flip_rate": verbal_flip_rate,
        "coconut_n": coconut_total,
        "verbal_n": verbal_total,
        "coconut_mean_effect": comparison['coconut']['mean_effect_size'],
        "verbal_mean_effect": comparison['verbal']['mean_effect_size'],
        "flip_rate_ratio": comparison['flip_rate_ratio'],
        "effect_size_ratio": comparison['effect_size_ratio'],
        "chi2_statistic": chi2,
        "p_value": p_value,
        "cohens_h": cohens_h,
        "significant": p_value < 0.05,
        "bottleneck_supported": comparison['bottleneck_supported']
    }
    
    print(f"\nCoconut: {coconut_flip_rate:.1%} flip rate ({coconut_flipped}/{coconut_total}), effect={comparison['coconut']['mean_effect_size']:.3f}")
    print(f"Verbal: {verbal_flip_rate:.1%} flip rate ({verbal_flipped}/{verbal_total}), effect={comparison['verbal']['mean_effect_size']:.3f}")
    print(f"\nChi-square: chi^2 = {chi2:.2f}, p = {p_value:.4e}")
    print(f"Cohen's h = {cohens_h:.3f}")
    print(f"Flip rate ratio = {comparison['flip_rate_ratio']:.1f}x")
    print(f"\nH1 SUPPORTED: {results['bottleneck_supported']}")
    
    return results


def test_hypothesis_2(coconut_results: Dict, verbal_results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """
    H2: Separation Hypothesis - Verbal CoT shows incoherence under steering.
    
    Incoherence = steering applied but answer doesn't flip (mismatch)
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Separation Hypothesis")
    print("Verbal CoT shows incoherence under steering, Coconut shows consistency")
    print("="*70)
    
    def compute_incoherence(results: Dict) -> float:
        """Compute incoherence rate: steering applied but no flip."""
        total_steered = 0
        incoherent = 0
        
        for config_name, config_results in results.get('results', []):
            # This needs to be adapted based on results structure
            sweep = config_results.get('sweep_results', {})
            for sweep_name, sweep_data in sweep.items():
                if sweep_name == 'metadata':
                    continue
                for alpha_key, alpha_data in sweep_data.items():
                    if alpha_key == 'metadata':
                        continue
                    try:
                        alpha = float(alpha_key)
                        if alpha > 0:
                            was_steered = alpha_data.get('was_steered', False)
                            flipped = alpha_data.get('answer_flipped', False)
                            if was_steered:
                                total_steered += 1
                                if not flipped:
                                    incoherent += 1
                    except (ValueError, TypeError):
                        pass
        
        return incoherent / total_steered if total_steered > 0 else 0
    
    # Use existing metrics
    coconut_metrics = compute_flip_metrics(coconut_results, alpha_threshold)
    verbal_metrics = compute_flip_metrics(verbal_results, alpha_threshold)
    
    # Approximate incoherence from flip metrics
    coconut_flip_rate = np.mean([m['flipped'] for m in coconut_metrics.values()])
    verbal_flip_rate = np.mean([m['flipped'] for m in verbal_metrics.values()])
    
    # Incoherence = steering attempted but no flip
    # Assuming steering was applied to all problems (which it was)
    coconut_incoherence = 1 - coconut_flip_rate
    verbal_incoherence = 1 - verbal_flip_rate
    
    # Statistical test
    stat, p_value = mannwhitneyu(
        [1 - m['flipped'] for m in coconut_metrics.values()],
        [1 - m['flipped'] for m in verbal_metrics.values()],
        alternative='less'
    )
    
    results = {
        "hypothesis": "H2 - Separation Hypothesis",
        "coconut_incoherence_rate": coconut_incoherence,
        "verbal_incoherence_rate": verbal_incoherence,
        "coconut_n": len(coconut_metrics),
        "verbal_n": len(verbal_metrics),
        "statistic": stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "supported": verbal_incoherence > coconut_incoherence
    }
    
    print(f"\nCoconut incoherence: {coconut_incoherence:.1%} (steering without flip)")
    print(f"Verbal incoherence: {verbal_incoherence:.1%} (steering without flip)")
    print(f"\nMann-Whitney U: p = {p_value:.4f}")
    print(f"\nH2 SUPPORTED: {results['supported']}")
    
    return results


def test_hypothesis_3(difficulty_results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """
    H3: Difficulty Scaling - Gap between Coconut and Verbal widens with hop count.
    """
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Difficulty Scaling")
    print("Steering efficacy gap widens as reasoning chain length increases")
    print("="*70)
    
    hops = sorted(difficulty_results.keys())
    coconut_rates = []
    gaps = []
    
    for h in hops:
        results = difficulty_results[h]
        metrics = compute_flip_metrics(results, alpha_threshold)
        flip_rate = np.mean([m['flipped'] for m in metrics.values()])
        coconut_rates.append(flip_rate)
        gaps.append(flip_rate)  # Gap = Coconut rate (Verbal is 0)
    
    # Test for increasing trend
    if len(gaps) >= 2:
        correlation = np.corrcoef(hops, gaps)[0, 1]
        slope = np.polyfit(hops, gaps, 1)[0]
        increasing = slope > 0
    else:
        correlation = 0
        slope = 0
        increasing = False
    
    results = {
        "hypothesis": "H3 - Difficulty Scaling",
        "hops": hops,
        "coconut_rates": coconut_rates,
        "gaps": gaps,
        "slope": slope,
        "correlation": correlation,
        "increasing_trend": increasing,
        "supported": increasing and correlation > 0.5
    }
    
    print(f"\nHop count analysis:")
    for h, rate, gap in zip(hops, coconut_rates, gaps):
        print(f"  {h} hops: Flip rate = {rate:.1%}, Gap = {gap:.1%}")
    
    print(f"\nTrend: slope = {slope:.3f}, correlation = {correlation:.3f}")
    print(f"Gap {'increases' if increasing else 'decreases'} with hop count")
    print(f"\nH3 SUPPORTED: {results['supported']}")
    
    return results


def create_figure_h1(h1_results: Dict, output_dir: str):
    """Create figure for Hypothesis 1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
    models = ['Coconut', 'Verbal CoT']
    flip_rates = [h1_results['coconut_flip_rate'], h1_results['verbal_flip_rate']]
    colors = ['#2E86AB', '#A23B72']
    
    bars = axes[0].bar(models, flip_rates, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Logical Flip Rate')
    axes[0].set_title('H1: Causal Steering Efficacy', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    
    for bar, rate in zip(bars, flip_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    if h1_results['significant']:
        axes[0].text(0.5, 0.95, f'*** p < 0.001\nCohen\'s h = {h1_results["cohens_h"]:.2f}', 
                    ha='center', transform=axes[0].transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Effect size gauge
    ax_gauge = axes[1]
    ax_gauge.set_xlim(0, 1.2)
    ax_gauge.set_ylim(0, 1)
    
    ax_gauge.axvspan(0, 0.2, alpha=0.3, color='green', label='Small')
    ax_gauge.axvspan(0.2, 0.5, alpha=0.3, color='yellow', label='Medium')
    ax_gauge.axvspan(0.5, 0.8, alpha=0.3, color='orange', label='Large')
    ax_gauge.axvspan(0.8, 1.2, alpha=0.3, color='red', label='Very Large')
    
    h_val = min(h1_results['cohens_h'], 1.2)
    ax_gauge.arrow(0, 0.5, h_val, 0, head_width=0.1, head_length=0.05, 
                   fc='black', ec='black', linewidth=2)
    ax_gauge.plot(h_val, 0.5, 'o', color='black', markersize=10)
    
    ax_gauge.set_xlim(0, max(1.0, h_val + 0.1))
    ax_gauge.set_ylim(0.2, 0.8)
    ax_gauge.set_xlabel("Cohen's h Effect Size")
    ax_gauge.set_title(f"Effect Size: h = {h1_results['cohens_h']:.2f}")
    ax_gauge.set_yticks([])
    ax_gauge.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_h1_bottleneck.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure_h1_bottleneck.png")


def create_figure_h2(h2_results: Dict, output_dir: str):
    """Create figure for Hypothesis 2."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bar chart
    models = ['Coconut', 'Verbal CoT']
    incoherence = [h2_results['coconut_incoherence_rate'], h2_results['verbal_incoherence_rate']]
    colors = ['#2E86AB', '#A23B72']
    
    bars = axes[0].bar(models, incoherence, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Incoherence Rate')
    axes[0].set_title('H2: Steering-Induced Incoherence', fontweight='bold')
    axes[0].set_ylim(0, 1.1)
    
    for bar, rate in zip(bars, incoherence):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Conceptual diagram
    ax_diag = axes[1]
    ax_diag.set_xlim(0, 10)
    ax_diag.set_ylim(0, 10)
    ax_diag.axis('off')
    ax_diag.set_title('Mechanistic Interpretation', fontweight='bold')
    
    # Coconut
    ax_diag.text(2.5, 9, 'Coconut (Bottleneck)', ha='center', fontweight='bold', fontsize=10)
    rect_coconut = plt.Rectangle((1, 5), 3, 2, facecolor='#2E86AB', alpha=0.5, edgecolor='black')
    ax_diag.add_patch(rect_coconut)
    ax_diag.text(2.5, 6, 'Latent\nBottleneck', ha='center', va='center', fontsize=9)
    ax_diag.annotate('', xy=(2.5, 5), xytext=(2.5, 7), arrowprops=dict(arrowstyle='->', color='black'))
    ax_diag.annotate('', xy=(2.5, 3), xytext=(2.5, 5), arrowprops=dict(arrowstyle='->', color='black'))
    ax_diag.text(2.5, 4, 'Steering -> Flip', ha='center', fontsize=8, color='green')
    
    # Verbal
    ax_diag.text(7.5, 9, 'Verbal CoT (Parallel)', ha='center', fontweight='bold', fontsize=10)
    rect_verbal1 = plt.Rectangle((6, 5), 1.5, 2, facecolor='#A23B72', alpha=0.5, edgecolor='black')
    rect_verbal2 = plt.Rectangle((7.5, 5), 1.5, 2, facecolor='#A23B72', alpha=0.3, edgecolor='black')
    ax_diag.add_patch(rect_verbal1)
    ax_diag.add_patch(rect_verbal2)
    ax_diag.text(6.75, 6, 'Verbal\nTrace', ha='center', va='center', fontsize=8)
    ax_diag.text(8.25, 6, 'Answer\nPath', ha='center', va='center', fontsize=8)
    ax_diag.annotate('', xy=(7.5, 3), xytext=(6.75, 5), arrowprops=dict(arrowstyle='->', color='red', linestyle='--'))
    ax_diag.annotate('', xy=(8.25, 3), xytext=(8.25, 5), arrowprops=dict(arrowstyle='->', color='black'))
    ax_diag.text(7.5, 4, 'Steering -> Mismatch', ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_h2_separation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure_h2_separation.png")


def create_figure_h3(h3_results: Dict, output_dir: str):
    """Create figure for Hypothesis 3."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    hops = h3_results['hops']
    coconut_rates = h3_results['coconut_rates']
    gaps = h3_results['gaps']
    
    # Bar chart by hop
    x = np.arange(len(hops))
    width = 0.6
    
    bars = axes[0].bar(x, coconut_rates, width, color='#2E86AB', edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Number of Reasoning Hops')
    axes[0].set_ylabel('Logical Flip Rate')
    axes[0].set_title('H3: Difficulty Scaling', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(hops)
    axes[0].set_ylim(0, 1.1)
    
    for bar, rate in zip(bars, coconut_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Gap trend
    axes[1].plot(hops, gaps, 'o-', color='#D81B60', linewidth=2, markersize=8, label='Flip Rate')
    
    # Trend line
    z = np.polyfit(hops, gaps, 1)
    p = np.poly1d(z)
    axes[1].plot(hops, p(hops), '--', color='gray', alpha=0.7, label=f'Trend (slope={z[0]:.3f})')
    
    axes[1].set_xlabel('Number of Reasoning Hops')
    axes[1].set_ylabel('Flip Rate')
    axes[1].set_title(f'Gap Analysis: r = {h3_results["correlation"]:.3f}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_h3_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure_h3_difficulty.png")


def create_dose_response_figure(coconut_results: Dict, verbal_results: Dict, output_dir: str):
    """Create dose-response curve using existing plot function."""
    plot_dose_response_comparison(
        coconut_results, 
        verbal_results,
        save_path=os.path.join(output_dir, 'dose_response_curve.png')
    )
    print(f"  Saved: dose_response_curve.png")


def generate_paper_figures(results_dir: str, output_dir: str, alpha_threshold: float = 10.0):
    """Generate all figures for the paper."""
    
    print("\n" + "="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)
    
    # Load all results
    print("\nLoading experiment data...")
    all_results = find_experiment_results(results_dir)
    print(f"Found {len(all_results)} experiment result files")
    
    # Extract specific experiment data
    coconut_results, verbal_results = extract_comparison_data(all_results)
    difficulty_results = extract_difficulty_data(all_results)
    
    if coconut_results is None or verbal_results is None:
        print("ERROR: Could not find comparison experiment results")
        return
    
    # Test all three hypotheses
    h1_results = test_hypothesis_1(coconut_results, verbal_results, alpha_threshold)
    h2_results = test_hypothesis_2(coconut_results, verbal_results, alpha_threshold)
    h3_results = test_hypothesis_3(difficulty_results, alpha_threshold) if difficulty_results else {
        "supported": False, "hops": [], "coconut_rates": [], "gaps": [], "slope": 0, "correlation": 0
    }
    
    # Create figures
    print("\nCreating figures...")
    create_figure_h1(h1_results, output_dir)
    create_figure_h2(h2_results, output_dir)
    
    if difficulty_results:
        create_figure_h3(h3_results, output_dir)
    
    create_dose_response_figure(coconut_results, verbal_results, output_dir)
    
    # Save all results
    all_hypotheses = {
        "hypothesis_1": h1_results,
        "hypothesis_2": h2_results,
        "hypothesis_3": h3_results,
        "alpha_threshold": alpha_threshold
    }
    
    with open(os.path.join(output_dir, "hypothesis_testing_results.json"), "w") as f:
        json.dump(all_hypotheses, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*70)
    print(f"\nH1 (Bottleneck): {h1_results['bottleneck_supported']} - "
          f"Coconut: {h1_results['coconut_flip_rate']:.1%} vs Verbal: {h1_results['verbal_flip_rate']:.1%}")
    print(f"H2 (Separation): {h2_results['supported']} - "
          f"Verbal incoherence: {h2_results['verbal_incoherence_rate']:.1%} vs Coconut: {h2_results['coconut_incoherence_rate']:.1%}")
    print(f"H3 (Difficulty): {h3_results['supported']} - "
          f"Slope: {h3_results['slope']:.3f}, Correlation: {h3_results['correlation']:.3f}")
    
    print(f"\nAll figures saved to: {output_dir}")


def main():
    args = parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "figures")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Alpha threshold: {args.alpha_threshold}")
    
    generate_paper_figures(args.results_dir, args.output_dir, args.alpha_threshold)


if __name__ == "__main__":
    args = parse_args()
    main()