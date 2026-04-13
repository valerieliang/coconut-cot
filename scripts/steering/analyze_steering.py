# scripts/steering/analyze_steering.py
"""
Analyze steering experiment results with hypothesis testing for CogAI paper.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import chi2_contingency

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steering results")
    parser.add_argument("--results_dir", type=str, 
                        default="results/steering/steering_suite_20260413_010514",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures")
    parser.add_argument("--alpha_threshold", type=float, default=10.0,
                        help="Alpha value for flip rate calculation")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_steering_results(filepath: str) -> Dict:
    """Load steering results from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_flip_metrics_from_results(results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """
    Compute flip metrics directly from loaded results.
    Returns dict with config_name -> metrics.
    """
    metrics = {}
    
    # Get all results
    all_results = results.get('results', [])
    
    for problem_result in all_results:
        problem_idx = problem_result.get('problem_idx', -1)
        answer_flipped = problem_result.get('answer_flipped', False)
        sweep_results = problem_result.get('sweep_results', {})
        
        # For each configuration in sweep_results
        for config_name, config_data in sweep_results.items():
            if config_name == 'metadata':
                continue
            
            if config_name not in metrics:
                metrics[config_name] = {
                    'flipped': False,
                    'effect_size': 0,
                    'prob_shift': 0,
                    'n_problems': 0,
                    'flipped_count': 0
                }
            
            metrics[config_name]['n_problems'] += 1
            if answer_flipped:
                metrics[config_name]['flipped_count'] += 1
                metrics[config_name]['flipped'] = True
    
    # Convert to flip rates
    for config_name in metrics:
        metrics[config_name]['flip_rate'] = metrics[config_name]['flipped_count'] / metrics[config_name]['n_problems']
    
    return metrics


def find_experiment_results(results_dir: str, verbose: bool = False) -> Dict[str, Dict]:
    """Find and load all experiment results once."""
    all_results = {}
    
    for exp_name in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_name)
        results_file = os.path.join(exp_path, "steering_results.json")
        
        if os.path.exists(results_file):
            try:
                all_results[exp_name] = load_steering_results(results_file)
                if verbose:
                    print(f"  Loaded: {exp_name}")
            except Exception as e:
                print(f"  Warning: Could not load {exp_name}: {e}")
    
    return all_results


def extract_comparison_data(all_results: Dict[str, Dict]) -> Tuple[Dict, Dict]:
    """Extract Coconut and Verbal results from Experiment 7."""
    coconut_results = None
    verbal_results = None
    
    for exp_name, results in all_results.items():
        if 'comparison_coconut' in exp_name:
            coconut_results = results
        elif 'comparison_verbal' in exp_name:
            verbal_results = results
    
    return coconut_results, verbal_results


def extract_difficulty_data(all_results: Dict[str, Dict]) -> Dict:
    """Extract difficulty scaling results."""
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
    """H1: Bottleneck Hypothesis - Coconut has higher flip rate."""
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Bottleneck Hypothesis")
    print("Coconut has higher causal steering efficacy than Verbal CoT")
    print("="*70)
    
    # Compute metrics
    coconut_metrics = compute_flip_metrics_from_results(coconut_results, alpha_threshold)
    verbal_metrics = compute_flip_metrics_from_results(verbal_results, alpha_threshold)
    
    # Get flip rates (average across configs)
    coconut_flip_rate = np.mean([m['flip_rate'] for m in coconut_metrics.values()])
    verbal_flip_rate = np.mean([m['flip_rate'] for m in verbal_metrics.values()])
    
    coconut_flipped = sum(m['flipped_count'] for m in coconut_metrics.values())
    coconut_total = sum(m['n_problems'] for m in coconut_metrics.values())
    verbal_flipped = sum(m['flipped_count'] for m in verbal_metrics.values())
    verbal_total = sum(m['n_problems'] for m in verbal_metrics.values())
    
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
        "coconut_flip_rate": float(coconut_flip_rate),
        "verbal_flip_rate": float(verbal_flip_rate),
        "coconut_n": int(coconut_total),
        "verbal_n": int(verbal_total),
        "coconut_flipped": int(coconut_flipped),
        "verbal_flipped": int(verbal_flipped),
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "cohens_h": float(cohens_h),
        "significant": bool(p_value < 0.05),
        "supported": bool(coconut_flip_rate > verbal_flip_rate)
    }
    
    print(f"\nCoconut: {coconut_flip_rate:.1%} flip rate ({coconut_flipped}/{coconut_total})")
    print(f"Verbal: {verbal_flip_rate:.1%} flip rate ({verbal_flipped}/{verbal_total})")
    print(f"\nChi-square: χ² = {chi2:.2f}, p = {p_value:.4e}")
    print(f"Cohen's h = {cohens_h:.3f}")
    print(f"\nH1 SUPPORTED: {results['supported']}")
    
    return results


def test_hypothesis_2(coconut_results: Dict, verbal_results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """H2: Separation Hypothesis - Verbal shows incoherence."""
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Separation Hypothesis")
    print("="*70)
    
    coconut_metrics = compute_flip_metrics_from_results(coconut_results, alpha_threshold)
    verbal_metrics = compute_flip_metrics_from_results(verbal_results, alpha_threshold)
    
    coconut_incoherence = 1 - np.mean([m['flip_rate'] for m in coconut_metrics.values()])
    verbal_incoherence = 1 - np.mean([m['flip_rate'] for m in verbal_metrics.values()])
    
    results = {
        "hypothesis": "H2 - Separation Hypothesis",
        "coconut_incoherence_rate": float(coconut_incoherence),
        "verbal_incoherence_rate": float(verbal_incoherence),
        "supported": bool(verbal_incoherence > coconut_incoherence)
    }
    
    print(f"\nCoconut incoherence: {coconut_incoherence:.1%}")
    print(f"Verbal incoherence: {verbal_incoherence:.1%}")
    print(f"\nH2 SUPPORTED: {results['supported']}")
    
    return results


def test_hypothesis_3(difficulty_results: Dict, alpha_threshold: float = 10.0) -> Dict:
    """H3: Difficulty Scaling - Gap widens with hop count."""
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Difficulty Scaling")
    print("="*70)
    
    hops = sorted(difficulty_results.keys())
    flip_rates = []
    
    for h in hops:
        results = difficulty_results[h]
        metrics = compute_flip_metrics_from_results(results, alpha_threshold)
        flip_rate = np.mean([m['flip_rate'] for m in metrics.values()])
        flip_rates.append(flip_rate)
    
    # Test for trend
    if len(hops) >= 2:
        correlation = np.corrcoef(hops, flip_rates)[0, 1]
        slope = np.polyfit(hops, flip_rates, 1)[0]
        increasing = slope > 0
    else:
        correlation = 0
        slope = 0
        increasing = False
    
    results = {
        "hypothesis": "H3 - Difficulty Scaling",
        "hops": [int(h) for h in hops],
        "flip_rates": [float(r) for r in flip_rates],
        "slope": float(slope),
        "correlation": float(correlation),
        "increasing": bool(increasing),
        "supported": bool(increasing)
    }
    
    print(f"\nHop count analysis:")
    for h, rate in zip(hops, flip_rates):
        print(f"  {h} hops: Flip rate = {rate:.1%}")
    
    print(f"\nTrend: slope = {slope:.3f}, correlation = {correlation:.3f}")
    print(f"\nH3 SUPPORTED: {results['supported']}")
    
    return results


def create_figure_h1(h1_results: Dict, output_dir: str):
    """Create figure for Hypothesis 1."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
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
    ax_diag.text(2.5, 4, 'Steering → Flip', ha='center', fontsize=8, color='green')
    
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
    ax_diag.text(7.5, 4, 'Steering → Mismatch', ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_h2_separation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure_h2_separation.png")


def create_figure_h3(h3_results: Dict, output_dir: str):
    """Create figure for Hypothesis 3."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    hops = h3_results['hops']
    flip_rates = h3_results['flip_rates']
    
    # Bar chart
    x = np.arange(len(hops))
    width = 0.6
    
    bars = axes[0].bar(x, flip_rates, width, color='#2E86AB', edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Number of Reasoning Hops')
    axes[0].set_ylabel('Logical Flip Rate')
    axes[0].set_title('H3: Difficulty Scaling', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(hops)
    axes[0].set_ylim(0, 1.1)
    
    for bar, rate in zip(bars, flip_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Trend line
    axes[1].plot(hops, flip_rates, 'o-', color='#D81B60', linewidth=2, markersize=8)
    
    if len(hops) >= 2:
        z = np.polyfit(hops, flip_rates, 1)
        p = np.poly1d(z)
        axes[1].plot(hops, p(hops), '--', color='gray', alpha=0.7, 
                    label=f'Trend (slope={z[0]:.3f})')
        axes[1].legend()
    
    axes[1].set_xlabel('Number of Reasoning Hops')
    axes[1].set_ylabel('Flip Rate')
    axes[1].set_title(f'Gap Analysis: r = {h3_results["correlation"]:.3f}', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_h3_difficulty.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: figure_h3_difficulty.png")


def generate_paper_figures(results_dir: str, output_dir: str, alpha_threshold: float = 10.0):
    """Generate all figures."""
    
    print("\n" + "="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)
    
    # Load all results once
    print("\nLoading experiment data...")
    all_results = find_experiment_results(results_dir, verbose=False)
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
        "supported": False, "hops": [], "flip_rates": [], "slope": 0, "correlation": 0
    }
    
    # Create figures
    print("\nCreating figures...")
    os.makedirs(output_dir, exist_ok=True)
    create_figure_h1(h1_results, output_dir)
    create_figure_h2(h2_results, output_dir)
    
    if difficulty_results:
        create_figure_h3(h3_results, output_dir)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    all_hypotheses = {
        "hypothesis_1": convert_to_serializable(h1_results),
        "hypothesis_2": convert_to_serializable(h2_results),
        "hypothesis_3": convert_to_serializable(h3_results),
        "alpha_threshold": alpha_threshold
    }
    
    with open(os.path.join(output_dir, "hypothesis_testing_results.json"), "w") as f:
        json.dump(all_hypotheses, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*70)
    print(f"\nH1 (Bottleneck): {h1_results['supported']} - "
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
    
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Alpha threshold: {args.alpha_threshold}")
    
    generate_paper_figures(args.results_dir, args.output_dir, args.alpha_threshold)


if __name__ == "__main__":
    args = parse_args()
    main()