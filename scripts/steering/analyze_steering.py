# scripts/analyze_steering.py
"""
Analyze steering experiment results.
Loads saved results and produces metrics, visualizations, and hypothesis tests.

Usage:
    python scripts/analyze_steering.py --results_dir results/steering/raw/experiment_name
    python scripts/analyze_steering.py --results_file results/steering/raw/experiment_name/steering_results.json
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from src.steering_analysis import (
    compute_flip_metrics,
    compute_dose_response_curve,
    compare_coconut_vs_verbal,
    visualize_steering_sweep,
    plot_dose_response_comparison,
    load_steering_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steering experiment results")
    
    # Input
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing steering_results.json")
    parser.add_argument("--results_file", type=str, default=None,
                        help="Direct path to steering_results.json")
    parser.add_argument("--comparison_file", type=str, default=None,
                        help="Optional: Compare with another results file")
    
    # Analysis parameters
    parser.add_argument("--alpha_threshold", type=float, default=5.0,
                        help="Alpha value for flip rate calculation")
    parser.add_argument("--config_filter", type=str, default=None,
                        help="Filter to specific config (e.g., 'step_0')")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for analysis outputs (default: results_dir/analysis)")
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--show_plots", action="store_true", default=False)
    
    # Analysis options
    parser.add_argument("--analyze_by_hop", action="store_true",
                        help="Break down results by hop count")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def load_results_from_path(results_path: str) -> Dict:
    """Load results from file or directory."""
    if os.path.isdir(results_path):
        file_path = os.path.join(results_path, "steering_results.json")
    else:
        file_path = results_path
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    
    return load_steering_results(file_path)


def aggregate_metrics_by_model(
    all_results: List[Dict], 
    alpha_threshold: float,
    config_filter: Optional[str] = None
) -> pd.DataFrame:
    """Aggregate metrics across all problems, grouped by model type."""
    
    rows = []
    
    for result in all_results:
        model_type = result.get("model_type", "unknown")
        problem_idx = result.get("problem_idx", -1)
        n_hops = result.get("n_hops", 0)
        sweep_results = result.get("sweep_results", {})
        
        if not sweep_results:
            continue
        
        # Filter configs if specified
        configs_to_analyze = sweep_results.keys()
        if config_filter:
            configs_to_analyze = [c for c in configs_to_analyze if config_filter in c]
        
        for config_name in configs_to_analyze:
            config_results = sweep_results.get(config_name, {})
            if not config_results:
                continue
            
            metrics = compute_flip_metrics({config_name: config_results}, alpha_threshold)
            
            for cfg, m in metrics.items():
                rows.append({
                    "model_type": model_type,
                    "problem_idx": problem_idx,
                    "n_hops": n_hops,
                    "config": cfg,
                    "flipped": m.get("flipped", False),
                    "effect_size": m.get("effect_size", 0.0),
                    "prob_shift": m.get("prob_shift", 0.0),
                    "logit_shift": m.get("logit_shift", 0.0),
                    "baseline_prob": m.get("baseline_prob_a", 0.5),
                    "steered_prob": m.get("steered_prob_a", 0.5),
                    "was_steered": m.get("was_steered", False),
                })
    
    return pd.DataFrame(rows)


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics by model type."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for model_type in df["model_type"].unique():
        model_df = df[df["model_type"] == model_type]
        
        print(f"\n{model_type.upper()}:")
        print(f"  Problems analyzed: {model_df['problem_idx'].nunique()}")
        print(f"  Total configurations: {len(model_df)}")
        print(f"  Flip rate: {model_df['flipped'].mean():.3f} ({model_df['flipped'].sum()}/{len(model_df)})")
        print(f"  Mean effect size: {model_df['effect_size'].mean():.3f} ± {model_df['effect_size'].std():.3f}")
        print(f"  Mean |prob shift|: {model_df['prob_shift'].abs().mean():.3f}")
        print(f"  Mean |logit shift|: {model_df['logit_shift'].abs().mean():.3f}")
        
        # By config
        if "config" in model_df.columns and model_df["config"].nunique() > 1:
            print("\n  By configuration:")
            for config in sorted(model_df["config"].unique()):
                cfg_df = model_df[model_df["config"] == config]
                print(f"    {config}: flip={cfg_df['flipped'].mean():.3f}, effect={cfg_df['effect_size'].mean():.3f}")


def analyze_by_hop_count(df: pd.DataFrame, output_dir: str):
    """Analyze and plot results broken down by hop count."""
    
    print("\n" + "="*60)
    print("ANALYSIS BY HOP COUNT")
    print("="*60)
    
    hop_summary = []
    
    for n_hops in sorted(df["n_hops"].unique()):
        hop_df = df[df["n_hops"] == n_hops]
        
        for model_type in hop_df["model_type"].unique():
            model_hop_df = hop_df[hop_df["model_type"] == model_type]
            
            summary = {
                "n_hops": n_hops,
                "model_type": model_type,
                "n_problems": model_hop_df["problem_idx"].nunique(),
                "flip_rate": model_hop_df["flipped"].mean(),
                "mean_effect": model_hop_df["effect_size"].mean(),
                "std_effect": model_hop_df["effect_size"].std(),
            }
            hop_summary.append(summary)
            
            print(f"\n{n_hops} hops - {model_type}:")
            print(f"  Problems: {summary['n_problems']}")
            print(f"  Flip rate: {summary['flip_rate']:.3f}")
            print(f"  Effect size: {summary['mean_effect']:.3f} ± {summary['std_effect']:.3f}")
    
    # Create hop comparison plot
    if output_dir and len(hop_summary) > 0:
        summary_df = pd.DataFrame(hop_summary)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Flip rate by hop
        for model_type in summary_df["model_type"].unique():
            model_data = summary_df[summary_df["model_type"] == model_type]
            axes[0].plot(model_data["n_hops"], model_data["flip_rate"], 
                        'o-', label=model_type, linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of Hops")
        axes[0].set_ylabel("Flip Rate")
        axes[0].set_title("Logical Flip Rate by Hop Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Effect size by hop
        for model_type in summary_df["model_type"].unique():
            model_data = summary_df[summary_df["model_type"] == model_type]
            axes[1].errorbar(model_data["n_hops"], model_data["mean_effect"], 
                           yerr=model_data["std_effect"], 
                           label=model_type, linewidth=2, markersize=8, capsize=5)
        axes[1].set_xlabel("Number of Hops")
        axes[1].set_ylabel("Effect Size")
        axes[1].set_title("Steering Effect Size by Hop Count")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hop_analysis.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved hop analysis plot to {output_dir}/hop_analysis.png")
        plt.close()


def test_hypotheses(df: pd.DataFrame) -> Dict:
    """Test the three hypotheses from the proposal."""
    
    print("\n" + "="*60)
    print("HYPOTHESIS TESTING")
    print("="*60)
    
    results = {}
    
    # Separate by model type
    coconut_df = df[df["model_type"] == "coconut"] if "coconut" in df["model_type"].unique() else None
    verbal_df = df[df["model_type"] == "verbal"] if "verbal" in df["model_type"].unique() else None
    
    if coconut_df is not None and verbal_df is not None:
        # H1: Bottleneck Hypothesis - Coconut has higher flip rate
        coconut_flip = coconut_df["flipped"].mean()
        verbal_flip = verbal_df["flipped"].mean()
        results["H1"] = {
            "description": "Coconut flip rate > Verbal flip rate",
            "coconut_flip_rate": float(coconut_flip),
            "verbal_flip_rate": float(verbal_flip),
            "supported": coconut_flip > verbal_flip,
            "ratio": float(coconut_flip / (verbal_flip + 1e-6)),
        }
        print(f"\nH1 (Bottleneck):")
        print(f"  Coconut flip rate: {coconut_flip:.3f}")
        print(f"  Verbal flip rate: {verbal_flip:.3f}")
        print(f"  Supported: {results['H1']['supported']}")
        print(f"  Ratio: {results['H1']['ratio']:.2f}x")
        
        # H2: Separation Hypothesis - Coconut effect > 1.5x Verbal
        coconut_effect = coconut_df["effect_size"].abs().mean()
        verbal_effect = verbal_df["effect_size"].abs().mean()
        effect_ratio = coconut_effect / (verbal_effect + 1e-6)
        results["H2"] = {
            "description": "Coconut effect size > 1.5x Verbal effect size",
            "coconut_effect": float(coconut_effect),
            "verbal_effect": float(verbal_effect),
            "ratio": float(effect_ratio),
            "supported": effect_ratio > 1.5,
        }
        print(f"\nH2 (Separation):")
        print(f"  Coconut effect: {coconut_effect:.3f}")
        print(f"  Verbal effect: {verbal_effect:.3f}")
        print(f"  Ratio: {effect_ratio:.2f}x")
        print(f"  Supported: {results['H2']['supported']}")
        
        # H3: Difficulty Scaling - Gap widens with hops
        if "n_hops" in df.columns:
            hop_gaps = []
            for n_hops in sorted(df["n_hops"].unique()):
                c_flip = coconut_df[coconut_df["n_hops"] == n_hops]["flipped"].mean()
                v_flip = verbal_df[verbal_df["n_hops"] == n_hops]["flipped"].mean()
                gap = c_flip - v_flip
                hop_gaps.append({"n_hops": n_hops, "coconut": c_flip, "verbal": v_flip, "gap": gap})
            
            # Check if gap increases with hops
            if len(hop_gaps) >= 2:
                gaps = [h["gap"] for h in hop_gaps]
                increasing = all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))
            else:
                increasing = None
            
            results["H3"] = {
                "description": "Gap between Coconut and Verbal increases with hop count",
                "hop_gaps": hop_gaps,
                "increasing_trend": increasing,
                "supported": increasing if increasing is not None else False,
            }
            print(f"\nH3 (Difficulty Scaling):")
            for h in hop_gaps:
                print(f"  {h['n_hops']} hops: Coconut={h['coconut']:.3f}, Verbal={h['verbal']:.3f}, Gap={h['gap']:.3f}")
            print(f"  Gap increasing: {increasing}")
            print(f"  Supported: {results['H3']['supported']}")
    
    elif coconut_df is not None:
        print("\nOnly Coconut results available - skipping comparative hypotheses")
        results["H1"] = {"supported": None, "note": "No verbal results for comparison"}
        results["H2"] = {"supported": None, "note": "No verbal results for comparison"}
    
    return results


def visualize_sample_dose_response(all_results: List[Dict], output_dir: str):
    """Create dose-response curves for sample problems."""
    
    # Find a problem with both model types if available
    problem_indices = {}
    for r in all_results:
        idx = r.get("problem_idx", -1)
        model = r.get("model_type", "unknown")
        if idx not in problem_indices:
            problem_indices[idx] = set()
        problem_indices[idx].add(model)
    
    # Pick first problem with results
    sample_idx = None
    for idx, models in problem_indices.items():
        if len(models) > 0:
            sample_idx = idx
            break
    
    if sample_idx is None:
        return
    
    # Extract results for this problem
    problem_results = {}
    for r in all_results:
        if r.get("problem_idx") == sample_idx:
            model = r.get("model_type")
            sweep = r.get("sweep_results", {})
            # Get first config (e.g., step_0)
            if sweep:
                first_config = list(sweep.keys())[0]
                problem_results[model] = sweep[first_config]
    
    if len(problem_results) >= 2:
        # Compare Coconut vs Verbal
        if "coconut" in problem_results and "verbal" in problem_results:
            plot_dose_response_comparison(
                problem_results["coconut"],
                problem_results["verbal"],
                save_path=os.path.join(output_dir, "sample_dose_response.png")
            )
            print(f"\nSaved sample dose-response plot")


def main():
    args = parse_args()
    
    # Determine results path
    if args.results_dir:
        results_path = args.results_dir
    elif args.results_file:
        results_path = args.results_file
    else:
        raise ValueError("Must provide --results_dir or --results_file")
    
    # Load results
    print(f"Loading results from: {results_path}")
    data = load_results_from_path(results_path)
    
    # Set output directory
    if args.output_dir is None:
        if os.path.isdir(results_path):
            args.output_dir = os.path.join(results_path, "analysis")
        else:
            args.output_dir = os.path.join(os.path.dirname(results_path), "analysis")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Output directory: {args.output_dir}")
    
    # Extract results list
    all_results = data.get("results", [])
    if not all_results:
        # Handle old format
        all_results = data if isinstance(data, list) else [data]
    
    print(f"Loaded {len(all_results)} result entries")
    
    # Aggregate metrics
    df = aggregate_metrics_by_model(
        all_results, 
        args.alpha_threshold,
        args.config_filter
    )
    
    if len(df) == 0:
        print("No metrics could be computed. Check results format.")
        return
    
    # Print summary
    print_summary_statistics(df)
    
    # Test hypotheses
    hypothesis_results = test_hypotheses(df)
    
    # Analyze by hop count if requested
    if args.analyze_by_hop and "n_hops" in df.columns:
        analyze_by_hop_count(df, args.output_dir)
    
    # Visualizations
    if args.save_plots:
        # Save aggregated dataframe
        df.to_csv(os.path.join(args.output_dir, "aggregated_metrics.csv"), index=False)
        print(f"\nSaved aggregated metrics to {args.output_dir}/aggregated_metrics.csv")
        
        # Sample dose-response
        visualize_sample_dose_response(all_results, args.output_dir)
        
        # Create summary plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Flip rate comparison
        model_types = df["model_type"].unique()
        flip_rates = [df[df["model_type"] == m]["flipped"].mean() for m in model_types]
        axes[0].bar(model_types, flip_rates, color=['blue', 'orange'][:len(model_types)])
        axes[0].set_ylabel("Flip Rate")
        axes[0].set_title(f"Logical Flip Rate (α={args.alpha_threshold})")
        axes[0].set_ylim(0, 1)
        
        # Effect size distribution
        for i, model in enumerate(model_types):
            effects = df[df["model_type"] == model]["effect_size"].values
            axes[1].hist(effects, bins=20, alpha=0.5, label=model)
        axes[1].set_xlabel("Effect Size")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Effect Size Distribution")
        axes[1].legend()
        axes[1].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "summary_comparison.png"), dpi=150, bbox_inches="tight")
        print(f"Saved summary comparison to {args.output_dir}/summary_comparison.png")
        plt.close()
    
    # Save hypothesis results
    with open(os.path.join(args.output_dir, "hypothesis_results.json"), "w") as f:
        json.dump(hypothesis_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()