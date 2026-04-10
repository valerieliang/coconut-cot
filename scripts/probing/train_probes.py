# scripts/probing/train_probes.py
import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from src.probe_utils import (
    load_reps, 
    run_binary_probe_grid, 
    run_binary_cross_step_probe,
    build_progress_aggregated_dataset,
    evaluate_progress_probe,
    compare_models_statistical
)

# Create results directories
os.makedirs("results/probing", exist_ok=True)


def format_matrix(matrix, p_values=None, sig_level=0.05):
    """Pretty print matrix with significance markers."""
    rows, cols = matrix.shape
    lines = []
    for i in range(rows):
        row_str = []
        for j in range(cols):
            val = matrix[i, j]
            if np.isnan(val):
                row_str.append("  nan ")
            else:
                s = f"{val:.3f}"
                if p_values is not None and not np.isnan(p_values[i, j]) and p_values[i, j] < sig_level:
                    s += "*"
                row_str.append(s)
        lines.append("[" + " ".join(row_str) + "]")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING PROBES")
    print("=" * 60)
    
    all_results = {}
    
    for model_name, path, rep_key in [
        ("verbal_cot", "data/verbal_reps.json", "step_reps"),
        ("coconut", "data/coconut_reps.json", "thought_reps"),
    ]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        records = load_reps(path)
        print(f"Loaded {len(records)} problems")
        
        # Binary Similarity Probe
        print("\n-- Binary Similarity Probe --", flush=True)
        binary_results = run_binary_probe_grid(records, max_steps=5, rep_key=rep_key)
        with open(f"results/probing/{model_name}_binary_probe.json", "w") as f:
            json.dump(binary_results, f, indent=2)
        print(f"  Saved to results/probing/{model_name}_binary_probe.json")
        
        # Binary Cross-Step Matrix
        print("\n-- Binary Cross-Step Matrix --", flush=True)
        matrix, p_values = run_binary_cross_step_probe(records, rep_key=rep_key, max_steps=5)
        
        print(f"\nAcc@1 Matrix ({model_name}) (* = p<0.05):")
        print(format_matrix(matrix, p_values))
        
        np.save(f"results/probing/{model_name}_binary_matrix.npy", matrix)
        np.save(f"results/probing/{model_name}_binary_pvalues.npy", p_values)
        print(f"  Saved matrices to results/probing/")
        
        all_results[model_name] = {
            "binary_results": binary_results,
            "matrix": matrix,
            "p_values": p_values
        }
        
        # Progress Aggregation
        print("\n-- Progress-Based Aggregation --", flush=True)
        stages = build_progress_aggregated_dataset(records, rep_key=rep_key)
        progress_results = evaluate_progress_probe(stages)
        
        print("\nProgress Stage Accuracy:")
        for stage, res in progress_results.items():
            mean_val = res.get('mean', float('nan'))
            ci_high = res.get('ci_high', float('nan'))
            n = res.get('n', 0)
            chance = res.get('chance', float('nan'))
            
            if not np.isnan(mean_val):
                ci = ci_high - mean_val if not np.isnan(ci_high) else float('nan')
                sig = " **" if mean_val > chance + 0.1 else ""
                print(f"  {stage:8s}: acc={mean_val:.3f} +/- {ci:.3f} "
                      f"(n={n}, chance={chance:.3f}){sig}")
            else:
                print(f"  {stage:8s}: insufficient data (n={n})")
        
        with open(f"results/probing/{model_name}_progress_probe.json", "w") as f:
            json.dump(progress_results, f, indent=2)
        print(f"  Saved to results/probing/{model_name}_progress_probe.json")
        
        all_results[model_name]["progress_results"] = progress_results
    
    # Statistical Comparison
    print(f"\n{'='*60}")
    print("Statistical Comparison: Verbal CoT vs Coconut")
    print(f"{'='*60}")
    
    comparison = compare_models_statistical(
        all_results["verbal_cot"]["binary_results"],
        all_results["coconut"]["binary_results"]
    )
    
    if comparison and comparison.get("n_pairs", 0) > 0:
        print(f"\nDiagonal accuracies (mean +/- std):")
        print(f"  Verbal CoT: {comparison['verbal_mean']:.3f} +/- {comparison['verbal_std']:.3f}")
        print(f"  Coconut:    {comparison['coconut_mean']:.3f} +/- {comparison['coconut_std']:.3f}")
        print(f"  Difference: {comparison['difference']:+.3f}")
        print(f"\nStatistical tests (n={comparison['n_pairs']} pairs):")
        print(f"  Paired t-test: t={comparison['ttest_statistic']:.3f}, p={comparison['ttest_p']:.4f}")
        print(f"  Wilcoxon:      W={comparison['wilcoxon_statistic']:.1f}, p={comparison['wilcoxon_p']:.4f}")
        
        with open("results/probing/statistical_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
    else:
        print("\n  Insufficient data for statistical comparison")
    
    print("\n" + "=" * 60)
    print("PROBE TRAINING COMPLETE")
    print("Results saved to results/probing/")
    print("=" * 60)