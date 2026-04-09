# scripts\09_steering_stats_table.py

import json
import numpy as np
from scipy import stats
from collections import defaultdict
import os

def generate_stats_table(results_path="results/steering/enhanced/full_results.json",
                         output_json="results/steering/enhanced/stats_summary.json"):
    """
    Generate statistical summary table and save results to JSON.
    Removes non-keyboard characters from printed output.
    """
    with open(results_path) as f:
        results = json.load(f)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Prepare data structure for JSON export
    summary_data = {}
    
    print("\n" + "="*90)
    print("STEERING EXPERIMENT STATISTICAL SUMMARY")
    print("="*90)
    
    for model in ["verbal", "coconut"]:
        print(f"\n{model.upper()} CoT")
        print("-"*70)
        print(f"{'Direction':<12} {'Rel Step':<10} {'Raw Delta':<12} {'Norm Delta':<12} {'% Change':<12} {'p-val':<8} {'N':<5}")
        print("-"*70)
        
        # Aggregate by direction type and relative step
        agg = defaultdict(lambda: {"raw": [], "norm": [], "pct": []})
        
        for res in results:
            if res["model"] != model:
                continue
            
            dir_type = res["direction_type"]
            rel_step = res["relative_step"]
            effects = res["result"]["effects"]
            
            if "5.0" in effects:
                eff = effects["5.0"]
                agg[(dir_type, rel_step)]["raw"].append(eff.get("raw_change", 0))
                agg[(dir_type, rel_step)]["norm"].append(eff.get("std_normalized", 0))
                agg[(dir_type, rel_step)]["pct"].append(eff.get("percent_change", 0))
        
        # Store model data in summary dict
        model_data = {}
        
        for (dir_type, rel_step), data in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1])):
            raw_mean = np.mean(data["raw"])
            raw_sem = np.std(data["raw"]) / np.sqrt(len(data["raw"])) if len(data["raw"]) > 1 else 0.0
            norm_mean = np.mean(data["norm"])
            pct_mean = np.mean(data["pct"])
            
            if len(data["raw"]) >= 3:
                t_stat, p_val = stats.ttest_1samp(data["raw"], 0)
            else:
                p_val = 1.0
            
            sig = "**" if p_val < 0.05 else ("*" if p_val < 0.10 else "")
            
            # Print with +/- instead of plus-minus sign
            print(f"{dir_type:<12} {rel_step:<10} {raw_mean:+.3f}+/-{raw_sem:.3f}  "
                  f"{norm_mean:+.3f}     {pct_mean:+.1%}     {p_val:.4f}  {len(data['raw']):<5} {sig}")
            
            # Store for JSON
            key = f"{dir_type}_step_{rel_step}"
            model_data[key] = {
                "direction_type": dir_type,
                "relative_step": rel_step,
                "raw_change_mean": raw_mean,
                "raw_change_sem": raw_sem,
                "norm_change_mean": norm_mean,
                "percent_change_mean": pct_mean,
                "p_value": p_val,
                "n": len(data["raw"])
            }
        
        summary_data[model] = model_data
    
    # Summary statistics for paper
    print("\n" + "="*90)
    print("PAPER-READY SUMMARY STATISTICS")
    print("="*90)
    
    paper_stats = {}
    for model in ["coconut"]:  # Focus on Coconut for key claim
        print(f"\n{model.upper()}: Step-invariant steerability")
        
        true_all_steps = []
        true_concept_step = []
        
        for res in results:
            if res["model"] != model or res["direction_type"] != "true":
                continue
            if "5.0" in res["result"]["effects"]:
                eff = res["result"]["effects"]["5.0"]["std_normalized"]
                true_all_steps.append(eff)
                if res["relative_step"] == 0:
                    true_concept_step.append(eff)
        
        model_paper = {}
        if true_all_steps:
            mean_all = np.mean(true_all_steps)
            sem_all = np.std(true_all_steps) / np.sqrt(len(true_all_steps))
            t_all, p_all = stats.ttest_1samp(true_all_steps, 0)
            print(f"  All steps: d = {mean_all:.3f} +/- {sem_all:.3f}, p = {p_all:.4f}, n = {len(true_all_steps)}")
            model_paper["all_steps"] = {
                "mean": mean_all,
                "sem": sem_all,
                "p_value": p_all,
                "n": len(true_all_steps)
            }
        
        if true_concept_step:
            mean_step = np.mean(true_concept_step)
            sem_step = np.std(true_concept_step) / np.sqrt(len(true_concept_step))
            print(f"  Concept step only: d = {mean_step:.3f} +/- {sem_step:.3f}, n = {len(true_concept_step)}")
            model_paper["concept_step_only"] = {
                "mean": mean_step,
                "sem": sem_step,
                "n": len(true_concept_step)
            }
        
        paper_stats[model] = model_paper
    
    summary_data["paper_summary"] = paper_stats
    
    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nStatistics saved to {output_json}")
    print("="*90)

if __name__ == "__main__":
    generate_stats_table()