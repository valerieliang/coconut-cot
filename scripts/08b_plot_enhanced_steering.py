# scripts/08b_plot_enhanced_steering.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats
import os

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

COLORS = {
    "verbal": "#4C78A8", 
    "coconut": "#F58518", 
    "random": "#999999",
    "orthogonal": "#72B66B"
}


def plot_dose_response(results_path="results/steering/enhanced/full_results.json"):
    with open(results_path) as f:
        results = json.load(f)
    
    alphas = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    # =========================
    # Figure 1: Dose-response
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model in enumerate(["verbal", "coconut"]):
        ax = axes[idx]
        
        agg = defaultdict(lambda: {"changes": [], "normalized": []})
        
        for res in results:
            if res["model"] != model:
                continue
            
            dir_type = res["direction_type"]
            effects = res["result"]["effects"]
            
            for alpha_str, effect_dict in effects.items():
                alpha = float(alpha_str)
                if alpha == 0.0:
                    continue
                
                raw_change = effect_dict.get("raw_change", 0)
                std_norm = effect_dict.get("std_normalized", 0)
                
                key = (dir_type, alpha)
                agg[key]["changes"].append(raw_change)
                agg[key]["normalized"].append(std_norm)
        
        for dir_type, color, label in [
            ("true", COLORS[model], "True Direction"),
            ("orthogonal", COLORS["orthogonal"], "Orthogonal Control"),
            ("random", COLORS["random"], "Random Control"),
        ]:
            means, sems = [], []
            for alpha in alphas:
                key = (dir_type, alpha)
                if key in agg and len(agg[key]["changes"]) >= 3:
                    changes = agg[key]["changes"]
                    means.append(np.mean(changes))
                    sems.append(np.std(changes) / np.sqrt(len(changes)))
                else:
                    means.append(np.nan)
                    sems.append(np.nan)
            
            if not all(np.isnan(means)):
                ax.errorbar(alphas, means, yerr=sems,
                            marker='o', capsize=4,
                            color=color, label=label,
                            linewidth=2, markersize=6)
        
        ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel("Steering Strength (α)")
        ax.set_ylabel("Raw Effect Size (Δ Logit Difference)")
        ax.set_title(f"{model.capitalize()} CoT", fontweight='semibold')
        ax.set_xscale('log')
        ax.legend(loc='best', frameon=True)
    
    plt.suptitle("Dose-Response: Steering Effect by Strength", fontweight='semibold')
    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig("figures/steering_dose_response.png", bbox_inches='tight')
    plt.savefig("figures/steering_dose_response.pdf", bbox_inches='tight')
    plt.show()
    
    # =========================
    # Figure 2: Normalized effects
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model in enumerate(["verbal", "coconut"]):
        ax = axes[idx]
        
        agg = defaultdict(list)
        for res in results:
            if res["model"] != model:
                continue
            
            dir_type = res["direction_type"]
            relative_step = res["relative_step"]
            effects = res["result"]["effects"]
            
            if "5.0" in effects:
                std_norm = effects["5.0"].get("std_normalized", 0)
                key = (dir_type, relative_step)
                agg[key].append(std_norm)
        
        rel_steps = sorted(set(k[1] for k in agg.keys() if k[0] == "true"))
        
        true_means, true_sems = [], []
        random_means, random_sems = [], []
        
        for step in rel_steps:
            key_true = ("true", step)
            key_rand = ("random", step)
            
            if key_true in agg:
                true_means.append(np.mean(agg[key_true]))
                true_sems.append(np.std(agg[key_true]) / np.sqrt(len(agg[key_true])))
            else:
                true_means.append(np.nan)
                true_sems.append(np.nan)
            
            if key_rand in agg:
                random_means.append(np.mean(agg[key_rand]))
                random_sems.append(np.std(agg[key_rand]) / np.sqrt(len(agg[key_rand])))
            else:
                random_means.append(np.nan)
                random_sems.append(np.nan)
        
        x = np.arange(len(rel_steps))
        width = 0.35
        
        ax.bar(x - width/2, true_means, width, yerr=true_sems,
               color=COLORS[model], alpha=0.8, capsize=3,
               label='True Direction', edgecolor='white')
        ax.bar(x + width/2, random_means, width, yerr=random_sems,
               color=COLORS["random"], alpha=0.6, capsize=3,
               label='Random Control', edgecolor='white')
        
        for i, step in enumerate(rel_steps):
            key_true = ("true", step)
            if key_true in agg and len(agg[key_true]) >= 3:
                t_stat, p_val = stats.ttest_1samp(agg[key_true], 0)
                if p_val < 0.05:
                    ax.text(i,
                            true_means[i] + true_sems[i] + 0.1,
                            '*',
                            ha='center',
                            fontsize=14,
                            fontweight='bold')
        
        ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel("Steering Step Relative to Concept Derivation")
        ax.set_ylabel("Normalized Effect Size (Cohen's d)")
        ax.set_title(f"{model.capitalize()} CoT: Normalized Effects",
                     fontweight='semibold')
        ax.set_xticks(x)
        ax.set_xticklabels(rel_steps)
        ax.legend(loc='best', frameon=True)
    
    plt.suptitle("Normalized Steering Effects by Step Position", fontweight='semibold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    plt.savefig("figures/steering_normalized_effects.png", bbox_inches='tight')
    plt.savefig("figures/steering_normalized_effects.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_dose_response()