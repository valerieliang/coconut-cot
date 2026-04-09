# scripts/08_plot_steering.py

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

COLORS = {"verbal": "#4C78A8", "coconut": "#F58518", "random": "#999999"}

def plot_steering_results(summary_path="results/steering/summary.json"):
    with open(summary_path) as f:
        summary = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model in enumerate(["verbal", "coconut"]):
        ax = axes[idx]
        
        # Extract data
        rel_steps_true = []
        effects_true = []
        errors_true = []
        sig_true = []
        
        rel_steps_random = []
        effects_random = []
        errors_random = []
        
        for key, stats in summary.items():
            if stats["model"] != model:
                continue
            
            rel_step = stats["relative_step"]
            
            if stats["direction_type"] == "true" and stats["n"] >= 3:
                rel_steps_true.append(rel_step)
                effects_true.append(stats["mean_effect"])
                errors_true.append(stats["sem"])
                sig_true.append(stats["significant"])
            elif stats["direction_type"] == "random" and stats["n"] >= 3:
                rel_steps_random.append(rel_step)
                effects_random.append(stats["mean_effect"])
                errors_random.append(stats["sem"])
        
        # Sort by relative step
        if rel_steps_true:
            order = np.argsort(rel_steps_true)
            rel_steps_true = np.array(rel_steps_true)[order]
            effects_true = np.array(effects_true)[order]
            errors_true = np.array(errors_true)[order]
            sig_true = np.array(sig_true)[order]
            
            # Plot true direction
            bars = ax.bar(np.array(rel_steps_true) - 0.2, effects_true, width=0.4,
                         yerr=errors_true, color=COLORS[model], alpha=0.8,
                         capsize=3, label='True Direction', edgecolor='white')
            
            # Add significance stars
            for i, (step, effect, err, sig) in enumerate(zip(rel_steps_true, effects_true, errors_true, sig_true)):
                if sig:
                    ax.text(step - 0.2, effect + err + (0.02 * max(effects_true)), 
                           '*', ha='center', fontsize=14, fontweight='bold')
        
        if rel_steps_random:
            order = np.argsort(rel_steps_random)
            rel_steps_random = np.array(rel_steps_random)[order]
            effects_random = np.array(effects_random)[order]
            errors_random = np.array(errors_random)[order]
            
            # Plot random direction
            ax.bar(np.array(rel_steps_random) + 0.2, effects_random, width=0.4,
                  yerr=errors_random, color=COLORS["random"], alpha=0.6,
                  capsize=3, label='Random Direction', edgecolor='white')
        
        ax.axhline(0, linestyle='--', color='gray', alpha=0.5, linewidth=1)
        ax.set_xlabel("Steering Step Relative to Concept Derivation")
        ax.set_ylabel("Effect Size (Δ Logit Difference)")
        ax.set_title(f"{model.capitalize()} CoT", fontweight='semibold')
        ax.legend(loc='best', frameon=True)
        
        # Set x-ticks
        all_steps = sorted(set(list(rel_steps_true) + list(rel_steps_random)))
        ax.set_xticks(all_steps)
    
    plt.suptitle("Causal Steering: Step-Invariant Effects in Coconut", 
                 fontweight='semibold', fontsize=14)
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/steering_relative_step.png", bbox_inches='tight')
    plt.savefig("figures/steering_relative_step.pdf", bbox_inches='tight')
    plt.show()
    
    # Also create a focused Coconut plot (since verbal scale is different)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Filter for Coconut true direction only
    rel_steps = []
    effects = []
    errors = []
    sigs = []
    
    for key, stats in summary.items():
        if stats["model"] == "coconut" and stats["direction_type"] == "true" and stats["n"] >= 3:
            rel_steps.append(stats["relative_step"])
            effects.append(stats["mean_effect"])
            errors.append(stats["sem"])
            sigs.append(stats["significant"])
    
    if rel_steps:
        order = np.argsort(rel_steps)
        rel_steps = np.array(rel_steps)[order]
        effects = np.array(effects)[order]
        errors = np.array(errors)[order]
        sigs = np.array(sigs)[order]
        
        bars = ax.bar(rel_steps, effects, yerr=errors, 
                     color=COLORS["coconut"], alpha=0.8, capsize=5,
                     edgecolor='white', linewidth=1)
        
        for i, (step, effect, err, sig) in enumerate(zip(rel_steps, effects, errors, sigs)):
            if sig:
                ax.text(step, effect + err + 0.01, '*', ha='center', 
                       fontsize=16, fontweight='bold')
        
        ax.axhline(0, linestyle='--', color='gray', alpha=0.5)
        ax.set_xlabel("Steering Step Relative to Concept Derivation")
        ax.set_ylabel("Effect Size (Δ Logit Difference)")
        ax.set_title("Coconut: Causal Evidence for Superposition", fontweight='semibold')
        ax.set_xticks(rel_steps)
    
    plt.tight_layout()
    plt.savefig("figures/coconut_steering_focused.png", bbox_inches='tight')
    plt.savefig("figures/coconut_steering_focused.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    import os
    plot_steering_results()