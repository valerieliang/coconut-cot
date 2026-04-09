
import json
import numpy as np
from scipy import stats
from collections import defaultdict

def final_analysis(results_path="results/steering/enhanced/full_results.json"):
    with open(results_path) as f:
        results = json.load(f)

    print("="*80)
    print("FINAL STEERING ANALYSIS")
    print("="*80)

    for model in ["verbal", "coconut"]:
        print(f"\n{model.upper()} CoT")
        print("-"*60)

        agg = defaultdict(lambda: {"true": [], "random": []})

        for res in results:
            if res["model"] != model:
                continue

            rel_step = res["relative_step"]
            dir_type = res["direction_type"]
            effects = res["result"]["effects"]

            if "5.0" in effects:
                raw_change = effects["5.0"].get("raw_change", 0)
                if dir_type in ["true", "random"]:
                    agg[rel_step][dir_type].append(raw_change)

        print(f"{'Rel Step':<10} {'True Δ':<15} {'Random Δ':<15} {'Ratio':<10} {'p (true≠random)':<15} {'N':<5}")
        print("-"*60)

        for rel_step in sorted(agg.keys()):
            true_vals = agg[rel_step]["true"]
            rand_vals = agg[rel_step]["random"]

            if len(true_vals) >= 3 and len(rand_vals) >= 3:
                true_mean = np.mean(true_vals)
                rand_mean = np.mean(rand_vals)
                ratio = true_mean / rand_mean if abs(rand_mean) > 1e-6 else np.inf

                t_stat, p_val = stats.ttest_ind(true_vals, rand_vals, equal_var=False)

                sig = "**" if p_val < 0.05 else ""
                print(f"{rel_step:<10} {true_mean:+.3f} +/- {np.std(true_vals)/np.sqrt(len(true_vals)):.3f}   "
                      f"{rand_mean:+.3f} +/- {np.std(rand_vals)/np.sqrt(len(rand_vals)):.3f}   "
                      f"{ratio:.2f}x     p={p_val:.4f}     {len(true_vals)} {sig}")

    print("\n" + "="*80)
    print("PAPER-READY STATISTICS")
    print("="*80)

    coconut_true_all = []
    coconut_rand_all = []
    verbal_true_all = []
    verbal_rand_all = []

    for res in results:
        model = res["model"]
        dir_type = res["direction_type"]
        if "5.0" in res["result"]["effects"]:
            effect = res["result"]["effects"]["5.0"].get("raw_change", 0)
            if dir_type == "true":
                if model == "coconut":
                    coconut_true_all.append(effect)
                else:
                    verbal_true_all.append(effect)
            elif dir_type == "random":
                if model == "coconut":
                    coconut_rand_all.append(effect)
                else:
                    verbal_rand_all.append(effect)

    print("\nCOCONUT:")
    print(f"  True direction:  {np.mean(coconut_true_all):.3f} +/- {np.std(coconut_true_all)/np.sqrt(len(coconut_true_all)):.3f} (n={len(coconut_true_all)})")
    print(f"  Random direction: {np.mean(coconut_rand_all):.3f} +/- {np.std(coconut_rand_all)/np.sqrt(len(coconut_rand_all)):.3f} (n={len(coconut_rand_all)})")
    t_coco, p_coco = stats.ttest_ind(coconut_true_all, coconut_rand_all, equal_var=False)
    print(f"  Difference: {np.mean(coconut_true_all) - np.mean(coconut_rand_all):.3f}, p = {p_coco:.4f}")

    pooled_std = np.sqrt((np.var(coconut_true_all) + np.var(coconut_rand_all)) / 2)
    cohens_d = (np.mean(coconut_true_all) - np.mean(coconut_rand_all)) / pooled_std
    print(f"  Cohen's d = {cohens_d:.3f}")

    print("\nVERBAL:")
    print(f"  True direction:  {np.mean(verbal_true_all):.1f} +/- {np.std(verbal_true_all)/np.sqrt(len(verbal_true_all)):.1f} (n={len(verbal_true_all)})")
    print(f"  Random direction: {np.mean(verbal_rand_all):.1f} +/- {np.std(verbal_rand_all)/np.sqrt(len(verbal_rand_all)):.1f} (n={len(verbal_rand_all)})")
    t_verb, p_verb = stats.ttest_ind(verbal_true_all, verbal_rand_all, equal_var=False)
    print(f"  Difference: {np.mean(verbal_true_all) - np.mean(verbal_rand_all):.1f}, p = {p_verb:.6f}")
    print(f"  Signal-to-noise ratio: {np.mean(verbal_true_all)/np.mean(verbal_rand_all):.1f}x")

    print("\n" + "="*80)
    print("STEP-INVARIANCE TEST (COCONUT)")
    print("="*80)

    coconut_by_step = defaultdict(list)
    for res in results:
        if res["model"] == "coconut" and res["direction_type"] == "true":
            if "5.0" in res["result"]["effects"]:
                coconut_by_step[res["relative_step"]].append(
                    res["result"]["effects"]["5.0"]["raw_change"]
                )

    steps = sorted([s for s in coconut_by_step.keys() if len(coconut_by_step[s]) >= 5])
    print(f"Steps with N>=5: {steps}")

    if len(steps) >= 2:
        groups = [coconut_by_step[s] for s in steps]
        f_stat, p_anova = stats.f_oneway(*groups)
        print(f"One-way ANOVA across steps: F = {f_stat:.3f}, p = {p_anova:.4f}")

        if p_anova > 0.05:
            print("-> No significant difference across steps (supports step-invariance)")
        else:
            print("-> Significant differences across steps")

        print("\nMean effects by step:")
        for step in steps:
            vals = coconut_by_step[step]
            print(f"  Step {step:2d}: {np.mean(vals):.3f} +/- {np.std(vals)/np.sqrt(len(vals)):.3f} (n={len(vals)})")

if __name__ == "__main__":
    final_analysis()