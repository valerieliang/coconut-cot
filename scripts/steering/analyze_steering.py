# scripts/steering/analyze_steering.py
"""
Analyze steering experiment results with hypothesis testing for CogAI paper.

Generates one figure per hypothesis with all alpha thresholds overlaid,
so a single run replaces the old per-alpha loop.

Usage:
    python analyze_steering.py --results_dir results/steering/steering_suite_...
    python analyze_steering.py --results_dir ... --alphas 1 2 5 10 20 50
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Publication-quality style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

COCONUT_COLOR = '#4C72B0'
VERBAL_COLOR  = '#C44E52'
ACCENT_COLOR  = '#55A868'
NEUTRAL_COLOR = '#8C8C8C'
ALPHA_CMAP    = plt.cm.viridis

plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.size':          12,
    'axes.labelsize':     13,
    'axes.titlesize':     14,
    'axes.titleweight':   'bold',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'grid.linewidth':     0.8,
    'legend.fontsize':    10,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.8',
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze steering results")
    parser.add_argument("--results_dir", type=str,
                        default="results/steering/steering_suite_20260413_010514",
                        help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for figures")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[1, 2, 5, 10, 20, 50],
                        help="Alpha thresholds to overlay (default: 1 2 5 10 20 50)")
    # Keep old --alpha_threshold for backwards-compat: treated as a single alpha
    parser.add_argument("--alpha_threshold", type=float, default=None,
                        help="(Legacy) single alpha threshold -- use --alphas instead")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_steering_results(filepath: str) -> Dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_experiment_results(results_dir: str, verbose: bool = False) -> Dict[str, Dict]:
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
    coconut_results = None
    verbal_results = None
    for exp_name, results in all_results.items():
        if 'comparison_coconut' in exp_name:
            coconut_results = results
        elif 'comparison_verbal' in exp_name:
            verbal_results = results
    return coconut_results, verbal_results


def extract_difficulty_data(all_results: Dict[str, Dict]) -> Dict:
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


# ---------------------------------------------------------------------------
# Per-alpha metric computation
# ---------------------------------------------------------------------------

def compute_flip_metrics_from_results(results: Dict, alpha_threshold: float) -> Dict:
    """
    Return config_name -> metrics for a single alpha threshold.
    Looks inside sweep_results[config][alpha] for per-alpha flip data;
    falls back to top-level answer_flipped when that structure is absent.
    """
    config_data_map: Dict[str, Dict] = {}

    for problem_result in results.get('results', []):
        sweep_results = problem_result.get('sweep_results', {})
        top_flipped   = problem_result.get('answer_flipped', False)

        if not sweep_results:
            # No sweep structure — use top-level flag under a synthetic config
            key = '_global'
            if key not in config_data_map:
                config_data_map[key] = {'flipped': [], 'n': 0}
            config_data_map[key]['n'] += 1
            config_data_map[key]['flipped'].append(top_flipped)
            continue

        for config_name, config_content in sweep_results.items():
            if config_name == 'metadata' or not isinstance(config_content, dict):
                continue
            if config_name not in config_data_map:
                config_data_map[config_name] = {'flipped': [], 'n': 0}
            config_data_map[config_name]['n'] += 1

            # Find the closest stored alpha
            alpha_keys = []
            for k in config_content.keys():
                try:
                    alpha_keys.append(float(k))
                except ValueError:
                    pass

            if alpha_keys:
                closest = min(alpha_keys, key=lambda x: abs(x - alpha_threshold))
                entry = config_content.get(str(closest)) or config_content.get(closest) or {}
                flipped = entry.get('answer_flipped', False)
            else:
                flipped = top_flipped

            config_data_map[config_name]['flipped'].append(flipped)

    metrics = {}
    for config_name, data in config_data_map.items():
        n = data['n']
        fc = sum(data['flipped'])
        metrics[config_name] = {
            'flipped': fc > 0,
            'n_problems': n,
            'flipped_count': fc,
            'flip_rate': fc / n if n > 0 else 0.0,
        }
    return metrics


def flip_rate_across_alphas(results: Dict, alphas: List[float]) -> Dict[float, float]:
    """Return {alpha: mean_flip_rate} over all configs for a results dict."""
    out = {}
    for a in alphas:
        metrics = compute_flip_metrics_from_results(results, a)
        out[a] = float(np.mean([m['flip_rate'] for m in metrics.values()])) if metrics else 0.0
    return out


# ---------------------------------------------------------------------------
# Hypothesis testing
# ---------------------------------------------------------------------------

def test_hypothesis_1(coconut_results: Dict, verbal_results: Dict,
                      alphas: List[float]) -> Dict:
    print("\n" + "="*70)
    print("HYPOTHESIS 1: Bottleneck Hypothesis")
    print("Coconut has higher causal steering efficacy than Verbal CoT")
    print("="*70)

    coconut_by_alpha = flip_rate_across_alphas(coconut_results, alphas)
    verbal_by_alpha  = flip_rate_across_alphas(verbal_results,  alphas)

    mid_alpha = sorted(alphas)[len(alphas) // 2]
    c_metrics = compute_flip_metrics_from_results(coconut_results, mid_alpha)
    v_metrics = compute_flip_metrics_from_results(verbal_results,  mid_alpha)

    c_flipped = sum(m['flipped_count'] for m in c_metrics.values())
    c_total   = sum(m['n_problems']    for m in c_metrics.values())
    v_flipped = sum(m['flipped_count'] for m in v_metrics.values())
    v_total   = sum(m['n_problems']    for m in v_metrics.values())

    contingency = np.array([
        [c_flipped, c_total - c_flipped],
        [v_flipped, v_total - v_flipped],
    ])
    chi2, p_value, dof, _ = chi2_contingency(contingency)

    c_rate = coconut_by_alpha[mid_alpha]
    v_rate = verbal_by_alpha[mid_alpha]
    cohens_h = (2 * math.asin(math.sqrt(max(0, min(1, c_rate)))) -
                2 * math.asin(math.sqrt(max(0, min(1, v_rate)))))

    for a in sorted(alphas):
        print(f"  alpha={a:>5g}  Coconut {coconut_by_alpha[a]:.1%}  "
              f"Verbal {verbal_by_alpha[a]:.1%}")
    print(f"\nChi-square (alpha={mid_alpha:g}): chi2={chi2:.2f}, p={p_value:.4e}")
    print(f"Cohen's h = {cohens_h:.3f}")
    print(f"H1 SUPPORTED: {c_rate > v_rate}")

    return {
        "hypothesis": "H1 - Bottleneck Hypothesis",
        "alphas": sorted(alphas),
        "coconut_by_alpha": {str(a): v for a, v in coconut_by_alpha.items()},
        "verbal_by_alpha":  {str(a): v for a, v in verbal_by_alpha.items()},
        "coconut_flip_rate": float(c_rate),
        "verbal_flip_rate":  float(v_rate),
        "coconut_n": int(c_total), "verbal_n": int(v_total),
        "coconut_flipped": int(c_flipped), "verbal_flipped": int(v_flipped),
        "chi2_statistic": float(chi2), "p_value": float(p_value),
        "cohens_h": float(cohens_h),
        "significant": bool(p_value < 0.05),
        "supported": bool(c_rate > v_rate),
        "mid_alpha": float(mid_alpha),
    }


def test_hypothesis_2(coconut_results: Dict, verbal_results: Dict,
                      alphas: List[float]) -> Dict:
    print("\n" + "="*70)
    print("HYPOTHESIS 2: Separation Hypothesis")
    print("="*70)

    coconut_by_alpha = flip_rate_across_alphas(coconut_results, alphas)
    verbal_by_alpha  = flip_rate_across_alphas(verbal_results,  alphas)
    coconut_incoh = {a: 1 - v for a, v in coconut_by_alpha.items()}
    verbal_incoh  = {a: 1 - v for a, v in verbal_by_alpha.items()}

    mid_alpha = sorted(alphas)[len(alphas) // 2]
    for a in sorted(alphas):
        print(f"  alpha={a:>5g}  Coconut incoh {coconut_incoh[a]:.1%}  "
              f"Verbal incoh {verbal_incoh[a]:.1%}")
    print(f"\nH2 SUPPORTED: {verbal_incoh[mid_alpha] > coconut_incoh[mid_alpha]}")

    return {
        "hypothesis": "H2 - Separation Hypothesis",
        "alphas": sorted(alphas),
        "coconut_incoh_by_alpha": {str(a): v for a, v in coconut_incoh.items()},
        "verbal_incoh_by_alpha":  {str(a): v for a, v in verbal_incoh.items()},
        "coconut_incoherence_rate": float(coconut_incoh[mid_alpha]),
        "verbal_incoherence_rate":  float(verbal_incoh[mid_alpha]),
        "supported": bool(verbal_incoh[mid_alpha] > coconut_incoh[mid_alpha]),
    }


def test_hypothesis_3(difficulty_results: Dict, alphas: List[float]) -> Dict:
    print("\n" + "="*70)
    print("HYPOTHESIS 3: Difficulty Scaling")
    print("="*70)

    hops = sorted(difficulty_results.keys())
    flip_by_alpha: Dict[float, List[float]] = {a: [] for a in alphas}
    for h in hops:
        rates = flip_rate_across_alphas(difficulty_results[h], alphas)
        for a in alphas:
            flip_by_alpha[a].append(rates[a])

    mid_alpha = sorted(alphas)[len(alphas) // 2]
    mid_rates = flip_by_alpha[mid_alpha]
    slope = float(np.polyfit(hops, mid_rates, 1)[0]) if len(hops) >= 2 else 0.0
    corr  = float(np.corrcoef(hops, mid_rates)[0, 1]) if len(hops) >= 2 else 0.0

    print(f"\nHop analysis (alpha={mid_alpha:g}):")
    for h, r in zip(hops, mid_rates):
        print(f"  {h} hops: {r:.1%}")
    print(f"\nTrend: slope={slope:.3f}, r={corr:.3f}")
    print(f"H3 SUPPORTED: {slope > 0}")

    return {
        "hypothesis": "H3 - Difficulty Scaling",
        "alphas": sorted(alphas),
        "hops": [int(h) for h in hops],
        "flip_by_alpha": {str(a): flip_by_alpha[a] for a in alphas},
        "flip_rates": [float(r) for r in mid_rates],
        "slope": slope, "correlation": corr,
        "increasing": slope > 0, "supported": slope > 0,
        "mid_alpha": float(mid_alpha),
    }


# ---------------------------------------------------------------------------
# Figure creators — all alphas overlaid
# ---------------------------------------------------------------------------

def _alpha_colors(alphas):
    n = len(alphas)
    return [ALPHA_CMAP(i / max(n - 1, 1)) for i in range(n)]


def create_figure_h1(h1_results: Dict, output_dir: str):
    """
    Left:  grouped bar chart — Coconut vs Verbal at every alpha.
    Right: dose-response line chart — flip rate vs alpha for each model.
    """
    alphas  = h1_results['alphas']
    c_rates = [h1_results['coconut_by_alpha'][str(a)] for a in alphas]
    v_rates = [h1_results['verbal_by_alpha'][str(a)]  for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Grouped bars
    x, w = np.arange(len(alphas)), 0.35
    axes[0].bar(x - w/2, c_rates, width=w, color=COCONUT_COLOR,
                label='Coconut', edgecolor='white')
    axes[0].bar(x + w/2, v_rates, width=w, color=VERBAL_COLOR,
                label='Verbal CoT', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'a={a:g}' for a in alphas], rotation=30, ha='right')
    axes[0].set_ylabel('Logical Flip Rate')
    axes[0].set_title('H1: Flip Rate by Alpha Threshold')
    axes[0].set_ylim(0, 1.15)
    axes[0].legend()

    # Dose-response lines
    axes[1].plot(alphas, c_rates, 'o-', color=COCONUT_COLOR, linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                 label='Coconut')
    axes[1].plot(alphas, v_rates, 's--', color=VERBAL_COLOR, linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                 label='Verbal CoT')

    if h1_results['significant']:
        p   = h1_results['p_value']
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
        axes[1].annotate(
            f"{sig}  Cohen's h = {h1_results['cohens_h']:.2f}\n"
            f"(at a = {h1_results['mid_alpha']:g})",
            xy=(0.97, 0.97), xycoords='axes fraction', ha='right', va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF3FF',
                      edgecolor='#9BAFD9', alpha=0.9))

    axes[1].set_xlabel('Steering Multiplier (alpha)')
    axes[1].set_ylabel('Logical Flip Rate')
    axes[1].set_title('H1: Dose-Response Curve')
    axes[1].set_ylim(0, 1.15)
    axes[1].legend()

    fig.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'figure_h1_bottleneck.png'))
    plt.close()
    print("  Saved: figure_h1_bottleneck.png")


def create_figure_h2(h2_results: Dict, output_dir: str):
    """
    Left:  incoherence dose-response lines for both models across all alphas.
    Right: mechanistic diagram.
    """
    alphas  = h2_results['alphas']
    c_incoh = [h2_results['coconut_incoh_by_alpha'][str(a)] for a in alphas]
    v_incoh = [h2_results['verbal_incoh_by_alpha'][str(a)]  for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(alphas, c_incoh, 'o-', color=COCONUT_COLOR, linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                 label='Coconut')
    axes[0].plot(alphas, v_incoh, 's--', color=VERBAL_COLOR, linewidth=2.5,
                 markersize=8, markerfacecolor='white', markeredgewidth=2.5,
                 label='Verbal CoT')
    axes[0].set_xlabel('Steering Multiplier (alpha)')
    axes[0].set_ylabel('Incoherence Rate')
    axes[0].set_title('H2: Incoherence vs Alpha')
    axes[0].set_ylim(-0.05, 1.15)
    axes[0].legend()

    # Mechanistic diagram
    ax_diag = axes[1]
    ax_diag.set_xlim(0, 10)
    ax_diag.set_ylim(0, 10)
    ax_diag.axis('off')
    ax_diag.set_title('Mechanistic Interpretation')

    def rounded_box(ax, x, y, w, h, color, label, fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle='round,pad=0.15',
            facecolor=color, edgecolor='white', linewidth=0, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white', zorder=4)

    arrow_kw    = dict(arrowstyle='->', color='#444444', lw=1.5,
                       connectionstyle='arc3,rad=0')
    red_arrow   = dict(arrowstyle='->', color=VERBAL_COLOR, lw=1.5,
                       linestyle='dashed', connectionstyle='arc3,rad=0.25')
    green_arrow = dict(arrowstyle='->', color=ACCENT_COLOR, lw=1.5)

    ax_diag.text(2.5, 9.5, 'Coconut', ha='center', fontsize=10,
                 fontweight='bold', color=COCONUT_COLOR)
    rounded_box(ax_diag, 2.5, 7.5, 3.0, 1.0, COCONUT_COLOR, 'Latent Bottleneck')
    rounded_box(ax_diag, 2.5, 5.5, 3.0, 1.0, '#6A9FCC',     'Steered State')
    rounded_box(ax_diag, 2.5, 3.5, 3.0, 1.0, ACCENT_COLOR,  'Answer Flip (Y)')
    ax_diag.annotate('', xy=(2.5, 6.05), xytext=(2.5, 7.0), arrowprops=arrow_kw)
    ax_diag.annotate('', xy=(2.5, 4.05), xytext=(2.5, 5.0), arrowprops=green_arrow)

    ax_diag.text(7.5, 9.5, 'Verbal CoT', ha='center', fontsize=10,
                 fontweight='bold', color=VERBAL_COLOR)
    rounded_box(ax_diag, 6.5, 7.5, 2.2, 1.0, VERBAL_COLOR, 'Verbal\nTrace')
    rounded_box(ax_diag, 8.5, 7.5, 2.2, 1.0, '#E07070',    'Answer\nPath')
    rounded_box(ax_diag, 7.5, 5.2, 3.0, 1.0, '#C98060',    'Mismatch (N)')
    ax_diag.annotate('', xy=(7.2, 5.72), xytext=(6.5, 7.0), arrowprops=red_arrow)
    ax_diag.annotate('', xy=(7.8, 5.72), xytext=(8.5, 7.0), arrowprops=arrow_kw)

    fig.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'figure_h2_separation.png'))
    plt.close()
    print("  Saved: figure_h2_separation.png")


def create_figure_h3(h3_results: Dict, output_dir: str):
    """
    Left:  flip rate vs hop count — one line per alpha.
    Right: flip rate vs alpha   — one line per hop count.
    """
    alphas = h3_results['alphas']
    hops   = h3_results['hops']
    flip_by_alpha = {float(a): h3_results['flip_by_alpha'][str(a)] for a in alphas}

    colors_alpha = _alpha_colors(alphas)
    colors_hop   = [ALPHA_CMAP(i / max(len(hops) - 1, 1)) for i in range(len(hops))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Flip rate vs hops, one line per alpha
    for alpha, col in zip(alphas, colors_alpha):
        rates = flip_by_alpha[alpha]
        axes[0].plot(hops, rates, 'o-', color=col, linewidth=2,
                     markersize=7, markerfacecolor='white', markeredgewidth=2,
                     label=f'a={alpha:g}')
    axes[0].set_xlabel('Number of Reasoning Hops')
    axes[0].set_ylabel('Logical Flip Rate')
    axes[0].set_title('H3: Flip Rate vs Difficulty')
    axes[0].set_xticks(hops)
    axes[0].set_ylim(0, 1.15)
    axes[0].legend(fontsize=9, ncol=2)

    # Flip rate vs alpha, one line per hop count
    for hop, col in zip(hops, colors_hop):
        hop_idx = hops.index(hop)
        rates_for_hop = [flip_by_alpha[a][hop_idx] for a in alphas]
        axes[1].plot(alphas, rates_for_hop, 'o-', color=col, linewidth=2,
                     markersize=7, markerfacecolor='white', markeredgewidth=2,
                     label=f'{hop} hops')
    axes[1].set_xlabel('Steering Multiplier (alpha)')
    axes[1].set_ylabel('Logical Flip Rate')
    axes[1].set_title('H3: Dose-Response by Difficulty')
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(fontsize=9)

    fig.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, 'figure_h3_difficulty.png'))
    plt.close()
    print("  Saved: figure_h3_difficulty.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_paper_figures(results_dir: str, output_dir: str,
                            alphas: List[float]):
    print("\n" + "="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)

    print("\nLoading experiment data...")
    all_results = find_experiment_results(results_dir, verbose=False)
    print(f"Found {len(all_results)} experiment result files")

    coconut_results, verbal_results = extract_comparison_data(all_results)
    difficulty_results = extract_difficulty_data(all_results)

    if coconut_results is None or verbal_results is None:
        print("ERROR: Could not find comparison experiment results")
        return

    h1 = test_hypothesis_1(coconut_results, verbal_results, alphas)
    h2 = test_hypothesis_2(coconut_results, verbal_results, alphas)
    h3 = (test_hypothesis_3(difficulty_results, alphas) if difficulty_results
          else {"supported": False, "hops": [], "flip_rates": [],
                "slope": 0, "correlation": 0, "alphas": alphas,
                "flip_by_alpha": {str(a): [] for a in alphas},
                "mid_alpha": sorted(alphas)[len(alphas)//2]})

    print("\nCreating figures...")
    os.makedirs(output_dir, exist_ok=True)
    create_figure_h1(h1, output_dir)
    create_figure_h2(h2, output_dir)
    if difficulty_results:
        create_figure_h3(h3, output_dir)

    def _serial(obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, dict):  return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_serial(v) for v in obj]
        return obj

    with open(os.path.join(output_dir, "hypothesis_testing_results.json"), "w") as f:
        json.dump(_serial({"h1": h1, "h2": h2, "h3": h3, "alphas": alphas}), f, indent=2)

    mid = sorted(alphas)[len(alphas) // 2]
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("="*70)
    print(f"H1 (Bottleneck): {h1['supported']} -- "
          f"Coconut {h1['coconut_flip_rate']:.1%} vs Verbal {h1['verbal_flip_rate']:.1%}"
          f"  (alpha={mid:g})")
    print(f"H2 (Separation): {h2['supported']} -- "
          f"Verbal incoh {h2['verbal_incoherence_rate']:.1%} vs "
          f"Coconut {h2['coconut_incoherence_rate']:.1%}  (alpha={mid:g})")
    print(f"H3 (Difficulty): {h3['supported']} -- "
          f"Slope {h3['slope']:.3f}, r={h3['correlation']:.3f}  (alpha={mid:g})")
    print(f"\nAll figures saved to: {output_dir}")


def main():
    args = parse_args()

    # Legacy: if only --alpha_threshold was given, treat it as a single alpha
    if args.alpha_threshold is not None and args.alphas == [1, 2, 5, 10, 20, 50]:
        alphas = [args.alpha_threshold]
    else:
        alphas = sorted(set(args.alphas))

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, "figures")

    print(f"Results directory: {args.results_dir}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Alpha thresholds:  {alphas}")

    generate_paper_figures(args.results_dir, args.output_dir, alphas)


if __name__ == "__main__":
    main()