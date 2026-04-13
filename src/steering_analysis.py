# src/steering_analysis.py
"""
Enhanced steering analysis with normalized effect sizes, dose-response curves,
and comparative metrics between Coconut and Verbal CoT.
"""

import json
import os
import numpy as np
import torch
from scipy import stats
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Publication-quality style
# ---------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

COCONUT_COLOR = '#4C72B0'
VERBAL_COLOR  = '#C44E52'
ACCENT_COLOR  = '#55A868'
NEUTRAL_COLOR = '#8C8C8C'

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
    'legend.fontsize':    11,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.8',
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'figure.dpi':         150,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
})


# ---------------------------------------------------------------------------
# Key-coercion helpers
# ---------------------------------------------------------------------------
# BUG FIX: save_steering_results serialises all dict keys through str(), so
# float alpha keys (0.0, 1.0 ...) become strings ("0.0", "1.0" ...) in JSON.
# Every analysis function that looks up result[0.0] or filters
# `isinstance(a, float)` breaks silently on loaded data.
# The helpers below normalise keys so the rest of the code is transparent to
# whether results came from memory or disk.

def _alpha_keys(results: Dict) -> List[float]:
    """Return sorted float alpha keys from a results dict (handles str keys)."""
    alphas = []
    for k in results.keys():
        if isinstance(k, (int, float)):
            alphas.append(float(k))
        elif isinstance(k, str):
            try:
                alphas.append(float(k))
            except ValueError:
                pass
    return sorted(alphas)


def _get_alpha(results: Dict, alpha: float):
    """Get a per-alpha result, trying both float and string key."""
    v = results.get(alpha)
    if v is None:
        v = results.get(str(alpha))
    return v


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_normalized_effect(results_dict: Dict, baseline_logit_diff: float,
                              token_logit_std: Optional[float] = None,
                              n_samples: int = 10) -> Dict:
    """
    Compute normalized effect size.

    Options:
    1. Percent change: (steered - baseline) / |baseline|
    2. Std-normalized: (steered - baseline) / baseline_std
    """
    effects = {}

    for alpha in _alpha_keys(results_dict):
        if alpha == 0.0:
            continue
        res = _get_alpha(results_dict, alpha)
        if res is None:
            continue

        raw_change = res.get("change", 0)

        if abs(baseline_logit_diff) > 1e-6:
            percent_change = raw_change / abs(baseline_logit_diff)
        else:
            percent_change = raw_change

        if token_logit_std is not None and token_logit_std > 0:
            std_normalized = raw_change / token_logit_std
        else:
            std_normalized = raw_change

        effects[alpha] = {
            "raw_change": raw_change,
            "percent_change": percent_change,
            "std_normalized": std_normalized,
            "logit_diff": res.get("logit_diff", 0),
            "prob_a": res.get("prob_a", 0.5),
            "answer_flipped": res.get("answer_flipped", False),
            "effect_size": res.get("effect_size", 0.0),
        }

    return effects


def compute_flip_metrics(sweep_results: Dict, alpha_threshold: float = 5.0) -> Dict:
    """
    Compute logical flip rates and effect sizes across a sweep.
    """
    metrics = {}

    for config_name, results in sweep_results.items():
        if results is None:
            continue

        alphas = _alpha_keys(results)
        if not alphas:
            continue

        closest_alpha = min(alphas, key=lambda x: abs(x - alpha_threshold))
        result_at_alpha = _get_alpha(results, closest_alpha)
        baseline = _get_alpha(results, 0.0)

        if baseline and result_at_alpha:
            metrics[config_name] = {
                'flipped': result_at_alpha.get('answer_flipped', False),
                'flipped_direction': result_at_alpha.get('flipped_direction'),
                'effect_size': result_at_alpha.get('effect_size', 0),
                'prob_shift': result_at_alpha.get('prob_a', 0.5) - baseline.get('prob_a', 0.5),
                'logit_shift': result_at_alpha.get('change', 0),
                'baseline_prob_a': baseline.get('prob_a', 0.5),
                'steered_prob_a': result_at_alpha.get('prob_a', 0.5),
                'baseline_logit_diff': baseline.get('logit_diff', 0),
                'steered_logit_diff': result_at_alpha.get('logit_diff', 0),
                'was_steered': result_at_alpha.get('was_steered', False),
            }

    return metrics


def compute_dose_response_curve(
    results: Dict,
    normalize: bool = True
) -> Dict:
    """Extract dose-response curve from steering results."""
    alphas = []
    changes = []
    probs = []
    logit_diffs = []
    flipped = []

    baseline_logit_diff = None
    baseline_prob = None

    for alpha in _alpha_keys(results):
        res = _get_alpha(results, alpha)
        if res is None:
            continue

        if alpha == 0.0:
            baseline_logit_diff = res.get('logit_diff', 0)
            baseline_prob = res.get('prob_a', 0.5)

        alphas.append(alpha)
        changes.append(res.get('change', 0))
        probs.append(res.get('prob_a', 0.5))
        logit_diffs.append(res.get('logit_diff', 0))
        flipped.append(res.get('answer_flipped', False))

    if normalize and baseline_logit_diff and abs(baseline_logit_diff) > 1e-6:
        normalized_changes = [c / abs(baseline_logit_diff) for c in changes]
    else:
        normalized_changes = changes

    return {
        'alphas': alphas,
        'changes': changes,
        'normalized_changes': normalized_changes,
        'probs': probs,
        'logit_diffs': logit_diffs,
        'flipped': flipped,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_prob': baseline_prob,
    }


def compare_coconut_vs_verbal(
    coconut_sweep: Dict,
    verbal_sweep: Dict,
    alpha_threshold: float = 5.0
) -> Dict:
    """Compare steering efficacy between Coconut and Verbal CoT."""
    coconut_metrics = compute_flip_metrics(coconut_sweep, alpha_threshold)
    verbal_metrics = compute_flip_metrics(verbal_sweep, alpha_threshold)

    comparison = {
        'coconut': {
            'flip_rate': sum(1 for m in coconut_metrics.values() if m['flipped']) / len(coconut_metrics) if coconut_metrics else 0,
            'mean_effect_size': np.mean([m['effect_size'] for m in coconut_metrics.values()]) if coconut_metrics else 0,
            'mean_prob_shift': np.mean([abs(m['prob_shift']) for m in coconut_metrics.values()]) if coconut_metrics else 0,
        },
        'verbal': {
            'flip_rate': sum(1 for m in verbal_metrics.values() if m['flipped']) / len(verbal_metrics) if verbal_metrics else 0,
            'mean_effect_size': np.mean([m['effect_size'] for m in verbal_metrics.values()]) if verbal_metrics else 0,
            'mean_prob_shift': np.mean([abs(m['prob_shift']) for m in verbal_metrics.values()]) if verbal_metrics else 0,
        }
    }

    if comparison['verbal']['flip_rate'] > 0:
        comparison['flip_rate_ratio'] = comparison['coconut']['flip_rate'] / comparison['verbal']['flip_rate']
    else:
        comparison['flip_rate_ratio'] = float('inf') if comparison['coconut']['flip_rate'] > 0 else 1.0

    if comparison['verbal']['mean_effect_size'] != 0:
        comparison['effect_size_ratio'] = comparison['coconut']['mean_effect_size'] / comparison['verbal']['mean_effect_size']
    else:
        comparison['effect_size_ratio'] = float('inf')

    comparison['bottleneck_supported'] = comparison['coconut']['flip_rate'] > comparison['verbal']['flip_rate']
    comparison['separation_supported'] = comparison['effect_size_ratio'] > 1.5

    return comparison


def visualize_steering_sweep(
    sweep_results: Dict,
    save_path: Optional[str] = None,
    title_prefix: str = "",
    alpha: float = 5.0
) -> None:
    """
    Create visualization of steering sweep results.

    Shows:
        - Heatmap of effect sizes (step x layer) if both dimensions present
        - Dose-response curves for each configuration
        - Flip rate by intervention point
    """
    configs = list(sweep_results.keys())

    if not configs:
        print("No results to visualize")
        return

    has_step = any('step' in c and 'layer' not in c for c in configs)
    has_layer = any('layer' in c for c in configs)
    has_both = any('step' in c and 'layer' in c for c in configs)

    if has_both:
        steps = set()
        layers = set()
        effect_matrix = {}

        for config_name, results in sweep_results.items():
            res_at_alpha = _get_alpha(results, alpha) if results else None
            if res_at_alpha:
                parts = config_name.split('_')
                step = int(parts[0].replace('step', ''))
                layer = int(parts[1].replace('layer', ''))
                steps.add(step)
                layers.add(layer)
                effect_matrix[(step, layer)] = res_at_alpha.get('effect_size', 0)

        steps = sorted(steps)
        layers = sorted(layers)

        matrix = np.zeros((len(steps), len(layers)))
        for i, step in enumerate(steps):
            for j, layer in enumerate(layers):
                matrix[i, j] = effect_matrix.get((step, layer), 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        vmax = max(abs(matrix).max(), 1e-6)
        im = axes[0].imshow(matrix, cmap='RdBu_r', aspect='auto',
                            vmin=-vmax, vmax=vmax)
        axes[0].set_xticks(range(len(layers)))
        axes[0].set_xticklabels(layers)
        axes[0].set_yticks(range(len(steps)))
        axes[0].set_yticklabels(steps)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Step')
        axes[0].set_title(f'{title_prefix}Effect Size — Step × Layer  (α={alpha})')
        cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=10)

        flip_rates = []
        for step in steps:
            step_flips = []
            for layer in layers:
                if (step, layer) in effect_matrix:
                    res_at_alpha = _get_alpha(
                        sweep_results.get(f'step{step}_layer{layer}', {}), alpha
                    ) or {}
                    step_flips.append(1.0 if res_at_alpha.get('answer_flipped', False) else 0.0)
            flip_rates.append(np.mean(step_flips) if step_flips else 0)

        axes[1].bar(range(len(steps)), flip_rates, color=COCONUT_COLOR,
                    width=0.6, edgecolor='white')
        axes[1].set_xticks(range(len(steps)))
        axes[1].set_xticklabels(steps)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Flip Rate')
        axes[1].set_title(f'{title_prefix}Logical Flip Rate by Step')
        axes[1].set_ylim(0, 1.05)

        fig.tight_layout(pad=2.0)

    elif has_step:
        steps = []
        effect_sizes = []
        flip_rates = []
        prob_shifts = []

        for config_name, results in sweep_results.items():
            res_at_alpha = _get_alpha(results, alpha) if results else None
            baseline = _get_alpha(results, 0.0) if results else None
            if res_at_alpha:
                step = int(config_name.split('_')[1])
                steps.append(step)
                effect_sizes.append(res_at_alpha.get('effect_size', 0))
                flip_rates.append(1.0 if res_at_alpha.get('answer_flipped', False) else 0.0)
                if baseline:
                    prob_shifts.append(res_at_alpha['prob_a'] - baseline['prob_a'])
                else:
                    prob_shifts.append(0)

        sorted_indices = np.argsort(steps)
        steps = np.array(steps)[sorted_indices]
        effect_sizes = np.array(effect_sizes)[sorted_indices]
        flip_rates = np.array(flip_rates)[sorted_indices]
        prob_shifts = np.array(prob_shifts)[sorted_indices]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        bar_colors_es = [VERBAL_COLOR if es < 0 else COCONUT_COLOR
                         for es in effect_sizes]
        axes[0].bar(steps, effect_sizes, color=bar_colors_es,
                    edgecolor='white', linewidth=0)
        axes[0].axhline(y=0, color='#333333', linewidth=1.2)
        axes[0].set_xlabel('Reasoning Step')
        axes[0].set_ylabel('Effect Size')
        axes[0].set_title(f'{title_prefix}Steering Effect by Step  (α={alpha})')

        axes[1].bar(steps, flip_rates, color=ACCENT_COLOR,
                    alpha=0.85, edgecolor='white')
        axes[1].set_xlabel('Reasoning Step')
        axes[1].set_ylabel('Flip Rate')
        axes[1].set_title(f'{title_prefix}Logical Flip Rate by Step')
        axes[1].set_ylim(0, 1.05)

        bar_colors_ps = [VERBAL_COLOR if ps < 0 else COCONUT_COLOR
                         for ps in prob_shifts]
        axes[2].bar(steps, prob_shifts, color=bar_colors_ps,
                    edgecolor='white', linewidth=0)
        axes[2].axhline(y=0, color='#333333', linewidth=1.2)
        axes[2].set_xlabel('Reasoning Step')
        axes[2].set_ylabel('Probability Shift')
        axes[2].set_title(f'{title_prefix}Probability Shift by Step')

        fig.tight_layout(pad=2.0)

    elif has_layer:
        layers = []
        effect_sizes = []
        flip_rates = []

        for config_name, results in sweep_results.items():
            res_at_alpha = _get_alpha(results, alpha) if results else None
            if res_at_alpha:
                layer = int(config_name.split('_')[1])
                layers.append(layer)
                effect_sizes.append(res_at_alpha.get('effect_size', 0))
                flip_rates.append(1.0 if res_at_alpha.get('answer_flipped', False) else 0.0)

        sorted_indices = np.argsort(layers)
        layers = np.array(layers)[sorted_indices]
        effect_sizes = np.array(effect_sizes)[sorted_indices]
        flip_rates = np.array(flip_rates)[sorted_indices]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        bar_colors_es = [VERBAL_COLOR if es < 0 else COCONUT_COLOR
                         for es in effect_sizes]
        axes[0].bar(layers, effect_sizes, color=bar_colors_es,
                    edgecolor='white', linewidth=0)
        axes[0].axhline(y=0, color='#333333', linewidth=1.2)
        axes[0].set_xlabel('Transformer Layer')
        axes[0].set_ylabel('Effect Size')
        axes[0].set_title(f'{title_prefix}Steering Effect by Layer  (α={alpha})')

        axes[1].bar(layers, flip_rates, color=ACCENT_COLOR,
                    alpha=0.85, edgecolor='white')
        axes[1].set_xlabel('Transformer Layer')
        axes[1].set_ylabel('Flip Rate')
        axes[1].set_title(f'{title_prefix}Logical Flip Rate by Layer')
        axes[1].set_ylim(0, 1.05)

        fig.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_dose_response_comparison(
    coconut_results: Dict,
    verbal_results: Dict,
    save_path: Optional[str] = None
) -> None:
    """Plot dose-response curves comparing Coconut and Verbal CoT."""
    coconut_curve = compute_dose_response_curve(coconut_results)
    verbal_curve = compute_dose_response_curve(verbal_results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    shared_line_kw = dict(linewidth=2.5, markersize=8)

    # Panel 0 — raw effect
    axes[0].plot(coconut_curve['alphas'], coconut_curve['changes'],
                 'o-', color=COCONUT_COLOR, label='Coconut', **shared_line_kw)
    axes[0].plot(verbal_curve['alphas'], verbal_curve['changes'],
                 's--', color=VERBAL_COLOR, label='Verbal CoT', **shared_line_kw)
    axes[0].axhline(y=0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle=':')
    axes[0].set_xlabel('Steering Multiplier (α)')
    axes[0].set_ylabel('Change in Logit Difference')
    axes[0].set_title('Raw Effect Size')
    axes[0].legend()

    # Panel 1 — normalized effect
    axes[1].plot(coconut_curve['alphas'], coconut_curve['normalized_changes'],
                 'o-', color=COCONUT_COLOR, label='Coconut', **shared_line_kw)
    axes[1].plot(verbal_curve['alphas'], verbal_curve['normalized_changes'],
                 's--', color=VERBAL_COLOR, label='Verbal CoT', **shared_line_kw)
    axes[1].axhline(y=0, color=NEUTRAL_COLOR, linewidth=1.0, linestyle=':')
    axes[1].set_xlabel('Steering Multiplier (α)')
    axes[1].set_ylabel('Normalized Change')
    axes[1].set_title('Normalized Effect Size')
    axes[1].legend()

    # Panel 2 — output probability
    axes[2].plot(coconut_curve['alphas'], coconut_curve['probs'],
                 'o-', color=COCONUT_COLOR, label='Coconut', **shared_line_kw)
    axes[2].plot(verbal_curve['alphas'], verbal_curve['probs'],
                 's--', color=VERBAL_COLOR, label='Verbal CoT', **shared_line_kw)
    axes[2].axhline(y=0.5, color=NEUTRAL_COLOR, linewidth=1.0,
                    linestyle=':', label='Chance')
    axes[2].set_xlabel('Steering Multiplier (α)')
    axes[2].set_ylabel('P(concept_a)')
    axes[2].set_title('Output Probability')
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    fig.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")

    plt.show()


def save_steering_results(
    results: Dict,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """Save steering results to JSON, handling numpy arrays and tensors."""
    # BUG FIX: guard against dirname being empty string (bare filename)
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable = convert_to_serializable(results)

    if metadata:
        serializable['_metadata'] = metadata

    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Results saved to {filepath}")


def load_steering_results(filepath: str) -> Dict:
    """
    Load steering results from JSON.

    BUG FIX: JSON forces all dict keys to strings, so float alpha keys (0.0,
    1.0 ...) become "0.0", "1.0" ... after a save/load round-trip.  The
    _alpha_keys() and _get_alpha() helpers throughout this module handle that
    transparently, so no key conversion is done here to keep the structure
    intact.
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


# ---------------------------------------------------------------------------
# Backward-compatibility: variance-estimation wrapper
# ---------------------------------------------------------------------------

def run_steering_with_variance_estimation(
    model, tokenizer, problem_row, concept_a, concept_b,
    latent_step_to_steer=0, alphas=None, device="cuda", latent_id=None,
    start_id=None, end_id=None, steering_vector=None, n_baseline_samples=10,
    model_type="coconut"
):
    """Original function kept for backward compatibility."""
    from src.coconut_steering import run_steering as run_coconut_steering
    from src.verbal_steering import run_verbal_steering

    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    baseline_logit_diffs = []
    baseline_probs = []

    for _ in range(n_baseline_samples):
        if model_type == "coconut":
            result = run_coconut_steering(
                model, tokenizer, problem_row, concept_a, concept_b,
                latent_step_to_steer, alphas=[0.0], device=device,
                latent_id=latent_id, start_id=start_id, end_id=end_id,
                steering_vector=steering_vector,
            )
        else:
            result = run_verbal_steering(
                model, tokenizer, problem_row, concept_a, concept_b,
                step_to_steer=latent_step_to_steer, alphas=[0.0],
                device=device, steering_vector=steering_vector,
            )

        if result:
            r0 = _get_alpha(result, 0.0)
            if r0:
                baseline_logit_diffs.append(r0["logit_diff"])
                baseline_probs.append(r0["prob_a"])

    if not baseline_logit_diffs:
        return None

    baseline_mean = np.mean(baseline_logit_diffs)
    baseline_std = np.std(baseline_logit_diffs)
    baseline_prob_mean = np.mean(baseline_probs)
    baseline_prob_std = np.std(baseline_probs)

    if model_type == "coconut":
        results = run_coconut_steering(
            model, tokenizer, problem_row, concept_a, concept_b,
            latent_step_to_steer, alphas=alphas, device=device,
            latent_id=latent_id, start_id=start_id, end_id=end_id,
            steering_vector=steering_vector,
        )
    else:
        results = run_verbal_steering(
            model, tokenizer, problem_row, concept_a, concept_b,
            step_to_steer=latent_step_to_steer, alphas=alphas,
            device=device, steering_vector=steering_vector,
        )

    if not results:
        return None

    normalized = compute_normalized_effect(
        results, baseline_mean, token_logit_std=baseline_std
    )

    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "baseline_prob_mean": baseline_prob_mean,
        "baseline_prob_std": baseline_prob_std,
        "effects": normalized,
        "n_baseline_samples": n_baseline_samples,
    }