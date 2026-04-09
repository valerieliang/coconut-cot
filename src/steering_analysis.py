# src/steering_analysis.py
"""
Enhanced steering analysis with normalized effect sizes and dose-response curves.
"""

import json
import numpy as np
import torch
from scipy import stats
from collections import defaultdict
from tqdm import tqdm


def compute_normalized_effect(results_dict, baseline_logit_diff, 
                               token_logit_std=None, n_samples=10):
    """
    Compute normalized effect size.
    
    Options:
    1. Cohen's d: (steered - baseline) / pooled_std
    2. Percent change: (steered - baseline) / |baseline|
    3. Std-normalized: (steered - baseline) / baseline_std
    """
    effects = {}
    
    for alpha, res in results_dict.items():
        if alpha == 0.0:
            continue
            
        raw_change = res.get("change", 0)
        
        # Option 1: Normalize by baseline magnitude
        if abs(baseline_logit_diff) > 1e-6:
            percent_change = raw_change / abs(baseline_logit_diff)
        else:
            percent_change = raw_change
        
        # Option 2: Normalize by token-level variance (if available)
        if token_logit_std is not None and token_logit_std > 0:
            std_normalized = raw_change / token_logit_std
        else:
            std_normalized = raw_change
        
        effects[alpha] = {
            "raw_change": raw_change,
            "percent_change": percent_change,
            "std_normalized": std_normalized,
            "logit_diff": res["logit_diff"],
            "prob_a": res["prob_a"],
        }
    
    return effects


def run_steering_with_variance_estimation(
    model, tokenizer, problem_row, concept_a, concept_b,
    latent_step_to_steer=0, alphas=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    device="cuda", latent_id=None, start_id=None, end_id=None,
    steering_vector=None, n_baseline_samples=10, model_type="coconut"
):
    """
    Run steering with multiple baseline samples to estimate variance.
    Returns normalized effects with confidence.
    """
    
    # First, estimate baseline variance by running unsteered forward pass multiple times
    baseline_logit_diffs = []
    baseline_probs = []
    
    for _ in range(n_baseline_samples):
        if model_type == "coconut":
            result = run_coconut_steering_single(
                model, tokenizer, problem_row, concept_a, concept_b,
                latent_step_to_steer, alphas=[0.0], device=device,
                latent_id=latent_id, start_id=start_id, end_id=end_id,
                steering_vector=steering_vector,
            )
        else:
            result = run_verbal_steering_single(
                model, tokenizer, problem_row, concept_a, concept_b,
                step_to_steer=latent_step_to_steer, alphas=[0.0],
                device=device, steering_vector=steering_vector,
            )
        
        if result and 0.0 in result:
            baseline_logit_diffs.append(result[0.0]["logit_diff"])
            baseline_probs.append(result[0.0]["prob_a"])
    
    if not baseline_logit_diffs:
        return None
    
    baseline_mean = np.mean(baseline_logit_diffs)
    baseline_std = np.std(baseline_logit_diffs)
    baseline_prob_mean = np.mean(baseline_probs)
    baseline_prob_std = np.std(baseline_probs)
    
    # Now run steering at each alpha (single sample is fine, we have baseline variance)
    if model_type == "coconut":
        results = run_coconut_steering_single(
            model, tokenizer, problem_row, concept_a, concept_b,
            latent_step_to_steer, alphas=alphas, device=device,
            latent_id=latent_id, start_id=start_id, end_id=end_id,
            steering_vector=steering_vector,
        )
    else:
        results = run_verbal_steering_single(
            model, tokenizer, problem_row, concept_a, concept_b,
            step_to_steer=latent_step_to_steer, alphas=alphas,
            device=device, steering_vector=steering_vector,
        )
    
    if not results:
        return None
    
    # Normalize effects
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


def run_coconut_steering_single(
    model, tokenizer, problem_row, concept_a, concept_b,
    latent_step_to_steer=0, alphas=[0.0], device="cuda",
    latent_id=None, start_id=None, end_id=None, steering_vector=None,
):
    """Single-run version for variance estimation."""
    from src.coconut_steering import run_steering
    return run_steering(
        model, tokenizer, problem_row, concept_a, concept_b,
        latent_step_to_steer, alphas, device,
        latent_id, start_id, end_id, steering_vector
    )


def run_verbal_steering_single(
    model, tokenizer, problem_row, concept_a, concept_b,
    step_to_steer=0, alphas=[0.0], device="cuda", steering_vector=None,
):
    """Single-run version for variance estimation."""
    from src.verbal_steering import run_verbal_steering
    return run_verbal_steering(
        model, tokenizer, problem_row, concept_a, concept_b,
        step_to_steer, alphas, device, steering_vector=steering_vector
    )

def get_orthogonal_direction(embed_matrix, tokenizer, concept_a, concept_b, concept_c):
    """
    Get a direction that is orthogonal to A-B but still semantically meaningful.
    Concept C should be from a different ontological family.
    """
    # Get A-B direction
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    tokens_c = tokenizer.encode(" " + concept_c, add_special_tokens=False)
    
    if not (tokens_a and tokens_b and tokens_c):
        return None
    
    embed_a = embed_matrix[tokens_a[0]].detach().cpu().numpy()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu().numpy()
    embed_c = embed_matrix[tokens_c[0]].detach().cpu().numpy()
    
    dir_ab = embed_a - embed_b
    dir_ab = dir_ab / np.linalg.norm(dir_ab)
    
    # Get A-C direction
    dir_ac = embed_a - embed_c
    dir_ac = dir_ac / np.linalg.norm(dir_ac)
    
    # Project out A-B component to make it orthogonal
    # Gram-Schmidt: dir_orth = dir_ac - (dir_ac · dir_ab) * dir_ab
    projection = np.dot(dir_ac, dir_ab) * dir_ab
    dir_orth = dir_ac - projection
    dir_orth = dir_orth / np.linalg.norm(dir_orth)
    
    return dir_orth


def find_orthogonal_concept(df_test, concept_a, concept_b):
    """
    Find a concept C that:
    - Appears in test set
    - Has different suffix than A and B (different ontological family)
    - Has comparable frequency
    """
    from collections import Counter
    from src.probe_utils import extract_derived_concept
    
    all_concepts = Counter()
    suffix_a = concept_a[-3:] if len(concept_a) >= 3 else ""
    suffix_b = concept_b[-3:] if len(concept_b) >= 3 else ""
    
    for _, row in df_test.iterrows():
        for step in row["steps"]:
            concept = extract_derived_concept(step)
            all_concepts[concept] += 1
    
    candidates = []
    for concept, count in all_concepts.most_common():
        if len(concept) >= 3:
            suffix = concept[-3:]
            if suffix != suffix_a and suffix != suffix_b:
                candidates.append((concept, count))
    
    if candidates:
        return candidates[0][0]  # Most frequent
    return None