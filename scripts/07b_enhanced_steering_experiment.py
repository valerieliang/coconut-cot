# scripts/07b_enhanced_steering_experiment.py
"""
Enhanced steering experiment with:
- Normalized effect sizes
- Dose-response curves
- Orthogonal direction control
- Multiple baseline samples for variance estimation
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import json
import numpy as np
import torch
from collections import defaultdict
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from src.model_utils import load_coconut_model, load_verbal_cot_model
from src.data_utils import load_split
from src.probe_utils import extract_derived_concept
from src.steering_analysis import (
    run_steering_with_variance_estimation,
    get_orthogonal_direction,
    find_orthogonal_concept,
)


def get_embedding_direction(model, tokenizer, concept_a, concept_b, device="cuda"):
    """Get normalized embedding direction."""
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        tokens_a = tokenizer.encode(concept_a, add_special_tokens=False)
        tokens_b = tokenizer.encode(concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        return None
    
    if hasattr(model, 'base_causallm'):
        embed_matrix = model.base_causallm.transformer.wte.weight
    else:
        embed_matrix = model.transformer.wte.weight
    
    embed_a = embed_matrix[tokens_a[0]].detach().cpu().numpy()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu().numpy()
    
    direction = embed_a - embed_b
    return direction / (np.linalg.norm(direction) + 1e-8)


def run_enhanced_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*70)
    print("ENHANCED CAUSAL STEERING EXPERIMENT")
    print("="*70)
    
    # Load data
    df_test = load_split("data/prontoqa_split.csv", split="test")
    print(f"Loaded {len(df_test)} test problems")
    
    # Select concept pair
    from collections import Counter
    test_concepts = Counter()
    for _, row in df_test.iterrows():
        for step in row["steps"]:
            test_concepts[extract_derived_concept(step)] += 1
    
    # Find top concepts with same suffix
    suffix_groups = defaultdict(list)
    for concept, count in test_concepts.most_common():
        if len(concept) >= 3:
            suffix = concept[-3:]
            suffix_groups[suffix].append((concept, count))
    
    # Pick the largest group
    best_suffix = max(suffix_groups.keys(), key=lambda s: sum(c for _, c in suffix_groups[s]))
    concepts = [c for c, _ in suffix_groups[best_suffix][:2]]
    
    if len(concepts) < 2:
        print("Could not find suitable concept pair")
        return
    
    concept_a, concept_b = concepts[0], concepts[1]
    concept_c = find_orthogonal_concept(df_test, concept_a, concept_b)
    
    print(f"\nSelected concepts:")
    print(f"  A: {concept_a} (count: {test_concepts[concept_a]})")
    print(f"  B: {concept_b} (count: {test_concepts[concept_b]})")
    print(f"  C (orthogonal): {concept_c} (count: {test_concepts.get(concept_c, 0)})")
    
    # Load models
    print("\n" + "="*70)
    print("Loading models...")
    verbal_model, verbal_tokenizer = load_verbal_cot_model(device)
    coconut_model, coconut_tokenizer = load_coconut_model(device)
    
    latent_id = coconut_tokenizer.convert_tokens_to_ids("<latent>")
    start_id = coconut_tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id = coconut_tokenizer.convert_tokens_to_ids("<end_latent>")
    
    # Get directions
    print("\nComputing steering directions...")
    
    dir_verbal_ab = get_embedding_direction(verbal_model, verbal_tokenizer, concept_a, concept_b)
    dir_coconut_ab = get_embedding_direction(coconut_model.base_causallm, coconut_tokenizer, concept_a, concept_b)
    
    # Random control
    hidden_dim = verbal_model.config.n_embd
    dir_random = np.random.randn(hidden_dim)
    dir_random = dir_random / np.linalg.norm(dir_random)
    
    # Orthogonal control (if concept_c available)
    if concept_c:
        dir_verbal_orth = get_orthogonal_direction(
            verbal_model.transformer.wte.weight, verbal_tokenizer,
            concept_a, concept_b, concept_c
        )
        dir_coconut_orth = get_orthogonal_direction(
            coconut_model.base_causallm.transformer.wte.weight, coconut_tokenizer,
            concept_a, concept_b, concept_c
        )
    else:
        dir_verbal_orth = None
        dir_coconut_orth = None
    
    # Collect test problems
    test_problems = []
    for _, row in df_test.iterrows():
        row_concepts = [extract_derived_concept(step) for step in row["steps"]]
        if concept_a in row_concepts or concept_b in row_concepts:
            for i, step in enumerate(row["steps"]):
                derived = extract_derived_concept(step)
                if derived in [concept_a, concept_b]:
                    test_problems.append({
                        "row": row,
                        "concept_a": concept_a,
                        "concept_b": concept_b,
                        "concept_step": i,
                        "actual_concept": derived,
                    })
                    break
    
    print(f"\nFound {len(test_problems)} test problems")
    if len(test_problems) > 15:
        test_problems = test_problems[:15]
        print(f"Limited to {len(test_problems)} for experiment")
    
    # Experiment parameters
    alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    n_baseline_samples = 10
    all_results = []
    
    # Test each problem
    for prob_data in tqdm(test_problems, desc="Running experiments"):
        row = prob_data["row"]
        concept_step = prob_data["concept_step"]
        
        # Test multiple steering steps
        for model_type, model, tokenizer, dir_ab, dir_orth in [
            ("verbal", verbal_model, verbal_tokenizer, dir_verbal_ab, dir_verbal_orth),
            ("coconut", coconut_model, coconut_tokenizer, dir_coconut_ab, dir_coconut_orth),
        ]:
            if dir_ab is None:
                continue
            
            # Determine test steps
            if model_type == "verbal":
                test_steps = []
                for offset in [-1, 0, 1]:
                    s = concept_step + offset
                    if 0 <= s < row["n_hops"]:
                        test_steps.append(s)
            else:
                test_steps = list(range(row["n_hops"]))
            
            for steer_step in test_steps:
                relative_step = steer_step - concept_step
                
                # 1. True direction (A-B)
                result_true = run_steering_with_variance_estimation(
                    model, tokenizer, row, concept_a, concept_b,
                    latent_step_to_steer=steer_step, alphas=alphas,
                    device=device, latent_id=latent_id, start_id=start_id,
                    end_id=end_id, steering_vector=dir_ab,
                    n_baseline_samples=n_baseline_samples,
                    model_type=model_type,
                )
                
                if result_true:
                    all_results.append({
                        "problem_id": str(row.name),
                        "model": model_type,
                        "direction_type": "true",
                        "steer_step": steer_step,
                        "concept_step": concept_step,
                        "relative_step": relative_step,
                        "concept_a": concept_a,
                        "concept_b": concept_b,
                        "n_hops": row["n_hops"],
                        "result": result_true,
                    })
                
                # 2. Random direction
                result_random = run_steering_with_variance_estimation(
                    model, tokenizer, row, concept_a, concept_b,
                    latent_step_to_steer=steer_step, alphas=alphas,
                    device=device, latent_id=latent_id, start_id=start_id,
                    end_id=end_id, steering_vector=dir_random,
                    n_baseline_samples=n_baseline_samples,
                    model_type=model_type,
                )
                
                if result_random:
                    all_results.append({
                        "problem_id": str(row.name),
                        "model": model_type,
                        "direction_type": "random",
                        "steer_step": steer_step,
                        "concept_step": concept_step,
                        "relative_step": relative_step,
                        "concept_a": concept_a,
                        "concept_b": concept_b,
                        "n_hops": row["n_hops"],
                        "result": result_random,
                    })
                
                # 3. Orthogonal direction (if available)
                if dir_orth is not None:
                    result_orth = run_steering_with_variance_estimation(
                        model, tokenizer, row, concept_a, concept_b,
                        latent_step_to_steer=steer_step, alphas=alphas,
                        device=device, latent_id=latent_id, start_id=start_id,
                        end_id=end_id, steering_vector=dir_orth,
                        n_baseline_samples=n_baseline_samples,
                        model_type=model_type,
                    )
                    
                    if result_orth:
                        all_results.append({
                            "problem_id": str(row.name),
                            "model": model_type,
                            "direction_type": "orthogonal",
                            "steer_step": steer_step,
                            "concept_step": concept_step,
                            "relative_step": relative_step,
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "n_hops": row["n_hops"],
                            "result": result_orth,
                        })
    
    # Save results
    os.makedirs("results/steering/enhanced", exist_ok=True)
    
    # Convert to serializable format
    clean_results = []
    for res in all_results:
        clean = res.copy()
        if "result" in clean:
            clean["result"] = {
                "baseline_mean": float(clean["result"]["baseline_mean"]),
                "baseline_std": float(clean["result"]["baseline_std"]),
                "baseline_prob_mean": float(clean["result"]["baseline_prob_mean"]),
                "baseline_prob_std": float(clean["result"]["baseline_prob_std"]),
                "n_baseline_samples": clean["result"]["n_baseline_samples"],
                "effects": {
                    str(k): {k2: float(v2) if isinstance(v2, (np.floating, float)) else v2
                             for k2, v2 in v.items()}
                    for k, v in clean["result"]["effects"].items()
                }
            }
        clean_results.append(clean)
    
    with open("results/steering/enhanced/full_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to results/steering/enhanced/")
    print(f"  - {len(clean_results)} total trials")
    
    return clean_results


if __name__ == "__main__":
    run_enhanced_experiment()