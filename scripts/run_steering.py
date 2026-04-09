# scripts/07_steering_experiment.py
"""
Causal Steering Experiment: Comparing Verbal CoT vs Coconut

Key fix: Use embedding-space directions instead of trained probes when
training data is insufficient. This is actually more principled--we want
the semantic direction between two concepts in embedding space.
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
from src.probe_utils import load_reps, extract_derived_concept
from src.coconut_steering import run_steering as run_coconut_steering
from src.verbal_steering import run_verbal_steering


def get_embedding_direction(model, tokenizer, concept_a, concept_b, device="cuda"):
    """
    Get steering direction directly from embedding space.
    This is more reliable than training a probe when data is limited.
    
    Direction = embed(concept_a) - embed(concept_b), normalized.
    """
    # Tokenize concepts (with leading space for GPT-2 style tokenization)
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        # Try without space
        tokens_a = tokenizer.encode(concept_a, add_special_tokens=False)
        tokens_b = tokenizer.encode(concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        return None
    
    # Get the embedding matrix
    if hasattr(model, 'base_causallm'):
        embed_matrix = model.base_causallm.transformer.wte.weight
    else:
        embed_matrix = model.transformer.wte.weight
    
    # Use the first token of each concept
    embed_a = embed_matrix[tokens_a[0]].detach().cpu().numpy()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu().numpy()
    
    direction = embed_a - embed_b
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    
    return direction


def get_random_direction(dim=1024):
    """Generate random unit vector for control condition."""
    direction = np.random.randn(dim)
    return direction / (np.linalg.norm(direction) + 1e-8)


def find_concept_pairs_from_test_data(df_test, min_freq=2):
    """
    Extract concept pairs directly from test data.
    Returns pairs of concepts that appear in the same ontology family.
    """
    from collections import Counter, defaultdict
    
    # Count all derived concepts in test set
    test_concepts = Counter()
    problem_concepts = defaultdict(set)
    
    for idx, row in df_test.iterrows():
        for step in row["steps"]:
            concept = extract_derived_concept(step)
            test_concepts[concept] += 1
            problem_concepts[idx].add(concept)
    
    # Group concepts by suffix (ontology family in ProntoQA)
    families = defaultdict(list)
    for concept in test_concepts.keys():
        if len(concept) >= 3:
            suffix = concept[-3:]  # e.g., "pus", "ump", etc.
            families[suffix].append(concept)
    
    # Create pairs within families
    pairs = []
    for suffix, concepts in families.items():
        # Sort by frequency
        concepts_sorted = sorted(concepts, key=lambda c: test_concepts[c], reverse=True)
        for i, ca in enumerate(concepts_sorted):
            for cb in concepts_sorted[i+1:]:
                if test_concepts[ca] >= min_freq and test_concepts[cb] >= min_freq:
                    pairs.append((ca, cb, test_concepts[ca], test_concepts[cb]))
    
    # Sort pairs by combined frequency
    pairs.sort(key=lambda x: x[2] + x[3], reverse=True)
    
    return pairs


def run_full_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ============================================================
    # 1. Load data
    # ============================================================
    print("="*70)
    print("CAUSAL STEERING EXPERIMENT")
    print("="*70)
    
    df_test = load_split("data/prontoqa_split.csv", split="test")
    print(f"Loaded {len(df_test)} test problems")
    
    # ============================================================
    # 2. Find concept pairs from test data
    # ============================================================
    pairs = find_concept_pairs_from_test_data(df_test, min_freq=2)
    
    print(f"\nFound {len(pairs)} potential concept pairs from test data:")
    for ca, cb, cnt_a, cnt_b in pairs[:5]:
        print(f"  {ca} ({cnt_a}) vs {cb} ({cnt_b})")
    
    if not pairs:
        print("\nNo suitable pairs found. Using fallback: first two concepts.")
        all_concepts = set()
        for _, row in df_test.iterrows():
            for step in row["steps"]:
                all_concepts.add(extract_derived_concept(step))
        concepts_list = list(all_concepts)[:4]
        pairs = [(concepts_list[0], concepts_list[1], 1, 1)]
    
    # Use top 2 pairs for the experiment
    selected_pairs = pairs[:2]
    print(f"\nSelected pairs for experiment: {[p[:2] for p in selected_pairs]}")
    
    # ============================================================
    # 3. Load models and get embedding directions
    # ============================================================
    print("\n" + "="*70)
    print("Loading models...")
    
    verbal_model, verbal_tokenizer = load_verbal_cot_model(device)
    coconut_model, coconut_tokenizer = load_coconut_model(device)
    
    latent_id = coconut_tokenizer.convert_tokens_to_ids("<latent>")
    start_id = coconut_tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id = coconut_tokenizer.convert_tokens_to_ids("<end_latent>")
    
    # Get hidden dimension for random direction
    hidden_dim = verbal_model.config.n_embd  # 1024 for GPT-2 medium
    
    # ============================================================
    # 4. Prepare directions for each pair
    # ============================================================
    directions = {}
    
    for concept_a, concept_b, _, _ in selected_pairs:
        pair_key = f"{concept_a}_vs_{concept_b}"
        print(f"\nPreparing directions for {concept_a} vs {concept_b}...")
        
        # Get embedding-space directions for both models
        dir_verbal = get_embedding_direction(
            verbal_model, verbal_tokenizer, concept_a, concept_b, device
        )
        dir_coconut = get_embedding_direction(
            coconut_model.base_causallm, coconut_tokenizer, concept_a, concept_b, device
        )
        
        if dir_verbal is None or dir_coconut is None:
            print(f"  Skipping pair: could not tokenize one or both concepts")
            continue
        
        directions[pair_key] = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "verbal": dir_verbal,
            "coconut": dir_coconut,
            "random": get_random_direction(hidden_dim),
        }
        
        print(f"  Direction norms: verbal={np.linalg.norm(dir_verbal):.3f}, "
              f"coconut={np.linalg.norm(dir_coconut):.3f}")
    
    if not directions:
        print("\nERROR: No valid concept pairs. Exiting.")
        return
    
    # ============================================================
    # 5. Collect test problems containing these concepts
    # ============================================================
    test_problems = []
    
    for _, row in df_test.iterrows():
        steps_text = " ".join(row["steps"])
        row_concepts = [extract_derived_concept(step) for step in row["steps"]]
        
        for pair_key, dirs in directions.items():
            concept_a, concept_b = dirs["concept_a"], dirs["concept_b"]
            
            if concept_a in row_concepts or concept_b in row_concepts:
                # Find the step where the concept appears
                for i, step in enumerate(row["steps"]):
                    derived = extract_derived_concept(step)
                    if derived == concept_a or derived == concept_b:
                        test_problems.append({
                            "row": row,
                            "pair_key": pair_key,
                            "concept_a": concept_a,
                            "concept_b": concept_b,
                            "concept_step": i,
                            "actual_concept": derived,
                        })
                        break
    
    print(f"\nFound {len(test_problems)} test problem instances with target concepts")
    
    # Limit to reasonable number for experiment
    if len(test_problems) > 30:
        test_problems = test_problems[:30]
        print(f"Limited to {len(test_problems)} problems for experiment")
    
    # ============================================================
    # 6. Run steering experiment
    # ============================================================
    alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    all_results = []
    
    for prob_data in tqdm(test_problems, desc="Steering experiments"):
        row = prob_data["row"]
        concept_a = prob_data["concept_a"]
        concept_b = prob_data["concept_b"]
        concept_step = prob_data["concept_step"]
        pair_key = prob_data["pair_key"]
        
        if pair_key not in directions:
            continue
        
        dirs = directions[pair_key]
        
        # Test both models
        for model_type, model, tokenizer, dir_key in [
            ("verbal", verbal_model, verbal_tokenizer, "verbal"),
            ("coconut", coconut_model, coconut_tokenizer, "coconut"),
        ]:
            direction = dirs[dir_key]
            random_dir = dirs["random"]
            
            if direction is None:
                continue
            
            # Determine which steps to test
            if model_type == "verbal":
                # Test at the concept step and adjacent steps
                test_steps = []
                for offset in [-1, 0, 1]:
                    s = concept_step + offset
                    if 0 <= s < row["n_hops"]:
                        test_steps.append(s)
            else:  # coconut
                # Test all latent steps
                test_steps = list(range(row["n_hops"]))
            
            for steer_step in test_steps:
                # Steer with true direction
                try:
                    if model_type == "coconut":
                        result_true = run_coconut_steering(
                            model=coconut_model,
                            tokenizer=tokenizer,
                            problem_row=row,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            latent_step_to_steer=steer_step,
                            alphas=alphas,
                            device=device,
                            latent_id=latent_id,
                            start_id=start_id,
                            end_id=end_id,
                            steering_vector=direction,
                        )
                        
                        # Control with random direction
                        result_random = run_coconut_steering(
                            model=coconut_model,
                            tokenizer=tokenizer,
                            problem_row=row,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            latent_step_to_steer=steer_step,
                            alphas=alphas,
                            device=device,
                            latent_id=latent_id,
                            start_id=start_id,
                            end_id=end_id,
                            steering_vector=random_dir,
                        )
                    
                    else:  # verbal
                        result_true = run_verbal_steering(
                            model=verbal_model,
                            tokenizer=tokenizer,
                            problem_row=row,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            step_to_steer=steer_step,
                            alphas=alphas,
                            device=device,
                            steering_vector=direction,
                        )
                        
                        result_random = run_verbal_steering(
                            model=verbal_model,
                            tokenizer=tokenizer,
                            problem_row=row,
                            concept_a=concept_a,
                            concept_b=concept_b,
                            step_to_steer=steer_step,
                            alphas=alphas,
                            device=device,
                            steering_vector=random_dir,
                        )
                    
                    if result_true is not None:
                        all_results.append({
                            "problem_id": int(row.name) if hasattr(row, 'name') else str(row.name),
                            "model": model_type,
                            "pair": [concept_a, concept_b],
                            "steer_step": steer_step,
                            "concept_step": concept_step,
                            "n_hops": row["n_hops"],
                            "direction_type": "true",
                            "results": result_true,
                        })
                    
                    if result_random is not None:
                        all_results.append({
                            "problem_id": int(row.name) if hasattr(row, 'name') else str(row.name),
                            "model": model_type,
                            "pair": [concept_a, concept_b],
                            "steer_step": steer_step,
                            "concept_step": concept_step,
                            "n_hops": row["n_hops"],
                            "direction_type": "random",
                            "results": result_random,
                        })
                        
                except Exception as e:
                    print(f"\n  Error on problem {row.name}, step {steer_step}: {e}")
                    continue
    
    # ============================================================
    # 7. Aggregate results
    # ============================================================
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    # Group by model, direction_type, and relative step
    aggregated = defaultdict(lambda: {"effects": [], "changes": []})
    
    for res in all_results:
        model = res["model"]
        direction_type = res["direction_type"]
        steer_step = res["steer_step"]
        concept_step = res["concept_step"]
        relative_step = steer_step - concept_step
        
        # Extract effect at alpha=5.0
        if res["results"] and 5.0 in res["results"]:
            effect = res["results"][5.0].get("change", 0)
            key = (model, direction_type, relative_step)
            aggregated[key]["effects"].append(effect)
            
            # Also collect changes across all alphas for dose-response
            if "changes_by_alpha" not in aggregated[key]:
                aggregated[key]["changes_by_alpha"] = defaultdict(list)
            for alpha in [1.0, 2.0, 5.0, 10.0]:
                if alpha in res["results"]:
                    aggregated[key]["changes_by_alpha"][alpha].append(
                        res["results"][alpha].get("change", 0)
                    )
    
    # Compute statistics
    summary = {}
    for (model, dir_type, rel_step), data in aggregated.items():
        effects = data["effects"]
        if len(effects) >= 2:
            mean_effect = np.mean(effects)
            sem_effect = np.std(effects) / np.sqrt(len(effects))
            t_stat, p_val = stats.ttest_1samp(effects, 0)
            
            summary[f"{model}_{dir_type}_rel{rel_step}"] = {
                "model": model,
                "direction_type": dir_type,
                "relative_step": int(rel_step),
                "mean_effect": float(mean_effect),
                "sem": float(sem_effect),
                "p_value": float(p_val),
                "n": len(effects),
                "significant": p_val < 0.05,
            }
    
    # Print summary
    print("\nEffect of steering (alpha=5.0) by relative step position:")
    print("-"*80)
    print(f"{'Model':<10} {'Direction':<10} {'Rel Step':<10} {'Effect':<15} {'p-value':<10} {'N':<5} {'Sig':<5}")
    print("-"*80)
    
    for key, stats_dict in sorted(summary.items(), key=lambda x: (x[1]['model'], x[1]['relative_step'])):
        sig_str = "**" if stats_dict["significant"] else ""
        print(f"{stats_dict['model']:<10} {stats_dict['direction_type']:<10} "
              f"{stats_dict['relative_step']:<10} "
              f"{stats_dict['mean_effect']:+.3f} +/- {stats_dict['sem']:.3f}   "
              f"{stats_dict['p_value']:.4f}   {stats_dict['n']:<5} {sig_str}")
    
    if not summary:
        print("\nWARNING: No results aggregated. Check that steering experiments completed successfully.")
    
    # ============================================================
    # 8. Save results
    # ============================================================

    os.makedirs("results/steering", exist_ok=True)
    
    clean_results = []
    for res in all_results:
        clean_res = res.copy()
        if "results" in clean_res and clean_res["results"]:
            clean_res["results"] = {str(k): v for k, v in clean_res["results"].items()}
        # Convert any numpy types to Python native types
        for key in ["problem_id", "steer_step", "concept_step", "n_hops"]:
            if key in clean_res:
                if hasattr(clean_res[key], 'item'):
                    clean_res[key] = clean_res[key].item()
        clean_results.append(clean_res)
    
    with open("results/steering/full_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)
    
    # Convert summary to JSON-serializable format
    clean_summary = {}
    for key, stats_dict in summary.items():
        clean_summary[key] = {
            "model": str(stats_dict["model"]),
            "direction_type": str(stats_dict["direction_type"]),
            "relative_step": int(stats_dict["relative_step"]),
            "mean_effect": float(stats_dict["mean_effect"]),
            "sem": float(stats_dict["sem"]),
            "p_value": float(stats_dict["p_value"]),
            "n": int(stats_dict["n"]),
            "significant": bool(stats_dict["significant"]),  # Convert np.bool_ to bool
        }
    
    with open("results/steering/summary.json", "w") as f:
        json.dump(clean_summary, f, indent=2)
    
    print(f"\nResults saved to results/steering/")
    print(f"  - full_results.json ({len(all_results)} trials)")
    print(f"  - summary.json ({len(clean_summary)} conditions)")
    
    return clean_summary, clean_results


if __name__ == "__main__":
    summary, all_results = run_full_experiment()