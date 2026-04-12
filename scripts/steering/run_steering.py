# scripts/steering/run_steering.py
"""
Run steering experiments on Coconut and/or Verbal CoT models.
Updated with proper persistent latent steering and control conditions.
"""

import sys
import os

# Add the project root to path FIRST, before any other imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Use the existing model_utils
from src.model_utils import load_coconut_model, load_verbal_cot_model

from src.coconut_steering import (
    run_steering_sweep as coconut_sweep,
    construct_contrast_vector_from_embeddings,
    construct_contrast_vector_from_negation,
    run_steering_with_persistence,
)
from src.verbal_steering import (
    run_verbal_steering_sweep as verbal_sweep,
)
from src.data_utils import (
    load_prontoqa,
    extract_premise_and_negation,
    get_problem_by_idx,
)
from src.steering_analysis import save_steering_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run steering experiments")
    
    # Model selection
    parser.add_argument("--model_type", type=str, default="coconut", 
                        choices=["coconut", "verbal", "both"])
    parser.add_argument("--coconut_checkpoint", type=str, 
                        default="checkpoints/coconut/final")
    parser.add_argument("--verbal_checkpoint", type=str,
                        default="checkpoints/verbal_cot/final")
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/prontoqa_split.csv")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--n_problems", type=int, default=10)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--hop_filter", type=int, nargs="+", default=None,
                        help="Only run on problems with these hop counts")
    
    # Steering parameters
    parser.add_argument("--alphas", type=float, nargs="+", 
                        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--sweep_dim", type=str, default="step",
                        choices=["step", "layer", "both", "all_steps"])
    parser.add_argument("--steering_method", type=str, default="contrast",
                        choices=["contrast", "embedding"])
    
    # Control conditions (for proposal Section 5)
    parser.add_argument("--random_control", action="store_true",
                        help="Use random steering vectors as control condition")
    parser.add_argument("--no_steering_control", action="store_true",
                        help="Include baseline (alpha=0) as control")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/steering/raw")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--save_latent_vectors", action="store_true",
                        help="Save all latent vectors for propagation analysis")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Debug
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def run_coconut_experiment(
    model, tokenizer, problem_dict: dict, args, 
    latent_id: int, start_id: int, end_id: int
) -> dict:
    """Run steering experiment on Coconut model."""
    
    concept_a = "True"
    concept_b = "False"
    
    # Construct steering vector
    steering_vector = None
    vector_source = "embedding"
    
    if args.steering_method == "contrast":
        premise, negated = extract_premise_and_negation(problem_dict, step_idx=0)
        if premise and negated:
            if args.verbose:
                print(f"  Constructing contrast vector from negation")
                print(f"    Premise: {premise}")
                print(f"    Negated: {negated}")
            
            n_hops = problem_dict.get("n_hops", len(problem_dict.get("steps", [])))
            steering_vector = construct_contrast_vector_from_negation(
                model, tokenizer, premise, negated,
                0, n_hops, latent_id, start_id, end_id, args.device
            )
            if steering_vector is not None:
                vector_source = "contrast"
    
    # Fall back to embedding difference
    if steering_vector is None:
        if args.verbose:
            print("  Using embedding difference for steering vector")
        steering_vector = construct_contrast_vector_from_embeddings(
            model, tokenizer, concept_a, concept_b
        )
        vector_source = "embedding"
    
    if steering_vector is None:
        print("  Failed to construct steering vector")
        return None
    
    # Run sweep
    sweep_results = coconut_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas, device=args.device,
        latent_id=latent_id, start_id=start_id, end_id=end_id,
        steering_vector=steering_vector, sweep_dim=args.sweep_dim,
        random_control=args.random_control
    )
    
    # Add metadata about vector source
    if sweep_results:
        for config_name in sweep_results:
            if isinstance(sweep_results[config_name], dict):
                if 'metadata' in sweep_results[config_name]:
                    sweep_results[config_name]['metadata']['vector_source'] = vector_source
                    sweep_results[config_name]['metadata']['random_control'] = args.random_control
    
    return sweep_results


def run_verbal_experiment(
    model, tokenizer, problem_dict: dict, args
) -> dict:
    """Run steering experiment on Verbal CoT model."""
    
    concept_a = "True"
    concept_b = "False"
    
    # Construct steering vector from embedding difference
    steering_vector = construct_contrast_vector_from_embeddings(
        model, tokenizer, concept_a, concept_b
    )
    
    if steering_vector is None:
        print("  Failed to construct steering vector")
        return None
    
    # Run sweep
    sweep_results = verbal_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas, device=args.device,
        steering_vector=steering_vector, sweep_dim=args.sweep_dim
    )
    
    return sweep_results


def load_model_safe(model_type: str, checkpoint_path: str, device: str):
    """Safely load model without changing directory."""
    abs_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    
    if model_type == "coconut":
        return load_coconut_model(checkpoint_dir=abs_path, device=device)
    elif model_type == "verbal":
        return load_verbal_cot_model(checkpoint_dir=abs_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def construct_semantic_steering_vector(model, tokenizer, concept_a, concept_b, device="cuda"):
    """
    Construct a steering vector that is semantically aligned with the model's latent space.
    Uses the model's OWN representations rather than static embeddings.
    """
    # Get the model's internal representation of each concept
    # by running them through the model with a neutral context
    
    neutral_context = "The concept of"
    
    # Get representation for concept A
    text_a = f"{neutral_context} {concept_a}"
    inputs_a = tokenizer(text_a, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if hasattr(model, 'base_causallm'):
            outputs_a = model.base_causallm(**inputs_a, output_hidden_states=True)
        else:
            outputs_a = model(**inputs_a, output_hidden_states=True)
    
    # Use last layer hidden state at the concept token
    hidden_a = outputs_a.hidden_states[-1][0, -1, :]
    
    # Get representation for concept B
    text_b = f"{neutral_context} {concept_b}"
    inputs_b = tokenizer(text_b, return_tensors="pt").to(device)
    
    with torch.no_grad():
        if hasattr(model, 'base_causallm'):
            outputs_b = model.base_causallm(**inputs_b, output_hidden_states=True)
        else:
            outputs_b = model(**inputs_b, output_hidden_states=True)
    
    hidden_b = outputs_b.hidden_states[-1][0, -1, :]
    
    # Direction from B to A
    direction = hidden_a - hidden_b
    direction = direction / torch.norm(direction)
    
    # Check alignment with latent space
    cos_sim = torch.dot(hidden_a / torch.norm(hidden_a), hidden_b / torch.norm(hidden_b)).item()
    print(f"  Semantic steering: cos({concept_a}, {concept_b}) = {cos_sim:.4f}")
    
    return direction.cpu()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        control_suffix = "_random_control" if args.random_control else ""
        args.experiment_name = f"{args.model_type}_{args.sweep_dim}_{args.steering_method}{control_suffix}_{timestamp}"
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("="*60)
    print(f"STEERING EXPERIMENT: {args.experiment_name}")
    print("="*60)
    print(f"Model type: {args.model_type}")
    print(f"Sweep dimension: {args.sweep_dim}")
    print(f"Steering method: {args.steering_method}")
    print(f"Random control: {args.random_control}")
    print(f"Alphas: {args.alphas}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    
    # Load data
    data_path = os.path.join(PROJECT_ROOT, args.data_path)
    print(f"\nLoading data from {data_path} ({args.data_split} split)")
    df = load_prontoqa(data_path, split=args.data_split)
    
    # Filter by hop count if specified
    if args.hop_filter:
        df = df[df['n_hops'].isin(args.hop_filter)]
        print(f"Filtered to hop counts: {args.hop_filter}")
        print(f"Remaining problems: {len(df)}")
    
    # Select problems
    end_idx = min(args.start_idx + args.n_problems, len(df))
    df_selected = df.iloc[args.start_idx:end_idx].reset_index(drop=True)
    print(f"Selected {len(df_selected)} problems (indices {args.start_idx}-{end_idx-1})")
    
    # Show hop distribution
    hop_counts = df_selected['n_hops'].value_counts().sort_index()
    print(f"Hop distribution: {dict(hop_counts)}")
    
    # Load models
    models_loaded = {}
    tokenizers = {}
    coconut_special_tokens = None
    
    if args.model_type in ["coconut", "both"]:
        print("\n" + "-"*40)
        print("Loading Coconut model...")
        try:
            model, tokenizer = load_model_safe(
                "coconut", args.coconut_checkpoint, args.device
            )
            
            # Extract special tokens
            latent_id = tokenizer.convert_tokens_to_ids("<latent>")
            start_id = tokenizer.convert_tokens_to_ids("<start_latent>")
            end_id = tokenizer.convert_tokens_to_ids("<end_latent>")
            
            # Handle cases where tokens might have different names
            if start_id == tokenizer.unk_token_id:
                start_id = tokenizer.convert_tokens_to_ids("<start>")
            if end_id == tokenizer.unk_token_id:
                end_id = tokenizer.convert_tokens_to_ids("<end>")
            
            models_loaded["coconut"] = model
            tokenizers["coconut"] = tokenizer
            coconut_special_tokens = (latent_id, start_id, end_id)
            print(f"  Loaded from {args.coconut_checkpoint}")
            print(f"  Special tokens: start={start_id}, end={end_id}, latent={latent_id}")
        except Exception as e:
            print(f"  Error loading Coconut model: {e}")
            import traceback
            traceback.print_exc()
    
    if args.model_type in ["verbal", "both"]:
        print("\n" + "-"*40)
        print("Loading Verbal CoT model...")
        try:
            model, tokenizer = load_model_safe(
                "verbal", args.verbal_checkpoint, args.device
            )
            models_loaded["verbal"] = model
            tokenizers["verbal"] = tokenizer
            print(f"  Loaded from {args.verbal_checkpoint}")
        except Exception as e:
            print(f"  Error loading Verbal model: {e}")
            import traceback
            traceback.print_exc()
    
    if not models_loaded:
        print("\nERROR: No models could be loaded!")
        return
    
    # Run experiments
    all_results = {
        "experiment_name": args.experiment_name,
        "args": vars(args),
        "data_info": {
            "data_path": args.data_path,
            "data_split": args.data_split,
            "n_problems": len(df_selected),
            "start_idx": args.start_idx,
            "hop_filter": args.hop_filter,
            "hop_distribution": dict(hop_counts),
        },
        "results": []
    }
    
    for model_type, model in models_loaded.items():
        print("\n" + "="*60)
        print(f"RUNNING {model_type.upper()} EXPERIMENTS")
        print("="*60)
        
        tokenizer = tokenizers[model_type]
        
        for idx in tqdm(range(len(df_selected)), desc=f"{model_type}"):
            problem_dict = get_problem_by_idx(df_selected, idx)
            
            if args.verbose:
                print(f"\nProblem {args.start_idx + idx}: {problem_dict.get('n_hops', '?')} hops")
                print(f"  Question: {problem_dict.get('question', '')[:100]}...")
            
            try:
                if model_type == "coconut":
                    if coconut_special_tokens is None:
                        print(f"  Skipping: No special tokens for Coconut")
                        continue
                    latent_id, start_id, end_id = coconut_special_tokens
                    result = run_coconut_experiment(
                        model, tokenizer, problem_dict, args,
                        latent_id, start_id, end_id
                    )
                else:
                    result = run_verbal_experiment(
                        model, tokenizer, problem_dict, args
                    )
                
                if result:
                    # Remove large activation data if not requested
                    if not args.save_activations and not args.save_latent_vectors:
                        for config_name, config_results in result.items():
                            if isinstance(config_results, dict):
                                for alpha, res in config_results.items():
                                    if isinstance(res, dict):
                                        if "steered_activations" in res:
                                            del res["steered_activations"]
                    
                    # Check if steering propagated
                    propagation_check = False
                    if model_type == "coconut":
                        for config_name, config_results in result.items():
                            if isinstance(config_results, dict) and 'metadata' in config_results:
                                if config_results['metadata'].get('steering_propagated', False):
                                    propagation_check = True
                                    break
                    
                    all_results["results"].append({
                        "model_type": model_type,
                        "problem_idx": args.start_idx + idx,
                        "n_hops": problem_dict.get("n_hops", 0),
                        "question": problem_dict.get("question", "")[:200],
                        "true_answer": problem_dict.get("answer", ""),
                        "steering_propagated": propagation_check,
                        "sweep_results": result
                    })
                else:
                    if args.verbose:
                        print(f"  Warning: Failed on problem {args.start_idx + idx}")
                    
            except Exception as e:
                print(f"  Error on problem {args.start_idx + idx}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
    
    # Save results
    print("\n" + "-"*40)
    print("Saving results...")
    
    results_path = os.path.join(output_dir, "steering_results.json")
    save_steering_results(
        all_results,
        results_path,
        metadata={
            "experiment_name": args.experiment_name,
            "model_type": args.model_type,
            "sweep_dim": args.sweep_dim,
            "steering_method": args.steering_method,
            "random_control": args.random_control,
            "n_problems": len(df_selected),
        }
    )
    
    # Summary with better propagation reporting
    successful = len(all_results["results"])
    print(f"\nExperiment complete!")
    print(f"  Successful problems: {successful}/{len(df_selected)}")
    print(f"  Results saved to: {output_dir}")
    
    for model_type in models_loaded.keys():
        model_results = [r for r in all_results["results"] if r["model_type"] == model_type]
        print(f"\n  {model_type.upper()}: {len(model_results)} results")
        
        if len(model_results) == 0:
            continue
            
        # Analyze steering effects
        total_steered = 0
        propagated_count = 0
        answer_flipped_count = 0
        max_effect = 0.0
        
        for r in model_results:
            sweep_results = r.get("sweep_results", {})
            for config_name, config_results in sweep_results.items():
                if config_name == "metadata" or not isinstance(config_results, dict):
                    continue
                for alpha_key, alpha_results in config_results.items():
                    if alpha_key == "metadata" or not isinstance(alpha_results, dict):
                        continue
                    try:
                        alpha = float(alpha_key)
                        if alpha > 0:
                            if alpha_results.get("was_steered", False):
                                total_steered += 1
                            if alpha_results.get("steering_propagated", False):
                                propagated_count += 1
                            if alpha_results.get("answer_flipped", False):
                                answer_flipped_count += 1
                            effect = abs(alpha_results.get("effect_size", 0))
                            if effect > max_effect:
                                max_effect = effect
                    except (ValueError, TypeError):
                        pass
        
        if total_steered > 0:
            print(f"    Steering applied: {total_steered} times")
            print(f"    Propagated: {propagated_count} ({100*propagated_count/total_steered:.0f}%)")
            print(f"    Answer flipped: {answer_flipped_count}")
            print(f"    Max effect size: {max_effect:.3f}")
            
            if answer_flipped_count > 0:
                print(f"    SUCCESS: Steering flipped the answer!")
            elif propagated_count > 0:
                print(f"    Steering propagated through reasoning chain")
            else:
                print(f"    Steering applied but did not propagate to output")
        else:
            print(f"    No steering applied (check alpha values)")


if __name__ == "__main__":
    main()