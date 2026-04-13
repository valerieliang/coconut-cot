# scripts/steering/run_steering.py
"""
Run steering experiments on Coconut and/or Verbal CoT models.
Fixed with proper per-step contrast vectors, stronger steering, and controlled random vectors.
"""

import sys
import os
from typing import Optional, Dict, List, Tuple 

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
                        default=[0.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    parser.add_argument("--sweep_dim", type=str, default="step",
                        choices=["step", "layer", "both", "all_steps"])
    parser.add_argument("--steering_method", type=str, default="contrast",
                        choices=["contrast", "embedding"])
    
    # Control conditions
    parser.add_argument("--random_control", action="store_true",
                        help="Use random steering vectors as control condition")
    parser.add_argument("--random_scale", type=float, default=0.1,
                        help="Scale factor for random control vectors (default: 0.1)")
    parser.add_argument("--no_steering_control", action="store_true",
                        help="Include baseline (alpha=0) as control")
    
    # Options for better steering
    parser.add_argument("--no_blending", action="store_true",
                        help="Disable surprise-based blending (use pure steering)")
    parser.add_argument("--per_step_vectors", action="store_true",
                        help="Construct separate contrast vectors for each step")
    parser.add_argument("--steer_all", action="store_true",
                        help="Steer all steps from target onward")
    parser.add_argument("--acceptance_threshold", type=float, default=0.5,
                        help="Threshold for surprise-based blending (higher = more steering)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/steering/raw")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_activations", action="store_true")
    parser.add_argument("--save_latent_vectors", action="store_true")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    # Debug
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def construct_contrast_vector_from_embeddings(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
    random_control: bool = False,
    random_scale: float = 0.1,
) -> Optional[torch.Tensor]:
    """
    Construct a contrast vector from embedding differences.
    FIXED: Proper random control with controlled magnitude.
    """
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)

    if not tokens_a or not tokens_b:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None

    # Find embedding matrix
    if hasattr(model, 'base_causallm'):
        embed_matrix = model.base_causallm.transformer.wte.weight
    elif hasattr(model, 'transformer'):
        embed_matrix = model.transformer.wte.weight
    elif hasattr(model, 'model'):
        embed_matrix = model.model.transformer.wte.weight
    elif hasattr(model, 'wte'):
        embed_matrix = model.wte.weight
    else:
        for name, param in model.named_parameters():
            if 'wte' in name or 'word_embeddings' in name:
                embed_matrix = param
                break
        else:
            print("Cannot find embedding matrix")
            return None

    embed_a = embed_matrix[tokens_a[0]].detach().cpu()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu()

    if random_control:
        # FIXED: Generate random vector with controlled magnitude
        direction = torch.randn_like(embed_a)
        direction = direction / (torch.norm(direction) + 1e-8)
        direction = direction * random_scale
        if random_scale <= 0.5:
            print(f"  Random control vector (scale={random_scale}) - should cause minimal flips")
        else:
            print(f"  Random control vector (scale={random_scale}) - WARNING: large scale may cause spurious flips")
    else:
        direction = embed_a - embed_b
        direction = direction / (torch.norm(direction) + 1e-8)
        print(f"  Semantic steering vector (norm=1.0)")

    return direction


def construct_per_step_steering_vectors(
    model, tokenizer, problem_dict: dict, n_hops: int,
    latent_id: int, start_id: int, end_id: int, device: str,
    random_control: bool = False,
    random_scale: float = 0.1,
    verbose: bool = False
) -> dict:
    """
    Construct separate contrast vectors for EACH reasoning step.
    FIXED: Now supports random control with proper scaling.
    """
    step_vectors = {}
    
    for step in range(n_hops):
        if random_control:
            # For random control, just create random vectors
            # Get dimension from a dummy forward pass
            dummy_vec = torch.randn(768)  # GPT-2 hidden size
            vector = dummy_vec / (torch.norm(dummy_vec) + 1e-8)
            vector = vector * random_scale
            step_vectors[step] = vector
            if verbose:
                print(f"    Step {step}: Random vector (scale={random_scale})")
        else:
            premise, negated = extract_premise_and_negation(problem_dict, step_idx=step)
            
            if premise and negated:
                if verbose:
                    print(f"    Step {step}: premise='{premise[:50]}...' negated='{negated[:50]}...'")
                
                vector = construct_contrast_vector_from_negation(
                    model, tokenizer, premise, negated,
                    step, n_hops, latent_id, start_id, end_id, device
                )
                
                if vector is not None:
                    step_vectors[step] = vector
                    if verbose:
                        print(f"      Vector norm: {vector.norm().item():.4f}")
                else:
                    if verbose:
                        print(f"      Failed to construct vector for step {step}")
            else:
                if verbose:
                    print(f"    Step {step}: No premise/negation found")
    
    return step_vectors


def run_coconut_experiment(
    model, tokenizer, problem_dict: dict, args, 
    latent_id: int, start_id: int, end_id: int
) -> dict:
    """Run steering experiment on Coconut model with improved vector construction."""
    
    concept_a = "True"
    concept_b = "False"
    n_hops = problem_dict.get("n_hops", len(problem_dict.get("steps", [])))
    
    # FIXED: Use per-step vectors if requested
    if args.per_step_vectors:
        if args.verbose:
            print(f"  Constructing per-step contrast vectors...")
        
        step_vectors = construct_per_step_steering_vectors(
            model, tokenizer, problem_dict, n_hops,
            latent_id, start_id, end_id, args.device,
            random_control=args.random_control,
            random_scale=args.random_scale,
            verbose=args.verbose
        )
        
        if not step_vectors:
            print(f"  Warning: No step vectors constructed, falling back to embedding")
            steering_vector = construct_contrast_vector_from_embeddings(
                model, tokenizer, concept_a, concept_b,
                random_control=args.random_control,
                random_scale=args.random_scale
            )
            step_vectors = {0: steering_vector}
    else:
        # Original behavior: single vector
        if args.random_control:
            steering_vector = construct_contrast_vector_from_embeddings(
                model, tokenizer, concept_a, concept_b,
                random_control=True,
                random_scale=args.random_scale
            )
        elif args.steering_method == "contrast":
            premise, negated = extract_premise_and_negation(problem_dict, step_idx=0)
            if premise and negated:
                steering_vector = construct_contrast_vector_from_negation(
                    model, tokenizer, premise, negated,
                    0, n_hops, latent_id, start_id, end_id, args.device
                )
            else:
                steering_vector = None
        else:  # embedding method
            steering_vector = construct_contrast_vector_from_embeddings(
                model, tokenizer, concept_a, concept_b
            )
        
        if steering_vector is None:
            steering_vector = construct_contrast_vector_from_embeddings(
                model, tokenizer, concept_a, concept_b
            )
        
        step_vectors = {0: steering_vector}
    
    # Run sweep
    sweep_results = coconut_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas, device=args.device,
        latent_id=latent_id, start_id=start_id, end_id=end_id,
        step_vectors=step_vectors,
        sweep_dim=args.sweep_dim,
        random_control=args.random_control,
        use_blending=not args.no_blending,
        acceptance_threshold=args.acceptance_threshold,
        steer_all=args.steer_all,
    )
    
    # Add metadata about vector source
    if sweep_results:
        for config_name in sweep_results:
            if isinstance(sweep_results[config_name], dict):
                if 'metadata' in sweep_results[config_name]:
                    sweep_results[config_name]['metadata']['vector_source'] = "per_step" if args.per_step_vectors else args.steering_method
                    sweep_results[config_name]['metadata']['random_control'] = args.random_control
                    sweep_results[config_name]['metadata']['random_scale'] = args.random_scale if args.random_control else None
                    sweep_results[config_name]['metadata']['use_blending'] = not args.no_blending
    
    return sweep_results

def run_verbal_experiment(
    model, tokenizer, problem_dict: dict, args
) -> dict:
    """Run steering experiment on Verbal CoT model with fixed intervention."""
    
    concept_a = "True"
    concept_b = "False"
    
    # Use semantic steering for verbal model
    steering_vector = construct_semantic_steering_vector(
        model, tokenizer, concept_a, concept_b, args.device
    )
    
    if steering_vector is None:
        print("  Failed to construct steering vector")
        return None
    
    if args.verbose:
        print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")
    
    # Make sure to use args.alphas, not the function default
    sweep_results = verbal_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas,  # This is key - use args.alphas
        device=args.device,
        steering_vector=steering_vector, 
        sweep_dim=args.sweep_dim,
        verbose=args.verbose
    )
    
    return sweep_results

def construct_semantic_steering_vector(
    model, tokenizer, concept_a: str, concept_b: str, device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Construct a steering vector using the model's OWN representations.
    """
    # Find the model's submodule
    if hasattr(model, 'base_causallm'):
        base_model = model.base_causallm
    elif hasattr(model, 'model'):
        base_model = model.model
    else:
        base_model = model
    
    # Get embedding layer
    if hasattr(base_model, 'transformer'):
        embed_layer = base_model.transformer.wte
    elif hasattr(base_model, 'wte'):
        embed_layer = base_model.wte
    else:
        # Fall back to old method
        return construct_contrast_vector_from_embeddings(model, tokenizer, concept_a, concept_b)
    
    # Tokenize concepts
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        return None
    
    # Get embeddings
    embed_a = embed_layer.weight[tokens_a[0]].detach()
    embed_b = embed_layer.weight[tokens_b[0]].detach()
    
    # Get direction and normalize
    direction = embed_a - embed_b
    direction = direction / (torch.norm(direction) + 1e-8)
    
    # Verify the direction makes sense
    cos_sim = torch.dot(embed_a, embed_b) / (torch.norm(embed_a) * torch.norm(embed_b) + 1e-8)
    print(f"  Semantic steering: cos({concept_a}, {concept_b}) = {cos_sim.item():.4f}")
    
    return direction


def load_model_safe(model_type: str, checkpoint_path: str, device: str):
    """Safely load model without changing directory."""
    abs_path = os.path.join(PROJECT_ROOT, checkpoint_path)
    
    if model_type == "coconut":
        return load_coconut_model(checkpoint_dir=abs_path, device=device)
    elif model_type == "verbal":
        return load_verbal_cot_model(checkpoint_dir=abs_path, device=device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        control_suffix = "_random_control" if args.random_control else ""
        blend_suffix = "_no_blend" if args.no_blending else ""
        per_step_suffix = "_per_step" if args.per_step_vectors else ""
        scale_suffix = f"_scale{args.random_scale}" if args.random_control and args.random_scale != 0.1 else ""
        args.experiment_name = f"{args.model_type}_{args.sweep_dim}_{args.steering_method}{control_suffix}{blend_suffix}{per_step_suffix}{scale_suffix}_{timestamp}"
    
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
    if args.random_control:
        print(f"Random scale: {args.random_scale}")
    print(f"Use blending: {not args.no_blending}")
    print(f"Per-step vectors: {args.per_step_vectors}")
    print(f"Steer all from target: {args.steer_all}")
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
                print(f"  True answer: {problem_dict.get('answer', '')}")
            
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
                    answer_flipped = False
                    if model_type == "coconut":
                        for config_name, config_results in result.items():
                            if isinstance(config_results, dict) and 'metadata' in config_results:
                                if config_results['metadata'].get('steering_propagated', False):
                                    propagation_check = True
                                    break
                        
                        # Check for answer flips
                        for config_name, config_results in result.items():
                            if isinstance(config_results, dict):
                                for alpha_key, alpha_results in config_results.items():
                                    if alpha_key != "metadata" and isinstance(alpha_results, dict):
                                        if alpha_results.get("answer_flipped", False):
                                            answer_flipped = True
                                            break
                    
                    all_results["results"].append({
                        "model_type": model_type,
                        "problem_idx": args.start_idx + idx,
                        "n_hops": problem_dict.get("n_hops", 0),
                        "question": problem_dict.get("question", "")[:200],
                        "true_answer": problem_dict.get("answer", ""),
                        "steering_propagated": propagation_check,
                        "answer_flipped": answer_flipped,
                        "sweep_results": result
                    })
                    
                    if answer_flipped and args.verbose:
                        print(f"  *** ANSWER FLIPPED! ***")
                    
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
            "random_scale": args.random_scale if args.random_control else None,
            "use_blending": not args.no_blending,
            "per_step_vectors": args.per_step_vectors,
            "n_problems": len(df_selected),
        }
    )
    
    # Summary
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
            if r.get("answer_flipped", False):
                answer_flipped_count += 1
            
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
                            effect = abs(alpha_results.get("effect_size", 0))
                            if effect > max_effect:
                                max_effect = effect
                    except (ValueError, TypeError):
                        pass
        
        print(f"    Answer flipped: {answer_flipped_count}/{len(model_results)} problems")
        if total_steered > 0:
            print(f"    Steering applied: {total_steered} times")
            print(f"    Propagated: {propagated_count} ({100*propagated_count/total_steered:.0f}%)")
            print(f"    Max effect size: {max_effect:.3f}")


if __name__ == "__main__":
    main()