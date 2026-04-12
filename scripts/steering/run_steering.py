# scripts/run_steering.py
"""
Run steering experiments on Coconut and/or Verbal CoT models.
Saves raw results for later analysis.

Usage:
    python scripts/run_steering.py --model_type coconut --n_problems 50
    python scripts/run_steering.py --model_type both --sweep_dim step
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from datetime import datetime

from src.coconut_steering import (
    run_steering_sweep as coconut_sweep,
    construct_contrast_vector_from_negation,
    construct_contrast_vector_from_embeddings,
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
                        default="checkpoints/coconut_prontoqa")
    parser.add_argument("--verbal_checkpoint", type=str,
                        default="checkpoints/verbal_cot_prontoqa")
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/prontoqa_split.csv")
    parser.add_argument("--data_split", type=str, default="test")
    parser.add_argument("--n_problems", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    
    # Steering parameters
    parser.add_argument("--alphas", type=float, nargs="+", 
                        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    parser.add_argument("--sweep_dim", type=str, default="step",
                        choices=["step", "layer", "both", "position", "normalization", "all_steps"])
    parser.add_argument("--steering_method", type=str, default="contrast",
                        choices=["contrast", "embedding"])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/steering/raw")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_activations", action="store_true",
                        help="Save intermediate activations (larger files)")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    
    # Debug
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def load_coconut_model(checkpoint_path: str, device: str):
    """Load Coconut model."""
    try:
        from coconut.coconut import CoconutModel, CoconutConfig
        
        config = CoconutConfig.from_pretrained(checkpoint_path)
        model = CoconutModel.from_pretrained(checkpoint_path, config=config)
    except:
        from coconut.coconut import CoconutModel
        model = CoconutModel.from_pretrained(checkpoint_path)
    
    model.to(device)
    model.eval()
    return model


def load_verbal_model(checkpoint_path: str, device: str):
    """Load Verbal CoT model (fine-tuned GPT-2)."""
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    return model


def get_tokenizer():
    """Get tokenizer with special tokens for Coconut."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = {"additional_special_tokens": ["<start>", "<end>", "<latent>"]}
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer


def run_coconut_experiment(
    model, tokenizer, problem_dict: dict, args, 
    latent_id: int, start_id: int, end_id: int
) -> dict:
    """Run steering experiment on Coconut model."""
    
    concept_a = "True"
    concept_b = "False"
    
    # Construct steering vector
    if args.steering_method == "contrast":
        premise, negated = extract_premise_and_negation(problem_dict, step_idx=0)
        if premise and negated and args.verbose:
            print(f"  Premise: {premise}")
            print(f"  Negated: {negated}")
        
        if premise and negated:
            steering_vector = construct_contrast_vector_from_negation(
                model, tokenizer, premise, negated,
                latent_id, start_id, end_id,
                problem_dict.get("n_hops", len(problem_dict.get("steps", []))),
                step_to_extract=0, layer=-1, device=args.device
            )
        else:
            steering_vector = None
    else:
        steering_vector = None
    
    if steering_vector is None:
        if args.verbose:
            print("  Falling back to embedding difference")
        steering_vector = construct_contrast_vector_from_embeddings(
            model, tokenizer, concept_a, concept_b
        )
    
    if steering_vector is None:
        return None
    
    # Run sweep
    sweep_results = coconut_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas, device=args.device,
        latent_id=latent_id, start_id=start_id, end_id=end_id,
        steering_vector=steering_vector, sweep_dim=args.sweep_dim
    )
    
    return sweep_results


def run_verbal_experiment(
    model, tokenizer, problem_dict: dict, args
) -> dict:
    """Run steering experiment on Verbal CoT model."""
    
    concept_a = "True"
    concept_b = "False"
    
    # Construct steering vector
    if args.steering_method == "contrast":
        premise, negated = extract_premise_and_negation(problem_dict, step_idx=0)
        if premise and negated and args.verbose:
            print(f"  Premise: {premise}")
            print(f"  Negated: {negated}")
        
        # For verbal, we still use embedding difference as contrast vector
        # (Could extend to activation-based later)
        steering_vector = construct_contrast_vector_from_embeddings(
            model, tokenizer, concept_a, concept_b
        )
    else:
        steering_vector = construct_contrast_vector_from_embeddings(
            model, tokenizer, concept_a, concept_b
        )
    
    if steering_vector is None:
        return None
    
    # Run sweep
    sweep_results = verbal_sweep(
        model, tokenizer, problem_dict, concept_a, concept_b,
        alphas=args.alphas, device=args.device,
        steering_vector=steering_vector, sweep_dim=args.sweep_dim
    )
    
    return sweep_results


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model_type}_{args.sweep_dim}_{timestamp}"
    
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
    print(f"Alphas: {args.alphas}")
    print(f"Output directory: {output_dir}")
    
    # Load tokenizer
    tokenizer = get_tokenizer()
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id = tokenizer.convert_tokens_to_ids("<start>")
    end_id = tokenizer.convert_tokens_to_ids("<end>")
    
    # Load data
    print(f"\nLoading data from {args.data_path} ({args.data_split} split)")
    df = load_prontoqa(args.data_path, split=args.data_split)
    
    # Select problems
    end_idx = min(args.start_idx + args.n_problems, len(df))
    df_selected = df.iloc[args.start_idx:end_idx].reset_index(drop=True)
    print(f"Selected {len(df_selected)} problems (indices {args.start_idx}-{end_idx-1})")
    
    # Load models
    models_loaded = {}
    
    if args.model_type in ["coconut", "both"]:
        print("\n" + "-"*40)
        print("Loading Coconut model...")
        models_loaded["coconut"] = load_coconut_model(args.coconut_checkpoint, args.device)
        print(f"  Loaded from {args.coconut_checkpoint}")
    
    if args.model_type in ["verbal", "both"]:
        print("\n" + "-"*40)
        print("Loading Verbal CoT model...")
        models_loaded["verbal"] = load_verbal_model(args.verbal_checkpoint, args.device)
        print(f"  Loaded from {args.verbal_checkpoint}")
    
    # Run experiments
    all_results = {
        "experiment_name": args.experiment_name,
        "args": vars(args),
        "data_info": {
            "data_path": args.data_path,
            "data_split": args.data_split,
            "n_problems": len(df_selected),
            "start_idx": args.start_idx,
            "hop_distribution": df_selected["n_hops"].value_counts().to_dict() if "n_hops" in df_selected.columns else {},
        },
        "results": []
    }
    
    for model_type, model in models_loaded.items():
        print("\n" + "="*60)
        print(f"RUNNING {model_type.upper()} EXPERIMENTS")
        print("="*60)
        
        for idx in tqdm(range(len(df_selected)), desc=f"{model_type}"):
            problem_dict = get_problem_by_idx(df_selected, idx)
            
            if args.verbose:
                print(f"\nProblem {args.start_idx + idx}: {problem_dict.get('n_hops', '?')} hops")
            
            try:
                if model_type == "coconut":
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
                    if not args.save_activations:
                        for config_name, config_results in result.items():
                            if isinstance(config_results, dict):
                                for alpha, res in config_results.items():
                                    if isinstance(res, dict) and "steered_activations" in res:
                                        del res["steered_activations"]
                    
                    all_results["results"].append({
                        "model_type": model_type,
                        "problem_idx": args.start_idx + idx,
                        "n_hops": problem_dict.get("n_hops", 0),
                        "sweep_results": result
                    })
                else:
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
            "n_problems": len(df_selected),
        }
    )
    
    # Summary
    successful = len([r for r in all_results["results"]])
    print(f"\nExperiment complete!")
    print(f"  Successful problems: {successful}/{len(df_selected)}")
    print(f"  Results saved to: {output_dir}")
    
    # Quick summary by model type
    for model_type in models_loaded.keys():
        model_results = [r for r in all_results["results"] if r["model_type"] == model_type]
        print(f"  {model_type}: {len(model_results)} results")


if __name__ == "__main__":
    main()