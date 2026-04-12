import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import json
from src.model_utils import load_coconut_model
from src.data_utils import load_prontoqa
from src.coconut_steering import run_steering_multilevel, construct_contrast_vector_from_embeddings

def main():
    device = "cuda"
    n_problems = 30
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("Loading Coconut model...")
    model, tokenizer = load_coconut_model("checkpoints/coconut/final", device)
    
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id = tokenizer.convert_tokens_to_ids("<end_latent>")
    
    print("Loading data...")
    df = load_prontoqa("data/prontoqa_split.csv", split="test")
    df = df.head(n_problems)
    
    print("Creating steering vector...")
    steering_vector = construct_contrast_vector_from_embeddings(model, tokenizer, "True", "False")
    
    config_single = {'step': 0, 'layer': -1, 'position': 'latent', 'normalization': 'activation', 'steer_all_steps': False}
    config_all = {'step': 0, 'layer': -1, 'position': 'latent', 'normalization': 'activation', 'steer_all_steps': True}
    
    results = {'single_step': {}, 'all_steps': {}}
    
    for idx, (_, problem) in enumerate(df.iterrows()):
        print(f"Problem {idx+1}/{len(df)}")
        problem_dict = problem.to_dict()
        
        print(f"  Single step steering...")
        results['single_step'][idx] = run_steering_multilevel(
            model, tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=steering_vector, steer_config=config_single
        )
        
        print(f"  All steps steering...")
        results['all_steps'][idx] = run_steering_multilevel(
            model, tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=steering_vector, steer_config=config_all
        )
    
    os.makedirs("results/final_experiments/coconut_steer_all", exist_ok=True)
    with open("results/final_experiments/coconut_steer_all/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Done! Results saved to results/final_experiments/coconut_steer_all/")

if __name__ == "__main__":
    main()