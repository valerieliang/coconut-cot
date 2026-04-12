import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import json
from src.model_utils import load_coconut_model, load_verbal_cot_model
from src.data_utils import load_prontoqa
from src.coconut_steering import run_steering_multilevel, construct_contrast_vector_from_embeddings
from src.verbal_steering import run_verbal_steering_multilevel

def main():
    device = "cuda"
    n_problems = 20
    alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    print("Loading models...")
    coconut_model, coconut_tokenizer = load_coconut_model("checkpoints/coconut/final", device)
    verbal_model, verbal_tokenizer = load_verbal_cot_model("checkpoints/verbal_cot/final", device)
    
    latent_id = coconut_tokenizer.convert_tokens_to_ids("<latent>")
    start_id = coconut_tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id = coconut_tokenizer.convert_tokens_to_ids("<end_latent>")
    
    print("Loading data...")
    df = load_prontoqa("data/prontoqa_split.csv", split="test")
    df = df.head(n_problems)
    
    print("Creating steering vectors...")
    semantic_vec = construct_contrast_vector_from_embeddings(coconut_model, coconut_tokenizer, "True", "False")
    random_vec = torch.randn_like(semantic_vec)
    random_vec = random_vec / torch.norm(random_vec) * torch.norm(semantic_vec)
    
    config_coconut = {'step': 0, 'layer': -1, 'position': 'latent', 'normalization': 'activation', 'steer_all_steps': False}
    config_verbal = {'step': 0, 'layer': -1, 'position': 'step_end', 'normalization': 'activation'}
    
    results = {'coconut_semantic': {}, 'coconut_random': {}, 'verbal_semantic': {}, 'verbal_random': {}}
    
    for idx, (_, problem) in enumerate(df.iterrows()):
        print(f"Problem {idx+1}/{len(df)}")
        problem_dict = problem.to_dict()
        
        print(f"  Coconut semantic...")
        results['coconut_semantic'][idx] = run_steering_multilevel(
            coconut_model, coconut_tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=semantic_vec, steer_config=config_coconut
        )
        
        print(f"  Coconut random...")
        results['coconut_random'][idx] = run_steering_multilevel(
            coconut_model, coconut_tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=random_vec, steer_config=config_coconut
        )
        
        print(f"  Verbal semantic...")
        results['verbal_semantic'][idx] = run_verbal_steering_multilevel(
            verbal_model, verbal_tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device,
            steering_vector=semantic_vec.cpu(), steer_config=config_verbal
        )
        
        print(f"  Verbal random...")
        results['verbal_random'][idx] = run_verbal_steering_multilevel(
            verbal_model, verbal_tokenizer, problem_dict, "True", "False",
            alphas=alphas, device=device,
            steering_vector=random_vec.cpu(), steer_config=config_verbal
        )
    
    os.makedirs("results/final_experiments/random_control", exist_ok=True)
    with open("results/final_experiments/random_control/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()