# src/coconut_steering.py
"""
Enhanced Coconut steering with multi-step and multi-layer intervention capabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union


def construct_contrast_vector_from_embeddings(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
) -> Optional[torch.Tensor]:
    """Construct a simple contrast vector from embedding differences."""
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)

    if not tokens_a or not tokens_b:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None

    if hasattr(model, 'base_causallm'):
        embed_matrix = model.base_causallm.transformer.wte.weight
    elif hasattr(model, 'transformer'):
        embed_matrix = model.transformer.wte.weight
    elif hasattr(model, 'wte'):
        embed_matrix = model.wte.weight
    else:
        print("Cannot find embedding matrix")
        return None

    embed_a = embed_matrix[tokens_a[0]].detach().cpu()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu()

    direction = embed_a - embed_b
    direction = direction / (torch.norm(direction) + 1e-8)

    return direction

class SelectiveHook:
    """
    Hook that selectively steers at specific latent steps.
    Includes debug logging.
    """
    def __init__(self, target_step_idx, steering_vec, alpha_val, 
                 steer_all_steps, n_hops_val, position_mode_val, norm_mode_val,
                 question_len, latent_start_pos):
        self.target_step_idx = target_step_idx
        self.latent_step_counter = -1
        self.steering_vec = steering_vec
        self.alpha_val = alpha_val
        self.steer_all = steer_all_steps
        self.n_hops = n_hops_val
        self.position_mode = position_mode_val
        self.norm_mode = norm_mode_val
        self.question_len = question_len
        self.latent_start_pos = latent_start_pos
        self.was_steered = False
        self.steered_activations = []
        self.processed_latents = set()
        self.hook_call_count = 0  # Debug counter
    
    def hook_fn(self, module, input, output):
        self.hook_call_count += 1
        
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
        else:
            hidden_states = output.clone()
        
        seq_len = hidden_states.shape[1]
        
        # DEBUG: Print first few calls
        if self.hook_call_count <= 3:
            print(f"  [DEBUG] Hook call #{self.hook_call_count}: seq_len={seq_len}, "
                  f"latent_start_pos={self.latent_start_pos}, n_hops={self.n_hops}")
        
        # Find all latent token positions
        latent_positions_found = []
        for step_idx in range(self.n_hops):
            latent_pos = self.latent_start_pos + step_idx
            if latent_pos < seq_len:
                latent_positions_found.append((step_idx, latent_pos))
        
        # DEBUG: Print if we found latent positions
        if latent_positions_found and self.hook_call_count <= 3:
            print(f"  [DEBUG] Found latent positions: {latent_positions_found}")
        
        for step_idx, latent_pos in latent_positions_found:
            if latent_pos not in self.processed_latents:
                self.processed_latents.add(latent_pos)
                
                # Determine if we should steer at this step
                should_steer = False
                if self.steer_all:
                    should_steer = True
                else:
                    should_steer = (step_idx == self.target_step_idx)
                
                if should_steer:
                    if self.hook_call_count <= 3:
                        print(f"  [DEBUG] STEERING at step {step_idx}, pos {latent_pos}, alpha={self.alpha_val}")
                    
                    pre_steer_activation = hidden_states[0, latent_pos, :].clone()
                    
                    if self.steering_vec.device != hidden_states.device:
                        self.steering_vec = self.steering_vec.to(
                            device=hidden_states.device, 
                            dtype=hidden_states.dtype
                        )
                    
                    if self.norm_mode == 'activation':
                        activation_norm = torch.norm(pre_steer_activation)
                        if activation_norm > 1e-6:
                            scaled = self.steering_vec * self.alpha_val * activation_norm
                        else:
                            scaled = self.steering_vec * self.alpha_val
                    else:
                        scaled = self.steering_vec * self.alpha_val
                    
                    hidden_states[0, latent_pos, :] += scaled
                    self.was_steered = True
                    
                    self.steered_activations.append({
                        'step': step_idx,
                        'alpha': self.alpha_val,
                        'position': latent_pos,
                        'pre_steer_norm': torch.norm(pre_steer_activation).item(),
                        'steering_magnitude': torch.norm(scaled).item(),
                    })
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states
def run_steering_multilevel(
    model,
    tokenizer,
    problem_row: Dict,
    concept_a: str,
    concept_b: str,
    alphas: List[float] = None,
    device: str = "cuda",
    latent_id: Optional[int] = None,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    steering_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
    steer_config: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Run steering with configurable intervention points.
    """

    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]

    if steer_config is None:
        steer_config = {
            'step': 0,
            'layer': -1,
            'position': 'latent',
            'normalization': 'activation',
            'steer_all_steps': False
        }

    # Get token IDs for concepts
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None

    if token_id_a is None or token_id_b is None:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None

    # Prepare steering direction
    if steering_vector is not None:
        if isinstance(steering_vector, np.ndarray):
            direction = torch.from_numpy(steering_vector).float()
        else:
            direction = steering_vector.float()
    else:
        direction = construct_contrast_vector_from_embeddings(
            model, tokenizer, concept_a, concept_b
        )
        if direction is None:
            return None

    direction = direction.to(device)

    # Get problem info
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    question = problem_row["question"]

    # Build input sequence
    question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
    question_len = len(question_ids)
    
    # The first latent token appears after question
    latent_start_pos = question_len
    
    input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn_mask = torch.ones_like(input_tensor)
    position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
    labels = input_tensor.clone()

    results = {}
    baseline_logit_diff = None
    baseline_probs = None

    # Get transformer layers
    if hasattr(model, 'base_causallm'):
        layers = model.base_causallm.transformer.h
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        print("Cannot find transformer layers")
        return None

    # Parse config
    target_step = steer_config.get('step', 0)
    position_mode = steer_config.get('position', 'latent')
    norm_mode = steer_config.get('normalization', 'activation')
    steer_all = steer_config.get('steer_all_steps', False)
    target_layer_idx = steer_config.get('layer', -1)
    if target_layer_idx == -1:
        target_layer_idx = len(layers) - 1

    for alpha in alphas:
        with torch.no_grad():
            # Create hook for this alpha
            selective_hook = SelectiveHook(
                target_step, direction.clone(), alpha, 
                steer_all, n_hops, position_mode, norm_mode,
                question_len, latent_start_pos  # Now latent_start_pos = question_len
            )
            
            target_layer = layers[target_layer_idx]
            handle = target_layer.register_forward_hook(selective_hook.hook_fn)
            
            try:
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    labels=labels,
                    position_ids=position_ids,
                    return_dict=True
                )
            except Exception as e:
                handle.remove()
                print(f"Error in forward pass: {e}")
                return None
            
            handle.remove()
            
            # Extract logits and compute metrics
            logits = outputs.logits
            last_logits = logits[0, -1, :]
            
            logit_a = last_logits[token_id_a].item()
            logit_b = last_logits[token_id_b].item()
            logit_diff = logit_a - logit_b
            
            relevant_logits = torch.stack([last_logits[token_id_a], last_logits[token_id_b]])
            probs = F.softmax(relevant_logits, dim=0)
            prob_a = probs[0].item()
            prob_b = probs[1].item()
            
            if alpha == 0.0:
                baseline_logit_diff = logit_diff
                baseline_probs = {'prob_a': prob_a, 'prob_b': prob_b}
            
            change = logit_diff - baseline_logit_diff if baseline_logit_diff is not None else 0
            
            answer_flipped = False
            flipped_direction = None
            if baseline_probs is not None:
                baseline_answer = concept_a if baseline_probs['prob_a'] > 0.5 else concept_b
                current_answer = concept_a if prob_a > 0.5 else concept_b
                answer_flipped = (baseline_answer != current_answer)
                if answer_flipped:
                    flipped_direction = f"{baseline_answer} -> {current_answer}"
            
            effect_size = 0.0
            if baseline_logit_diff and abs(baseline_logit_diff) > 1e-6:
                effect_size = change / abs(baseline_logit_diff)
            
            results[alpha] = {
                "logit_diff": logit_diff,
                "logit_a": logit_a,
                "logit_b": logit_b,
                "prob_a": prob_a,
                "prob_b": prob_b,
                "change": change,
                "answer_flipped": answer_flipped,
                "flipped_direction": flipped_direction,
                "effect_size": effect_size,
                "was_steered": selective_hook.was_steered,
                "steered_activations": selective_hook.steered_activations,
            }
    
    results['metadata'] = {
        'steer_config': steer_config,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_probs': baseline_probs,
        'n_hops': n_hops,
        'concept_a': concept_a,
        'concept_b': concept_b,
        'token_id_a': token_id_a,
        'token_id_b': token_id_b,
        'question_length': question_len,
        'latent_start_pos': latent_start_pos,
    }
    
    return results


def run_steering_sweep(
    model,
    tokenizer,
    problem_row: Dict,
    concept_a: str,
    concept_b: str,
    alphas: List[float] = None,
    device: str = "cuda",
    latent_id: Optional[int] = None,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    steering_vector: Optional[torch.Tensor] = None,
    sweep_dim: str = 'step',
) -> Dict:
    """Run a systematic sweep over intervention parameters."""
    
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    
    sweep_results = {}
    
    if sweep_dim == 'step':
        print(f"Sweeping over {n_hops} reasoning steps...")
        for step in range(n_hops):
            config = {
                'step': step,
                'layer': -1,
                'position': 'latent',
                'normalization': 'activation',
                'steer_all_steps': False
            }
            result = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
            if result:
                sweep_results[f'step_{step}'] = result
    
    return sweep_results


# Backward compatibility wrapper
def run_steering(
    model,
    tokenizer,
    problem_row,
    concept_a,
    concept_b,
    latent_step_to_steer=0,
    alphas=None,
    device="cuda",
    latent_id=None,
    start_id=None,
    end_id=None,
    steering_vector=None,
):
    """Backward-compatible wrapper for original run_steering interface."""
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    config = {
        'step': latent_step_to_steer,
        'layer': -1,
        'position': 'latent',
        'normalization': 'activation',
        'steer_all_steps': False
    }
    
    return run_steering_multilevel(
        model, tokenizer, problem_row, concept_a, concept_b,
        alphas=alphas, device=device, latent_id=latent_id,
        start_id=start_id, end_id=end_id,
        steering_vector=steering_vector, steer_config=config
    )