# src/verbal_steering.py
"""
Enhanced steering for verbal CoT models using activation patching.
Supports multi-step and multi-layer interventions for comparison with Coconut.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union


def run_verbal_steering_multilevel(
    model,
    tokenizer,
    problem_row: Dict,
    concept_a: str,
    concept_b: str,
    alphas: List[float] = None,
    device: str = "cuda",
    steering_vector: Optional[Union[np.ndarray, torch.Tensor]] = None,
    steer_config: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Steer verbal CoT by modifying hidden states at specified reasoning steps.
    
    Args:
        model: Verbal CoT model (GPT2LMHeadModel)
        tokenizer: Tokenizer
        problem_row: Dictionary with 'question', 'steps'
        concept_a: First concept for logit comparison
        concept_b: Second concept for logit comparison
        alphas: List of steering multipliers
        device: Device to run on
        steering_vector: Pre-computed steering direction
        steer_config: Dictionary specifying intervention parameters:
            - 'step': Which reasoning step to steer (0-indexed)
            - 'layer': Which transformer layer to apply steering
            - 'position': 'last' or 'step_end' (which token to steer)
            - 'normalization': 'activation' or 'fixed'
    
    Returns:
        Results dictionary with detailed steering metrics
    """
    
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    # Default steering configuration
    if steer_config is None:
        steer_config = {
            'step': 0,
            'layer': -1,
            'position': 'last',
            'normalization': 'activation'
        }
    
    # Prepare steering vector
    if steering_vector is not None:
        if isinstance(steering_vector, np.ndarray):
            direction = torch.from_numpy(steering_vector).float()
        else:
            direction = steering_vector.float()
    else:
        # Fallback: use embedding difference
        tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
        tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
        token_id_a = tokens_a[0] if tokens_a else None
        token_id_b = tokens_b[0] if tokens_b else None
        
        if token_id_a is None or token_id_b is None:
            print(f"Cannot tokenize {concept_a} or {concept_b}")
            return None
        
        # Get embedding matrix - handle different model architectures
        if hasattr(model, 'transformer'):
            embed_matrix = model.transformer.wte.weight
        elif hasattr(model, 'base_causallm'):
            embed_matrix = model.base_causallm.transformer.wte.weight
        else:
            # Try common attribute names
            if hasattr(model, 'wte'):
                embed_matrix = model.wte.weight
            elif hasattr(model, 'word_embeddings'):
                embed_matrix = model.word_embeddings.weight
            else:
                print("Cannot find embedding matrix")
                return None
        
        embed_a = embed_matrix[token_id_a].detach().cpu()
        embed_b = embed_matrix[token_id_b].detach().cpu()
        direction = embed_a - embed_b
        direction = direction / torch.norm(direction)
    
    direction = direction.to(device)
    
    # Get token IDs for concepts
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None
    
    if token_id_a is None or token_id_b is None:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None
    
    # Build text up to the target step
    steps = problem_row["steps"]
    question = problem_row["question"]
    
    target_step_idx = steer_config.get('step', 0)
    
    text_parts = [question]
    for i in range(target_step_idx + 1):
        if i < len(steps):
            text_parts.append(steps[i])
    text = "\n".join(text_parts)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    results = {}
    baseline_logit_diff = None
    baseline_probs = None
    
    for alpha in alphas:
        with torch.no_grad():
            
            class VerbalSteeringHook:
                def __init__(self, config: Dict, steering_vec: torch.Tensor, alpha_val: float):
                    self.config = config
                    self.steering_vec = steering_vec
                    self.alpha_val = alpha_val
                    self.was_steered = False
                    self.steered_activation = None
                
                def hook_fn(self, module, input, output):
                    # Handle different output types
                    if isinstance(output, tuple):
                        hidden_states = output[0].clone()
                    else:
                        hidden_states = output.clone()
                    
                    if self.steering_vec.device != hidden_states.device:
                        self.steering_vec = self.steering_vec.to(
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )
                    
                    # Determine position to steer
                    position_mode = self.config.get('position', 'last')
                    if position_mode == 'last':
                        pos = hidden_states.shape[1] - 1
                    elif position_mode == 'step_end':
                        # Steer at the end of the target step's tokens
                        pos = hidden_states.shape[1] - 1
                    else:
                        pos = -1
                    
                    # Store pre-steering activation
                    pre_steer = hidden_states[0, pos, :].clone()
                    
                    # Apply steering
                    norm_mode = self.config.get('normalization', 'activation')
                    if norm_mode == 'activation':
                        activation_norm = torch.norm(pre_steer)
                        steer_norm = torch.norm(self.steering_vec)
                        if steer_norm > 1e-6 and activation_norm > 1e-6:
                            scaled = self.steering_vec * (self.alpha_val * activation_norm / steer_norm)
                        else:
                            scaled = self.steering_vec * self.alpha_val
                    else:
                        scaled = self.steering_vec * self.alpha_val
                    
                    hidden_states[0, pos, :] += scaled
                    self.was_steered = True
                    self.steered_activation = hidden_states[0, pos, :].clone()
                    
                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    else:
                        return hidden_states
            
            hook = VerbalSteeringHook(steer_config, direction.clone(), alpha)
            
            # Find the transformer layers
            if hasattr(model, 'transformer'):
                layers = model.transformer.h
            elif hasattr(model, 'base_causallm'):
                layers = model.base_causallm.transformer.h
            else:
                print("Cannot find transformer layers")
                return None
            
            # Register hook on specified layer
            target_layer_idx = steer_config.get('layer', -1)
            if target_layer_idx == -1:
                target_layer_idx = len(layers) - 1
            target_layer = layers[target_layer_idx]
            handle = target_layer.register_forward_hook(hook.hook_fn)
            
            try:
                outputs = model(**inputs)
            except Exception as e:
                handle.remove()
                print(f"Error in forward pass: {e}")
                return None
            
            handle.remove()
            
            # Get logits for next token prediction
            last_logits = outputs.logits[0, -1, :]
            
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
            
            # Determine if answer flipped
            answer_flipped = False
            flipped_direction = None
            if baseline_probs is not None:
                baseline_answer = concept_a if baseline_probs['prob_a'] > 0.5 else concept_b
                current_answer = concept_a if prob_a > 0.5 else concept_b
                answer_flipped = (baseline_answer != current_answer)
                if answer_flipped:
                    flipped_direction = f"{baseline_answer} -> {current_answer}"
            
            # Calculate effect size
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
                "was_steered": hook.was_steered,
            }
    
    # Add metadata
    results['metadata'] = {
        'steer_config': steer_config,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_probs': baseline_probs,
        'concept_a': concept_a,
        'concept_b': concept_b,
        'n_steps': len(steps),
    }
    
    return results


def run_verbal_steering_sweep(
    model,
    tokenizer,
    problem_row: Dict,
    concept_a: str,
    concept_b: str,
    alphas: List[float] = None,
    device: str = "cuda",
    steering_vector: Optional[torch.Tensor] = None,
    sweep_dim: str = 'step',
) -> Dict:
    """
    Run a systematic sweep over intervention parameters for verbal CoT.
    
    Args:
        sweep_dim: Which dimension to sweep over
            - 'step': Sweep over reasoning steps
            - 'layer': Sweep over transformer layers
            - 'both': Sweep over both (step × layer grid)
    
    Returns:
        Dictionary with results for each configuration
    """
    
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    n_steps = len(problem_row.get("steps", []))
    
    # Find number of layers
    if hasattr(model, 'transformer'):
        n_layers = len(model.transformer.h)
    elif hasattr(model, 'base_causallm'):
        n_layers = len(model.base_causallm.transformer.h)
    else:
        n_layers = 12  # Default for GPT-2
    
    sweep_results = {}
    
    if sweep_dim == 'step':
        print(f"Sweeping over {n_steps} reasoning steps...")
        for step in range(n_steps):
            config = {
                'step': step,
                'layer': -1,
                'position': 'last',
                'normalization': 'activation'
            }
            sweep_results[f'step_{step}'] = run_verbal_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'layer':
        print(f"Sweeping over {n_layers} layers...")
        for layer in range(n_layers):
            config = {
                'step': 0,
                'layer': layer,
                'position': 'last',
                'normalization': 'activation'
            }
            sweep_results[f'layer_{layer}'] = run_verbal_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'both':
        print(f"Sweeping over {n_steps} steps × {n_layers} layers...")
        for step in range(n_steps):
            for layer in range(n_layers):
                config = {
                    'step': step,
                    'layer': layer,
                    'position': 'last',
                    'normalization': 'activation'
                }
                sweep_results[f'step{step}_layer{layer}'] = run_verbal_steering_multilevel(
                    model, tokenizer, problem_row, concept_a, concept_b,
                    alphas=alphas, device=device,
                    steering_vector=steering_vector, steer_config=config
                )
    
    return sweep_results


# Backward compatibility wrapper
def run_verbal_steering(
    model,
    tokenizer,
    problem_row,
    concept_a,
    concept_b,
    step_to_steer=0,
    alphas=None,
    device="cuda",
    layer=-1,
    steering_vector=None,
):
    """
    Backward-compatible wrapper for original run_verbal_steering interface.
    """
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    config = {
        'step': step_to_steer,
        'layer': layer,
        'position': 'last',
        'normalization': 'activation'
    }
    
    return run_verbal_steering_multilevel(
        model, tokenizer, problem_row, concept_a, concept_b,
        alphas=alphas, device=device,
        steering_vector=steering_vector, steer_config=config
    )