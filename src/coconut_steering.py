# src/coconut_steering.py
"""
Enhanced Coconut steering with multi-step and multi-layer intervention capabilities.
Supports steering at different levels of reasoning as described in the proposal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Any


def construct_contrast_vector_from_negation(
    model,
    tokenizer,
    premise_text: str,
    negated_premise_text: str,
    latent_id: int,
    start_id: int,
    end_id: int,
    n_hops: int,
    step_to_extract: int = 0,
    layer: int = -1,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Construct a steering vector by contrasting activations from a premise
    and its negation at a specific reasoning step.
    
    Args:
        premise_text: The original premise (e.g., "Alex is a reptile.")
        negated_premise_text: The negated premise (e.g., "Alex is not a reptile.")
        step_to_extract: Which continuous thought step to extract from
        layer: Which transformer layer to extract from
    
    Returns:
        Steering vector (v_neg - v_pos) normalized to unit length
    """
    
    def get_activation_at_step(text: str) -> Optional[torch.Tensor]:
        """Extract activation at specified step and layer."""
        question_ids = tokenizer.encode(text + "\n", add_special_tokens=True)
        input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
        
        # Find the position of the target latent step
        question_len = len(question_ids)
        target_pos = question_len + step_to_extract
        
        activation = None
        
        def hook_fn(module, input, output):
            nonlocal activation
            activation = output[0][0, target_pos, :].detach().clone()
        
        target_layer = model.base_causallm.transformer.h[layer]
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = model(
                    input_ids=input_tensor,
                    position_ids=position_ids,
                )
        except Exception as e:
            print(f"Error extracting activation: {e}")
            return None
        finally:
            handle.remove()
        
        return activation
    
    v_pos = get_activation_at_step(premise_text)
    v_neg = get_activation_at_step(negated_premise_text)
    
    if v_pos is None or v_neg is None:
        print("Failed to extract one or both activations")
        return None
    
    steering_vector = v_neg - v_pos
    steering_vector = steering_vector / torch.norm(steering_vector)
    
    return steering_vector


def construct_contrast_vector_from_embeddings(
    model,
    tokenizer,
    concept_a: str,
    concept_b: str,
) -> Optional[torch.Tensor]:
    """
    Construct a simple contrast vector from embedding differences.
    Fallback method when activation-based construction isn't available.
    
    Args:
        concept_a: First concept (e.g., "True")
        concept_b: Second concept (e.g., "False")
    
    Returns:
        Normalized steering vector
    """
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    
    if not tokens_a or not tokens_b:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None
    
    embed_matrix = model.base_causallm.transformer.wte.weight
    embed_a = embed_matrix[tokens_a[0]].detach().cpu()
    embed_b = embed_matrix[tokens_b[0]].detach().cpu()
    
    direction = embed_a - embed_b
    direction = direction / torch.norm(direction)
    
    return direction


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
    
    Args:
        model: Coconut model
        tokenizer: Tokenizer
        problem_row: Dictionary with 'question', 'steps', 'n_hops'
        concept_a: First concept for logit comparison (e.g., "True")
        concept_b: Second concept for logit comparison (e.g., "False")
        alphas: List of steering multipliers
        device: Device to run on
        latent_id: Token ID for <latent>
        start_id: Token ID for <start>
        end_id: Token ID for <end>
        steering_vector: Pre-computed steering direction
        steer_config: Dictionary specifying intervention parameters:
            - 'step': Which continuous thought step to steer (0-indexed)
            - 'layer': Which transformer layer to apply steering
            - 'position': 'last' or 'latent' (which token position to steer)
            - 'normalization': 'activation' or 'fixed'
            - 'steer_all_steps': Boolean to steer all steps
    
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
    
    # Prepare steering vector
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
    
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    question = problem_row["question"]
    
    question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
    input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn_mask = torch.ones_like(input_tensor)
    position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
    labels = input_tensor.clone()
    
    results = {}
    baseline_logit_diff = None
    baseline_probs = None
    
    for alpha in alphas:
        with torch.no_grad():
            
            class ConfigurableSteeringHook:
                def __init__(self, config: Dict, steering_vec: torch.Tensor, 
                             alpha_val: float, question_len: int):
                    self.config = config
                    self.target_step = config.get('step', 0)
                    self.target_layer = config.get('layer', -1)
                    self.position_mode = config.get('position', 'latent')
                    self.norm_mode = config.get('normalization', 'activation')
                    self.steer_all = config.get('steer_all_steps', False)
                    self.current_step = 0
                    self.steering_vec = steering_vec
                    self.alpha_val = alpha_val
                    self.question_len = question_len
                    self.steered_activations = []
                    self.was_steered = False
                
                def hook_fn(self, module, input, output):
                    hidden_states = output[0].clone()
                    
                    # Determine if we should steer at this step
                    should_steer = False
                    if self.steer_all:
                        should_steer = True
                    elif self.current_step == self.target_step:
                        should_steer = True
                    
                    if should_steer:
                        if self.steering_vec.device != hidden_states.device:
                            self.steering_vec = self.steering_vec.to(
                                device=hidden_states.device, dtype=hidden_states.dtype
                            )
                        
                        # Determine which position to steer
                        if self.position_mode == 'last':
                            pos_to_steer = hidden_states.shape[1] - 1
                        elif self.position_mode == 'latent':
                            pos_to_steer = self.question_len + self.current_step
                        else:
                            pos_to_steer = -1
                        
                        # Store pre-steering activation
                        pre_steer_activation = hidden_states[0, pos_to_steer, :].clone()
                        
                        # Apply steering with chosen normalization
                        if self.norm_mode == 'activation':
                            activation_norm = torch.norm(pre_steer_activation)
                            steer_norm = torch.norm(self.steering_vec)
                            if steer_norm > 1e-6 and activation_norm > 1e-6:
                                scaled = self.steering_vec * (self.alpha_val * activation_norm / steer_norm)
                            else:
                                scaled = self.steering_vec * self.alpha_val
                        else:  # 'fixed'
                            scaled = self.steering_vec * self.alpha_val
                        
                        hidden_states[0, pos_to_steer, :] += scaled
                        self.was_steered = True
                        
                        # Store post-steering activation
                        post_steer_activation = hidden_states[0, pos_to_steer, :].clone()
                        
                        self.steered_activations.append({
                            'step': self.current_step,
                            'alpha': self.alpha_val,
                            'position': pos_to_steer,
                            'pre_steer_norm': torch.norm(pre_steer_activation).item(),
                            'steering_magnitude': torch.norm(scaled).item(),
                            'cosine_sim': float(torch.cosine_similarity(
                                scaled.unsqueeze(0), pre_steer_activation.unsqueeze(0)
                            )[0]) if torch.norm(scaled) > 0 and torch.norm(pre_steer_activation) > 0 else 0.0,
                        })
                    
                    self.current_step += 1
                    return (hidden_states,) + output[1:]
            
            # Create hook with configuration
            hook = ConfigurableSteeringHook(
                steer_config, direction.clone(), alpha, len(question_ids)
            )
            
            # Register hook on specified layer
            target_layer_idx = steer_config.get('layer', -1)
            target_layer = model.base_causallm.transformer.h[target_layer_idx]
            handle = target_layer.register_forward_hook(hook.hook_fn)
            
            try:
                outputs = model(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    labels=labels,
                    position_ids=position_ids,
                )
            except Exception as e:
                handle.remove()
                print(f"Error in forward pass: {e}")
                return None
            
            handle.remove()
            
            # Extract logits and probabilities
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
            
            # Determine if answer flipped
            answer_flipped = False
            flipped_direction = None
            if baseline_probs is not None:
                baseline_answer = concept_a if baseline_probs['prob_a'] > 0.5 else concept_b
                current_answer = concept_a if prob_a > 0.5 else concept_b
                answer_flipped = (baseline_answer != current_answer)
                if answer_flipped:
                    flipped_direction = f"{baseline_answer} -> {current_answer}"
            
            # Calculate effect size (Cohen's d style)
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
                "steered_activations": hook.steered_activations,
            }
    
    # Add metadata
    results['metadata'] = {
        'steer_config': steer_config,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_probs': baseline_probs,
        'n_hops': n_hops,
        'concept_a': concept_a,
        'concept_b': concept_b,
        'token_id_a': token_id_a,
        'token_id_b': token_id_b,
        'question_length': len(question_ids),
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
    """
    Run a systematic sweep over intervention parameters.
    
    Args:
        sweep_dim: Which dimension to sweep over
            - 'step': Sweep over reasoning steps (0 to n_hops-1)
            - 'layer': Sweep over transformer layers (0 to n_layers-1)
            - 'both': Sweep over both (step × layer grid)
            - 'position': Compare 'latent' vs 'last' positions
            - 'normalization': Compare 'activation' vs 'fixed' normalization
    
    Returns:
        Dictionary with results for each configuration
    """
    
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    n_layers = len(model.base_causallm.transformer.h)
    
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
            sweep_results[f'step_{step}'] = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'layer':
        print(f"Sweeping over {n_layers} layers...")
        for layer in range(n_layers):
            config = {
                'step': 0,
                'layer': layer,
                'position': 'latent',
                'normalization': 'activation',
                'steer_all_steps': False
            }
            sweep_results[f'layer_{layer}'] = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'both':
        print(f"Sweeping over {n_hops} steps × {n_layers} layers...")
        for step in range(n_hops):
            for layer in range(n_layers):
                config = {
                    'step': step,
                    'layer': layer,
                    'position': 'latent',
                    'normalization': 'activation',
                    'steer_all_steps': False
                }
                sweep_results[f'step{step}_layer{layer}'] = run_steering_multilevel(
                    model, tokenizer, problem_row, concept_a, concept_b,
                    alphas=alphas, device=device, latent_id=latent_id,
                    start_id=start_id, end_id=end_id,
                    steering_vector=steering_vector, steer_config=config
                )
    
    elif sweep_dim == 'position':
        print("Comparing 'latent' vs 'last' positions...")
        for pos_mode in ['latent', 'last']:
            config = {
                'step': 0,
                'layer': -1,
                'position': pos_mode,
                'normalization': 'activation',
                'steer_all_steps': False
            }
            sweep_results[f'position_{pos_mode}'] = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'normalization':
        print("Comparing normalization methods...")
        for norm_mode in ['activation', 'fixed']:
            config = {
                'step': 0,
                'layer': -1,
                'position': 'latent',
                'normalization': norm_mode,
                'steer_all_steps': False
            }
            sweep_results[f'norm_{norm_mode}'] = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
    
    elif sweep_dim == 'all_steps':
        print("Testing steering at all steps simultaneously...")
        config = {
            'step': 0,
            'layer': -1,
            'position': 'latent',
            'normalization': 'activation',
            'steer_all_steps': True
        }
        sweep_results['all_steps'] = run_steering_multilevel(
            model, tokenizer, problem_row, concept_a, concept_b,
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=steering_vector, steer_config=config
        )
    
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
    """
    Backward-compatible wrapper for original run_steering interface.
    """
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