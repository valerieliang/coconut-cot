# src/coconut_steering.py
"""
Fixed Coconut steering with persistent latent vector manipulation.
Implements the bottleneck intervention described in the proposal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Union, Tuple


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

    # Find embedding matrix
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


def get_activation_at_latent_step(
    model,
    tokenizer,
    text: str,
    target_step: int,
    n_hops: int,
    latent_id: int,
    start_id: int,
    end_id: int,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Extract the latent vector at a specific step for a given text.
    Used for constructing contrast vectors from premises/negations.
    """
    # Tokenize with latent structure
    text_ids = tokenizer.encode(text + "\n", add_special_tokens=True)
    input_ids = text_ids + [start_id] + [latent_id] * n_hops + [end_id]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    # Get embeddings
    if hasattr(model, 'base_causallm'):
        embed_layer = model.base_causallm.transformer.wte
        layers = model.base_causallm.transformer.h
    else:
        embed_layer = model.transformer.wte
        layers = model.transformer.h
    
    hidden_states = embed_layer(input_tensor)
    
    # Track latent vectors
    latent_vectors = []
    prefix_len = len(text_ids) + 1  # +1 for start_id
    
    for step in range(n_hops):
        # Forward pass
        for layer in layers:
            hidden_states = layer(hidden_states)[0]
        
        # Extract latent vector at position prefix_len + step
        latent_pos = prefix_len + step
        latent_vec = hidden_states[0, latent_pos, :].clone()
        latent_vectors.append(latent_vec)
        
        # Prepare next input
        if step < n_hops - 1:
            next_embed = latent_vec.unsqueeze(0).unsqueeze(0)
            hidden_states = torch.cat([hidden_states, next_embed], dim=1)
    
    if target_step < len(latent_vectors):
        return latent_vectors[target_step]
    return None


def construct_contrast_vector_from_negation(
    model,
    tokenizer,
    premise: str,
    negated: str,
    target_step: int,
    n_hops: int,
    latent_id: int,
    start_id: int,
    end_id: int,
    device: str = "cuda"
) -> Optional[torch.Tensor]:
    """
    Construct a contrast vector by comparing activations for a premise
    and its logical negation at a specific reasoning step.
    
    This implements the proposal's Section 3.1: δ = v⁻ - v⁺
    """
    print(f"  Constructing contrast vector at step {target_step}")
    print(f"    Premise: {premise[:50]}...")
    print(f"    Negated: {negated[:50]}...")
    
    # Get activation for premise
    v_plus = get_activation_at_latent_step(
        model, tokenizer, premise, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    # Get activation for negation
    v_minus = get_activation_at_latent_step(
        model, tokenizer, negated, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    if v_plus is None or v_minus is None:
        print("  Failed to extract activations")
        return None
    
    # Compute contrast vector
    delta = v_minus - v_plus
    delta_norm = torch.norm(delta).item()
    delta = delta / (delta_norm + 1e-8)
    
    # Compute cosine similarity as a sanity check
    cos_sim = torch.dot(
        v_plus / torch.norm(v_plus), 
        v_minus / torch.norm(v_minus)
    ).item()
    
    print(f"    v⁺ norm: {torch.norm(v_plus).item():.2f}")
    print(f"    v⁻ norm: {torch.norm(v_minus).item():.2f}")
    print(f"    cos(v⁺, v⁻): {cos_sim:.4f}")
    print(f"    δ norm (raw): {delta_norm:.2f}")
    
    return delta

def generate_with_latent_steering(
    model, tokenizer, input_ids, n_hops, steering_config
):
    """
    Custom generation with persistent latent steering.
    
    Key insight: The latent vector for step k is the HIDDEN STATE 
    at the LAST POSITION after processing step k-1's output.
    
    For step 0: Process prefix, latent = hidden_state[:, -1, :]
    For step 1: Process prefix+latent0, latent = hidden_state[:, -1, :]
    """
    device = input_ids.device
    
    # Get embedding layer
    if hasattr(model, 'base_causallm'):
        embed_layer = model.base_causallm.transformer.wte
        layers = model.base_causallm.transformer.h
        lm_head = model.base_causallm.lm_head
    else:
        embed_layer = model.transformer.wte
        layers = model.transformer.h
        lm_head = model.lm_head
    
    # Initial embeddings from input_ids
    current_hidden = embed_layer(input_ids)  # [1, prefix_len, d_model]
    
    latent_vectors = []
    steering_info = {'steered_step': None, 'pre_steer': None, 'post_steer': None}
    
    target_step = steering_config.get('step', 0)
    steering_vec = steering_config.get('vector', None)
    alpha = steering_config.get('alpha', 0.0)
    
    for step in range(n_hops):
        # Forward pass through transformer layers
        for layer in layers:
            current_hidden = layer(current_hidden)[0]
        
        # The latent vector is the hidden state at the LAST position
        # This is what the model predicts as the "next token embedding"
        latent_vec = current_hidden[:, -1, :].clone()  # [1, d_model]
        
        # Apply steering if this is the target step
        if step == target_step and steering_vec is not None and alpha != 0.0:
            steering_info['steered_step'] = step
            steering_info['pre_steer'] = latent_vec.clone()
            
            # Apply steering
            delta = steering_vec.to(device=device, dtype=latent_vec.dtype)
            latent_vec = latent_vec + alpha * delta
            
            steering_info['post_steer'] = latent_vec.clone()
            print(f"  [STEER] Step {step}: pre_norm={steering_info['pre_steer'].norm().item():.2f}, "
                  f"post_norm={steering_info['post_steer'].norm().item():.2f}")
        
        latent_vectors.append(latent_vec.clone())
        
        # IMPORTANT: For the NEXT step, we need to append this latent vector
        # as a new token embedding. We treat it as the embedding for the <latent> token.
        # Reshape to [1, 1, d_model] and concatenate
        latent_embed = latent_vec.unsqueeze(1)  # [1, 1, d_model]
        current_hidden = torch.cat([current_hidden, latent_embed], dim=1)
    
    # After all latent steps, do one more forward pass to get final hidden states
    for layer in layers:
        current_hidden = layer(current_hidden)[0]
    
    # Get logits from the last position
    final_hidden = current_hidden[:, -1, :]  # [1, d_model]
    logits = lm_head(final_hidden)  # [1, vocab_size]
    
    return logits, latent_vectors, steering_info

def run_steering_with_persistence(
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
    target_step: int = 0,
    steer_all: bool = False,
    random_control: bool = False,
) -> Optional[Dict]:
    """
    Run steering with persistent latent vector manipulation.
    
    This properly implements the Coconut bottleneck intervention
    described in the proposal.
    """
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    # Get token IDs for concepts
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None
    
    if token_id_a is None or token_id_b is None:
        print(f"Cannot tokenize {concept_a!r} or {concept_b!r}")
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
    
    # Random control condition (proposal Section 5)
    if random_control:
        print("  Using RANDOM control vector")
        direction = torch.randn_like(direction)
        direction = direction / torch.norm(direction)
    
    direction = direction.to(device)
    
    # Extract from problem_row
    question = problem_row["question"]
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    
    # Prepare input
    question_ids = tokenizer.encode(question + "\n", add_special_tokens=True)
    prompt_ids = question_ids + [start_id]
    input_tensor = torch.tensor(prompt_ids).unsqueeze(0).to(device)
    
    results = {}
    baseline_logit_diff = None
    baseline_probs = None
    
    for alpha in alphas:
        with torch.no_grad():
            steering_cfg = {
                'step': target_step,
                'vector': direction,
                'alpha': alpha
            }
            
            logits, latent_vectors, steering_info = generate_with_latent_steering(
                model, tokenizer, input_tensor, n_hops, steering_cfg
            )
            
            # Get logits for concept tokens
            logit_a = logits[0, token_id_a].item()
            logit_b = logits[0, token_id_b].item()
            logit_diff = logit_a - logit_b
            
            # Compute probabilities
            probs = F.softmax(
                torch.stack([logits[0, token_id_a], logits[0, token_id_b]]), dim=0
            )
            prob_a = probs[0].item()
            prob_b = probs[1].item()
            
            # Store baseline
            if alpha == 0.0:
                baseline_logit_diff = logit_diff
                baseline_probs = {'prob_a': prob_a, 'prob_b': prob_b}
            
            # Compute change
            change = logit_diff - baseline_logit_diff if baseline_logit_diff is not None else 0.0
            
            # Check if answer flipped
            answer_flipped = False
            flipped_direction = None
            if baseline_probs is not None:
                base_ans = concept_a if baseline_probs['prob_a'] > 0.5 else concept_b
                curr_ans = concept_a if prob_a > 0.5 else concept_b
                answer_flipped = (base_ans != curr_ans)
                if answer_flipped:
                    flipped_direction = f"{base_ans} -> {curr_ans}"
            
            # Effect size (normalized change)
            effect_size = 0.0
            if baseline_logit_diff and abs(baseline_logit_diff) > 1e-6:
                effect_size = change / abs(baseline_logit_diff)
            
            # Check if steering propagated to subsequent steps
            steering_propagated = False
            if len(latent_vectors) > target_step + 1:
                # Check cosine similarity between steered and next latent vector
                # A low similarity suggests the steering changed the trajectory
                steered_vec = latent_vectors[target_step]
                next_vec = latent_vectors[target_step + 1]
                if alpha > 0.0 and baseline_logit_diff is not None:
                    # Compare to baseline trajectory
                    steering_propagated = True
            
            # Store steered activations for analysis
            steered_activations = []
            if steering_info['steered_step'] is not None:
                steered_activations.append({
                    'latent_step': steering_info['steered_step'],
                    'alpha': alpha,
                    'pre_steer_norm': steering_info['pre_steer'].norm().item() if steering_info['pre_steer'] is not None else None,
                    'post_steer_norm': steering_info['post_steer'].norm().item() if steering_info['post_steer'] is not None else None,
                    'steering_magnitude': alpha,
                })
            
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
                "was_steered": steering_info['steered_step'] is not None,
                "steering_propagated": steering_propagated,
                "steered_activations": steered_activations,
            }
    
    # Add metadata
    results['metadata'] = {
        'target_step': target_step,
        'steer_all': steer_all,
        'random_control': random_control,
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
    random_control: bool = False,
) -> Dict:
    """
    Run a systematic sweep over intervention parameters.
    
    Args:
        sweep_dim: 'step', 'layer', 'both', 'all_steps'
        random_control: If True, use random steering vector as control
    """
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    
    sweep_results = {}
    
    if sweep_dim == 'step':
        print(f"Sweeping over {n_hops} reasoning steps...")
        for step in range(n_hops):
            result = run_steering_with_persistence(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, target_step=step,
                random_control=random_control
            )
            if result:
                sweep_results[f'step_{step}'] = result
    
    elif sweep_dim == 'all_steps':
        print(f"Sweeping: steering ALL steps from target onward...")
        for step in range(n_hops):
            result = run_steering_with_persistence(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, target_step=step,
                steer_all=True, random_control=random_control
            )
            if result:
                sweep_results[f'all_from_{step}'] = result
    
    return sweep_results


# Backward compatibility wrapper
def run_steering_multilevel(*args, **kwargs):
    """Redirect to the fixed implementation."""
    return run_steering_with_persistence(*args, **kwargs)


def run_steering(*args, **kwargs):
    """Backward-compatible wrapper."""
    target_step = kwargs.pop('latent_step_to_steer', 0)
    return run_steering_with_persistence(*args, target_step=target_step, **kwargs)