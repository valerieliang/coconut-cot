# src/coconut_steering.py
"""
Fixed Coconut steering with persistent latent vector manipulation.
Implements introspection-aware steering based on metacognition principles.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
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
    """Extract the latent vector at a specific step for a given text."""
    text_ids = tokenizer.encode(text + "\n", add_special_tokens=True)
    input_ids = text_ids + [start_id] + [latent_id] * n_hops + [end_id]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    
    if hasattr(model, 'base_causallm'):
        embed_layer = model.base_causallm.transformer.wte
        layers = model.base_causallm.transformer.h
    elif hasattr(model, 'transformer'):
        embed_layer = model.transformer.wte
        layers = model.transformer.h
    elif hasattr(model, 'model'):
        embed_layer = model.model.transformer.wte
        layers = model.model.transformer.h
    else:
        print("Cannot find transformer layers")
        return None
    
    with torch.no_grad():
        hidden_states = embed_layer(input_tensor)
        prefix_len = len(text_ids) + 1
        
        for step in range(n_hops):
            for layer in layers:
                hidden_states = layer(hidden_states)[0]
            
            latent_pos = prefix_len + step
            if latent_pos < hidden_states.shape[1]:
                latent_vec = hidden_states[0, latent_pos, :].clone()
            else:
                latent_vec = hidden_states[0, -1, :].clone()
            
            if step == target_step:
                return latent_vec
            
            if step < n_hops - 1:
                next_embed = latent_vec.unsqueeze(0).unsqueeze(0)
                hidden_states = torch.cat([hidden_states, next_embed], dim=1)
    
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
    """Construct a contrast vector by comparing activations for a premise and its negation."""
    print(f"  Constructing contrast vector at step {target_step}")
    print(f"    Premise: {premise[:50]}...")
    print(f"    Negated: {negated[:50]}...")
    
    v_plus = get_activation_at_latent_step(
        model, tokenizer, premise, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    v_minus = get_activation_at_latent_step(
        model, tokenizer, negated, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    if v_plus is None or v_minus is None:
        print("  Failed to extract activations")
        return None
    
    delta = v_minus - v_plus
    delta_norm = torch.norm(delta).item()
    delta = delta / (delta_norm + 1e-8)
    
    cos_sim = torch.dot(
        v_plus / torch.norm(v_plus), 
        v_minus / torch.norm(v_minus)
    ).item()
    
    print(f"    v⁺ norm: {torch.norm(v_plus).item():.2f}")
    print(f"    v⁻ norm: {torch.norm(v_minus).item():.2f}")
    print(f"    cos(v⁺, v⁻): {cos_sim:.4f}")
    print(f"    δ norm (raw): {delta_norm:.2f}")
    
    return delta


def compute_steering_surprise(
    original_vec: torch.Tensor,
    steered_vec: torch.Tensor
) -> Dict:
    """
    Measure how "surprising" the steering would be to the model.
    Based on metacognition principles - high surprise triggers correction.
    
    Returns:
        Dict with surprise metrics
    """
    # Ensure 1D
    orig = original_vec.squeeze()
    steer = steered_vec.squeeze()
    if orig.dim() > 1:
        orig = orig[0]
    if steer.dim() > 1:
        steer = steer[0]
    
    # Cosine similarity
    orig_norm = torch.norm(orig)
    steer_norm = torch.norm(steer)
    
    cos_sim = torch.dot(orig / (orig_norm + 1e-8), steer / (steer_norm + 1e-8)).item()
    
    # Norm ratio
    norm_ratio = steer_norm.item() / (orig_norm.item() + 1e-8)
    
    # Surprise score: combines angular change and magnitude change
    angular_surprise = 1.0 - cos_sim
    magnitude_surprise = abs(math.log(max(norm_ratio, 1e-8)))
    
    surprise = angular_surprise * magnitude_surprise
    
    return {
        'cosine_similarity': cos_sim,
        'norm_ratio': norm_ratio,
        'angular_surprise': angular_surprise,
        'magnitude_surprise': magnitude_surprise,
        'total_surprise': surprise,
        'orig_norm': orig_norm.item(),
        'steered_norm': steer_norm.item(),
    }


def generate_with_introspection_aware_steering(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    n_hops: int,
    steering_config: Dict,
    norm_mode: str = 'activation',
    acceptance_threshold: float = 0.1,
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
    """
    Custom generation with introspection-aware steering.
    
    The model may actively resist steering that creates too much "surprise".
    This implements metacognitive monitoring from the lecture slides.
    
    Args:
        model: Coconut model
        tokenizer: Tokenizer (unused, kept for API consistency)
        input_ids: Input token IDs [1, seq_len]
        n_hops: Number of latent reasoning steps
        steering_config: Dict with 'step', 'vector', 'alpha'
        norm_mode: 'activation' or 'unit' for steering scaling
        acceptance_threshold: Maximum surprise before blending
    
    Returns:
        logits: Final logits [1, vocab_size]
        latent_vectors: List of latent vectors from each step
        steering_info: Dict with steering metadata
    """
    device = input_ids.device
    
    # Get model components
    if hasattr(model, 'base_causallm'):
        embed_layer = model.base_causallm.transformer.wte
        layers = model.base_causallm.transformer.h
        lm_head = model.base_causallm.lm_head
    elif hasattr(model, 'transformer'):
        embed_layer = model.transformer.wte
        layers = model.transformer.h
        lm_head = model.lm_head
    elif hasattr(model, 'model'):
        embed_layer = model.model.transformer.wte
        layers = model.model.transformer.h
        lm_head = model.model.lm_head
    else:
        raise ValueError("Cannot find model components")
    
    # Initial embeddings
    current_hidden = embed_layer(input_ids)
    
    latent_vectors = []
    steering_info = {
        'steered_step': None, 
        'pre_steer': None, 
        'post_steer': None,
        'steering_applied': False,
        'surprise_metrics': [],
        'blend_ratios': [],
    }
    
    target_step = steering_config.get('step', 0)
    steering_vec = steering_config.get('vector', None)
    alpha = steering_config.get('alpha', 0.0)
    steer_all = steering_config.get('steer_all', False)
    use_blending = steering_config.get('use_blending', True)
    
    for step in range(n_hops):
        # Forward pass through transformer layers
        for layer in layers:
            current_hidden = layer(current_hidden)[0]
        
        # Extract latent vector
        latent_vec = current_hidden[:, -1, :].clone()
        
        # Determine if we should steer
        should_steer = (steer_all and step >= target_step) or (step == target_step)
        
        if should_steer and steering_vec is not None and alpha != 0.0:
            steering_info['steered_step'] = step
            steering_info['pre_steer'] = latent_vec.clone()
            steering_info['steering_applied'] = True
            
            # Compute candidate steered vector
            delta = steering_vec.to(device=device, dtype=latent_vec.dtype)
            
            if norm_mode == 'activation':
                activation_norm = torch.norm(latent_vec)
                if activation_norm > 1e-6:
                    scaled_delta = delta * alpha * activation_norm
                else:
                    scaled_delta = delta * alpha
            else:
                scaled_delta = delta * alpha
            
            steered_candidate = latent_vec + scaled_delta
            
            # Introspection check: measure surprise
            surprise_metrics = compute_steering_surprise(latent_vec, steered_candidate)
            steering_info['surprise_metrics'].append(surprise_metrics)
            
            # Apply blending if surprise exceeds threshold
            if use_blending and surprise_metrics['total_surprise'] > acceptance_threshold:
                # Blend to reduce surprise
                blend_ratio = min(1.0, acceptance_threshold / (surprise_metrics['total_surprise'] + 1e-8))
                steered_vec = (1 - blend_ratio) * latent_vec + blend_ratio * steered_candidate
                steering_info['blend_ratios'].append(blend_ratio)
                
                print(f"  [BLEND] Step {step}: surprise={surprise_metrics['total_surprise']:.4f}, "
                      f"blend={blend_ratio:.3f}, cos_sim={surprise_metrics['cosine_similarity']:.4f}")
            else:
                steered_vec = steered_candidate
                if should_steer:
                    print(f"  [STEER] Step {step}: accepted (surprise={surprise_metrics['total_surprise']:.4f}, "
                          f"cos_sim={surprise_metrics['cosine_similarity']:.4f})")
            
            steering_info['post_steer'] = steered_vec.clone()
            
            if step == target_step or (steer_all and step == target_step):
                pre_norm = steering_info['pre_steer'].norm().item()
                post_norm = steering_info['post_steer'].norm().item()
                print(f"           pre_norm={pre_norm:.2f}, post_norm={post_norm:.2f}, "
                      f"change={post_norm - pre_norm:.2f}")
            
            latent_vec = steered_vec
        
        latent_vectors.append(latent_vec.clone())
        
        # Append for next step
        latent_embed = latent_vec.unsqueeze(1)
        current_hidden = torch.cat([current_hidden, latent_embed], dim=1)
    
    # Final forward pass
    for layer in layers:
        current_hidden = layer(current_hidden)[0]
    
    final_hidden = current_hidden[:, -1, :]
    logits = lm_head(final_hidden)
    
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
    norm_mode: str = 'activation',
    use_blending: bool = True,
    acceptance_threshold: float = 0.1,
) -> Optional[Dict]:
    """
    Run steering with persistent latent vector manipulation.
    
    Args:
        use_blending: If True, blend steering when surprise exceeds threshold
        acceptance_threshold: Maximum surprise before blending (lower = more conservative)
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
    baseline_latent_vectors = None
    
    for alpha in alphas:
        with torch.no_grad():
            steering_cfg = {
                'step': target_step,
                'vector': direction,
                'alpha': alpha,
                'steer_all': steer_all,
                'use_blending': use_blending,
            }
            
            logits, latent_vectors, steering_info = generate_with_introspection_aware_steering(
                model, tokenizer, input_tensor, n_hops, steering_cfg, 
                norm_mode=norm_mode, acceptance_threshold=acceptance_threshold
            )
            
            # Convert to CPU
            latent_cpu = [v.cpu() for v in latent_vectors]
            
            # Get logits
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
                baseline_latent_vectors = latent_cpu
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
            
            # Effect size
            effect_size = 0.0
            if baseline_logit_diff and abs(baseline_logit_diff) > 1e-6:
                effect_size = change / abs(baseline_logit_diff)
            
            # Check propagation
            steering_propagated = False
            propagation_metrics = {}
            if alpha > 0.0 and baseline_latent_vectors is not None:
                steering_propagated, propagation_metrics = check_steering_propagation(
                    latent_cpu, target_step, baseline_latent_vectors
                )
            
            # Store steered activations
            steered_activations = []
            if steering_info['steering_applied']:
                pre_norm = steering_info['pre_steer'].norm().item() if steering_info['pre_steer'] is not None else None
                post_norm = steering_info['post_steer'].norm().item() if steering_info['post_steer'] is not None else None
                steered_activations.append({
                    'latent_step': steering_info['steered_step'],
                    'alpha': alpha,
                    'pre_steer_norm': pre_norm,
                    'post_steer_norm': post_norm,
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
                "was_steered": steering_info['steering_applied'],
                "steering_propagated": steering_propagated,
                "propagation_metrics": propagation_metrics,
                "steered_activations": steered_activations,
                "latent_norms": [v.norm().item() for v in latent_cpu],
                "surprise_metrics": steering_info.get('surprise_metrics', []),
                "blend_ratios": steering_info.get('blend_ratios', []),
            }
    
    # Add metadata
    results['metadata'] = {
        'target_step': target_step,
        'steer_all': steer_all,
        'random_control': random_control,
        'norm_mode': norm_mode,
        'use_blending': use_blending,
        'acceptance_threshold': acceptance_threshold,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_probs': baseline_probs,
        'n_hops': n_hops,
        'concept_a': concept_a,
        'concept_b': concept_b,
        'token_id_a': token_id_a,
        'token_id_b': token_id_b,
        'question_length': len(question_ids),
        'prefix_len': len(prompt_ids),
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
    norm_mode: str = 'activation',
    use_blending: bool = True,
    acceptance_threshold: float = 0.1,
) -> Dict:
    """
    Run a systematic sweep over intervention parameters.
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
                random_control=random_control, norm_mode=norm_mode,
                use_blending=use_blending, acceptance_threshold=acceptance_threshold
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
                steer_all=True, random_control=random_control, norm_mode=norm_mode,
                use_blending=use_blending, acceptance_threshold=acceptance_threshold
            )
            if result:
                sweep_results[f'all_from_{step}'] = result
    
    elif sweep_dim == 'layer':
        if hasattr(model, 'base_causallm'):
            n_layers = len(model.base_causallm.transformer.h)
        elif hasattr(model, 'transformer'):
            n_layers = len(model.transformer.h)
        else:
            n_layers = 12
        
        print(f"Sweeping over {n_layers} layers...")
        for layer in [n_layers - 1]:
            result = run_steering_with_persistence(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, target_step=0,
                random_control=random_control, norm_mode=norm_mode,
                use_blending=use_blending, acceptance_threshold=acceptance_threshold
            )
            if result:
                sweep_results[f'layer_{layer}'] = result
    
    elif sweep_dim == 'both':
        if hasattr(model, 'base_causallm'):
            n_layers = len(model.base_causallm.transformer.h)
        elif hasattr(model, 'transformer'):
            n_layers = len(model.transformer.h)
        else:
            n_layers = 12
        
        print(f"Sweeping over steps and layers...")
        for step in range(min(n_hops, 3)):
            for layer in [0, n_layers // 2, n_layers - 1]:
                result = run_steering_with_persistence(
                    model, tokenizer, problem_row, concept_a, concept_b,
                    alphas=alphas, device=device, latent_id=latent_id,
                    start_id=start_id, end_id=end_id,
                    steering_vector=steering_vector, target_step=step,
                    random_control=random_control, norm_mode=norm_mode,
                    use_blending=use_blending, acceptance_threshold=acceptance_threshold
                )
                if result:
                    sweep_results[f'step{step}_layer{layer}'] = result
    
    return sweep_results


# Backward compatibility wrappers
def run_steering_multilevel(*args, **kwargs):
    """Redirect to the fixed implementation."""
    return run_steering_with_persistence(*args, **kwargs)


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
    """Backward-compatible wrapper."""
    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0]
    
    return run_steering_with_persistence(
        model, tokenizer, problem_row, concept_a, concept_b,
        alphas=alphas, device=device, latent_id=latent_id,
        start_id=start_id, end_id=end_id,
        steering_vector=steering_vector, target_step=latent_step_to_steer
    )