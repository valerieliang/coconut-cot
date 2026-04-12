# src/coconut_steering.py
"""
Fixed Coconut steering with persistent latent vector manipulation.
Includes fixes for propagation detection and steering strength.
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
    if premise is None or negated is None:
        return None
        
    v_plus = get_activation_at_latent_step(
        model, tokenizer, premise, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    v_minus = get_activation_at_latent_step(
        model, tokenizer, negated, target_step, n_hops,
        latent_id, start_id, end_id, device
    )
    
    if v_plus is None or v_minus is None:
        return None
    
    delta = v_minus - v_plus
    delta_norm = torch.norm(delta).item()
    if delta_norm > 1e-8:
        delta = delta / delta_norm
    
    return delta


def compute_steering_surprise(
    original_vec: torch.Tensor,
    steered_vec: torch.Tensor
) -> Dict:
    """Measure how "surprising" the steering would be to the model."""
    orig = original_vec.squeeze()
    steer = steered_vec.squeeze()
    if orig.dim() > 1:
        orig = orig[0]
    if steer.dim() > 1:
        steer = steer[0]
    
    orig_norm = torch.norm(orig)
    steer_norm = torch.norm(steer)
    
    cos_sim = torch.dot(orig / (orig_norm + 1e-8), steer / (steer_norm + 1e-8)).item()
    norm_ratio = steer_norm.item() / (orig_norm.item() + 1e-8)
    
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


def check_steering_propagation(
    latent_vectors: List[torch.Tensor],
    target_step: int,
    baseline_latents: Optional[List[torch.Tensor]] = None
) -> Tuple[bool, Dict]:
    """
    FIXED: Check if steering actually changed the trajectory.
    Now uses stricter criteria for propagation detection.
    """
    if len(latent_vectors) <= target_step:
        return False, {'error': 'Not enough latent vectors'}
    
    metrics = {}
    
    if baseline_latents is not None and len(baseline_latents) > target_step:
        steered = latent_vectors[target_step].squeeze()
        baseline_steered = baseline_latents[target_step].squeeze()
        
        if steered.dim() > 1:
            steered = steered[0]
        if baseline_steered.dim() > 1:
            baseline_steered = baseline_steered[0]
        
        steered_norm = torch.norm(steered).item()
        baseline_norm = torch.norm(baseline_steered).item()
        norm_ratio = steered_norm / (baseline_norm + 1e-8)
        
        metrics['steered_norm'] = steered_norm
        metrics['baseline_norm'] = baseline_norm
        metrics['norm_ratio'] = norm_ratio
        
        try:
            steer_vs_baseline_cos = torch.dot(
                steered / (torch.norm(steered) + 1e-8),
                baseline_steered / (torch.norm(baseline_steered) + 1e-8)
            ).item()
            metrics['steer_vs_baseline_cos'] = steer_vs_baseline_cos
            
            # FIXED: Stricter propagation criteria
            # Need at least 10% change in norm OR 0.1 change in cosine similarity
            norm_changed = abs(norm_ratio - 1.0) > 0.10  # Was 0.005
            vector_changed = steer_vs_baseline_cos < 0.90  # Was 0.995
            
            propagated = norm_changed or vector_changed
            
            # Check downstream effect if available
            if len(latent_vectors) > target_step + 1 and len(baseline_latents) > target_step + 1:
                next_vec = latent_vectors[target_step + 1].squeeze()
                baseline_next = baseline_latents[target_step + 1].squeeze()
                
                if next_vec.dim() > 1:
                    next_vec = next_vec[0]
                if baseline_next.dim() > 1:
                    baseline_next = baseline_next[0]
                
                next_vs_baseline_next_cos = torch.dot(
                    next_vec / (torch.norm(next_vec) + 1e-8),
                    baseline_next / (torch.norm(baseline_next) + 1e-8)
                ).item()
                metrics['next_vs_baseline_cos'] = next_vs_baseline_next_cos
                
                # Propagation requires downstream change
                next_changed = next_vs_baseline_next_cos < 0.95
                propagated = propagated and next_changed
                
        except Exception as e:
            metrics['error'] = str(e)
    else:
        propagated = False
        metrics['note'] = 'no baseline for comparison'
    
    return propagated, metrics


def generate_with_introspection_aware_steering(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    n_hops: int,
    steering_config: Dict,
    norm_mode: str = 'activation',
    acceptance_threshold: float = 0.5,  # FIXED: Higher threshold = more steering
    verbose: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
    """Custom generation with introspection-aware steering."""
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
        for layer in layers:
            current_hidden = layer(current_hidden)[0]
        
        latent_vec = current_hidden[:, -1, :].clone()
        
        # FIXED: Support per-step vectors
        step_vectors = steering_config.get('step_vectors', {})
        step_vec = step_vectors.get(step, steering_vec)
        
        should_steer = (steer_all and step >= target_step) or (step == target_step)
        
        if should_steer and step_vec is not None and alpha != 0.0:
            steering_info['steered_step'] = step
            steering_info['pre_steer'] = latent_vec.clone()
            steering_info['steering_applied'] = True
            
            delta = step_vec.to(device=device, dtype=latent_vec.dtype)
            
            # FIXED: Stronger steering scaling
            if norm_mode == 'activation':
                activation_norm = torch.norm(latent_vec)
                if activation_norm > 1e-6:
                    # Scale by activation norm to have consistent effect across steps
                    scaled_delta = delta * alpha * (activation_norm / torch.norm(delta) + 1e-8)
                else:
                    scaled_delta = delta * alpha
            else:
                scaled_delta = delta * alpha
            
            steered_candidate = latent_vec + scaled_delta
            surprise_metrics = compute_steering_surprise(latent_vec, steered_candidate)
            steering_info['surprise_metrics'].append(surprise_metrics)
            
            # FIXED: Higher acceptance threshold means more steering
            if use_blending and surprise_metrics['total_surprise'] > acceptance_threshold:
                # Less aggressive blending when surprise is high
                blend_ratio = min(1.0, acceptance_threshold / (surprise_metrics['total_surprise'] + 1e-8))
                # FIXED: Blend less when alpha is large (more aggressive steering)
                if alpha > 100:
                    blend_ratio = blend_ratio * 0.5  # Half the blending for large alphas
                steered_vec = (1 - blend_ratio) * latent_vec + blend_ratio * steered_candidate
                steering_info['blend_ratios'].append(blend_ratio)
                
                if verbose:
                    print(f"  [BLEND] Step {step}: surprise={surprise_metrics['total_surprise']:.4f}, "
                          f"blend={blend_ratio:.3f}, alpha={alpha}")
            else:
                steered_vec = steered_candidate
                if verbose:
                    print(f"  [STEER] Step {step}: alpha={alpha}, surprise={surprise_metrics['total_surprise']:.4f}")
            
            steering_info['post_steer'] = steered_vec.clone()
            latent_vec = steered_vec
        
        latent_vectors.append(latent_vec.clone())
        latent_embed = latent_vec.unsqueeze(1)
        current_hidden = torch.cat([current_hidden, latent_embed], dim=1)
    
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
    step_vectors: Optional[Dict[int, torch.Tensor]] = None,  # FIXED: New parameter
    target_step: int = 0,
    steer_all: bool = False,
    random_control: bool = False,
    norm_mode: str = 'activation',
    use_blending: bool = True,
    acceptance_threshold: float = 0.5,
    verbose: bool = False,
) -> Optional[Dict]:
    """Run steering with persistent latent vector manipulation."""
    if alphas is None:
        alphas = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0]  # FIXED: Larger alphas
    
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None
    
    if token_id_a is None or token_id_b is None:
        return None
    
    # FIXED: Handle step_vectors parameter
    if step_vectors is not None:
        # Use per-step vectors
        direction_dict = {}
        for step, vec in step_vectors.items():
            if isinstance(vec, np.ndarray):
                direction_dict[step] = torch.from_numpy(vec).float().to(device)
            else:
                direction_dict[step] = vec.float().to(device)
            
            if random_control:
                rand_vec = torch.randn_like(direction_dict[step])
                direction_dict[step] = rand_vec / (torch.norm(rand_vec) + 1e-8)
    else:
        # Use single vector
        if steering_vector is not None:
            if isinstance(steering_vector, np.ndarray):
                direction = torch.from_numpy(steering_vector).float()
            else:
                direction = steering_vector.float()
        else:
            direction = construct_contrast_vector_from_embeddings(model, tokenizer, concept_a, concept_b)
            if direction is None:
                return None
        
        if random_control:
            direction = torch.randn_like(direction)
            direction = direction / (torch.norm(direction) + 1e-8)
        
        direction = direction.to(device)
        direction_dict = {i: direction for i in range(10)}  # Replicate for all steps
    
    question = problem_row["question"]
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    
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
                'vector': steering_vector,
                'step_vectors': direction_dict,  # FIXED: Pass per-step vectors
                'alpha': alpha,
                'steer_all': steer_all,
                'use_blending': use_blending,
            }
            
            logits, latent_vectors, steering_info = generate_with_introspection_aware_steering(
                model, tokenizer, input_tensor, n_hops, steering_cfg, 
                norm_mode=norm_mode, acceptance_threshold=acceptance_threshold,
                verbose=verbose
            )
            
            latent_cpu = [v.cpu() for v in latent_vectors]
            
            logit_a = logits[0, token_id_a].item()
            logit_b = logits[0, token_id_b].item()
            logit_diff = logit_a - logit_b
            
            probs = F.softmax(torch.stack([logits[0, token_id_a], logits[0, token_id_b]]), dim=0)
            prob_a = probs[0].item()
            prob_b = probs[1].item()
            
            if alpha == 0.0:
                baseline_latent_vectors = latent_cpu
                baseline_logit_diff = logit_diff
                baseline_probs = {'prob_a': prob_a, 'prob_b': prob_b}
            
            change = logit_diff - baseline_logit_diff if baseline_logit_diff is not None else 0.0
            
            answer_flipped = False
            flipped_direction = None
            if baseline_probs is not None:
                base_ans = concept_a if baseline_probs['prob_a'] > 0.5 else concept_b
                curr_ans = concept_a if prob_a > 0.5 else concept_b
                answer_flipped = (base_ans != curr_ans)
                if answer_flipped:
                    flipped_direction = f"{base_ans} -> {curr_ans}"
            
            effect_size = 0.0
            if baseline_logit_diff and abs(baseline_logit_diff) > 1e-6:
                effect_size = change / abs(baseline_logit_diff)
            
            steering_propagated = False
            propagation_metrics = {}
            if alpha > 0.0 and baseline_latent_vectors is not None:
                steering_propagated, propagation_metrics = check_steering_propagation(
                    latent_cpu, target_step, baseline_latent_vectors
                )
            
            steered_activations = []
            if steering_info['steering_applied']:
                pre_norm = steering_info['pre_steer'].norm().item() if steering_info['pre_steer'] is not None else None
                post_norm = steering_info['post_steer'].norm().item() if steering_info['post_steer'] is not None else None
                steered_activations.append({
                    'latent_step': steering_info['steered_step'],
                    'alpha': alpha,
                    'pre_steer_norm': pre_norm,
                    'post_steer_norm': post_norm,
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
            }
    
    results['metadata'] = {
        'target_step': target_step,
        'steer_all': steer_all,
        'random_control': random_control,
        'norm_mode': norm_mode,
        'use_blending': use_blending,
        'baseline_logit_diff': baseline_logit_diff,
        'baseline_probs': baseline_probs,
        'n_hops': n_hops,
        'concept_a': concept_a,
        'concept_b': concept_b,
        'token_id_a': token_id_a,
        'token_id_b': token_id_b,
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
    step_vectors: Optional[Dict[int, torch.Tensor]] = None,  # FIXED: New parameter
    sweep_dim: str = 'step',
    random_control: bool = False,
    norm_mode: str = 'activation',
    use_blending: bool = True,
    acceptance_threshold: float = 0.5,
    steer_all: bool = False,
    verbose: bool = False,
) -> Dict:
    """Run a systematic sweep over intervention parameters."""
    if alphas is None:
        alphas = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0]  # FIXED: Larger alphas
    
    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))
    sweep_results = {}
    
    if sweep_dim == 'step':
        if verbose:
            print(f"Sweeping over {n_hops} reasoning steps...")
        for step in range(n_hops):
            result = run_steering_with_persistence(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector,
                step_vectors=step_vectors,  # FIXED: Pass through
                target_step=step,
                steer_all=steer_all,
                random_control=random_control, norm_mode=norm_mode,
                use_blending=use_blending, acceptance_threshold=acceptance_threshold,
                verbose=verbose
            )
            if result:
                sweep_results[f'step_{step}'] = result
    
    elif sweep_dim == 'all_steps':
        if verbose:
            print(f"Sweeping all steps from target onward...")
        for step in range(n_hops):
            result = run_steering_with_persistence(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector,
                step_vectors=step_vectors,  # FIXED: Pass through
                target_step=step,
                steer_all=True,
                random_control=random_control, norm_mode=norm_mode,
                use_blending=use_blending, acceptance_threshold=acceptance_threshold,
                verbose=verbose
            )
            if result:
                sweep_results[f'all_from_{step}'] = result
    
    return sweep_results


# Backward compatibility wrappers
def run_steering_multilevel(*args, **kwargs):
    return run_steering_with_persistence(*args, **kwargs)


def run_steering(
    model, tokenizer, problem_row, concept_a, concept_b,
    latent_step_to_steer=0, alphas=None, device="cuda",
    latent_id=None, start_id=None, end_id=None, steering_vector=None,
):
    if alphas is None:
        alphas = [0.0, 50.0, 100.0, 200.0, 500.0, 1000.0]  # FIXED
    
    return run_steering_with_persistence(
        model, tokenizer, problem_row, concept_a, concept_b,
        alphas=alphas, device=device, latent_id=latent_id,
        start_id=start_id, end_id=end_id,
        steering_vector=steering_vector, target_step=latent_step_to_steer
    )