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
    """

    def get_activation_at_step(text: str) -> Optional[torch.Tensor]:
        """Extract activation at specified step and layer."""
        question_ids = tokenizer.encode(text + "\n", add_special_tokens=True)
        input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)

        # BUG FIX: skip start_id (+1) to point at the correct latent token.
        # Previously this was question_len + step_to_extract, which pointed at
        # start_id instead of the first latent token.
        question_len = len(question_ids)
        target_pos = question_len + 1 + step_to_extract

        activation = None

        def hook_fn(module, input, output):
            nonlocal activation
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            activation = hidden_states[0, target_pos, :].detach().clone()

        if hasattr(model, 'base_causallm'):
            target_layer = model.base_causallm.transformer.h[layer]
        elif hasattr(model, 'transformer'):
            target_layer = model.transformer.h[layer]
        else:
            print("Cannot find transformer layers")
            return None

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
    Fallback method when activation-based construction is not available.
    """
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
    elif hasattr(model, 'word_embeddings'):
        embed_matrix = model.word_embeddings.weight
    else:
        print("Cannot find embedding matrix")
        return None

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
    """

    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    if steer_config is None:
        steer_config = {
            'step': 0,
            'layer': -1,
            'position': 'latent',
            'normalization': 'activation',
            'steer_all_steps': False
        }

    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None

    if token_id_a is None or token_id_b is None:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None

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

    if hasattr(model, 'base_causallm'):
        layers = model.base_causallm.transformer.h
    elif hasattr(model, 'transformer'):
        layers = model.transformer.h
    else:
        print("Cannot find transformer layers")
        return None

    for alpha in alphas:
        with torch.no_grad():

            # BUG FIX: The original code used a current_step counter inside the
            # hook to track which latent step was being processed.  However the
            # hook fires exactly ONCE per forward pass (one call per registered
            # layer), so current_step was always 0 when the check ran, meaning
            # target_step > 0 was never steered.  The latent token positions are
            # deterministic (question_len + 1 + step), so we calculate them
            # directly from target_step instead.
            target_step = steer_config.get('step', 0)
            position_mode = steer_config.get('position', 'latent')
            norm_mode = steer_config.get('normalization', 'activation')
            steer_all = steer_config.get('steer_all_steps', False)

            was_steered_flag = [False]
            steered_activations_list = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0].clone()
                else:
                    hidden_states = output.clone()

                sv = direction.clone()
                if sv.device != hidden_states.device:
                    sv = sv.to(device=hidden_states.device, dtype=hidden_states.dtype)

                steps_to_apply = list(range(n_hops)) if steer_all else [target_step]

                for s in steps_to_apply:
                    if s >= n_hops:
                        continue

                    if position_mode == 'last':
                        pos_to_steer = hidden_states.shape[1] - 1
                    elif position_mode == 'latent':
                        # +1 to skip start_id token
                        pos_to_steer = len(question_ids) + 1 + s
                    else:
                        pos_to_steer = hidden_states.shape[1] - 1

                    if pos_to_steer >= hidden_states.shape[1]:
                        continue

                    pre_steer_activation = hidden_states[0, pos_to_steer, :].clone()

                    if norm_mode == 'activation':
                        activation_norm = torch.norm(pre_steer_activation)
                        steer_norm = torch.norm(sv)
                        if steer_norm > 1e-6 and activation_norm > 1e-6:
                            scaled = sv * (alpha * activation_norm / steer_norm)
                        else:
                            scaled = sv * alpha
                    else:
                        scaled = sv * alpha

                    hidden_states[0, pos_to_steer, :] += scaled
                    was_steered_flag[0] = True

                    steered_activations_list.append({
                        'step': s,
                        'alpha': alpha,
                        'position': pos_to_steer,
                        'pre_steer_norm': torch.norm(pre_steer_activation).item(),
                        'steering_magnitude': torch.norm(scaled).item(),
                    })

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

            target_layer_idx = steer_config.get('layer', -1)
            if target_layer_idx == -1:
                target_layer_idx = len(layers) - 1
            target_layer = layers[target_layer_idx]
            handle = target_layer.register_forward_hook(hook_fn)

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
                "was_steered": was_steered_flag[0],
                "steered_activations": steered_activations_list,
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
    """

    if alphas is None:
        alphas = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]

    n_hops = problem_row.get("n_hops", len(problem_row.get("steps", [])))

    if hasattr(model, 'base_causallm'):
        n_layers = len(model.base_causallm.transformer.h)
    elif hasattr(model, 'transformer'):
        n_layers = len(model.transformer.h)
    else:
        n_layers = 12

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
            result = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
            if result:
                sweep_results[f'layer_{layer}'] = result

    elif sweep_dim == 'both':
        # ASCII replacement: 'x' instead of the multiplication sign
        print(f"Sweeping over {n_hops} steps x {n_layers} layers...")
        for step in range(n_hops):
            for layer in range(n_layers):
                config = {
                    'step': step,
                    'layer': layer,
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
                    sweep_results[f'step{step}_layer{layer}'] = result

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
            result = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
            if result:
                sweep_results[f'position_{pos_mode}'] = result

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
            result = run_steering_multilevel(
                model, tokenizer, problem_row, concept_a, concept_b,
                alphas=alphas, device=device, latent_id=latent_id,
                start_id=start_id, end_id=end_id,
                steering_vector=steering_vector, steer_config=config
            )
            if result:
                sweep_results[f'norm_{norm_mode}'] = result

    elif sweep_dim == 'all_steps':
        print("Testing steering at all steps simultaneously...")
        config = {
            'step': 0,
            'layer': -1,
            'position': 'latent',
            'normalization': 'activation',
            'steer_all_steps': True
        }
        result = run_steering_multilevel(
            model, tokenizer, problem_row, concept_a, concept_b,
            alphas=alphas, device=device, latent_id=latent_id,
            start_id=start_id, end_id=end_id,
            steering_vector=steering_vector, steer_config=config
        )
        if result:
            sweep_results['all_steps'] = result

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