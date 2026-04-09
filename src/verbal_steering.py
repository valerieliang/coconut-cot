# src/verbal_steering.py
"""Steering for verbal CoT models using activation patching."""

import torch
import torch.nn.functional as F
import numpy as np


def run_verbal_steering(
    model,
    tokenizer,
    problem_row,
    concept_a,
    concept_b,
    step_to_steer=0,
    alphas=[0.0, 1.0, 2.0, 5.0, 10.0],
    device="cuda",
    layer=-1,
    steering_vector=None,
):
    """
    Steer verbal CoT by modifying the hidden state at the end of a reasoning step.
    
    Args:
        steering_vector: Pre-computed direction vector
    """
    
    if steering_vector is None:
        print("No steering direction provided")
        return None
    
    # Convert to tensor
    if isinstance(steering_vector, np.ndarray):
        direction = torch.from_numpy(steering_vector).float()
    else:
        direction = steering_vector.float()
    
    # Tokenize concepts
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
    
    text_parts = [question]
    for i in range(step_to_steer + 1):
        if i < len(steps):
            text_parts.append(steps[i])
    text = "\n".join(text_parts)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    results = {}
    baseline_logit_diff = None
    
    for alpha in alphas:
        with torch.no_grad():
            class SteeringHook:
                def __init__(self, direction_vec, alpha_val):
                    self.direction = direction_vec
                    self.alpha = alpha_val
                
                def hook_fn(self, module, input, output):
                    hidden_states = output[0].clone()
                    
                    if self.direction.device != hidden_states.device:
                        self.direction = self.direction.to(
                            device=hidden_states.device, dtype=hidden_states.dtype
                        )
                    
                    # Steer the last position
                    pos = hidden_states.shape[1] - 1
                    
                    # Normalize steering magnitude relative to activation
                    activation_norm = torch.norm(hidden_states[0, pos, :])
                    steer_norm = torch.norm(self.direction)
                    if steer_norm > 0:
                        scaled = self.direction * (self.alpha * activation_norm / steer_norm)
                    else:
                        scaled = self.direction * self.alpha
                    
                    hidden_states[0, pos, :] += scaled
                    
                    return (hidden_states,) + output[1:]
            
            hook = SteeringHook(direction.clone(), alpha)
            
            # Register hook on specified layer
            target_layer = model.transformer.h[layer]
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
            
            if alpha == 0.0:
                baseline_logit_diff = logit_diff
            
            change = logit_diff - baseline_logit_diff if baseline_logit_diff is not None else 0
            
            results[alpha] = {
                "logit_diff": logit_diff,
                "prob_a": prob_a,
                "change": change
            }
    
    return results