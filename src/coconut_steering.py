# src/coconut_steering.py

import torch
import torch.nn.functional as F
import numpy as np


def run_steering(
    model,
    tokenizer,
    problem_row,
    concept_a,
    concept_b,
    latent_step_to_steer=0,
    alphas=[0.0, 1.0, 2.0, 5.0, 10.0],
    device="cuda",
    latent_id=None,
    start_id=None,
    end_id=None,
    steering_vector=None,  # Added parameter
):
    """
    Steer Coconut by modifying the hidden state that gets copied into the latent token.
    
    Args:
        steering_vector: Pre-computed direction vector (optional)
    """
    
    # Get token IDs for concepts
    tokens_a = tokenizer.encode(" " + concept_a, add_special_tokens=False)
    tokens_b = tokenizer.encode(" " + concept_b, add_special_tokens=False)
    token_id_a = tokens_a[0] if tokens_a else None
    token_id_b = tokens_b[0] if tokens_b else None
    
    if token_id_a is None or token_id_b is None:
        print(f"Cannot tokenize {concept_a} or {concept_b}")
        return None
    
    # Use provided steering_vector or compute from embeddings
    if steering_vector is not None:
        if isinstance(steering_vector, np.ndarray):
            direction = torch.from_numpy(steering_vector).float()
        else:
            direction = steering_vector.float()
    else:
        # Fallback: use embedding difference
        embed_matrix = model.base_causallm.transformer.wte.weight
        embed_a = embed_matrix[token_id_a].detach().cpu()
        embed_b = embed_matrix[token_id_b].detach().cpu()
        direction = embed_a - embed_b
        direction = direction / torch.norm(direction)
    
    n_hops = problem_row["n_hops"]
    question_ids = tokenizer.encode(problem_row["question"] + "\n", add_special_tokens=True)
    input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
    input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
    attn_mask = torch.ones_like(input_tensor)
    position_ids = torch.arange(len(input_ids), device=device).unsqueeze(0)
    labels = input_tensor.clone()
    
    results = {}
    baseline_logit_diff = None
    
    for alpha in alphas:
        with torch.no_grad():
            class SelectiveHook:
                def __init__(self, target_pass_idx, steering_vec, alpha_val):
                    self.target_pass_idx = target_pass_idx
                    self.current_pass = 0
                    self.steering_vec = steering_vec
                    self.alpha_val = alpha_val
                    self.handle = None
                
                def hook_fn(self, module, input, output):
                    if self.current_pass == self.target_pass_idx:
                        hidden_states = output[0].clone()
                        
                        if self.steering_vec.device != hidden_states.device:
                            self.steering_vec = self.steering_vec.to(
                                device=hidden_states.device, dtype=hidden_states.dtype
                            )
                        
                        pos_to_steer = hidden_states.shape[1] - 1
                        activation_norm = torch.norm(hidden_states[0, pos_to_steer, :])
                        steer_norm = torch.norm(self.steering_vec)
                        if steer_norm > 0:
                            scaled = self.steering_vec * (self.alpha_val * activation_norm / steer_norm)
                        else:
                            scaled = self.steering_vec * self.alpha_val
                        
                        hidden_states[0, pos_to_steer, :] += scaled
                        self.current_pass += 1
                        return (hidden_states,) + output[1:]
                    
                    self.current_pass += 1
                    return output
            
            selective_hook = SelectiveHook(
                latent_step_to_steer, 
                direction.clone(), 
                alpha
            )
            
            target_layer = model.base_causallm.transformer.h[-1]
            handle = target_layer.register_forward_hook(selective_hook.hook_fn)
            
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
            
            if alpha == 0.0:
                baseline_logit_diff = logit_diff
            
            change = logit_diff - baseline_logit_diff if baseline_logit_diff is not None else 0
            
            results[alpha] = {
                "logit_diff": logit_diff,
                "prob_a": prob_a,
                "change": change
            }
    
    return results