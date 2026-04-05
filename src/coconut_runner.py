import torch
import json
import sys, os
sys.path.insert(0, os.path.abspath("."))
from coconut.coconut import Coconut
from src.model_utils import load_coconut_model
from src.data_utils import load_split

def extract_coconut_representations(
    data_path="data/prontoqa_split.csv",
    split="test",
    output_path="data/coconut_reps.json",
    device="cuda",
):
    model, tokenizer = load_coconut_model(device=device)
    
    # Check if model loaded successfully
    if model is None or tokenizer is None:
        print("ERROR: Failed to load model or tokenizer")
        return None
    
    model.eval()

    df = load_split(data_path, split=split)

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")

    # Sanity-check: special tokens must resolve to real ids
    for name, tok_id in [("<latent>", latent_id), ("<start_latent>", start_id), ("<end_latent>", end_id)]:
        assert tok_id != tokenizer.unk_token_id, (
            f"Special token {name} not in tokenizer vocab. "
            f"Was the tokenizer saved with add_special_tokens=True in 03_train_coconut.py?"
        )

    results        = []
    n_missing_reps = 0

    for idx, row in df.iterrows():
        n_hops = int(row["n_hops"])

        # ── Build input: question + <start_latent> + [<latent>]*n_hops + <end_latent> ──
        question_ids   = tokenizer.encode(row["question"] + "\n", add_special_tokens=True)
        input_ids_list = question_ids + [start_id] + [latent_id] * n_hops + [end_id]

        print(f"\nProblem {idx}: n_hops={n_hops}, seq_len={len(input_ids_list)}")
        print(f"  Question tokens: {len(question_ids)}")
        print(f"  Special tokens: start(1) + latent({n_hops}) + end(1) = {n_hops+2}")

        # Pre-compute the index of every <latent> token
        latent_positions = [i for i, t in enumerate(input_ids_list) if t == latent_id]
        print(f"  Latent positions: {latent_positions}")
        
        assert len(latent_positions) == n_hops, (
            f"problem {idx}: expected {n_hops} latent positions, "
            f"found {len(latent_positions)} — check input construction."
        )

        input_tensor = torch.tensor(input_ids_list).unsqueeze(0).to(device)   # (1, seq)
        attn_mask    = torch.ones_like(input_tensor)
        position_ids = torch.arange(input_tensor.shape[1], device=device).unsqueeze(0)

        # Create dummy labels to satisfy Coconut.forward (it expects non-None labels)
        # We'll use a dummy label tensor that won't affect computation much
        dummy_labels = torch.full_like(input_tensor, -100)  # -100 is ignored in loss

        # ── Hook: extract hidden state at the k-th <latent> position ──────
        thought_buffer = []
        call_count     = [0]
        
        # We need to track the actual sequence length at each hook call
        # and find where the latent tokens are in the current sequence
        current_latent_positions = None

        def hook_fn(module, input, output):
            nonlocal current_latent_positions
            hidden = output[0].detach().cpu()   # (batch, seq_len, d_model)
            k = call_count[0]
            seq_len = hidden.shape[1]
            
            # On first call, determine where latent tokens are in this sequence
            if k == 0 and current_latent_positions is None:
                # The sequence might have been modified, let's find latent token positions
                # We need access to the current input_ids - but we don't have them here
                # Alternative: assume the model preserves the relative order and just find
                # the first few positions that might contain thought vectors
                current_latent_positions = []
                # Look for positions where the hidden state might be a thought vector
                # This is a heuristic - in practice, the model might compress the sequence
                print(f"    Hook call {k}: seq_len={seq_len}, original latent positions={latent_positions}")
            
            if k < n_hops:
                # For Coconut, thought vectors are typically at the last position
                # of the current sequence (since <end_latent> triggers the next step)
                # Let's try extracting from the last position
                tok_pos = seq_len - 1
                if tok_pos >= 0:
                    thought_vector = hidden[0, tok_pos, :].clone()
                    thought_buffer.append(thought_vector)
                    print(f"    Hook call {k}: extracted at position {tok_pos} (seq_len={seq_len})")
                else:
                    print(f"    Hook call {k}: ERROR - no valid position (seq_len={seq_len})")
            else:
                print(f"    Hook call {k}: final pass (k={k} >= n_hops={n_hops})")
            
            call_count[0] += 1

        # Register hook on the last transformer layer
        handle = model.base_causallm.transformer.h[-1].register_forward_hook(hook_fn)

        with torch.no_grad():
            # Pass dummy labels instead of None to avoid the NoneType error
            outputs = model(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                labels=dummy_labels,  # Use dummy labels instead of None
                position_ids=position_ids,
            )

        handle.remove()

        print(f"  Total hook calls: {call_count[0]}")
        print(f"  Thought vectors extracted: {len(thought_buffer)}/{n_hops}")

        # Validate extraction count
        if len(thought_buffer) != n_hops:
            print(f"  WARNING: expected {n_hops} thought vectors, captured {len(thought_buffer)}")
            n_missing_reps += 1

        thought_reps = [t.tolist() for t in thought_buffer[:n_hops]]

        results.append({
            "problem_id":         int(idx),
            "n_hops":             n_hops,
            "n_thought_reps":     len(thought_reps),
            "split":              row["split"],
            "ground_truth_steps": row["steps"],
            "final_answer":       row["answer"],
            "thought_reps":       thought_reps,
        })
        

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} problems → {output_path}")
    if n_missing_reps:
        print(
            f"  ↳ {n_missing_reps} problems had incomplete thought_reps. "
            f"If this is all problems, your checkpoint was likely trained with fixed K. "
            f"Set n_hops = K (= max_latent_stage from 03_train_coconut.py) for all rows."
        )
    return results


if __name__ == "__main__":
    extract_coconut_representations()