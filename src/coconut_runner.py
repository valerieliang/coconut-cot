import torch
import json
import pandas as pd
from src.model_utils import load_coconut_model
from src.data_utils import load_split

def extract_coconut_representations(
    data_path="data/prontoqa_split.csv",
    split="test",
    output_path="data/coconut_reps.json",
    device="cuda",
):
    model, tokenizer = load_coconut_model(device=device)
    df = load_split(data_path, split=split)

    # Hook captures the final layer hidden state at each continuous thought step.
    # Coconut's forward pass calls the transformer repeatedly in latent mode;
    # we capture the last-token hidden state at each such call.
    thought_buffer = []

    def hook_fn(module, input, output):
        # Capture last token of each forward pass through the final layer
        hidden = output[0].detach().cpu()   # (batch, seq_len, d_model)
        thought_buffer.append(hidden[0, -1, :].clone())   # scalar per step: (d_model,)

    target_layer = model.transformer.h[-1]

    results = []
    for _, row in df.iterrows():
        thought_buffer.clear()
        handle = target_layer.register_forward_hook(hook_fn)

        # Coconut takes the question only as input — it reasons in latent space
        # for K steps before decoding the answer. The repo's generate() or
        # forward() method drives the recurrence; verify the exact call signature
        # in coconut/src/ before running.
        inputs = tokenizer(row["question"], return_tensors="pt").to(device)

        with torch.no_grad():
            # ↓ replace with the repo's actual generation call if different
            _ = model.generate(
                **inputs,
                max_new_tokens=50,
                num_thought_steps=row["n_hops"],   # verify param name in repo
            )

        handle.remove()

        # thought_buffer now contains one vector per continuous thought step.
        # The hook fires on every forward pass through the final layer, which
        # includes non-thought passes (e.g. the initial encode). Filter to only
        # the K thought steps — in practice this means skipping the first
        # hook call if the model encodes the prompt separately.
        # VERIFY this offset against the repo's forward pass logic.
        thought_reps = [t.tolist() for t in thought_buffer]

        results.append({
            "problem_id":         int(row.name),
            "n_hops":             int(row["n_hops"]),
            "split":              row["split"],
            "ground_truth_steps": row["steps"],
            "final_answer":       row["answer"],
            "thought_reps":       thought_reps,   # list of d_model vectors, one per thought step
        })

    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} problems -> {output_path}")
    return results


if __name__ == "__main__":
    extract_coconut_representations()