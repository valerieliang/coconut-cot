import torch
import json
import pandas as pd
from src.model_utils import load_verbal_cot_model
from src.data_utils import load_split

def extract_verbal_representations(
    data_path="data/prontoqa_split.csv",
    split="test",
    output_path="data/verbal_reps.json",
    device="cuda",
    layer=-1,          # which transformer layer to extract from; -1 = final
):
    model, tokenizer = load_verbal_cot_model(device=device)
    df = load_split(data_path, split=split)

    # Register hook on the target layer to capture residual stream activations
    captured = {}
    def hook_fn(module, input, output):
        # output is (hidden_states, present_key_values, ...)
        # hidden_states: (batch, seq_len, d_model)
        captured["hidden"] = output[0].detach().cpu()

    target_layer = model.transformer.h[layer]
    handle = target_layer.register_forward_hook(hook_fn)

    results = []
    for _, row in df.iterrows():
        # Reconstruct the input text with delimiters — must match format_cot exactly
        steps = "".join(f" [STEP {i+1}] {s}" for i, s in enumerate(row["steps"]))
        text  = row["question"] + steps + " [ANS] " + row["answer"]

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        with torch.no_grad():
            model(**inputs)

        hidden = captured["hidden"][0]   # (seq_len, d_model)

        # Find token positions for each [STEP k] delimiter
        step_reps = []

        # Tokenize each step and the answer prefix to find boundaries.
        # Format: question\nstep1\nstep2\n...\n### answer
        # We want the activation at the last token of each step's content,
        # i.e. the token just before the \n that follows it.

        step_token_seqs = [
            tokenizer.encode(s + "\n", add_special_tokens=False)
            for s in row["steps"]
        ]
        ans_prefix_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode("\n### ", add_special_tokens=False)
        )

        for k in range(len(row["steps"])):
            # Find the start of step k's token sequence in the full token list
            step_toks = tokenizer.convert_ids_to_tokens(step_token_seqs[k])
            start_pos = find_subsequence(tokens, step_toks)

            if start_pos == -1:
                step_reps.append(None)
                print(f"WARNING: could not locate step {k+1} in problem {row.name}")
                continue

            # The step content ends at the token before the trailing \n
            # step_toks includes the \n, so the last content token is at:
            #   start_pos + len(step_toks) - 2
            # (subtract 1 for 0-indexing end, subtract 1 more to exclude the \n)
            last_content_pos = start_pos + len(step_toks) - 2

            if last_content_pos < 0 or last_content_pos >= hidden.shape[0]:
                step_reps.append(None)
                print(f"WARNING: step {k+1} boundary out of range in problem {row.name}")
                continue

            rep = hidden[last_content_pos].tolist()
            step_reps.append(rep)

        results.append({
            "problem_id":        int(row.name),
            "n_hops":            int(row["n_hops"]),
            "split":             row["split"],
            "ground_truth_steps": row["steps"],        # list of strings; truth value is in content
            "final_answer":      row["answer"],
            "step_reps":         step_reps,            # list of d_model vectors, one per step
        })

    handle.remove()   # always clean up hooks

    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} problems -> {output_path}")
    return results


def find_subsequence(token_list, subseq):
    """Return start index of first occurrence of subseq in token_list, or -1."""
    n, m = len(token_list), len(subseq)
    for i in range(n - m + 1):
        if token_list[i:i+m] == subseq:
            return i
    return -1


if __name__ == "__main__":
    extract_verbal_representations()