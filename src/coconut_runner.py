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
    df = load_split(data_path, split=split)

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")

    results = []

    for _, row in df.iterrows():
        n_hops = int(row["n_hops"])

        question_ids = tokenizer.encode(
            row["question"] + "\n", add_special_tokens=True
        )
        input_ids = (
            question_ids
            + [start_id]
            + [latent_id] * n_hops
            + [end_id]
        )
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
        attn_mask    = torch.ones_like(input_tensor)

        with torch.no_grad():
            _, filled_embeds = model.generate(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                max_new_tokens=20,
                output_embedding=True,
            )

        latent_start_pos = len(question_ids) + 1
        latent_positions  = list(range(latent_start_pos, latent_start_pos + n_hops))

        thought_reps = []
        for pos in latent_positions:
            if pos < filled_embeds.shape[1]:
                vec = filled_embeds[0, pos, :].detach().cpu().tolist()
                thought_reps.append(vec)
            else:
                thought_reps.append(None)
                print(f"WARNING: latent pos {pos} out of range for problem {row.name}",
                      flush=True)

        if len(thought_reps) != n_hops:
            print(f"WARNING: expected {n_hops} thought reps, got {len(thought_reps)} "
                  f"for problem {row.name}", flush=True)

        results.append({
            "problem_id":         int(row.name),
            "n_hops":             n_hops,
            "split":              row["split"],
            "ground_truth_steps": row["steps"],
            "final_answer":       row["answer"],
            "thought_reps":       thought_reps,
        })

    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} problems -> {output_path}", flush=True)
    return results

if __name__ == "__main__":
    extract_coconut_representations()