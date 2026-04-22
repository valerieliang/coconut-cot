"""
extract_vectors.py
Builds one steering vector per latent position using contrastive mean difference.
Run: python steering/extract_vectors.py --config steering/steering_config.yaml
"""
import argparse
import json
import torch
import yaml
from pathlib import Path


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_latent_token_index(input_ids, latent_pos, bot_token_id, eot_token_id):
    """Return the sequence index of the latent_pos-th continuous thought token."""
    ids = input_ids[0].tolist()
    count = 0
    for i, tok in enumerate(ids):
        if tok == bot_token_id:
            count += 1
            if count == latent_pos:
                return i + 1  # the token immediately after <bot>
    return None


def extract_vectors(cfg):
    from coconut import CoconutModel  # adjust import to match your module
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = CoconutModel.from_pretrained(cfg["model_path"]).to(device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")

    with open(cfg["train_path"]) as f:
        data = json.load(f)

    # limit to n_contrast samples that have enough steps
    max_pos = max(cfg["latent_positions"])
    data = [d for d in data if len(d["steps"]) >= max_pos][: cfg["n_contrast"]]

    vectors = {}

    for pos in cfg["latent_positions"]:
        correct_acts, wrong_acts = [], []

        def hook_fn(module, inp, output, store):
            idx = get_latent_token_index(
                current_input_ids, pos,
                tokenizer.convert_tokens_to_ids("<bot>"),
                tokenizer.convert_tokens_to_ids("<eot>"),
            )
            if idx is not None:
                store.append(output[0][0, idx, :].detach().cpu())

        for sample in data:
            # --- correct path ---
            store_c = []
            handle = model.transformer.h[cfg["layer"]].register_forward_hook(
                lambda m, i, o, s=store_c: hook_fn(m, i, o, s)
            )
            enc = tokenizer(sample["question"], return_tensors="pt").to(device)
            global current_input_ids
            current_input_ids = enc["input_ids"]
            with torch.no_grad():
                model(**enc)
            handle.remove()
            if store_c:
                correct_acts.append(store_c[0])

            # --- counterfactual path (swap answer to neg_target) ---
            store_w = []
            handle = model.transformer.h[cfg["layer"]].register_forward_hook(
                lambda m, i, o, s=store_w: hook_fn(m, i, o, s)
            )
            neg_sample = dict(sample)
            neg_sample["answer"] = sample.get("neg_answer", sample["answer"] + " [neg]")
            enc = tokenizer(neg_sample["question"], return_tensors="pt").to(device)
            current_input_ids = enc["input_ids"]
            with torch.no_grad():
                model(**enc)
            handle.remove()
            if store_w:
                wrong_acts.append(store_w[0])

        if correct_acts and wrong_acts:
            v = torch.stack(correct_acts).mean(0) - torch.stack(wrong_acts).mean(0)
            vectors[pos] = v
            print(f"  position {pos}: vector norm = {v.norm():.4f}")

    out_path = Path(cfg["output_dir"]) / "steering_vectors.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, out_path)
    print(f"Saved vectors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    extract_vectors(load_config(args.config))