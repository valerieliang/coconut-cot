"""
run_steering_eval.py
Injects steering vectors at each latent position and logs per-sample accuracy.
Run: python steering/run_steering_eval.py --config steering/steering_config.yaml
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_latent_token_index(input_ids, latent_pos, bot_token_id):
    ids = input_ids[0].tolist()
    count = 0
    for i, tok in enumerate(ids):
        if tok == bot_token_id:
            count += 1
            if count == latent_pos:
                return i + 1
    return None


def run_eval(cfg):
    from coconut import CoconutModel  # adjust to your module
    import transformers

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CoconutModel.from_pretrained(cfg["model_path"]).to(device).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("openai-community/gpt2")
    bot_token_id = tokenizer.convert_tokens_to_ids("<bot>")

    with open(cfg["val_path"]) as f:
        val_data = json.load(f)

    vectors = torch.load(Path(cfg["output_dir"]) / "steering_vectors.pt")

    out_path = Path(cfg["output_dir"]) / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["position", "alpha", "sample_idx", "correct", "n_steps"],
        )
        writer.writeheader()

        for sample_idx, sample in enumerate(val_data):
            n_steps = len(sample["steps"])
            enc = tokenizer(sample["question"], return_tensors="pt").to(device)

            # baseline (no hook)
            with torch.no_grad():
                out = model(**enc)
            pred = out.logits[0, -1].argmax().item()
            correct_tok = tokenizer.encode(sample["answer"])[-1]
            writer.writerow({
                "position": 0, "alpha": 0, "sample_idx": sample_idx,
                "correct": int(pred == correct_tok), "n_steps": n_steps,
            })

            # steering runs
            for pos in cfg["latent_positions"]:
                if pos > n_steps:
                    continue
                if pos not in vectors:
                    continue
                vec = vectors[pos].to(device)

                for alpha in cfg["alpha_sweep"]:
                    seq_idx = get_latent_token_index(enc["input_ids"], pos, bot_token_id)
                    if seq_idx is None:
                        continue

                    def hook_fn(module, inp, output, idx=seq_idx, v=vec, a=alpha):
                        h = output[0]
                        h[:, idx, :] = h[:, idx, :] + a * v
                        return (h,) + output[1:]

                    handle = model.transformer.h[cfg["layer"]].register_forward_hook(hook_fn)
                    with torch.no_grad():
                        out = model(**enc)
                    handle.remove()

                    pred = out.logits[0, -1].argmax().item()
                    writer.writerow({
                        "position": pos, "alpha": alpha, "sample_idx": sample_idx,
                        "correct": int(pred == correct_tok), "n_steps": n_steps,
                    })

            if sample_idx % 50 == 0:
                print(f"  {sample_idx}/{len(val_data)} samples done")

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_eval(load_config(args.config))