"""
run_steering_eval.py
Injects steering vectors at each latent position and logs per-sample accuracy.

For each (latent_position, alpha) pair, a forward hook adds:
  alpha * v_i  to the residual stream at the i-th latent token position.

Outputs results.csv with columns:
  position, alpha, sample_idx, correct, n_steps, chosen, expected

Run:
  python steering/run_steering_eval.py --config steering/steering_config.yaml
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_model(cfg, device):
    from coconut.coconut import Coconut

    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.resize_token_embeddings(VOCAB_SIZE)

    model = Coconut(
        base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID,
        start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID,
        eos_token_id=EOS_TOKEN_ID,
    )

    ckpt = torch.load(cfg["model_path"], map_location="cpu")
    if "base_causallm" in ckpt:
        ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.", ""): v for k, v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def load_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "additional_special_tokens": [
            "<|latent|>", "<|start-latent|>", "<|end-latent|>"
        ]
    })
    tok.pad_token = tok.eos_token
    return tok


def get_latent_index(input_ids: list[int], latent_pos: int) -> int | None:
    count = 0
    for i, tok in enumerate(input_ids):
        if tok == LATENT_TOKEN_ID:
            count += 1
            if count == latent_pos:
                return i
    return None


def get_answer_token(tokenizer, answer: str) -> int:
    """Return the token ID of the last word in the answer string."""
    last_word = answer.strip().rstrip(".").split()[-1]
    tokens = tokenizer.encode(" " + last_word)
    return tokens[-1]


def run_inference(model, enc, device) -> int:
    """Run a forward pass and return the argmax token at the final position."""
    with torch.no_grad():
        out = model.base_causallm(**enc)
    return out.logits[0, -1].argmax().item()


def run_eval(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    layer     = cfg["layer"]

    vectors_path = Path(cfg["output_dir"]) / "steering_vectors.pt"
    vectors = torch.load(vectors_path, map_location=device)
    print(f"Loaded vectors for positions: {sorted(vectors.keys())}")

    with open(cfg["val_path"]) as f:
        val_data = json.load(f)

    out_path = Path(cfg["output_dir"]) / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "position", "alpha", "sample_idx",
            "correct", "n_steps", "chosen_token", "expected_token",
        ])
        writer.writeheader()

        for sample_idx, sample in enumerate(val_data):
            n_steps       = len(sample.get("steps", []))
            expected_tok  = get_answer_token(tokenizer, sample["answer"])
            prompt        = sample["question"] + "\n<|start-latent|>"
            enc           = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids     = enc["input_ids"][0].tolist()

            # ── baseline (no hook) ───────────────────────────────────────────
            pred = run_inference(model, enc, device)
            writer.writerow({
                "position": 0, "alpha": 0, "sample_idx": sample_idx,
                "correct": int(pred == expected_tok), "n_steps": n_steps,
                "chosen_token": pred, "expected_token": expected_tok,
            })

            # ── steered runs ─────────────────────────────────────────────────
            for pos in cfg["latent_positions"]:
                if pos > n_steps:
                    continue
                if pos not in vectors:
                    continue

                seq_idx = get_latent_index(input_ids, pos)
                if seq_idx is None:
                    continue

                vec = vectors[pos].to(device)

                for alpha in cfg["alpha_sweep"]:
                    def hook_fn(module, inp, output,
                                idx=seq_idx, v=vec, a=float(alpha)):
                        h = output[0].clone()
                        h[:, idx, :] = h[:, idx, :] + a * v
                        return (h,) + output[1:]

                    handle = model.base_causallm.transformer.h[layer]\
                                  .register_forward_hook(hook_fn)
                    pred = run_inference(model, enc, device)
                    handle.remove()

                    writer.writerow({
                        "position": pos, "alpha": alpha,
                        "sample_idx": sample_idx,
                        "correct": int(pred == expected_tok),
                        "n_steps": n_steps,
                        "chosen_token": pred,
                        "expected_token": expected_tok,
                    })

            if (sample_idx + 1) % 50 == 0:
                print(f"  {sample_idx + 1}/{len(val_data)} samples done")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_eval(load_config(args.config))