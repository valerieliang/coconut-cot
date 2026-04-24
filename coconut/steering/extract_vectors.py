"""
extract_vectors.py
Builds one steering vector per latent position using contrastive mean difference.

For each sample, two forward passes are run:
  - correct path:        the sample as-is, reasoning toward target answer
  - counterfactual path: the sample with the question reframed toward neg_target

The steering vector for position Li is:
  v_i = mean(hidden states at Li | correct) - mean(hidden states at Li | wrong)

Run:
  python steering/extract_vectors.py --config steering/steering_config.yaml
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# ── token IDs (must match training) ─────────────────────────────────────────
LATENT_TOKEN_ID  = 50257
START_LATENT_ID  = 50258
END_LATENT_ID    = 50259
EOS_TOKEN_ID     = 50256
VOCAB_SIZE       = 50260


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
    """
    Return the sequence index of the latent_pos-th <|latent|> token (1-indexed).
    The model places latent tokens between <|start-latent|> and <|end-latent|>.
    """
    count = 0
    for i, tok in enumerate(input_ids):
        if tok == LATENT_TOKEN_ID:
            count += 1
            if count == latent_pos:
                return i
    return None


def build_prompt(question: str) -> str:
    return question + "\n<|start-latent|>"


def get_hidden(model, tokenizer, prompt: str, layer: int, latent_pos: int, device):
    """Run one forward pass and return the hidden state at the target latent index."""
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"][0].tolist()

    # The latent tokens are appended during the forward pass by Coconut.
    # We hook after the model has consumed the prompt up to <|start-latent|>,
    # then capture the hidden state at the latent token position.
    captured = []

    def hook(module, inp, output):
        h = output[0]  # (batch, seq, hidden)
        idx = get_latent_index(input_ids, latent_pos)
        if idx is not None and idx < h.shape[1]:
            captured.append(h[0, idx, :].detach().cpu())

    handle = model.base_causallm.transformer.h[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model.base_causallm(**enc)
    finally:
        handle.remove()

    return captured[0] if captured else None


def build_neg_prompt(sample: dict) -> str:
    """
    Construct a counterfactual prompt by appending a hint toward neg_target.
    Since ProsQA questions contain the answer options, we reorder them so
    neg_target appears first, nudging the model toward the wrong branch.
    """
    q = sample["question"]
    target_word = sample["answer"].split()[-1].rstrip(".")
    neg_word = sample.get("neg_answer", "")

    if not neg_word:
        # derive neg_answer from neg_target index if not stored
        idx_to_sym = sample.get("idx_to_symbol", [])
        neg_idx = sample.get("neg_target")
        if neg_idx is not None and neg_idx < len(idx_to_sym):
            neg_word = idx_to_sym[neg_idx]

    # swap the two answer options in the question string if possible
    if target_word and neg_word and target_word in q and neg_word in q:
        q_neg = q.replace(
            f"{target_word} or {neg_word}",
            f"{neg_word} or {target_word}",
        ).replace(
            f"{neg_word} or {target_word}",
            f"{neg_word} or {target_word}",  # already correct order
        )
        return q_neg + "\n<|start-latent|>"

    return build_prompt(sample["question"])  # fallback: same prompt


def extract_vectors(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    layer     = cfg["layer"]

    with open(cfg["train_path"]) as f:
        data = json.load(f)

    max_pos = max(cfg["latent_positions"])
    data = [d for d in data if len(d.get("steps", [])) >= max_pos]
    data = data[: cfg["n_contrast"]]
    print(f"Using {len(data)} samples for vector extraction (layer {layer})")

    vectors = {}

    for pos in cfg["latent_positions"]:
        correct_acts, wrong_acts = [], []

        for i, sample in enumerate(data):
            # correct path
            h_c = get_hidden(model, tokenizer, build_prompt(sample["question"]),
                             layer, pos, device)
            # counterfactual path
            h_w = get_hidden(model, tokenizer, build_neg_prompt(sample),
                             layer, pos, device)

            if h_c is not None:
                correct_acts.append(h_c)
            if h_w is not None:
                wrong_acts.append(h_w)

            if (i + 1) % 100 == 0:
                print(f"  pos {pos}: {i+1}/{len(data)} samples")

        if correct_acts and wrong_acts:
            v = torch.stack(correct_acts).mean(0) - torch.stack(wrong_acts).mean(0)
            vectors[pos] = v
            print(f"  position {pos}: vector norm = {v.norm():.4f}  "
                  f"({len(correct_acts)} correct, {len(wrong_acts)} wrong)")
        else:
            print(f"  position {pos}: no activations captured — check latent indexing")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steering_vectors.pt"
    torch.save(vectors, out_path)
    print(f"\nSaved {len(vectors)} vectors to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    extract_vectors(load_config(args.config))