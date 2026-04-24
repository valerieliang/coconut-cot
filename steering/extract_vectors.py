
"""
extract_vectors.py
Builds one steering vector per latent pass using contrastive mean difference.

From coconut.py:
  - forward() runs `max_n_latents` sequential passes through base_causallm
  - after each pass, the last hidden state is fed back as the embedding for
    the next latent slot: tensor_list[b][token_idx] = hidden_states[..., token_idx-1, :]
  - "steering position i" means intercepting that hidden state feedback after pass i

We monkey-patch Coconut.forward() to capture the continuous thought vector
at each pass — that's the true latent representation, not a token index.

Run:
  python steering/extract_vectors.py --config steering/steering_config.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coconut"))

import torch
import yaml
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer

LATENT_TOKEN_ID = 50257
START_LATENT_ID = 50258
END_LATENT_ID   = 50259
EOS_TOKEN_ID    = 50256
VOCAB_SIZE      = 50260


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_tokenizer():
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.add_special_tokens({
        "additional_special_tokens": [
            "<|latent|>", "<|start-latent|>", "<|end-latent|>"
        ]
    })
    tok.pad_token = tok.eos_token
    return tok


def load_model(cfg, device):
    from coconut import Coconut

    base = GPT2LMHeadModel.from_pretrained("gpt2")
    base.resize_token_embeddings(VOCAB_SIZE)
    model = Coconut(
        base_causallm=base,
        latent_token_id=LATENT_TOKEN_ID,
        start_latent_id=START_LATENT_ID,
        end_latent_id=END_LATENT_ID,
        eos_token_id=EOS_TOKEN_ID,
    )
    ckpt = torch.load(os.path.expanduser(cfg["model_path"]), map_location="cpu")
    if "base_causallm" in ckpt:
        ckpt = ckpt["base_causallm"]
    ckpt = {k.replace("base_causallm.", ""): v for k, v in ckpt.items()}
    model.base_causallm.load_state_dict(ckpt, strict=False)
    model.to(device).eval()
    return model


def build_input(tokenizer, question: str, n_latent: int, device):
    """Build the full token sequence Coconut expects during training/eval."""
    q_ids = tokenizer.encode(question)
    seq = (
        q_ids
        + [START_LATENT_ID]
        + [LATENT_TOKEN_ID] * n_latent
        + [END_LATENT_ID]
    )
    input_ids      = torch.tensor([seq], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    labels         = input_ids.clone()
    position_ids   = torch.arange(
        0, input_ids.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0)
    return input_ids, attention_mask, labels, position_ids


def forward_with_capture(model, input_ids, attention_mask, labels,
                         position_ids, capture_pass: int,
                         steer_vec=None, steer_alpha: float = 0.0):
    """
    Re-implementation of Coconut.forward() that:
      - captures the continuous thought vector produced at `capture_pass`
      - optionally injects `steer_alpha * steer_vec` at that same pass

    Returns (captured_vector, logits_last_token)
    """
    from coconut import Outputs

    captured = [None]
    logits_list = []

    latent_indices = (input_ids == model.latent_token_id).nonzero()
    latent_lists = [
        [idx[1].item() for idx in latent_indices if idx[0] == i]
        for i in range(input_ids.shape[0])
    ]
    max_n_latents = max([len(l) for l in latent_lists])

    next_compute_range = (0, input_ids.shape[1])
    inputs_embeds = model.embedding(input_ids)

    if max_n_latents > 0:
        next_compute_range = (0, latent_indices[:, 1].min().item())

    kv_cache = None

    for pass_idx in range(max_n_latents):
        if kv_cache is None:
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[
                    :, next_compute_range[0]:next_compute_range[1]],
                position_ids=position_ids[
                    :, next_compute_range[0]:next_compute_range[1]],
                output_hidden_states=True,
            )
            hidden_states_offset = 0
        else:
            past_kv = [
                (k[:, :, :next_compute_range[0], :],
                 v[:, :, :next_compute_range[0], :])
                for k, v in kv_cache
            ]
            outputs = model.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0]:next_compute_range[1], :],
                attention_mask=attention_mask[:, :next_compute_range[1]],
                position_ids=position_ids[
                    :, next_compute_range[0]:next_compute_range[1]],
                past_key_values=past_kv,
                output_hidden_states=True,
            )
            hidden_states_offset = next_compute_range[0]

        logits_list.append(outputs.logits)
        next_compute_range = (
            next_compute_range[1],
            (
                input_ids.shape[1]
                if pass_idx + 1 >= max_n_latents
                else next_compute_range[1] + 1
            ),
        )
        hidden_states = outputs.hidden_states[-1]
        kv_cache      = outputs.past_key_values

        filling_indices = [
            (b, mask_list[pass_idx])
            for b, mask_list in enumerate(latent_lists)
            if len(mask_list) > pass_idx
        ]

        tensor_list = [
            [inputs_embeds[b, pos, :] for pos in range(inputs_embeds.shape[1])]
            for b in range(inputs_embeds.shape[0])
        ]

        for batch_idx, token_idx in filling_indices:
            thought = hidden_states[
                batch_idx, token_idx - 1 - hidden_states_offset, :
            ]

            # capture
            if pass_idx == capture_pass:
                captured[0] = thought.detach().cpu()

            # optional injection
            if pass_idx == capture_pass and steer_vec is not None and steer_alpha != 0:
                thought = thought + steer_alpha * steer_vec.to(thought.device)

            tensor_list[batch_idx][token_idx] = thought

        inputs_embeds = torch.stack([
            torch.stack(tensor_list[b])
            for b in range(inputs_embeds.shape[0])
        ])

    # final pass
    past_kv = (
        [(k[:, :, :next_compute_range[0], :],
          v[:, :, :next_compute_range[0], :]) for k, v in kv_cache]
        if kv_cache else None
    )
    outputs = model.base_causallm(
        inputs_embeds=inputs_embeds[
            :, next_compute_range[0]:next_compute_range[1], :],
        attention_mask=attention_mask[:, :next_compute_range[1]],
        position_ids=position_ids[
            :, next_compute_range[0]:next_compute_range[1]],
        past_key_values=past_kv,
        output_hidden_states=True,
    )
    logits_list.append(outputs.logits)
    logits = torch.cat(logits_list, dim=-2)

    last_token_logits = logits[0, -1]
    return captured[0], last_token_logits


def build_neg_question(sample: dict) -> str:
    """Swap answer options in the question string to nudge toward neg_target."""
    q          = sample["question"]
    idx_to_sym = sample.get("idx_to_symbol", [])
    t_idx      = sample.get("target")
    n_idx      = sample.get("neg_target")
    t_word     = idx_to_sym[t_idx] if t_idx is not None and t_idx < len(idx_to_sym) else ""
    n_word     = idx_to_sym[n_idx] if n_idx is not None and n_idx < len(idx_to_sym) else ""
    if t_word and n_word and f"{t_word} or {n_word}" in q:
        q = q.replace(f"{t_word} or {n_word}", f"{n_word} or {t_word}")
    return q


def extract_vectors(cfg):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    n_latent  = 6  # prosqa_coconut.yaml uses max_latent_stage=6

    with open(os.path.expanduser(cfg["train_path"])) as f:
        data = json.load(f)

    max_pos = max(cfg["latent_positions"])
    data = [d for d in data if len(d.get("steps", [])) >= max_pos]
    data = data[:cfg["n_contrast"]]
    print(f"Using {len(data)} samples, n_latent={n_latent}")

    vectors = {}

    for pos in cfg["latent_positions"]:
        capture_pass = pos - 1  # positions are 1-indexed; passes are 0-indexed
        correct_acts, wrong_acts = [], []

        for i, sample in enumerate(data):
            inp_c = build_input(tokenizer, sample["question"], n_latent, device)
            inp_w = build_input(tokenizer, build_neg_question(sample), n_latent, device)

            with torch.no_grad():
                h_c, _ = forward_with_capture(model, *inp_c, capture_pass=capture_pass)
                h_w, _ = forward_with_capture(model, *inp_w, capture_pass=capture_pass)

            if h_c is not None:
                correct_acts.append(h_c)
            if h_w is not None:
                wrong_acts.append(h_w)

            if (i + 1) % 100 == 0:
                print(f"  pos {pos}: {i+1}/{len(data)} samples")

        if correct_acts and wrong_acts:
            v = torch.stack(correct_acts).mean(0) - torch.stack(wrong_acts).mean(0)
            vectors[pos] = v
            print(f"  position {pos}: norm={v.norm():.4f}  "
                  f"({len(correct_acts)} correct, {len(wrong_acts)} wrong)")
        else:
            print(f"  position {pos}: no vectors captured")

    out_dir  = Path(os.path.expanduser(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steering_vectors.pt"
    torch.save(vectors, out_path)
    print(f"\nSaved {len(vectors)} vectors -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    extract_vectors(load_config(args.config))
