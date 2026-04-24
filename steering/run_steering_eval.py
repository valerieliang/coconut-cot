"""
run_steering_eval.py
Injects steering vectors into Coconut's continuous thought feedback loop
and measures the effect on generated answer accuracy.

Run: 
    python steering/run_steering_eval.py --config steering/steering_config.yaml
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "coconut"))

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
    """Build the input sequence Coconut expects: question + latent block."""
    q_ids = tokenizer.encode(question)
    seq = (
        q_ids
        + [START_LATENT_ID]
        + [LATENT_TOKEN_ID] * n_latent
        + [END_LATENT_ID]
    )
    input_ids      = torch.tensor([seq], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def make_steering_forward(model, inject_pass: int, steer_vec, steer_alpha: float):
    """
    Return a patched version of model.forward() that injects
    `steer_alpha * steer_vec` into the continuous thought at `inject_pass`.
    The patch is used once then removed.
    """
    original_forward = model.forward

    def patched_forward(input_ids, attention_mask, labels, position_ids, **kwargs):
        from coconut import Outputs
        from torch.nn import CrossEntropyLoss

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
                [inputs_embeds[b, pos, :]
                 for pos in range(inputs_embeds.shape[1])]
                for b in range(inputs_embeds.shape[0])
            ]
            for batch_idx, token_idx in filling_indices:
                thought = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]
                # -- inject here --------------------------------------------
                if pass_idx == inject_pass:
                    thought = thought + steer_alpha * steer_vec.to(thought.device)
                tensor_list[batch_idx][token_idx] = thought

            inputs_embeds = torch.stack([
                torch.stack(tensor_list[b])
                for b in range(inputs_embeds.shape[0])
            ])

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

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    return original_forward, patched_forward


def generate_answer(model, input_ids, attention_mask,
                    steer_vec=None, inject_pass=None, steer_alpha=0.0):
    """
    Call model.generate(), optionally with a patched forward() that injects
    a steering vector at inject_pass. Returns the decoded token list.
    """
    original_forward = None
    if steer_vec is not None and steer_alpha != 0.0 and inject_pass is not None:
        original_forward, patched = make_steering_forward(
            model, inject_pass, steer_vec, steer_alpha
        )
        model.forward = patched

    try:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=16,
            )
    finally:
        if original_forward is not None:
            model.forward = original_forward

    # out_ids is a 1D tensor of token ids (Coconut.generate returns shape [1, n])
    return out_ids[0].tolist()


def extract_answer_word(tokens: list[int], tokenizer) -> str:
    """
    Coconut outputs: <question> ### <answer sentence>
    Decode and grab the last word after ###.
    """
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    if "###" in decoded:
        answer_part = decoded.split("###")[-1].strip()
    else:
        answer_part = decoded.strip()
    return answer_part.rstrip(".").split()[-1].lower() if answer_part else ""


def run_eval(cfg):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model     = load_model(cfg, device)
    tokenizer = load_tokenizer()
    n_latent  = 6

    vectors_path = Path(os.path.expanduser(cfg["output_dir"])) / "steering_vectors.pt"
    vectors = torch.load(vectors_path, map_location="cpu")
    print(f"Loaded vectors for positions: {sorted(vectors.keys())}")

    with open(os.path.expanduser(cfg["val_path"])) as f:
        val_data = json.load(f)

    out_path = Path(os.path.expanduser(cfg["output_dir"])) / "results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "position", "alpha", "sample_idx",
            "correct", "n_steps", "predicted_word", "expected_word",
        ])
        writer.writeheader()

        for sample_idx, sample in enumerate(val_data):
            n_steps      = len(sample.get("steps", []))
            idx_to_sym   = sample.get("idx_to_symbol", [])
            target_idx   = sample.get("target")
            expected_word = (
                idx_to_sym[target_idx].lower()
                if target_idx is not None and target_idx < len(idx_to_sym)
                else sample["answer"].rstrip(".").split()[-1].lower()
            )

            input_ids, attention_mask = build_input(
                tokenizer, sample["question"], n_latent, device
            )

            # -- baseline ----------------------------------------------
            tokens = generate_answer(model, input_ids, attention_mask)
            pred_word = extract_answer_word(tokens, tokenizer)
            writer.writerow({
                "position": 0, "alpha": 0, "sample_idx": sample_idx,
                "correct": int(pred_word == expected_word),
                "n_steps": n_steps,
                "predicted_word": pred_word,
                "expected_word": expected_word,
            })

            # -- steered runs -----------------------------------------
            for pos in cfg["latent_positions"]:
                if pos not in vectors:
                    continue
                vec          = vectors[pos]
                inject_pass  = pos - 1

                for alpha in cfg["alpha_sweep"]:
                    tokens = generate_answer(
                        model, input_ids, attention_mask,
                        steer_vec=vec, inject_pass=inject_pass,
                        steer_alpha=float(alpha),
                    )
                    pred_word = extract_answer_word(tokens, tokenizer)
                    writer.writerow({
                        "position": pos, "alpha": alpha,
                        "sample_idx": sample_idx,
                        "correct": int(pred_word == expected_word),
                        "n_steps": n_steps,
                        "predicted_word": pred_word,
                        "expected_word": expected_word,
                    })

            if (sample_idx + 1) % 50 == 0:
                print(f"  {sample_idx + 1}/{len(val_data)} samples done")

    # -- quick terminal summary -------------------------------------------
    import pandas as pd
    df       = pd.read_csv(out_path)
    baseline = df[df["alpha"] == 0].set_index("sample_idx")["correct"]
    steered  = df[df["alpha"] > 0].copy()
    steered["baseline"] = steered["sample_idx"].map(baseline)
    steered["delta"]    = steered["correct"] - steered["baseline"]

    print(f"\nBaseline accuracy: {baseline.mean():.4f} ({int(baseline.sum())}/300)")
    print("\nMean accuracy delta by position (avg across all alphas):")
    for pos, grp in steered.groupby("position"):
        delta = grp["delta"].mean()
        acc   = grp["correct"].mean()
        flag  = " <-" if abs(delta) > 0.05 else ""
        print(f"  L{pos}: delta={delta:+.4f}  steered_acc={acc:.4f}{flag}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_eval(load_config(args.config))