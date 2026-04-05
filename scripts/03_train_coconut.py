# scripts/03_train_coconut.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath("coconut"))   # repo root

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from coconut.dataset import get_dataset, get_cot_latent_dataset, MyCollator
from coconut.coconut import Coconut
from coconut.utils import Config, set_seed

set_seed(42)
device = "cuda"

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained("checkpoints/tokenizer")
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

latent_id = tokenizer.convert_tokens_to_ids("<latent>")
start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")

# ── Base model ───────────────────────────────────────────────────────────────
base_lm = AutoModelForCausalLM.from_pretrained("gpt2-medium")
base_lm.resize_token_embeddings(len(tokenizer))
model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id).to(device)

# ── Dataset ──────────────────────────────────────────────────────────────────
# get_dataset expects the repo's JSON format: list of {question, steps, answer}
# Point it at your prepared data — you may need a small adapter if your JSON
# structure differs from the repo's expected format.
base_dataset = get_dataset("coconut/data/prosqa_test.json", tokenizer)

configs = Config({
    "max_latent_stage":  5,      # max number of steps to replace with latents
    "c_thought":         1,      # latent tokens per replaced step (keep 1)
    "uniform_prob":      0.1,    # prob of randomly sampling stage during training
    "pad_latent_to_max": False,
    "no_cot":            False,
})

collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)

# ── Curriculum ───────────────────────────────────────────────────────────────
# Stage 0: full verbal CoT (no latent tokens) — warm up the model
# Stage k: first k steps replaced with latent tokens
# Train each stage for a fixed number of epochs before advancing

EPOCHS_PER_STAGE = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

for stage in range(configs["max_latent_stage"] + 1):
    print(f"\n{'='*50}")
    print(f"Curriculum stage {stage} / {configs['max_latent_stage']}")
    print(f"{'='*50}")

    stage_dataset = get_cot_latent_dataset(
        scheduled_stage=stage,
        base_dataset=base_dataset,
        configs=configs,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        shuffle=True,
    )

    loader = DataLoader(
        stage_dataset,
        batch_size=8,
        collate_fn=collator,
        shuffle=True,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(loader),
        num_training_steps=len(loader) * EPOCHS_PER_STAGE,
    )

    model.train()
    for epoch in range(EPOCHS_PER_STAGE):
        total_loss = 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                position_ids=batch["position_ids"],
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 20 == 0:
                print(f"  stage {stage} | epoch {epoch} | step {step} | loss {loss.item():.4f}")

        print(f"  → epoch {epoch} mean loss: {total_loss/len(loader):.4f}")

# ── Save ─────────────────────────────────────────────────────────────────────
os.makedirs("checkpoints/coconut/final", exist_ok=True)
# Save the inner base_causallm (standard HF format) + tokenizer
model.base_causallm.save_pretrained("checkpoints/coconut/final")
tokenizer.save_pretrained("checkpoints/coconut/final")
# Save the full Coconut wrapper state dict separately for inference
torch.save(model.state_dict(), "checkpoints/coconut/final/coconut_state_dict.pt")
print("\nSaved to checkpoints/coconut/final/")