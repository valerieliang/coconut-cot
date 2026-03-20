import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coconut"))

import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import json, yaml, gc

from coconut import Coconut
from dataset import get_dataset, get_question_latent_dataset, get_cot_latent_dataset, MyCollator

# ── config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_id":                  "openai-community/gpt2",
    "train_path":                "coconut/data/prosqa_train.json",
    "val_path":                  "coconut/data/prosqa_valid.json",
    "save_path":                 "checkpoints/coconut",
    "c_thought":                 1,
    "epochs_per_stage":          5,
    "max_latent_stage":          6,
    "pad_latent_to_max":         True,
    "num_epochs":                50,
    "batch_size_training":       8,    # reduced from 32 for single GPU
    "gradient_accumulation_steps": 4,  # effective batch = 32
    "lr":                        1e-4,
    "weight_decay":              0.01,
    "reset_optimizer":           True,
    "seed":                      0,
    "bf16":                      False,
    "uniform_prob":              0.0,
    "coconut":                   True,
    "cot":                       False,
    "no_thoughts":               False,
    "no_cot":                    False,
    "debug":                     False,
    "save_only_improve":         False,
}

from utils import Config, set_seed
configs = Config(CONFIG)
set_seed(configs.seed)

os.makedirs(configs.save_path, exist_ok=True)
device = torch.device("cuda")

# ── model + tokenizer ─────────────────────────────────────────────────────────
model     = AutoModelForCausalLM.from_pretrained(configs.model_id)
tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_tokens("<|start-latent|>")
tokenizer.add_tokens("<|end-latent|>")
tokenizer.add_tokens("<|latent|>")

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

model.resize_token_embeddings(len(tokenizer))
embeddings = model.get_input_embeddings()
target_id  = tokenizer.convert_tokens_to_ids("<<")
for token_id in [latent_id, start_id, end_id]:
    embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
    model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
model = model.to(device)

# ── data ──────────────────────────────────────────────────────────────────────
collator         = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
base_dataset_train = get_dataset(configs.train_path, tokenizer)
base_dataset_valid = get_dataset(configs.val_path,   tokenizer)

answers_val = [d["answer"].replace(",","").strip() for d in json.load(open(configs.val_path))]
cot_val     = ["\n".join(d["steps"])              for d in json.load(open(configs.val_path))]

# ── training loop ─────────────────────────────────────────────────────────────
best_acc  = 0.0
optimizer = None

for epoch in range(configs.num_epochs):
    scheduled_stage = epoch // configs.epochs_per_stage
    scheduled_stage = min(scheduled_stage, configs.max_latent_stage)
    print(f"\n── Epoch {epoch+1}/{configs.num_epochs}  stage={scheduled_stage} ──")

    # build datasets for this stage
    dataset_train = get_cot_latent_dataset(
        scheduled_stage, base_dataset_train, configs,
        start_id, latent_id, end_id, shuffle=True,
    )
    dataset_val_gen = get_question_latent_dataset(
        scheduled_stage, base_dataset_valid, configs,
        start_id, latent_id, end_id,
    )

    train_loader = DataLoader(
        dataset_train, batch_size=configs.batch_size_training,
        shuffle=True, collate_fn=collator, num_workers=0,
    )
    val_loader = DataLoader(
        dataset_val_gen, batch_size=1,
        shuffle=False, collate_fn=collator, num_workers=0,
    )

    # reset optimizer each stage (matches original)
    if configs.reset_optimizer:
        optimizer = optim.AdamW(
            model.parameters(), lr=configs.lr, weight_decay=configs.weight_decay
        )

    # ── train ────────────────────────────────────────────────────────────────
    model.base_causallm.train()
    optimizer.zero_grad()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items() if k != "idx"}
        outputs = model(**batch)
        loss    = outputs.loss / configs.gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item() * configs.gradient_accumulation_steps

        if (step + 1) % configs.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item() * configs.gradient_accumulation_steps:.4f}")

    print(f"Mean train loss: {total_loss / len(train_loader):.4f}")

    # ── val accuracy ─────────────────────────────────────────────────────────
    model.base_causallm.eval()
    cor, total = 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val"):
            idx    = batch["idx"][0].item()
            batch  = {k: v.to(device) for k, v in batch.items()
                      if k != "idx" and v is not None}
            output = model.generate(**batch, max_new_tokens=128)
            text   = tokenizer.decode(output[0], skip_special_tokens=True)
            pred   = text.split("#")[-1].replace(",","").strip()
            cor   += pred == answers_val[idx]
            total += 1

    acc = cor / total
    print(f"Val accuracy: {cor}/{total} = {acc:.3f}")

    # ── save ─────────────────────────────────────────────────────────────────
    if not configs.save_only_improve or acc > best_acc:
        ckpt_path = os.path.join(configs.save_path, f"checkpoint_{epoch+1}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved → {ckpt_path}")

    best_acc = max(best_acc, acc)

    gc.collect()
    torch.cuda.empty_cache()

print("Training complete.")