import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DATASETS_NUM_PROC"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "coconut"))

if __name__ == "__main__":
    import torch
    import torch.optim as optim
    import itertools
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    from tqdm import tqdm
    import json, gc, random

    from coconut import Coconut
    from utils import Config, set_seed

    # ── config ────────────────────────────────────────────────────────────────
    CONFIG = {
        "model_id":                    "openai-community/gpt2",
        "train_path":                  "coconut/data/prosqa_train.json",
        "val_path":                    "coconut/data/prosqa_valid.json",
        "save_path":                   "checkpoints/coconut",
        "c_thought":                   1,
        "epochs_per_stage":            5,
        "max_latent_stage":            6,
        "num_epochs":                  50,
        "batch_size_training":         4,
        "gradient_accumulation_steps": 4,
        "lr":                          1e-4,
        "weight_decay":                0.01,
        "seed":                        0,
        "max_train_samples":           3000,
        "max_val_samples":             200,
        "max_seq_len":                 600,
    }

    configs = Config(CONFIG)
    set_seed(configs.seed)
    os.makedirs(configs.save_path, exist_ok=True)
    device = torch.device("cuda")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id  = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id    = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    # ── dataset ───────────────────────────────────────────────────────────────
    class ProsQADataset(Dataset):
        def __init__(self, path, tokenizer, max_samples=None,
                     stage=0, max_seq_len=600):
            print(f"  Loading {path}...")
            with open(path) as f:
                data = json.load(f)
            if max_samples:
                random.shuffle(data)
                data = data[:max_samples]
            self.data        = data
            self.tokenizer   = tokenizer
            self.stage       = stage
            self.max_seq_len = max_seq_len
            self.latent_id   = tokenizer.convert_tokens_to_ids("<|latent|>")
            self.start_id    = tokenizer.convert_tokens_to_ids("<|start-latent|>")
            self.end_id      = tokenizer.convert_tokens_to_ids("<|end-latent|>")

        def set_stage(self, stage):
            self.stage = stage

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            ex       = self.data[idx]
            steps    = ex["steps"]
            n_latent = min(self.stage, len(steps))

            # match original coconut format exactly:
            # question + "\n", each step + "\n", "### " + answer
            q_ids = self.tokenizer.encode(
                ex["question"] + "\n", add_special_tokens=True
            )
            step_ids = [
                self.tokenizer.encode(s + "\n", add_special_tokens=False)
                for s in steps
            ]
            a_ids = self.tokenizer.encode(
                "### " + ex["answer"], add_special_tokens=False
            ) + [self.tokenizer.eos_token_id]

            if n_latent == 0:
                # pure CoT: question + all steps as text + answer
                flat_steps = list(itertools.chain.from_iterable(step_ids))
                input_ids  = q_ids + flat_steps + a_ids
                # supervise on steps + answer (not question)
                labels     = [-100] * len(q_ids) + flat_steps + a_ids

            else:
                # latent stage: replace first n_latent steps with latent tokens
                latent_block    = (
                    [self.start_id]
                    + [self.latent_id] * n_latent
                    + [self.end_id]
                )
                remaining_steps = list(
                    itertools.chain.from_iterable(step_ids[n_latent:])
                )
                input_ids = q_ids + latent_block + remaining_steps + a_ids
                # supervise on remaining steps + answer (not question or latent block)
                labels    = (
                    [-100] * (len(q_ids) + len(latent_block))
                    + remaining_steps
                    + a_ids
                )

            # if too long, trim from question start (always preserve steps+answer)
            if len(input_ids) > self.max_seq_len:
                overflow = len(input_ids) - self.max_seq_len
                q_ids    = q_ids[overflow:]
                if n_latent == 0:
                    input_ids = q_ids + flat_steps + a_ids
                    labels    = [-100] * len(q_ids) + flat_steps + a_ids
                else:
                    input_ids = q_ids + latent_block + remaining_steps + a_ids
                    labels    = (
                        [-100] * (len(q_ids) + len(latent_block))
                        + remaining_steps
                        + a_ids
                    )

            # pad to max_seq_len
            pad_len      = self.max_seq_len - len(input_ids)
            attn_mask    = [1] * len(input_ids) + [0] * pad_len
            input_ids    = input_ids + [self.tokenizer.pad_token_id] * pad_len
            labels       = labels    + [-100] * pad_len
            position_ids = list(range(self.max_seq_len))

            # safety checks
            assert len(input_ids) == self.max_seq_len, f"input_ids len {len(input_ids)}"
            assert len(labels)    == self.max_seq_len, f"labels len {len(labels)}"
            assert len(attn_mask) == self.max_seq_len, f"attn_mask len {len(attn_mask)}"
            assert any(l != -100 for l in labels),     f"all labels -100 at idx {idx}"

            return {
                "input_ids":      torch.tensor(input_ids,    dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask,    dtype=torch.long),
                "labels":         torch.tensor(labels,       dtype=torch.long),
                "position_ids":   torch.tensor(position_ids, dtype=torch.long),
                "idx":            torch.tensor(idx,          dtype=torch.long),
            }

    # ── model ─────────────────────────────────────────────────────────────────
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings()
    target_id  = tokenizer.convert_tokens_to_ids("<<")
    for token_id in [latent_id, start_id, end_id]:
        embeddings.weight.data[token_id] = embeddings.weight.data[target_id]
        model.lm_head.weight.data[token_id] = model.lm_head.weight.data[target_id]

    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model.to(device)   # do NOT chain .eval() — Coconut.eval() returns None
    print("Model ready.")

    # ── val data for generation eval ──────────────────────────────────────────
    with open(configs.val_path) as f:
        val_raw = json.load(f)
    random.seed(configs.seed)
    random.shuffle(val_raw)
    val_raw     = val_raw[:configs.max_val_samples]
    answers_val = [d["answer"].replace(",", "").strip() for d in val_raw]

    # ── build datasets ────────────────────────────────────────────────────────
    train_dataset = ProsQADataset(
        configs.train_path, tokenizer,
        max_samples=configs.max_train_samples,
        stage=0,
        max_seq_len=configs.max_seq_len,
    )
    val_dataset = ProsQADataset(
        configs.val_path, tokenizer,
        max_samples=configs.max_val_samples,
        stage=0,
        max_seq_len=configs.max_seq_len,
    )

    # ── training loop ─────────────────────────────────────────────────────────
    best_acc  = 0.0
    optimizer = None

    for epoch in range(configs.num_epochs):
        scheduled_stage = min(
            epoch // configs.epochs_per_stage,
            configs.max_latent_stage,
        )
        train_dataset.set_stage(scheduled_stage)
        val_dataset.set_stage(scheduled_stage)

        print(f"\n── Epoch {epoch+1}/{configs.num_epochs}  stage={scheduled_stage} ──")

        train_loader = DataLoader(
            train_dataset,
            batch_size=configs.batch_size_training,
            shuffle=True,
            num_workers=0,    # must be 0 on Windows
            pin_memory=True,
        )

        # reset optimizer at start of each new stage
        if optimizer is None or (
            epoch > 0 and epoch % configs.epochs_per_stage == 0
        ):
            optimizer = optim.AdamW(
                model.parameters(),
                lr=configs.lr,
                weight_decay=configs.weight_decay,
            )
            print("  Optimizer reset for new stage.")

        # ── train ─────────────────────────────────────────────────────────────
        model.base_causallm.train()
        optimizer.zero_grad()
        total_loss  = 0.0
        n_nan_skips = 0

        pbar = tqdm(train_loader, desc=f"Train epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            batch   = {k: v.to(device) for k, v in batch.items() if k != "idx"}
            outputs = model(**batch)
            loss    = outputs.loss / configs.gradient_accumulation_steps

            if torch.isnan(loss):
                n_nan_skips += 1
                optimizer.zero_grad()
                continue

            loss.backward()
            total_loss += loss.item() * configs.gradient_accumulation_steps

            if (step + 1) % configs.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                loss=f"{loss.item() * configs.gradient_accumulation_steps:.4f}",
                skips=n_nan_skips,
            )

        n_steps  = len(train_loader) - n_nan_skips
        avg_loss = total_loss / max(n_steps, 1)
        print(f"Mean train loss: {avg_loss:.4f}  (nan skips: {n_nan_skips})")

        # ── val accuracy ──────────────────────────────────────────────────────
        model.base_causallm.eval()
        cor, total = 0, 0

        with torch.no_grad():
            for i, ex in enumerate(tqdm(val_raw, desc="Val")):
                # prompt with "### " so model knows to produce the answer
                prompt    = ex["question"] + "\n### "
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
                attn_mask = torch.ones_like(input_ids)

                try:
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=32,
                    )
                    # decode only newly generated tokens
                    new_tokens = output[0][input_ids.shape[1]:]
                    pred = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                    pred = pred.split("\n")[0].strip()
                except Exception:
                    pred = ""

                cor   += int(pred == answers_val[i])
                total += 1

        acc = cor / total if total > 0 else 0.0
        print(f"Val accuracy: {cor}/{total} = {acc:.3f}")

        # ── save ──────────────────────────────────────────────────────────────
        if acc >= best_acc:
            ckpt_path = os.path.join(configs.save_path, f"checkpoint_{epoch+1}")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved → {ckpt_path}")
            best_acc = acc

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")