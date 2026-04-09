import sys, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))
from coconut.dataset import get_dataset, get_cot_latent_dataset, MyCollator
from coconut.coconut import Coconut
from coconut.utils import Config, set_seed

if __name__ == '__main__':
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    import datasets as hf_datasets
    _original_map = hf_datasets.Dataset.map
    def _single_proc_map(self, *args, **kwargs):
        kwargs['num_proc'] = 1
        return _original_map(self, *args, **kwargs)
    hf_datasets.Dataset.map = _single_proc_map

    set_seed(42)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/tokenizer")
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")

    base_lm = AutoModelForCausalLM.from_pretrained("gpt2-medium")
    base_lm.resize_token_embeddings(len(tokenizer))
    model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id).to(device)

    base_dataset = get_dataset("coconut/data/prosqa_test.json", tokenizer)

    def truncate(example):
        for key in ["question_tokenized", "answer_tokenized"]:
            if key in example and len(example[key]) > 256:
                example[key] = example[key][:256]
        return example
    base_dataset = base_dataset.map(truncate, num_proc=1)

    configs = Config({
        "max_latent_stage":  5,
        "c_thought":         1,
        "uniform_prob":      0.1,
        "pad_latent_to_max": False,
        "no_cot":            False,
    })

    collator = MyCollator(tokenizer=tokenizer, latent_id=latent_id)

    EPOCHS_PER_STAGE = 5
    BATCH_SIZE       = 2
    GRAD_ACCUM_STEPS = 4

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    for stage in range(configs.__dict__["max_latent_stage"] + 1):
        print(f"\n{'='*50}", flush=True)
        print(f"Curriculum stage {stage} / {configs.__dict__['max_latent_stage']}", flush=True)
        print(f"{'='*50}", flush=True)

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
            batch_size=BATCH_SIZE,
            collate_fn=collator,
            shuffle=True,
            num_workers=0,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(loader) // GRAD_ACCUM_STEPS,
            num_training_steps=(len(loader) // GRAD_ACCUM_STEPS) * EPOCHS_PER_STAGE,
        )

        model.train()
        for epoch in range(EPOCHS_PER_STAGE):
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    position_ids=batch["position_ids"],
                )
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                total_loss += loss.item() * GRAD_ACCUM_STEPS

                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % 20 == 0:
                    print(f"  stage {stage} | epoch {epoch} | step {step} | loss {loss.item() * GRAD_ACCUM_STEPS:.4f}", flush=True)

            print(f"  -> epoch {epoch} mean loss: {total_loss/len(loader):.4f}", flush=True)

    os.makedirs("checkpoints/coconut/final", exist_ok=True)
    model.base_causallm.save_pretrained("checkpoints/coconut/final")
    tokenizer.save_pretrained("checkpoints/coconut/final")
    torch.save(model.state_dict(), "checkpoints/coconut/final/coconut_state_dict.pt")
    print("\nSaved to checkpoints/coconut/final/", flush=True)