# scripts/02_finetune_cot.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_utils import load_base_model
from src.data_utils import load_split, format_cot
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

tokenizer = AutoTokenizer.from_pretrained("checkpoints/tokenizer")
tokenizer.pad_token = tokenizer.eos_token

model, _ = load_base_model()
model.resize_token_embeddings(len(tokenizer))

train_data = load_split("data/prontoqa_split.csv", split="train")
print(f"Training on {len(train_data)} examples")

dataset = format_cot(train_data, tokenizer)

args = TrainingArguments(
    output_dir="checkpoints/verbal_cot",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    fp16=True,
    learning_rate=5e-5,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collator,
)
trainer.train()

model.save_pretrained("checkpoints/verbal_cot/final")
tokenizer.save_pretrained("checkpoints/verbal_cot/final")
print("Done. Saved to checkpoints/verbal_cot/final/")