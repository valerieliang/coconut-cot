from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

special_tokens = ["<latent>", "<start_latent>", "<end_latent>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

os.makedirs("checkpoints/tokenizer", exist_ok=True)
tokenizer.save_pretrained("checkpoints/tokenizer")

print("Latent token id:      ", tokenizer.convert_tokens_to_ids("<latent>"))
print("Start latent token id:", tokenizer.convert_tokens_to_ids("<start_latent>"))
print("End latent token id:  ", tokenizer.convert_tokens_to_ids("<end_latent>"))