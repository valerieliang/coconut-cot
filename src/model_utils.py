from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys, os

def _load(path, device="cuda"):
    model     = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer

def load_base_model(device="cuda"):
    return _load("gpt2-medium", device)

def load_coconut_model(device="cuda"):

    sys.path.insert(0, os.path.abspath("."))
    from coconut.coconut import Coconut

    tokenizer = AutoTokenizer.from_pretrained("checkpoints/coconut/final")
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")

    base_lm = AutoModelForCausalLM.from_pretrained("checkpoints/coconut/final")
    base_lm.resize_token_embeddings(len(tokenizer))

    model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id)
    state = torch.load(
        "checkpoints/coconut/final/coconut_state_dict.pt",
        map_location=device,
    )
    model.load_state_dict(state)

    # Coconut.eval() doesn't return self, so don't chain it
    model.to(device)
    model.eval()
    return model, tokenizer
    
def load_verbal_cot_model(device="cuda"):
    return _load("checkpoints/verbal_cot/final", device)