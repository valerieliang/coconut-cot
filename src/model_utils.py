from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def _load(path, device="cuda"):
    model     = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer

def load_base_model(device="cuda"):
    return _load("gpt2-medium", device)

def load_coconut_model(device="cuda"):
    return _load("checkpoints/coconut", device)

def load_verbal_cot_model(device="cuda"):
    return _load("checkpoints/verbal_cot", device)