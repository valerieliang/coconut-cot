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
    import sys, os
    # Coconut uses a non-standard forward pass — must use the repo's own class,
    # not AutoModelForCausalLM, which would load weights without the recurrence logic.
    sys.path.insert(0, os.path.abspath("coconut"))      # path to cloned repo root
    from src.model import CoconutModel                  # repo's model class — verify exact import path

    tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
    tokenizer.pad_token = tokenizer.eos_token

    model = CoconutModel.from_pretrained("checkpoints/coconut")
    return model.to(device).eval(), tokenizer

def load_verbal_cot_model(device="cuda"):
    return _load("checkpoints/verbal_cot/final", device)