# src/model_utils.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os


def _load(path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return model.to(device).eval(), tokenizer


def load_base_model(device="cuda"):
    return _load("gpt2-medium", device)


def load_coconut_model(checkpoint_dir: str = "checkpoints/coconut/final", device: str = "cuda"):
    """
    Load Coconut model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to Coconut checkpoint directory (should contain model files and coconut_state_dict.pt)
        device: Device to load model on
    """
    sys.path.insert(0, os.path.abspath("."))
    from coconut.coconut import Coconut

    # Normalize the path - remove any trailing 'final' if it's doubled
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    
    # Check if the directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"  Loading from: {checkpoint_dir}")
    
    # Look for state dict
    state_dict_path = os.path.join(checkpoint_dir, "coconut_state_dict.pt")
    if not os.path.exists(state_dict_path):
        # Try alternative locations
        alt_paths = [
            os.path.join(checkpoint_dir, "coconut_state_dict.pt"),
            os.path.join(checkpoint_dir, "state_dict.pt"),
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                state_dict_path = alt
                break
        else:
            raise FileNotFoundError(f"Cannot find state dict in {checkpoint_dir}")
    
    print(f"  State dict: {state_dict_path}")
    
    # Load tokenizer from the checkpoint directory
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Get special token IDs
    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id = tokenizer.convert_tokens_to_ids("<end_latent>")

    # Handle cases where tokens might have different names
    if start_id == tokenizer.unk_token_id:
        start_id = tokenizer.convert_tokens_to_ids("<start>")
    if end_id == tokenizer.unk_token_id:
        end_id = tokenizer.convert_tokens_to_ids("<end>")
    if latent_id == tokenizer.unk_token_id:
        latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    
    print(f"  Special tokens: start={start_id}, end={end_id}, latent={latent_id}")

    # Load base model
    base_lm = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
    base_lm.resize_token_embeddings(len(tokenizer))

    # Create Coconut wrapper
    model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    # Load state dict
    state = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model, tokenizer


def load_verbal_cot_model(checkpoint_dir: str = "checkpoints/verbal_cot/final", device: str = "cuda"):
    """
    Load Verbal CoT model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to Verbal CoT checkpoint directory
        device: Device to load model on
    """
    # Normalize the path
    checkpoint_dir = os.path.normpath(checkpoint_dir)
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    print(f"  Loading from: {checkpoint_dir}")
    return _load(checkpoint_dir, device)


def load_model_by_type(model_type: str, checkpoint_dir: str, device: str = "cuda"):
    """
    Load model by type.
    
    Args:
        model_type: Either 'coconut' or 'verbal'
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on
    """
    if model_type == "coconut":
        return load_coconut_model(checkpoint_dir, device)
    elif model_type == "verbal":
        return load_verbal_cot_model(checkpoint_dir, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")