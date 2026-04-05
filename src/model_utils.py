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
    sys.path.insert(0, os.path.abspath("."))
    
    print("Debug: Importing Coconut...")
    from coconut.coconut import Coconut
    print("Debug: Coconut imported successfully")
    
    checkpoint_path = "checkpoints/coconut/final"
    print(f"Debug: Checking checkpoint path: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Path {checkpoint_path} does not exist!")
        return None, None
    
    print("Debug: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("Debug: Tokenizer loaded")

    latent_id = tokenizer.convert_tokens_to_ids("<latent>")
    start_id  = tokenizer.convert_tokens_to_ids("<start_latent>")
    end_id    = tokenizer.convert_tokens_to_ids("<end_latent>")
    print(f"Debug: Special token IDs - latent:{latent_id}, start:{start_id}, end:{end_id}")

    print("Debug: Loading base model...")
    base_lm = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    base_lm.resize_token_embeddings(len(tokenizer))
    print("Debug: Base model loaded")

    print("Debug: Creating Coconut wrapper...")
    model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id)
    
    state_path = os.path.join(checkpoint_path, "coconut_state_dict.pt")
    print(f"Debug: Loading state dict from {state_path}")
    
    if not os.path.exists(state_path):
        print(f"ERROR: {state_path} does not exist!")
        return None, None
    
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    print("Debug: State dict loaded successfully")
    
    print("Debug: Moving model to device...")
    model = model.to(device)
    print(f"Debug: After .to(device), model type: {type(model)}")
    
    print("Debug: Setting eval mode...")
    model = model.eval()  # This might return None if Coconut.eval() doesn't return self
    print(f"Debug: After .eval(), model type: {type(model)}")
    
    # FIX: If eval() returns None, just use model directly
    if model is None:
        print("Debug: eval() returned None, using original model reference")
        # Re-get the model reference (it was modified in-place)
        model = model if model is not None else model  # This won't work
    
    print("Debug: About to return model and tokenizer")
    print(f"Debug: Final model type: {type(model)}")
    print(f"Debug: Tokenizer type: {type(tokenizer)}")
    
    # FIX: Ensure we return the actual model, not None
    # The eval() call might have modified the model in-place but returned None
    # Let's reload the model reference
    
    # Better approach: Don't assign the result of eval()
    model = Coconut(base_lm, latent_id, start_id, end_id, tokenizer.eos_token_id)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model = model.to(device)
    model.eval()  # Call in-place, don't assign
    print(f"Debug: Final model type after in-place eval: {type(model)}")
    
    return model, tokenizer
    
def load_verbal_cot_model(device="cuda"):
    return _load("checkpoints/verbal_cot/final", device)