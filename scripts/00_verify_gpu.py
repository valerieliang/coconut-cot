import torch

print(torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("torch version:", torch.__version__)