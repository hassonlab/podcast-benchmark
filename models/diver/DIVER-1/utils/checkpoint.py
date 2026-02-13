import os
import torch

def load_model_checkpoint(model, device, checkpoint_path, deepspeed_pth_format=False):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if deepspeed_pth_format:
        checkpoint['model_state_dict'] = checkpoint['module']
    print("Loading checkpoint from:", checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded.")