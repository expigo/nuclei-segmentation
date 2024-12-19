from typing import Dict, Any
import torch

def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor") -> Dict[str, Any]:
    """Print detailed tensor information for debugging."""
    info = {
        "name": name,
        "shape": tensor.shape,
        "dtype": tensor.dtype,
        "device": tensor.device,
        "requires_grad": tensor.requires_grad,
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "has_nan": torch.isnan(tensor).any().item()
    }
    
    print(f"\n{name} info:")
    for k, v in info.items():
        print(f"{k}: {v}")
    
    return info