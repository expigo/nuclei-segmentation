import torch

def get_device_config():
    """Configure device based on available hardware."""
    if torch.cuda.is_available():
        return {
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '16-mixed'  # For NVIDIA GPUs
        }
    elif torch.backends.mps.is_available():
        return {
            'accelerator': 'mps',
            'devices': 1,
            'precision': '32'  # MPS doesn't support mixed precision yet
        }
    else:
        return {
            'accelerator': 'cpu',
            'devices': 1,
            'precision': '32'
        }

