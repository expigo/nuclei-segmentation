import pytest
import torch

@pytest.fixture
def sample_batch():
    # Generate random image
    image = torch.randn(4, 3, 256, 256)
    
    # Generate random binary mask (as float32 for BCE loss)
    mask = torch.randint(0, 2, (4, 1, 256, 256))
    
    return {
        'image': image,
        'mask': mask
    }

