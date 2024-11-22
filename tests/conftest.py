import pytest
import torch

@pytest.fixture
def sample_batch():
    return {
        'image': torch.randn(4, 3, 256, 256),
        'mask': torch.randint(0, 2, (4, 1, 256, 256)).float()
    }

# tests/models/test_unet.py
import pytest
from nuclei_segmentation.models.unet import UNet

def test_unet_output_shape(sample_batch):
    model = UNet()
    output = model(sample_batch['image'])
    assert output.shape == sample_batch['mask'].shape

def test_unet_training_step(sample_batch):
    model = UNet()
    loss = model.training_step((sample_batch['image'], sample_batch['mask']), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad