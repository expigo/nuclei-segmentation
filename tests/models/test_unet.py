# tests/models/test_unet.py
import pytest
import torch
from nuclei_segmentation.models.unet.base import UNet
# from nuclei_segmentation.models.unet.smp_unet import SMPUNet as UNet

def test_unet_output_dimensions(sample_batch):
    model = UNet()
    output = model(sample_batch['image'])
    
    # Print shapes for debugging
    print(f"Input shape: {sample_batch['image'].shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mask shape: {sample_batch['mask'].shape}")
    
    assert output.shape[-2:] == sample_batch['mask'].shape[-2:] # spatial dimensions
    assert output.shape[0] == sample_batch['mask'].shape[0] # batch size


def test_unet_training_step(sample_batch):
    model = UNet()
    
    # Verify input types
    assert sample_batch['image'].dtype == torch.float32
    # assert sample_batch['mask'].dtype == torch.long
    
    # Run training step
    loss = model.training_step((sample_batch['image'], sample_batch['mask']), 0)
    
    # Verify output
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
    assert loss.dtype == torch.float32
    
def test_unet_metrics(sample_batch):
    model = UNet()
    
    # Forward pass
    with torch.no_grad():
        output = model(sample_batch['image'])
        y_pred = (torch.sigmoid(output) > 0.5).long()
        
        # Test metrics
        dice_score = model.dice(y_pred, sample_batch['mask'])
        iou_score = model.iou(y_pred, sample_batch['mask'])
        
        assert isinstance(dice_score, torch.Tensor)
        assert isinstance(iou_score, torch.Tensor)