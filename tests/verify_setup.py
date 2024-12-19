import torch
import wandb
from pathlib import Path
from nuclei_segmentation.models.unet.base import UNet
from nuclei_segmentation.models.unet.smp_unet import SMPUNet

def test_cuda_availability():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

def test_unet_models():
    # Test basic UNet
    x = torch.randn(2, 3, 256, 256)
    
    print("\nTesting basic UNet...")
    model = UNet()
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    print("\nTesting SMP UNet...")
    smp_model = SMPUNet()
    y_smp = smp_model(x)
    print(f"SMP Output shape: {y_smp.shape}")

def test_wandb_connection():
    print("\nTesting W&B connection...")
    try:
        wandb.init(project="nuclei-segmentation-test", name="setup-test")
        print("Successfully connected to W&B")
        wandb.finish()
    except Exception as e:
        print(f"W&B connection failed: {str(e)}")

if __name__ == "__main__":
    print("Starting setup verification...\n")
    test_cuda_availability()
    test_unet_models()
    test_wandb_connection()