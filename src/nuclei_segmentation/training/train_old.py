import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from nuclei_segmentation.data.dataset import NucleiDataset
from nuclei_segmentation.models.unet.base import UNet

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="train")
def train(cfg: DictConfig):
    # Print config for debugging
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    
    # Convert OmegaConf to regular dictionary for wandb
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Initialize wandb
    wandb.init(project="nuclei-segmentation", config=cfg_dict)
    
    # Create datasets
    train_dataset = NucleiDataset(
        root_dir=cfg.data.root_dir,
        split='train',
        mask_type=cfg.data.mask_type
    )
    
    val_dataset = NucleiDataset(
        root_dir=cfg.data.root_dir,
        split='val',
        mask_type=cfg.data.mask_type
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Initialize model
    model = UNet()
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='mps' if cfg.training.device == 'mps' else 'gpu',
        devices=1,
        logger=pl.loggers.WandbLogger(project="nuclei-segmentation")
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    wandb.finish()

if __name__ == "__main__":
    train()