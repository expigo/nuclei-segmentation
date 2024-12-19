import os
from pathlib import Path
from typing import Dict, Optional

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from nuclei_segmentation.config.machine_config import get_machine_config
from nuclei_segmentation.data.dataset import NucleiDataset
from nuclei_segmentation.models import MODEL_REGISTRY

class NucleiSegmentation(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model from registry
        self.model = MODEL_REGISTRY[config.model.name](**config.model)
        
        # Set up machine-specific config
        self.machine_config = get_machine_config()
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        # Calculate loss based on mask type
        if self.config.data.mask_type == 'binaries':
            loss = self.model.binary_loss(outputs, masks)
        elif self.config.data.mask_type == 'cell_types':
            loss = self.model.multiclass_loss(outputs, masks)
        else:
            raise ValueError(f"Unsupported mask type: {self.config.data.mask_type}")
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        # Calculate metrics
        metrics = self.model.calculate_metrics(outputs, masks)
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f'val_{name}', value, prog_bar=True)
        
        return metrics

@hydra.main(config_path="../configs", config_name="train")
def train(config: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(config))
    
    # Get machine-specific config
    machine_config = get_machine_config()
    
    # Update config with machine-specific settings
    config.training.batch_size = machine_config["batch_size"]
    config.training.num_workers = machine_config["num_workers"]
    
    # Initialize wandb
    wandb_logger = WandbLogger(
        project="nuclei-segmentation",
        name=f"{config.model.name}_{config.data.mask_type}",
        config=OmegaConf.to_container(config, resolve=True)
    )
    
    # Create datasets
    train_dataset = NucleiDataset(
        root_dir=config.data.root_dir,
        split='train',
        mask_type=config.data.mask_type
    )
    
    val_dataset = NucleiDataset(
        root_dir=config.data.root_dir,
        split='val',
        mask_type=config.data.mask_type
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = NucleiSegmentation(config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=f"models/checkpoints/{config.model.name}",
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min"
        )
    ]
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'mps',
        devices=1,
        precision=machine_config["precision"],
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=True
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Save artifacts to wandb
    wandb.save(f"models/checkpoints/{config.model.name}/*")

if __name__ == "__main__":
    train()