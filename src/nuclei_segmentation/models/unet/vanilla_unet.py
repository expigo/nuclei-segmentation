import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional

from nuclei_segmentation.metrics.mask_metrics import MaskMetrics

class DoubleConv(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class VanillaUNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        init_features: int = 64,
        n_classes: int = 1,
        depth: int = 4,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-4,
        mask_type: str = 'binaries',
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize metrics
        self.metrics = MaskMetrics.get_metrics(mask_type)
        
        # Set up loss function based on mask type
        if mask_type == 'binaries':
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([1/0.15]) if class_weights is None else class_weights
            )
        elif mask_type == 'cell_types':
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights
            )
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")
        
        # Encoder path
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        features = init_features
        for _ in range(depth):
            self.encoder.append(DoubleConv(current_channels, features, dropout_rate))
            current_channels = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = DoubleConv(current_channels, features, dropout_rate)
        
        # Decoder path
        self.decoder = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for _ in range(depth):
            self.upconvs.append(
                nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2)
            )
            self.decoder.append(
                DoubleConv(features, features//2, dropout_rate)
            )
            features //= 2
            
        # Final convolution
        self.final_conv = nn.Conv2d(features, n_classes, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2)
            
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for easier access
        
        for idx, (decoder_block, upconv) in enumerate(zip(self.decoder, self.upconvs)):
            x = upconv(x)
            skip = skip_connections[idx]
            
            # Handle cases where input size is not perfectly divisible by 2**depth
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
                
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)
            
        return self.final_conv(x)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            masks = masks.float()
            loss = self.criterion(outputs, masks)
        else:  # cell_types
            masks = masks.long()
            loss = self.criterion(outputs, masks)
        
        # Calculate and log metrics
        with torch.no_grad():
            metrics = self.metrics.calculate_metrics(
                outputs.sigmoid() if self.hparams.mask_type == 'binaries' else outputs,
                masks
            )
            
            self.log('train_loss', loss, prog_bar=True)
            for metric_name, value in metrics.items():
                self.log(f'train_{metric_name}', value, prog_bar=True)
            
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            masks = masks.float()
            loss = self.criterion(outputs, masks)
        else:  # cell_types
            masks = masks.long()
            loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            outputs.sigmoid() if self.hparams.mask_type == 'binaries' else outputs,
            masks
        )
        
        metrics['loss'] = loss
        
        # Log all metrics
        self.log('val_loss', loss, prog_bar=True)
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
        
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        images = batch['image']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            return outputs.sigmoid()
        return outputs