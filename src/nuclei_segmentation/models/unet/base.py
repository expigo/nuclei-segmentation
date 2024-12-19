# src/nuclei_segmentation/models/unet/base.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
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

    def forward(self, x):
        return self.conv(x)

class UNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        init_features: int = 64,
        n_classes: int = 1,
        depth: int = 4,
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Contracting path
        self.encoder = nn.ModuleList()
        current_channels = in_channels
        features = init_features
        for _ in range(depth):
            self.encoder.append(DoubleConv(current_channels, features, dropout_rate))
            current_channels = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = DoubleConv(current_channels, features, dropout_rate)
        
        # Expanding path
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
        
        # Define loss function with class weights due to imbalance
        pos_weight = torch.tensor([1/0.15])  # Based on foreground ratio
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, x):
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
    
    def training_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        # Ensure masks are binary
        masks = masks.float()
        
        loss = self.criterion(outputs, masks)
        
        # Log metrics
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice = self._dice_coefficient(preds, masks)
            self.log('train_loss', loss, prog_bar=True)
            self.log('train_dice', dice, prog_bar=True)
            
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        masks = masks.float()
        loss = self.criterion(outputs, masks)
        
        preds = (torch.sigmoid(outputs) > 0.5).float()
        dice = self._dice_coefficient(preds, masks)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice', dice, prog_bar=True)
        
        return {'val_loss': loss, 'val_dice': dice}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
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
    
    def _dice_coefficient(self, pred, target, smooth=1e-6):
        intersection = (pred * target).sum(dim=(2,3))
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean()