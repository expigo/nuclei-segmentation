from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Dice, JaccardIndex

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        features: List[int] = [64, 128, 256, 512],
        learning_rate: float = 1e-3
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part
        in_c = in_channels
        for feature in features:
            self.downs.append(DoubleConv(in_c, feature))
            in_c = feature
            
        # Up part
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        
        # Metrics
        self.dice = Dice(
            num_classes=1,
            # multiclass=False,
            threshold=0.5
        )
        self.iou = JaccardIndex(
            task="binary",
            threshold=0.5
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]
            
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx+1](x)
            
        return self.final_conv(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step_old(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        
        # Ensure y is float32 for BCE loss
        y = y.float()
        
        # Ensure shapes match
        if len(y_hat.shape) != len(y.shape):
            y_hat = y_hat.squeeze(1)
            
        # Compute loss
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        
        # For metrics, convert predictions to binary
        with torch.no_grad():
            y_pred = (torch.sigmoid(y_hat) > 0.5).long()
            y_int = y.long()
            
            self.log('train_loss', loss)
            self.log('train_dice', self.dice(y_pred, y_int))
            self.log('train_iou', self.iou(y_pred, y_int))
        
        return loss


    def training_step(self, batch, batch_idx):
        # Update this line to handle dictionary input
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        # Ensure proper shapes
        outputs = outputs.squeeze(1)  # Remove channel dim if necessary
        masks = masks.squeeze(1)      # Remove channel dim if necessary

        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_dice', self.dice(outputs.sigmoid(), masks.long()))
        self.log('train_iou', self.iou(outputs.sigmoid(), masks.long()))
        
        return loss

    # Also add validation_step for proper validation
    def validation_step(self, batch, batch_idx):
        images, masks = batch['image'], batch['mask']
        outputs = self(images)

        # Ensure proper shapes
        outputs = outputs.squeeze(1)
        masks = masks.squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
        
        self.log('val_loss', loss)
        self.log('val_dice', self.dice(outputs.sigmoid(), masks.long()))
        self.log('val_iou', self.iou(outputs.sigmoid(), masks.long()))
        
        return loss