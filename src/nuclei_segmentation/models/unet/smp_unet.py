import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from typing import Tuple, Optional
import torch
from torchmetrics import Dice, JaccardIndex

class SMPUNet(pl.LightningModule):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        learning_rate: float = 1e-3
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1
        )
        
        # Metrics (same as base UNet)
        self.dice = Dice(average='micro')
        self.iou = JaccardIndex(task='binary', num_classes=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_dice', self.dice(y_hat.sigmoid(), y))
        self.log('train_iou', self.iou(y_hat.sigmoid(), y))
        
        return loss