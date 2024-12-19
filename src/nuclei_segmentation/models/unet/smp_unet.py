import segmentation_models_pytorch as smp
import torch
from typing import List, Dict, Any, Optional
import logging

from ..base import BaseSegmentationModel

logger = logging.getLogger(__name__)

class SMPUNet(BaseSegmentationModel):
    """U-Net implementation using segmentation-models-pytorch."""
    
    @classmethod
    def get_required_parameters(cls) -> List[str]:
        """Get required parameters, adding SMP-specific ones to base requirements."""
        return super().get_required_parameters() + [
            'encoder_name',
        ]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SMPUNet':
        """Create SMPUNet instance from config dictionary."""
        # Get only the parameters we need
        model_params = {
            'in_channels': config['in_channels'],
            'n_classes': config['n_classes'],
            'mask_type': config['mask_type'],
            'learning_rate': config['learning_rate'],
            'encoder_name': config['encoder_name'],
            'encoder_weights': config.get('encoder_weights', 'imagenet')
        }
    
        logger.info(f"SMPUNet params after filtering: {model_params}")
        return cls(**model_params)
    
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        mask_type: str,
        learning_rate: float,
        encoder_name: str,
        encoder_weights: str = "imagenet",
        **kwargs
    ):
        # Pass ALL parameters to base class
        super().__init__(**{
            'in_channels': in_channels,
            'n_classes': n_classes,
            'mask_type': mask_type,
            'learning_rate': learning_rate,
            'encoder_name': encoder_name,  # Include model-specific params
            'encoder_weights': encoder_weights,
            **kwargs
        })
        
        # Initialize SMP U-Net
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
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