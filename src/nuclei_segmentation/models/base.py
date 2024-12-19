from abc import ABC, abstractmethod
import pytorch_lightning as pl
from typing import Dict, Any, List, Optional
import logging
import torch

logger = logging.getLogger(__name__)

class BaseSegmentationModel(pl.LightningModule, ABC):
    """Base class for all segmentation models."""
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseSegmentationModel':
        """Create model instance from config dictionary.
        
        Each model implementation should handle its own config parsing.
        """
        pass
    
    @classmethod
    def get_required_parameters(cls) -> List[str]:
        """Return list of required base parameters."""
        return [
            'in_channels',
            'n_classes',
            'mask_type',
            'learning_rate'
        ]
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required_params = cls.get_required_parameters()
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {cls.__name__}: {missing_params}\n"
                f"Provided config: {config}"
            )
    
    def __init__(self, **kwargs):
        super().__init__()
        logger.info(f"Base model received kwargs: {kwargs}")
        self.validate_config(kwargs)
        self.save_hyperparameters()
        
        # Initialize metrics based on mask type
        from nuclei_segmentation.metrics.mask_metrics import MaskMetrics
        self.metrics = MaskMetrics.get_metrics(self.hparams.mask_type)
        
        # Setup loss function
        self.criterion = self.setup_criterion(
            self.hparams.mask_type,
            getattr(self.hparams, 'class_weights', None)
        )
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required_params = cls.get_required_parameters()
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {cls.__name__}: {missing_params}\n"
                f"Provided config: {config}"
            )
    
    def setup_criterion(self, mask_type: str, class_weights: Any = None) -> Any:
        """Setup loss function based on mask type."""
        import torch.nn as nn
        
        if mask_type == 'binaries':
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([1/0.15]) if class_weights is None else class_weights
            )
        elif mask_type == 'cell_types':
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Common training step."""
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            masks = masks.float()
            loss = self.criterion(outputs, masks)
        else:
            masks = masks.long()
            loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            outputs.sigmoid() if self.hparams.mask_type == 'binaries' else outputs,
            masks
        )
        
        # Log only scalar metrics
        self.log('train_loss', loss, prog_bar=True, batch_size=batch['image'].shape[0])
        self.log('train_dice', metrics['dice'], prog_bar=True, batch_size=batch['image'].shape[0])
        self.log('train_iou', metrics['iou'], prog_bar=True, batch_size=batch['image'].shape[0])
                
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Common validation step."""
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            masks = masks.float()
            loss = self.criterion(outputs, masks)
        else:
            masks = masks.long()
            loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            outputs.sigmoid() if self.hparams.mask_type == 'binaries' else outputs,
            masks
        )
        
        # Log only scalar metrics
        self.log('val_loss', loss, prog_bar=True, batch_size=batch['image'].shape[0])
        self.log('val_dice', metrics['dice'], prog_bar=True, batch_size=batch['image'].shape[0])
        self.log('val_iou', metrics['iou'], prog_bar=True, batch_size=batch['image'].shape[0])
        
        # Return all metrics for potential later use
        return {
            'val_loss': loss,
            'val_dice': metrics['dice'],
            'val_iou': metrics['iou'],
            # Store per-sample metrics without logging them
            'val_dice_per_sample': metrics['dice_per_sample'],
            'val_iou_per_sample': metrics['iou_per_sample']
        }
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
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
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Common test step."""
        images, masks = batch['image'], batch['mask']
        outputs = self(images)
        
        if self.hparams.mask_type == 'binaries':
            masks = masks.float()
            loss = self.criterion(outputs, masks)
        else:
            masks = masks.long()
            loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            outputs.sigmoid() if self.hparams.mask_type == 'binaries' else outputs,
            masks
        )
        
        # Store batch data for visualization
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
        
        # Create batch predictions record
        batch_preds = {
            'images': images.cpu(),
            'masks': masks.cpu(),
            'predictions': outputs.cpu(),
            'metrics': metrics
        }
        self.test_predictions.append(batch_preds)
        
        # Create test metrics dict with all statistics
        test_metrics = {
            'test_loss': loss,
            'test_dice': metrics['dice'],
            'test_iou': metrics['iou'],
            'test_dice_std': metrics['dice_std'],
            'test_iou_std': metrics['iou_std'],
            'test_dice_min': metrics['dice_min'],
            'test_dice_max': metrics['dice_max'],
            'test_iou_min': metrics['iou_min'],
            'test_iou_max': metrics['iou_max']
        }
        
        self.log_dict(
            test_metrics,
            prog_bar=['test_loss', 'test_dice', 'test_iou'],
            batch_size=batch['image'].shape[0]
        )
        
        return test_metrics
    # def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
    #     """Aggregate test metrics at epoch end."""
    #     logger.info("Aggregating test metrics")
        
    #     # Gather all batch metrics
    #     all_metrics = {
    #         'test_loss': [],
    #         'test_dice': [],
    #         'test_iou': [],
    #         'test_dice_per_sample': [],
    #         'test_iou_per_sample': []
    #     }
        
    #     for output in outputs:
    #         for key in all_metrics:
    #             if key in output:
    #                 all_metrics[key].append(output[key])
        
    #     # Stack metrics and compute statistics
    #     aggregated_metrics = {}
    #     for key in ['test_dice_per_sample', 'test_iou_per_sample']:
    #         if all_metrics[key]:
    #             values = torch.cat(all_metrics[key])
    #             base_key = key.replace('_per_sample', '')
    #             aggregated_metrics[f'{base_key}_std'] = values.std()
    #             aggregated_metrics[f'{base_key}_min'] = values.min()
    #             aggregated_metrics[f'{base_key}_max'] = values.max()
        
    #     # Log aggregated metrics
    #     for key, value in aggregated_metrics.items():
    #         self.log(key, value)
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation to be defined by child classes."""
        pass