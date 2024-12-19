from typing import Dict, Optional
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import Metric
import numpy as np
from scipy.optimize import linear_sum_assignment

class MaskMetrics:
    """Metrics factory for different mask types."""
    
    @staticmethod
    def get_metrics(mask_type: str) -> Dict[str, Metric]:
        if mask_type == 'binaries':
            return BinaryMaskMetrics()
        elif mask_type == 'cell_types':
            return CellTypeMaskMetrics()
        elif mask_type == 'multi_instance':
            return InstanceMaskMetrics()
        elif mask_type == 'contours':
            return ContourMaskMetrics()
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

class BinaryMaskMetrics:
    """Metrics for binary segmentation masks."""
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, 
                        threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
        """
        Calculate Dice coefficient for each sample in batch.
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
            threshold: Binary threshold for predictions
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Tensor of shape [B] with Dice score for each sample
        """
        pred = (pred > threshold).float()
        intersection = (pred * target).sum(dim=(2,3))  # Sum over H,W dimensions
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice  # Shape: [B]

    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor,
                  threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
        """
        Calculate IoU score for each sample in batch.
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
            threshold: Binary threshold for predictions
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Tensor of shape [B] with IoU score for each sample
        """
        pred = (pred > threshold).float()
        intersection = (pred * target).sum(dim=(2,3))  # Sum over H,W dimensions
        union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou  # Shape: [B]
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate comprehensive metrics for binary segmentation.
        
        Args:
            pred: Predicted masks [B, 1, H, W]
            target: Ground truth masks [B, 1, H, W]
            
        Returns:
            Dictionary with metrics and their batch statistics
        """
        # Ensure predictions are binary
        pred_binary = (pred > 0.5).float()
        
        # Calculate basic metrics
        dice_scores = self.dice_coefficient(pred_binary, target)  # Returns scores per sample
        iou_scores = self.iou_score(pred_binary, target)  # Returns scores per sample

        # print(f"Calculated raw metrics - dice_scores shape: {dice_scores.shape}")
        
        def safe_stats(tensor):
            if tensor.numel() > 1:
                return {
                    'mean': tensor.mean(),
                    'std': tensor.std(unbiased=True) if tensor.numel() > 1 else torch.tensor(0.0),
                    'min': tensor.min(),
                    'max': tensor.max()
                }
            return {
                'mean': tensor.mean(),
                'std': torch.tensor(0.0),
                'min': tensor,
                'max': tensor
            }

        dice_stats = safe_stats(dice_scores)
        iou_stats = safe_stats(iou_scores)

        metrics = {
            # Mean metrics
            'dice': dice_stats['mean'],
            'iou': iou_stats['mean'],
            
            # Per-sample scores for distribution analysis
            'dice_per_sample': dice_scores,
            'iou_per_sample': iou_scores,
            
            'dice_std': dice_stats['std'],
            'iou_std': iou_stats['std'],
            'dice_min': dice_stats['min'],
            'dice_max': dice_stats['max'],
            'iou_min': iou_stats['min'],
            'iou_max': iou_stats['max']
        }

        # print(f"Returning metrics with keys: {list(metrics.keys())}")
        return metrics

class CellTypeMaskMetrics(BinaryMaskMetrics):
    """Metrics for cell type segmentation masks."""
    
    def __init__(self, num_classes: int = 21):  # 20 cell types + background
        self.num_classes = num_classes
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        binary_metrics = super().calculate_metrics(pred, target)
        
        # Add cell type specific metrics
        pred_classes = pred.argmax(dim=1)
        conf_matrix = self.confusion_matrix(pred_classes, target)
        per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(dim=1)
        
        binary_metrics.update({
            'mean_accuracy': per_class_acc.mean(),
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': conf_matrix
        })
        return binary_metrics

class InstanceMaskMetrics(BinaryMaskMetrics):
    """Metrics for instance segmentation masks."""
    
    @staticmethod
    def calculate_aji(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Aggregated Jaccard Index."""
        pred_instances = pred.unique()[1:]  # Exclude background
        target_instances = target.unique()[1:]  # Exclude background
        
        if len(pred_instances) == 0 or len(target_instances) == 0:
            return torch.tensor(0.0)
        
        # Calculate IoU matrix
        iou_matrix = torch.zeros((len(pred_instances), len(target_instances)))
        for i, p in enumerate(pred_instances):
            for j, t in enumerate(target_instances):
                pred_mask = (pred == p)
                target_mask = (target == t)
                intersection = (pred_mask & target_mask).sum()
                union = (pred_mask | target_mask).sum()
                iou_matrix[i, j] = intersection / union
        
        # Use Hungarian algorithm
        matched_indices = linear_sum_assignment(-iou_matrix.cpu().numpy())
        matched_iou = iou_matrix[matched_indices]
        
        return matched_iou.mean()
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        binary_metrics = super().calculate_metrics(pred, target)
        binary_metrics['aji'] = self.calculate_aji(pred, target)
        return binary_metrics

class ContourMaskMetrics(BinaryMaskMetrics):
    """Metrics for contour segmentation masks."""
    
    @staticmethod
    def boundary_iou(pred: torch.Tensor, target: torch.Tensor, 
                     threshold: float = 1.5) -> torch.Tensor:
        """Calculate IoU with distance threshold for boundaries."""
        from scipy.ndimage import distance_transform_edt
        
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
        
        pred_boundary = pred == 1
        target_boundary = target == 1
        
        pred_dt = distance_transform_edt(~pred_boundary)
        target_dt = distance_transform_edt(~target_boundary)
        
        pred_mask = pred_dt < threshold
        target_mask = target_dt < threshold
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        return torch.tensor(intersection / union if union > 0 else 0.0)
    
    def calculate_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        binary_metrics = super().calculate_metrics(pred, target)
        binary_metrics['boundary_iou'] = self.boundary_iou(pred, target)
        return binary_metrics
    

class MaskMetrics:
    """Metrics factory for different mask types."""
    
    @staticmethod
    def get_metrics(mask_type: str) -> BinaryMaskMetrics:
        if mask_type == 'binaries':
            return BinaryMaskMetrics()
        else:
            raise ValueError(f"Unsupported mask type: {mask_type}")