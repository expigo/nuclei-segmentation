from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from torchmetrics import Metric
from scipy.optimize import linear_sum_assignment

class NucleusMetrics:
    """Collection of metrics for nucleus segmentation evaluation."""
    
    @staticmethod
    def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute Dice coefficient.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            Dice coefficient
        """
        pred = pred.float()
        target = target.float()
        
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice

    @staticmethod
    def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Compute Intersection over Union (IoU).
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            smooth: Smoothing factor to avoid division by zero
            
        Returns:
            IoU score
        """
        pred = pred.float()
        target = target.float()
        
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou

    @staticmethod
    def compute_aji(pred: np.ndarray, target: np.ndarray) -> float:
        """Compute Aggregated Jaccard Index (AJI).
        
        Args:
            pred: Predicted instance segmentation mask
            target: Ground truth instance segmentation mask
            
        Returns:
            AJI score
        """
        def get_instances(mask):
            instances = []
            for id in np.unique(mask):
                if id == 0:  # background
                    continue
                instances.append(mask == id)
            return instances

        pred_instances = get_instances(pred)
        target_instances = get_instances(target)
        
        if not pred_instances or not target_instances:
            return 0.0

        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_instances), len(target_instances)))
        for i, pred_inst in enumerate(pred_instances):
            for j, target_inst in enumerate(target_instances):
                intersection = np.sum(pred_inst & target_inst)
                union = np.sum(pred_inst | target_inst)
                iou_matrix[i, j] = intersection / union if union > 0 else 0

        # Use Hungarian algorithm for matching
        pred_idx, target_idx = linear_sum_assignment(-iou_matrix)
        
        # Compute AJI
        matched_ious = iou_matrix[pred_idx, target_idx]
        return float(np.mean(matched_ious))

class AggregatedJaccardIndex(Metric):
    """PyTorch Lightning compatible AJI metric."""
    
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("aji_sum", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """Update metric states."""
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        
        aji = self.compute_aji(preds, target)
        self.aji_sum += aji
        self.total += 1

    def compute(self):
        """Compute final metric value."""
        return self.aji_sum / self.total

class SegmentationMetrics:
    """Wrapper class for all segmentation metrics."""
    
    def __init__(self):
        self.metrics = {
            'dice': NucleusMetrics.compute_dice,
            'iou': NucleusMetrics.compute_iou,
            'aji': NucleusMetrics.compute_aji
        }
    
    def compute_metrics(
        self, 
        pred: Union[torch.Tensor, np.ndarray], 
        target: Union[torch.Tensor, np.ndarray],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute all or specified metrics.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            metrics: List of metrics to compute. If None, compute all.
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = list(self.metrics.keys())
            
        results = {}
        for metric_name in metrics:
            if metric_name not in self.metrics:
                raise ValueError(f"Unknown metric: {metric_name}")
                
            metric_fn = self.metrics[metric_name]
            results[metric_name] = metric_fn(pred, target)
            
        return results

def evaluate_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate model predictions using multiple metrics.
    
    Args:
        model: PyTorch model
        dataloader: Dataloader for evaluation
        device: Device to run evaluation on
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of average metric values
    """
    model.eval()
    metric_calculator = SegmentationMetrics()
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            batch_metrics = metric_calculator.compute_metrics(
                outputs, 
                targets,
                metrics
            )
            all_metrics.append(batch_metrics)
    
    # Average metrics
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
    
    return avg_metrics