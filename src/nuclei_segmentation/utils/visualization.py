import io
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import wandb
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image as PILImage

from nuclei_segmentation.metrics.mask_metrics import BinaryMaskMetrics

logger = logging.getLogger(__name__)

def create_mask_overlay(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.5
) -> List[np.ndarray]:
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    visualizations = []
    
    for img, pred, target in zip(images, predictions, targets):
        # Transpose from CHW to HWC
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        
        # Create figure with fixed DPI and backend
        plt.switch_backend('agg')  # Use non-interactive backend
        fig = plt.figure(figsize=(15, 5), dpi=100, frameon=False)
        
        # Create a grid of subplots that exactly fills the figure
        grid = ImageGrid(fig, 111,
                        nrows_ncols=(1, 3),
                        axes_pad=0.1)
        
        # Plot images
        grid[0].imshow(img)
        grid[0].set_title('Original Image')
        grid[0].axis('off')
        
        grid[1].imshow(img)
        mask = pred[0] > 0.5
        pred_overlay = np.zeros((*mask.shape, 3))
        pred_overlay[mask] = [1, 0, 0]  # Red
        grid[1].imshow(pred_overlay, alpha=alpha)
        grid[1].set_title('Prediction')
        grid[1].axis('off')
        
        grid[2].imshow(img)
        target_overlay = np.zeros((*target[0].shape, 3))
        target_overlay[target[0] > 0] = [0, 1, 0]  # Green
        grid[2].imshow(target_overlay, alpha=alpha)
        grid[2].set_title('Ground Truth')
        grid[2].axis('off')
        
        # Save to buffer
        fig.canvas.draw()
        
        # Get the correct dimensions
        width, height = fig.canvas.get_width_height()
        buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        # Reshape using exact dimensions from canvas
        plot = buffer.reshape(height, width, 3)
        visualizations.append(plot)
        
        plt.close(fig)
    
    return visualizations

def create_enhanced_mask_overlay(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    confidence_maps: Optional[torch.Tensor] = None
) -> List[Dict]:
    """Creates enhanced visualization with metrics."""
    batch_visualizations = []
    metrics_calculator = BinaryMaskMetrics()
    
    for idx in range(len(images)):
        # Convert tensors to numpy
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        pred = predictions[idx].cpu().numpy()
        target = targets[idx].cpu().numpy()
        
        # Normalize image for display
        img = (img - img.min()) / (img.max() - img.min())
        
        # Calculate metrics
        pred_tensor = predictions[idx:idx+1]
        target_tensor = targets[idx:idx+1]
        metrics = metrics_calculator.calculate_metrics(pred_tensor, target_tensor)
        
        # Create figure with fixed DPI
        plt.ioff()
        fig = plt.figure(figsize=(12, 12), dpi=100)
        fig.suptitle(f'Dice: {metrics["dice"]:.3f}, IoU: {metrics["iou"]:.3f}')
        
        # Original image
        ax1 = plt.subplot(2, 2, 1)
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Prediction overlay
        ax2 = plt.subplot(2, 2, 2)
        ax2.imshow(img)
        pred_mask = np.ma.masked_where(pred[0] < 0.5, pred[0])
        ax2.imshow(pred_mask, alpha=0.5, cmap='Reds')
        ax2.set_title('Prediction')
        ax2.axis('off')
        
        # Ground truth overlay
        ax3 = plt.subplot(2, 2, 3)
        ax3.imshow(img)
        gt_mask = np.ma.masked_where(target[0] == 0, target[0])
        ax3.imshow(gt_mask, alpha=0.5, cmap='Greens')
        ax3.set_title('Ground Truth')
        ax3.axis('off')
        
        # Confidence map
        ax4 = plt.subplot(2, 2, 4)
        if confidence_maps is not None:
            conf = confidence_maps[idx].cpu().numpy()[0]
            ax4.imshow(conf, cmap='viridis')
            ax4.set_title('Confidence Map')
        else:
            error = np.abs(pred[0] - target[0])
            ax4.imshow(error, cmap='RdYlBu')
            ax4.set_title('Prediction Error')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Convert figure to numpy array
        width = int(fig.bbox.bounds[2])
        height = int(fig.bbox.bounds[3])
        
        fig.canvas.draw()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 3)
        
        batch_visualizations.append({
            'image': image_array,
            'metrics': {
                'dice': metrics['dice'].item(),
                'iou': metrics['iou'].item()
            }
        })
        
        plt.close(fig)
        plt.ion()
    
    return batch_visualizations

def create_grid_summary(images: List[np.ndarray], title: str, metrics: List[Dict]) -> np.ndarray:
    """
    Create a grid of images with metrics.
    
    Args:
        images: List of numpy arrays containing images
        title: Title for the grid
        metrics: List of metric dictionaries for each image
    
    Returns:
        np.ndarray: Grid image as numpy array
    """
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure with fixed DPI
    plt.ioff()  # Turn off interactive mode
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows), dpi=100)
    plt.suptitle(title, fontsize=16)
    
    for idx, (img, metric) in enumerate(zip(images, metrics)):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.imshow(img)  # Directly use numpy array
        ax.set_title(f'Dice: {metric["dice"]:.3f}\nIoU: {metric["iou"]:.3f}')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Get the size before drawing
    width = int(fig.bbox.bounds[2])
    height = int(fig.bbox.bounds[3])
    
    # Draw and convert to numpy array
    fig.canvas.draw()
    
    # Get the color buffer
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(height, width, 3)
    
    plt.close(fig)
    plt.ion()  # Turn interactive mode back on
    
    return img_array