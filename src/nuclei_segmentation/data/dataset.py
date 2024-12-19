from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from enum import Enum
import torch
from torch.utils.data import Dataset
import os

import logging

class MaskType(Enum):
    BINARY = "binaries"
    CELL_TYPES = "cell_types"
    CONTOURS = "contours"
    MULTI_INSTANCE = "multi_instance"

class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"

class NucleiDatasetInspector:
    """Class for inspecting nuclei segmentation dataset."""
    
    def __init__(self, data_root: Union[str, Path]):
        self.data_root = Path(data_root)
        self._validate_directory_structure()
    
    def _validate_directory_structure(self) -> None:
        """Validate if the directory structure is correct."""
        for split in DatasetSplit:
            split_dir = self.data_root / split.value
            assert split_dir.exists(), f"Missing {split.value} directory"
            assert (split_dir / "images").exists(), f"Missing images in {split.value}"
            assert (split_dir / "masks").exists(), f"Missing masks in {split.value}"
            
            for mask_type in MaskType:
                assert (split_dir / "masks" / mask_type.value).exists(), \
                    f"Missing {mask_type.value} in {split.value}/masks"
    
    def load_sample(self, 
                   split: DatasetSplit, 
                   index: int) -> Dict[str, np.ndarray]:
        """Load an image and all its corresponding masks."""
        split_path = self.data_root / split.value
        
        # Get image paths
        image_files = sorted(list((split_path / "images").glob("*.png")))
        if index >= len(image_files):
            raise IndexError(f"Index {index} out of range for {split.value} split")
            
        # Load image and masks
        sample = {
            "image": io.imread(image_files[index]),
            "masks": {}
        }
        
        # Load all mask types
        for mask_type in MaskType:
            mask_path = split_path / "masks" / mask_type.value / image_files[index].name
            sample["masks"][mask_type.value] = io.imread(mask_path)
            
        return sample

    def visualize_sample(self, 
                        split: DatasetSplit, 
                        index: int, 
                        figsize: Tuple[int, int] = (15, 10)) -> None:
        """Visualize an image and all its masks."""
        sample = self.load_sample(split, index)
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f"Sample {index} from {split.value} split")
        
        # Plot original image
        axes[0, 0].imshow(sample["image"])
        axes[0, 0].set_title("Original Image")
        
        # Plot masks
        for idx, (mask_type, mask) in enumerate(sample["masks"].items(), 1):
            row, col = divmod(idx, 3)
            ax = axes[row, col]
            
            if mask_type in [MaskType.CELL_TYPES.value, MaskType.MULTI_INSTANCE.value]:
                # For 16-bit masks, use a colormap and show unique values
                unique_values = np.unique(mask)
                im = ax.imshow(mask, cmap='tab20')
                ax.set_title(f"{mask_type}\nUnique values: {len(unique_values)-1}")
                plt.colorbar(im, ax=ax)
            else:
                # For binary masks
                ax.imshow(mask, cmap='gray')
                ax.set_title(mask_type)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.show()
        
    def get_mask_statistics(self, 
                          split: DatasetSplit, 
                          mask_type: MaskType) -> Dict:
        """Get statistics for a specific mask type in a split."""
        split_path = self.data_root / split.value / "masks" / mask_type.value
        mask_files = list(split_path.glob("*.png"))
        
        statistics = {
            "total_files": len(mask_files),
            "unique_values": set()
        }
        
        for mask_file in mask_files[:100]:  # Sample first 100 files
            mask = io.imread(mask_file)
            statistics["unique_values"].update(np.unique(mask))
            
        return statistics

logger = logging.getLogger(__name__)

class NucleiDataset(Dataset):
    """Dataset for nuclei segmentation."""
    
    AVAILABLE_DATASETS = ['MoNuSAC', 'NuCLS', 'PanNuke', 'TNBC']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        mask_type: str = 'binaries',
        dataset_name: str = 'all',
        transform = None
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.mask_type = mask_type
        self.transform = transform
        self.dataset_name = dataset_name
        self._worker_pool = None

        # Validate dataset name
        if dataset_name != 'all' and dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Invalid dataset name. Available options: {['all'] + self.AVAILABLE_DATASETS}")
        
        # Log initialization
        logger.info(f"Initializing NucleiDataset:")
        logger.info(f"  root_dir: {self.root_dir}")
        logger.info(f"  split: {split}")
        logger.info(f"  mask_type: {mask_type}")
        logger.info(f"  dataset: {dataset_name}")
        
        # Setup paths
        self.image_dir = self.root_dir / split / 'images'
        self.mask_dir = self.root_dir / split / 'masks' / mask_type
        
        # Verify directories exist
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise ValueError(f"Mask directory not found: {self.mask_dir}")
            
        # Get all image files
        image_files = sorted(list(self.image_dir.glob('*.png')))
        logger.info(f"Found {len(image_files)} total images")
        
        # Filter by dataset if specified
        if dataset_name != 'all':
            image_files = [
                f for f in image_files 
                if self._get_dataset_name(f.name) == dataset_name
            ]
            logger.info(f"Filtered to {len(image_files)} images from {dataset_name}")
        
        # Verify each image has a corresponding mask
        self.valid_pairs = []
        for img_path in image_files:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
            else:
                logger.warning(f"No mask found for image: {img_path}")
        
        logger.info(f"Final dataset size: {len(self.valid_pairs)} valid image-mask pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError(
                f"No valid image-mask pairs found in:\n"
                f"  Images: {self.image_dir}\n"
                f"  Masks: {self.mask_dir}"
            )
    
    def _load_and_preprocess_image(self, img_path: Path) -> torch.Tensor:
        """Load and preprocess an image ensuring RGB format."""
        try:
            # Load image with skimage
            from skimage import io
            image = io.imread(str(img_path))
            
            # Ensure RGB if image is grayscale
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            # Convert RGBA to RGB if needed
            elif image.shape[-1] == 4:
                image = image[..., :3]
                
            # Convert to torch tensor and normalize
            image = torch.from_numpy(image).float().permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
            image = image / 255.0  # Normalize to [0,1]
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise

    
    def _load_and_preprocess_mask(self, mask_path: Path) -> torch.Tensor:
        """Load and preprocess a mask based on mask type."""
        try:
            mask = np.array(io.imread(mask_path))
            
            if self.mask_type == 'binaries':
                mask = torch.from_numpy(mask).float().unsqueeze(0)  # Add channel dim
                mask = (mask > 0).float()  # Convert to binary
            else:
                mask = torch.from_numpy(mask).long()
                
            return mask
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {str(e)}")
            raise

    def _get_dataset_name(self, filename: str) -> str:
        """Extract dataset name from filename."""
        for dataset in self.AVAILABLE_DATASETS:
            if dataset in filename:
                return dataset
        return 'Unknown'
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        # Get paths
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load and preprocess image and mask
        image = self._load_and_preprocess_image(img_path)
        mask = self._load_and_preprocess_mask(mask_path)
            
        # Apply transforms if any
        if self.transform is not None:
            image, mask = self.transform(image, mask)
            
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
            'dataset': self._get_dataset_name(img_path.name)
        }
    def cleanup(self):
        """Explicit cleanup method"""
        if hasattr(self, '_worker_pool') and self._worker_pool is not None:
            self._worker_pool.close()
            self._worker_pool.join()
            self._worker_pool = None
            
    def __del__(self):
        """Fallback cleanup"""
        try:
            self.cleanup()
        except:
            pass