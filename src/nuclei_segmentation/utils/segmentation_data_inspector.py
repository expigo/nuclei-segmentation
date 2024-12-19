from skimage import measure
from scipy import ndimage
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import defaultdict
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


from nuclei_segmentation.utils.data_inspector import DataInspector

class SegmentationDataInspector(DataInspector):
    """Enhanced inspector with segmentation-specific analyses."""
    
    def analyze_segmentation_properties(self, mask_type: str = 'binaries') -> pd.DataFrame:
        """Analyze properties relevant for segmentation tasks."""
        properties = []
        
        for split in self.splits:
            image_dir = self.data_root / split / 'images'
            mask_dir = self.data_root / split / 'masks' / mask_type
            
            for img_path in tqdm(list(image_dir.glob('*.png')), desc=f'Analyzing {split} split'):
                img = np.array(Image.open(img_path))
                mask = np.array(Image.open(mask_dir / img_path.name))
                dataset_name = self._get_dataset_name(img_path.name)
                
                # Connected components analysis
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]  # Take first channel if multi-channel
                labels = measure.label(mask > 0)
                regions = measure.regionprops(labels)
                
                # Calculate properties
                prop_dict = {
                    'split': split,
                    'dataset': dataset_name,
                    'filename': img_path.name,
                    'num_instances': len(regions),
                    'avg_instance_size': np.mean([r.area for r in regions]) if regions else 0,
                    'std_instance_size': np.std([r.area for r in regions]) if regions else 0,
                    'min_instance_size': min([r.area for r in regions]) if regions else 0,
                    'max_instance_size': max([r.area for r in regions]) if regions else 0,
                    'foreground_ratio': (mask > 0).mean(),
                    'image_intensity_mean': img.mean(),
                    'image_intensity_std': img.std(),
                    'image_size': img.shape,
                    'mask_unique_values': len(np.unique(mask))
                }
                
                # Calculate instance spacing
                if len(regions) > 1:
                    centroids = np.array([r.centroid for r in regions])
                    distances = []
                    for i in range(len(centroids)):
                        dist = np.sqrt(np.sum((centroids - centroids[i])**2, axis=1))
                        dist = dist[dist > 0]  # Remove self-distance
                        if len(dist) > 0:
                            distances.append(np.min(dist))
                    
                    prop_dict.update({
                        'avg_instance_spacing': np.mean(distances) if distances else 0,
                        'min_instance_spacing': np.min(distances) if distances else 0
                    })
                else:
                    prop_dict.update({
                        'avg_instance_spacing': 0,
                        'min_instance_spacing': 0
                    })
                
                properties.append(prop_dict)
        
        return pd.DataFrame(properties)
    
    def analyze_class_balance(self, mask_type: str = 'cell_types') -> pd.DataFrame:
        """Analyze class distribution in masks."""
        class_stats = []
        
        for split in self.splits:
            mask_dir = self.data_root / split / 'masks' / mask_type
            
            for mask_path in tqdm(list(mask_dir.glob('*.png')), desc=f'Analyzing {split} classes'):
                mask = np.array(Image.open(mask_path))
                dataset_name = self._get_dataset_name(mask_path.name)
                
                unique, counts = np.unique(mask, return_counts=True)
                class_dist = dict(zip(unique, counts))
                
                class_stats.append({
                    'split': split,
                    'dataset': dataset_name,
                    'filename': mask_path.name,
                    'class_distribution': class_dist,
                    'num_classes': len(unique),
                    'majority_class': unique[np.argmax(counts)],
                    'majority_class_ratio': np.max(counts) / np.sum(counts)
                })
        
        return pd.DataFrame(class_stats)
    
    def suggest_unet_parameters(self, mask_type: str = 'binaries') -> Dict:
        """Suggest U-Net architecture parameters based on data analysis."""
        seg_props = self.analyze_segmentation_properties(mask_type)
        
        # Debug prints
        print("\nDataset Properties:")
        print(f"Number of images: {len(seg_props)}")
        print("\nColumns available:")
        print(seg_props.columns.tolist())
        print("\nSample statistics:")
        print(seg_props.describe())
        seg_props.describe().to_csv("./unet_dataset_sample_statistics.py")
        
        # Analyze image sizes
        image_sizes = seg_props['image_size'].unique()
        target_size = self._suggest_target_size(image_sizes)
        
        # Analyze instance sizes to suggest network depth
        avg_instance_size = seg_props['avg_instance_size'].median()
        suggested_depth = int(np.log2(min(target_size) / (avg_instance_size ** 0.5))) + 1
        
        # Calculate class weights if needed
        class_weights = None
        if mask_type == 'cell_types':
            class_stats = self.analyze_class_balance(mask_type)
            class_weights = self._calculate_class_weights(class_stats)
        
        suggestions = {
            'architecture': {
                'input_size': target_size,
                'depth': min(max(suggested_depth, 3), 5),  # Limit between 3 and 5
                'initial_features': 64,  # Standard U-Net starting point
                'dropout_rate': 0.2 if seg_props['foreground_ratio'].mean() < 0.3 else 0.1
            },
            'training': {
                'batch_size': self._suggest_batch_size(target_size),
                'suggested_learning_rate': 1e-4,
                'class_weights': class_weights,
                'suggested_augmentations': self._suggest_augmentations(seg_props)
            },
            'dataset_stats': {
                'avg_instances_per_image': seg_props['num_instances'].mean(),
                'foreground_ratio': seg_props['foreground_ratio'].mean(),
                'avg_instance_size': seg_props['avg_instance_size'].mean(),
                'size_variation': seg_props['std_instance_size'].mean() / seg_props['avg_instance_size'].mean()
            }
        }
        
        return suggestions
    
    def _suggest_target_size(self, image_sizes) -> Tuple[int, int]:
        """Suggest target size for network input."""
        sizes = np.array([eval(str(size)) for size in image_sizes])
        median_size = np.median(sizes, axis=0)
        
        # Round to nearest power of 2
        target_size = 2 ** np.ceil(np.log2(median_size))
        return tuple(int(s) for s in target_size[:2])
    
    def _suggest_batch_size(self, target_size: Tuple[int, int]) -> int:
        """Suggest batch size based on image size."""
        pixel_count = target_size[0] * target_size[1]
        if pixel_count > 512 * 512:
            return 8
        elif pixel_count > 256 * 256:
            return 16
        else:
            return 32
    
    def _suggest_augmentations(self, seg_props: pd.DataFrame) -> List[str]:
        """Suggest data augmentations based on dataset properties."""
        augmentations = ['random_flip', 'random_rotate']
        
        # Calculate size variation using std_instance_size and avg_instance_size
        size_variation = (
            seg_props['std_instance_size'] / 
            (seg_props['avg_instance_size'].where(seg_props['avg_instance_size'] > 0, 1))
        ).mean()
        
        if seg_props['image_intensity_std'].mean() > 20:
            augmentations.extend(['random_brightness', 'random_contrast'])
        
        if size_variation > 0.5:
            augmentations.append('random_scale')
                
        return augmentations
    
    def visualize_instance_statistics(self, mask_type: str = 'binaries') -> None:
        """Visualize key statistics for segmentation."""
        seg_props = self.analyze_segmentation_properties(mask_type)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Instance count distribution
        sns.boxplot(data=seg_props, x='dataset', y='num_instances', ax=axes[0,0])
        axes[0,0].set_title('Number of Instances per Image')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # Instance size distribution
        sns.boxplot(data=seg_props, x='dataset', y='avg_instance_size', ax=axes[0,1])
        axes[0,1].set_title('Average Instance Size')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        
        # Foreground ratio
        sns.boxplot(data=seg_props, x='dataset', y='foreground_ratio', ax=axes[1,0])
        axes[1,0].set_title('Foreground Ratio')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # Instance spacing
        sns.boxplot(data=seg_props, x='dataset', y='avg_instance_spacing', ax=axes[1,1])
        axes[1,1].set_title('Average Instance Spacing')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()

def analyze_dataset_for_unet(data_root: str, mask_type: str = 'binaries'):
    """Convenience function to analyze dataset and suggest U-Net parameters."""
    inspector = SegmentationDataInspector(data_root)
    
    print("Analyzing dataset...")
    suggestions = inspector.suggest_unet_parameters(mask_type)
    
    print("\n=== U-Net Architecture Suggestions ===")
    print(f"Recommended input size: {suggestions['architecture']['input_size']}")
    print(f"Recommended depth: {suggestions['architecture']['depth']}")
    print(f"Initial features: {suggestions['architecture']['initial_features']}")
    print(f"Dropout rate: {suggestions['architecture']['dropout_rate']}")
    
    print("\n=== Training Suggestions ===")
    print(f"Batch size: {suggestions['training']['batch_size']}")
    print(f"Learning rate: {suggestions['training']['suggested_learning_rate']}")
    print(f"Recommended augmentations: {', '.join(suggestions['training']['suggested_augmentations'])}")
    
    print("\n=== Dataset Statistics ===")
    print(f"Average instances per image: {suggestions['dataset_stats']['avg_instances_per_image']:.2f}")
    print(f"Average foreground ratio: {suggestions['dataset_stats']['foreground_ratio']:.2f}")
    
    # Visualize statistics
    inspector.visualize_instance_statistics(mask_type)
    
    return suggestions

if __name__ == "__main__":
    suggestions = analyze_dataset_for_unet("data/raw", "binaries")