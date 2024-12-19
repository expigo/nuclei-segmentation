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

class DataInspector:
    """Comprehensive data inspector for nuclei segmentation datasets."""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.splits = ['train', 'val', 'test']
        self.mask_types = ['binaries', 'cell_types', 'contours', 'multi_instance']
        
    def _get_dataset_name(self, filename: str) -> str:
        """Extract dataset name from filename based on pattern."""
        if 'TNBC' in filename:
            return 'TNBC'
        elif 'MoNuSAC' in filename:
            return 'MoNuSAC'
        elif 'NuCLS' in filename:
            return 'NuCLS'
        elif 'PanNuke' in filename:
            return 'PanNuke'
        return 'Unknown'
    
    def analyze_image_properties(self) -> pd.DataFrame:
        """Analyze image properties across all splits and datasets."""
        properties = []
        
        for split in self.splits:
            image_dir = self.data_root / split / 'images'
            for img_path in tqdm(list(image_dir.glob('*.png')), desc=f'Analyzing {split} split'):
                img = np.array(Image.open(img_path))
                dataset_name = self._get_dataset_name(img_path.name)
                
                properties.append({
                    'split': split,
                    'dataset': dataset_name,
                    'filename': img_path.name,
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'channels': img.shape[2] if len(img.shape) > 2 else 1,
                    'min_value': img.min(),
                    'max_value': img.max(),
                    'mean_value': img.mean(),
                    'std_value': img.std()
                })
        
        return pd.DataFrame(properties)
    
    def analyze_masks(self) -> Dict[str, pd.DataFrame]:
        """Analyze mask properties for each mask type."""
        mask_analyses = {}
        
        for mask_type in self.mask_types:
            properties = []
            
            for split in self.splits:
                mask_dir = self.data_root / split / 'masks' / mask_type
                for mask_path in tqdm(list(mask_dir.glob('*.png')), 
                                    desc=f'Analyzing {mask_type} masks in {split}'):
                    mask = np.array(Image.open(mask_path))
                    dataset_name = self._get_dataset_name(mask_path.name)
                    
                    properties.append({
                        'split': split,
                        'dataset': dataset_name,
                        'filename': mask_path.name,
                        'unique_values': len(np.unique(mask)),
                        'min_value': mask.min(),
                        'max_value': mask.max(),
                        'mean_value': mask.mean(),
                        'non_zero_ratio': (mask > 0).mean()
                    })
            
            mask_analyses[mask_type] = pd.DataFrame(properties)
        
        return mask_analyses
    
    def plot_dataset_distribution(self) -> None:
        """Plot distribution of images across datasets and splits."""
        all_files = []
        
        for split in self.splits:
            image_dir = self.data_root / split / 'images'
            for img_path in image_dir.glob('*.png'):
                all_files.append({
                    'split': split,
                    'dataset': self._get_dataset_name(img_path.name)
                })
        
        df = pd.DataFrame(all_files)
        
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='dataset', hue='split')
        plt.title('Distribution of Images Across Datasets and Splits')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def visualize_samples(self, num_samples: int = 3) -> None:
        """Visualize random samples from each dataset with all mask types."""
        for dataset in ['TNBC', 'MoNuSAC', 'NuCLS', 'PanNuke']:
            # Find samples from this dataset
            image_dir = self.data_root / 'train' / 'images'
            dataset_images = [f for f in image_dir.glob('*.png') 
                            if self._get_dataset_name(f.name) == dataset]
            
            if not dataset_images:
                continue
                
            selected_images = np.random.choice(dataset_images, 
                                             size=min(num_samples, len(dataset_images)), 
                                             replace=False)
            
            for img_path in selected_images:
                fig, axes = plt.subplots(1, len(self.mask_types) + 1, 
                                       figsize=(20, 4))
                
                # Load and display image
                img = np.array(Image.open(img_path))
                axes[0].imshow(img)
                axes[0].set_title(f'Original\n{img_path.name}')
                
                # Load and display masks
                for idx, mask_type in enumerate(self.mask_types, 1):
                    mask_path = self.data_root / 'train' / 'masks' / mask_type / img_path.name
                    mask = np.array(Image.open(mask_path))
                    axes[idx].imshow(mask)
                    axes[idx].set_title(f'{mask_type}\nUnique values: {len(np.unique(mask))}')
                
                plt.suptitle(f'{dataset} Sample')
                plt.tight_layout()
                plt.show()
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive report of the dataset."""
        report = {
            'basic_stats': {},
            'image_properties': None,
            'mask_analyses': None,
            'dataset_distribution': None
        }
        
        # Basic statistics
        total_images = 0
        dataset_counts = defaultdict(int)
        
        for split in self.splits:
            image_dir = self.data_root / split / 'images'
            split_count = len(list(image_dir.glob('*.png')))
            total_images += split_count
            
            for img_path in image_dir.glob('*.png'):
                dataset_counts[self._get_dataset_name(img_path.name)] += 1
        
        report['basic_stats'] = {
            'total_images': total_images,
            'dataset_distribution': dict(dataset_counts),
            'splits': {split: len(list((self.data_root / split / 'images').glob('*.png'))) 
                      for split in self.splits}
        }
        
        # Detailed analyses
        report['image_properties'] = self.analyze_image_properties()
        report['mask_analyses'] = self.analyze_masks()
        
        return report

def main():
    inspector = DataInspector("data/raw")
    
    # Generate and print report
    report = inspector.generate_report()
    
    print("\n=== Dataset Overview ===")
    print(f"Total images: {report['basic_stats']['total_images']}")
    print("\nDataset distribution:")
    for dataset, count in report['basic_stats']['dataset_distribution'].items():
        print(f"{dataset}: {count} images")
    
    # Visualize distributions
    inspector.plot_dataset_distribution()
    
    # Show sample images
    inspector.visualize_samples()

if __name__ == "__main__":
    main()