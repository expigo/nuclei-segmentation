from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_image_formats(data_root: str = "data/raw"):
    """Analyze image formats across datasets and splits."""
    root = Path(data_root)
    stats = defaultdict(lambda: defaultdict(int))
    
    # Analyze each split
    for split in ['train', 'val', 'test']:
        image_dir = root / split / 'images'
        logger.info(f"Analyzing {split} split...")
        
        for img_path in image_dir.glob("*.png"):
            # Get dataset name
            if "MoNuSAC" in img_path.name:
                dataset = "MoNuSAC"
            elif "NuCLS" in img_path.name:
                dataset = "NuCLS"
            elif "PanNuke" in img_path.name:
                dataset = "PanNuke"
            elif "TNBC" in img_path.name:
                dataset = "TNBC"
            else:
                dataset = "Unknown"
            
            # Load and check image
            img = np.array(Image.open(img_path))
            channels = img.shape[-1] if len(img.shape) > 2 else 1
            key = f"{dataset}_{split}"
            stats[key][channels] += 1
    
    # Convert to DataFrame for better visualization
    data = []
    for key, counts in stats.items():
        dataset, split = key.split('_')
        for channels, count in counts.items():
            data.append({
                'Dataset': dataset,
                'Split': split,
                'Channels': channels,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    return df

def main():
    # Analyze formats
    df = analyze_image_formats()
    
    # Print summary
    print("\nImage Format Analysis:")
    print(df.to_string())
    
    # Print recommendations
    print("\nRecommendations:")
    inconsistent_datasets = df.groupby('Dataset')['Channels'].nunique() > 1
    inconsistent_datasets = inconsistent_datasets[inconsistent_datasets].index
    
    if len(inconsistent_datasets) > 0:
        print("\nInconsistent channel counts found in datasets:")
        for dataset in inconsistent_datasets:
            print(f"\n{dataset}:")
            subset = df[df['Dataset'] == dataset]
            print(subset.to_string())
            
        print("\nRecommended fix: Convert all images to RGB during loading")
    else:
        print("All images have consistent channel counts within datasets")

if __name__ == "__main__":
    main()