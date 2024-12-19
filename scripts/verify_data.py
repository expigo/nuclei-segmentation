"""Script to verify dataset structure and contents."""
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_dataset_structure(data_root: str):
    """Verify that the dataset structure is correct."""
    root = Path(data_root)
    
    # Check root exists
    if not root.exists():
        logger.error(f"Data root does not exist: {root}")
        return False
        
    # Expected structure
    splits = ['train', 'val', 'test']
    mask_types = ['binaries', 'cell_types', 'contours', 'multi_instance']
    
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            logger.error(f"Split directory missing: {split_dir}")
            continue
            
        # Check images directory
        image_dir = split_dir / 'images'
        if not image_dir.exists():
            logger.error(f"Images directory missing: {image_dir}")
            continue
            
        images = list(image_dir.glob('*.png'))
        logger.info(f"Found {len(images)} images in {split} split")
        
        # Check masks directories
        for mask_type in mask_types:
            mask_dir = split_dir / 'masks' / mask_type
            if not mask_dir.exists():
                logger.error(f"Mask directory missing: {mask_dir}")
                continue
                
            masks = list(mask_dir.glob('*.png'))
            logger.info(f"Found {len(masks)} masks in {split}/{mask_type}")
            
            # Verify each image has a mask
            for img_path in images:
                mask_path = mask_dir / img_path.name
                if not mask_path.exists():
                    logger.warning(f"Missing mask for {img_path.name} in {mask_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/raw",
                      help="Path to dataset root directory")
    args = parser.parse_args()
    
    verify_dataset_structure(args.data_root)