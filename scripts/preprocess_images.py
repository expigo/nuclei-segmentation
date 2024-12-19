"""Script to convert all dataset images to RGB format."""
from pathlib import Path
import logging
from PIL import Image
from tqdm import tqdm
import shutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_images_to_rgb(data_root: str = "data/raw", backup: bool = True):
    """Convert all images in the dataset to RGB format.
    
    Args:
        data_root: Path to the data directory
        backup: If True, create a backup of original data
    """
    root = Path(data_root)
    
    # Create backup if requested
    if backup:
        backup_dir = root.parent / (root.name + "_backup")
        if not backup_dir.exists():
            logger.info(f"Creating backup in {backup_dir}")
            shutil.copytree(root, backup_dir)
        else:
            logger.warning(f"Backup directory {backup_dir} already exists, skipping backup")
    
    # Process each split
    for split in ['train', 'val', 'test']:
        image_dir = root / split / 'images'
        if not image_dir.exists():
            logger.warning(f"Directory not found: {image_dir}")
            continue
            
        logger.info(f"Processing {split} split...")
        
        # Convert each image
        image_files = list(image_dir.glob("*.png"))
        for img_path in tqdm(image_files, desc=f"Converting {split} images"):
            try:
                # Open image and convert to RGB
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        # Convert to RGB and save back
                        img_rgb = img.convert('RGB')
                        img_rgb.save(img_path)
                        
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")

def verify_conversion(data_root: str = "data/raw"):
    """Verify all images are in RGB format."""
    root = Path(data_root)
    all_rgb = True
    
    for split in ['train', 'val', 'test']:
        image_dir = root / split / 'images'
        if not image_dir.exists():
            continue
            
        logger.info(f"Verifying {split} split...")
        
        for img_path in tqdm(list(image_dir.glob("*.png")), desc=f"Checking {split} images"):
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    logger.error(f"Image {img_path} is not RGB (mode: {img.mode})")
                    all_rgb = False
    
    return all_rgb

def main():
    # Get data directory
    data_dir = "data/raw"
    
    # Create backup and convert images
    logger.info("Starting image conversion...")
    convert_images_to_rgb(data_dir)
    
    # Verify conversion
    logger.info("\nVerifying conversion...")
    if verify_conversion(data_dir):
        logger.info("All images successfully converted to RGB")
    else:
        logger.error("Some images were not converted successfully")
        
    logger.info("\nBackup of original data is stored in data/raw_backup")
    logger.info("If conversion results are satisfactory, you can remove the backup")

if __name__ == "__main__":
    main()