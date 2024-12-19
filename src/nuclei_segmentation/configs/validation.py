from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig

def validate_data_config(config: DictConfig) -> None:
    """Validate data configuration."""
    data_root = Path(config.data.root_dir)
    
    # Check if path exists
    if not data_root.exists():
        raise ValueError(
            f"Data root directory not found: {data_root}\n"
            "Please either:\n"
            "1. Update configs/data/default.yaml with correct path\n"
            "2. Set DATA_ROOT environment variable\n"
            "3. Move data to expected location"
        )
    
    # Check required subdirectories
    required_dirs = [
        data_root / "train" / "images",
        data_root / "train" / "masks" / config.data.mask_type,
        data_root / "val" / "images",
        data_root / "val" / "masks" / config.data.mask_type,
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise ValueError(f"Required directory not found: {dir_path}")
            
def validate_config(config: DictConfig) -> None:
    """Validate full configuration."""
    # Validate data configuration
    validate_data_config(config)
