from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DataConfig:
    image_size: int = 256
    batch_size: int = 32
    min_cells_per_image: int = 3
    datasets: List[str] = ('TNBC',)  
    
@dataclass
class ModelConfig:
    name: str = 'unet'
    in_channels: int = 3
    initial_features: int = 64
    
@dataclass
class TrainingConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    device: str = 'cuda'
    wandb_project: str = 'nuclei-segmentation'