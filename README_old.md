# Nuclei Segmentation

Deep Learning methods for detecting cells in histopathological images.

## Setup

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended)
- Mamba or Conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nuclei-segmentation
```

2. Create environment using mamba/conda:
```bash
mamba env create -f environment.yml
mamba activate nuclei-seg
```

3. Install package in development mode:
```bash
pip install -e ".[dev]"
```

4. Configure DVC
```bash
dvc init
```

5. Login to Weights&Biases
```bash
wandb login
```

### Project Structure
```
nuclei_segmentation/
├── configs/                    # Configuration files
│   ├── model_config.yaml      # Model parameters
│   └── train_config.yaml      # Training parameters
├── data/
│   ├── raw/                   # Original dataset (tracked by DVC)
│   └── processed/             # Processed dataset
├── src/
│   └── nuclei_segmentation/
│       ├── data/              # Data processing
│       ├── models/            # Model implementations
│       └── utils/             # Utility functions
├── tests/                     # Test files
├── metrics/                   # Training metrics
├── models/                    # Saved models
├── environment.yml            # Environment specification
├── pyproject.toml            # Project configuration
└── dvc.yaml                  # DVC pipeline

```

### Dataset structure
```
data/raw/
├── train/
│   ├── images/
│   └── masks/
│       ├── binaries/         # Binary masks (8-bit)
│       ├── cell_types/       # Cell type annotations (16-bit)
│       ├── contours/         # Cell contours (8-bit)
│       └── multi_instance/   # Instance segmentation (16-bit)
├── val/
└── test/
```


## Development
### Running tests

```bash
pytest tests/ -v
```

### DVC Pipeline

The pipeline consists of two stages:
1. Data preparation:
```bash
dvc repo prepare
```
2. Training baseline model:
```bash
dvc repo train_baseline
```

To run the full pipeline:
```bash
dvc repro
```

### Configuration
The project uses two main configuration files:

- configs/model_config.yaml: Model architecture parameters
- configs/train_config.yaml: Training hyperparameters

### Experiment Tracking
Experiments are tracked using Weights&Biases


# Guide: Adding New Models

## Step 1: Create Model Class
Create a new file in `src/nuclei_segmentation/models/your_model_name/` with your model implementation:

```python
# src/nuclei_segmentation/models/your_model_name/model.py
from typing import Dict, Any
import pytorch_lightning as pl
import torch.nn as nn

class YourModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize your model architecture here
        self.model = nn.Sequential(...)
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
```

## Step 2: Update Model Registry

Add new model to the registry in `src/nuclei_segmentation/models/__init__.py`:
```python
from .unet.base import UNet
from .your_model_name.model import YourModel

MODEL_REGISTRY = {
    'unet': UNet,
    'your_model': YourModel,
}
```
## Step 3: Add Model COnfiguration

Add the new model config to `configs/model_config.yaml`:
```yaml
new_model:
    param1: value1
    param2: value2
    # Add all model-specific parameters
```

### Step 4: Create Tests

Add tests for the new model in `tests/models/new_model.py`:
```python
import pytest
import torch
from nuclei_segmentation.models.your_model_name.model import YourModel

def test_new_model_output_shape(sample_batch):
    model = YourModel(config={})
    output = model(sample_batch['image'])
    assert output.shape == sample_batch['mask'].shape

def test_new_model_training_step(sample_batch):
    model = YourModel(config={})
    loss = model.training_step((sample_batch['image'], sample_batch['mask']), 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad
```

### Step 5: Add Documentation
## NewModel

### Description
Brief description of new model and its key features.

### Architecture
Explanation of the model architecture, possibly with a diagram.

### Configuration
List of all configuration parameters and their descriptions:
- param1: Description of param1
- param2: Description of param2

### Performance
Benchmark results on different datasets and/or mask types.

### Step 6: Update DVC Pipeline
If the model requires sprecial preprocessing or training stes, update `dvc.yaml`
```yaml
stages:
  train_your_model:
    cmd: python -m nuclei_segmentation.training.train --model your_model
    deps:
      - data/processed
      - src/nuclei_segmentation/models/your_model_name/model.py
    params:
      - configs/model_config.yaml
      - configs/train_config.yaml
    metrics:
      - metrics/your_model_metrics.json:
          cache: false
    outs:
      - models/your_model.pt
```