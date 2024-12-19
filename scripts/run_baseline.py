import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path

from nuclei_segmentation.training.experiment import ExperimentManager

WANDB_PROJECT = "nuclei-segmentation"

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_baseline(cfg: DictConfig):
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize experiment
    experiment = ExperimentManager(
        # config=OmegaConf.to_container(cfg, resolve=True),
        config=cfg,
        project_name=WANDB_PROJECT,
        experiment_name=f"baseline_unet_{cfg.data.mask_type}"
    )
    
    # Train model
    model = experiment.train()
    
    # Evaluate on each dataset separately
    datasets = ['MoNuSAC', 'NuCLS', 'PanNuke', 'TNBC']
    for dataset in datasets:
        print(f"\nEvaluating on {dataset}...")
        results = experiment.evaluate(model, dataset_name=dataset)
        print(f"Results for {dataset}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    
    # Evaluate on combined test set
    print("\nEvaluating on combined test set...")
    results = experiment.evaluate(model)
    print("Combined results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    train_baseline()