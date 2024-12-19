import wandb
from pathlib import Path
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelCheckpointManager:
    """
    Manages model checkpoints and their tracking in Weights & Biases.
    Handles saving checkpoints, logging metadata, and maintaining experiment history.
    """
    def __init__(self, 
                 experiment_name: str,
                 wandb_run: wandb.Run,
                 config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.wandb_run = wandb_run
        self.config = config
        
        # Create directory for local checkpoint storage
        self.checkpoint_dir = Path("models/checkpoints") / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized checkpoint manager for {experiment_name}")
        logger.info(f"Storing checkpoints in {self.checkpoint_dir}")

    def save_checkpoint(self,
                       model: pl.LightningModule,
                       metrics: Dict[str, float],
                       epoch: int,
                       is_best: bool = False) -> Path:
        """
        Saves a model checkpoint and logs it to W&B with appropriate metadata.
        
        Args:
            model: The PyTorch Lightning model to save
            metrics: Current performance metrics
            epoch: Current training epoch
            is_best: Whether this is the best model so far
        
        Returns:
            Path to saved checkpoint
        """
        # Create meaningful checkpoint name
        checkpoint_name = f"epoch_{epoch}"
        if is_best:
            checkpoint_name += "_best"
            
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.ckpt"
        
        # Save the checkpoint locally
        model.save_checkpoint(checkpoint_path)
        
        # Create W&B artifact
        artifact = wandb.Artifact(
            name=f"model-{self.experiment_name}",
            type="model",
            metadata={
                "epoch": epoch,
                "metrics": metrics,
                "model_config": self.config.model,
                "dataset_config": self.config.data,
                "hardware_config": self.config.hardware
            }
        )
        
        # Add checkpoint file to artifact
        artifact.add_file(checkpoint_path)
        
        # Set appropriate aliases
        aliases = ["latest"]
        if is_best:
            aliases.append("best")
            
        # Log artifact to W&B
        self.wandb_run.log_artifact(artifact, aliases=aliases)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path