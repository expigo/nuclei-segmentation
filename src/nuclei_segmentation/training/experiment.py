from omegaconf import OmegaConf
import wandb
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import logging
import matplotlib.pyplot as plt


from nuclei_segmentation.models import MODEL_REGISTRY
from nuclei_segmentation.data.dataset import NucleiDataset
from nuclei_segmentation.metrics.mask_metrics import MaskMetrics
from nuclei_segmentation.utils.checkpoint_manager import ModelCheckpointManager
from nuclei_segmentation.utils.visualization import create_mask_overlay, create_enhanced_mask_overlay, create_grid_summary

logger = logging.getLogger(__name__)

class CheckpointLoggingCallback(pl.callbacks.Callback):
    """
    Custom callback to handle checkpoint logging to W&B.
    Tracks training progress and saves checkpoints at appropriate times.
    """
    def __init__(self, checkpoint_manager: ModelCheckpointManager):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.best_val_loss = float('inf')
        
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called when validation ends, saves checkpoint if appropriate."""
        # Get current metrics
        metrics = trainer.callback_metrics
        
        # Check if this is the best model so far
        current_val_loss = metrics.get('val_loss', float('inf'))
        is_best = current_val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = current_val_loss
            
        # Save checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=pl_module,
            metrics=metrics,
            epoch=trainer.current_epoch,
            is_best=is_best
        )

class ExperimentManager:
    """Manages training experiments with wandb integration."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        project_name: str = "nuclei-segmentation",
        experiment_name: Optional[str] = None,
    ):
        self.config = config
        self.project_name = project_name
        self.experiment_name = experiment_name or f"{config['model']['name']}_{config['data']['mask_type']}"
        
        # Initialize wandb
        self.init_wandb()
        self.checkpoint_manager = ModelCheckpointManager(
            experiment_name=self.experiment_name,
            wandb_run=self.wandb_logger.experiment,
            config=self.config
        )
        
        # Setup metrics
        self.metrics = MaskMetrics.get_metrics(config['data']['mask_type'])
        
    def init_wandb(self):
        """Initialize wandb run."""
        # Convert OmegaConf to plain dictionary for wandb
        wandb_config = OmegaConf.to_container(
            self.config,
            resolve=True,
            throw_on_missing=True
        )
        
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=wandb_config,
            reinit=True
        )

        self.wandb_logger = WandbLogger(
            project=self.project_name,
            name=self.experiment_name,
            config=wandb_config
        )

    def setup_trainer(self, inference: bool = False) -> pl.Trainer:
        """Setup PyTorch Lightning trainer with appropriate hardware config."""
        # Hardware config
        accelerator = self.config.hardware.device.accelerator
        precision = self.config.hardware.device.precision
        
        # Setup callbacks
        callbacks = [] if inference else [
            ModelCheckpoint(
                dirpath=f"models/checkpoints/{self.experiment_name}",
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            pl.callbacks.RichModelSummary(max_depth=2),
            pl.callbacks.RichProgressBar(),
            CheckpointLoggingCallback(self.checkpoint_manager)

        ]
        
        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=None if inference else self.config.training.epochs,
            accelerator=accelerator,
            devices=1,
            precision=precision,
            logger=self.wandb_logger,
            callbacks=callbacks,
            deterministic=True,
            gradient_clip_val=1.0 if not inference else None,
            log_every_n_steps=10,  
            enable_model_summary=True,
            inference_mode=inference,
            enable_checkpointing=not inference,
        )
        
        return trainer
    
    def setup_data(self):
            """Setup data loaders."""
            logger.info("Setting up datasets...")
            
            # Get absolute path to data
            data_root = Path(self.config.data.root_dir).resolve()
            logger.info(f"Using data root: {data_root}")
            
            try:
                # Create datasets
                train_dataset = NucleiDataset(
                    root_dir=data_root,
                    split='train',
                    mask_type=self.config.data.mask_type,
                    dataset_name=self.config.data.dataset_name
                )
                
                val_dataset = NucleiDataset(
                    root_dir=data_root,
                    split='val',
                    mask_type=self.config.data.mask_type,
                    dataset_name=self.config.data.dataset_name
                )
                
                logger.info(f"Created datasets: train={len(train_dataset)}, val={len(val_dataset)}")
                
                # Create dataloaders with hardware config
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.config.hardware.memory.batch_size, 
                    shuffle=True,
                    num_workers=self.config.hardware.workers.num_workers,  
                    pin_memory=self.config.hardware.memory.pin_memory,  
                    persistent_workers=self.config.hardware.workers.persistent_workers  
                )
                
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.config.hardware.memory.batch_size,  # Updated to match config
                    shuffle=False,
                    num_workers=self.config.hardware.workers.num_workers,  # Updated to match config
                    pin_memory=self.config.hardware.memory.pin_memory,  # Updated to match config
                    persistent_workers=self.config.hardware.workers.persistent_workers  # Updated to match config
                )
                
                return train_loader, val_loader
                
            except Exception as e:
                logger.error(f"Error setting up datasets: {str(e)}")
                logger.error(f"Config being used:\n{OmegaConf.to_yaml(self.config)}")
                raise
            finally:
                train_dataset.cleanup()
                val_dataset.cleanup()

    def log_predictions(self, model, dataloader, dataset_name: str, num_samples: int = 4):
        """
        Log prediction visualizations to wandb, organized by dataset and performance.
        
        Args:
            model: The trained model
            dataloader: DataLoader for predictions
            dataset_name: Name of the dataset being evaluated
            num_samples: Number of samples to display per category (best/worst/median)
        """
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            # Store all predictions to sort them later
            all_predictions = []
            
            # First pass: collect all predictions
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                predictions = model(images)
                
                # Create enhanced overlays
                visualizations = create_enhanced_mask_overlay(
                    images.cpu(),
                    predictions.cpu(),
                    masks.cpu(),
                    confidence_maps=torch.sigmoid(predictions.cpu())
                )
                
                # Store predictions with their metrics
                for idx, viz in enumerate(visualizations):
                    all_predictions.append({
                        'image_id': f"{dataset_name}_{batch_idx}_{idx}",
                        'visualization': viz['image'],
                        'metrics': viz['metrics'],
                        'dice_score': viz['metrics']['dice']
                    })
            
            # Sort predictions by Dice score
            sorted_predictions = sorted(all_predictions, key=lambda x: x['dice_score'])
            total_samples = len(sorted_predictions)
            
            # Select best, worst and median samples
            worst_samples = sorted_predictions[:num_samples]
            best_samples = sorted_predictions[-num_samples:]
            median_idx = total_samples // 2
            median_samples = sorted_predictions[median_idx-num_samples//2:median_idx+num_samples//2]
            
            # Create performance-grouped tables
            performance_categories = {
                'best': best_samples,
                'worst': worst_samples,
                'median': median_samples
            }
            
        for category, samples in performance_categories.items():
            # Create table for this category
            table = wandb.Table(
                columns=["image_id", "visualization", "dice_score", "iou_score"]
            )
            
            # Collect images and metrics for grid summary
            category_images = []
            category_metrics = []
            
            # Log individual predictions
            for pred in samples:
                # Add to collections for grid summary
                category_images.append(pred['visualization'])
                category_metrics.append(pred['metrics'])
                
                # Log to table
                table.add_data(
                    pred['image_id'],
                    pred['visualization'],
                    pred['metrics']['dice'],
                    pred['metrics']['iou']
                )
                
                # Log individual predictions
                wandb.log({
                    f"predictions/{dataset_name}/{category}/images/{pred['image_id']}": pred['visualization'],
                    f"predictions/{dataset_name}/{category}/metrics/dice": pred['metrics']['dice'],
                    f"predictions/{dataset_name}/{category}/metrics/iou": pred['metrics']['iou']
                })
            
            # Create and log grid summary
            grid_summary = create_grid_summary(
                images=category_images,
                title=f"{dataset_name} - {category.capitalize()} Predictions",
                metrics=category_metrics
            )
            
            # Calculate average metrics
            avg_dice = np.mean([s['metrics']['dice'] for s in samples])
            avg_iou = np.mean([s['metrics']['iou'] for s in samples])
            
            # Log summary for this category
            wandb.log({
                f"predictions/{dataset_name}/summaries/{category}_grid": grid_summary,
                f"predictions/{dataset_name}/tables/{category}": table,
                f"predictions/{dataset_name}/summary/{category}/avg_dice": avg_dice,
                f"predictions/{dataset_name}/summary/{category}/avg_iou": avg_iou,
                # Add performance distribution
                f"predictions/{dataset_name}/summary/{category}/performance_distribution": wandb.plot.histogram(
                    [[s['metrics']['dice'] for s in samples]], 
                    title=f"{category.capitalize()} Dice Score Distribution"
                )
            })

    def log_predictionsxxx(self, model, dataloader, dataset_name: str, num_samples: int = 8):
        """Log enhanced predictions with metrics."""
        model.eval()
        device = next(model.parameters()).device
        
        predictions_by_performance = {
            'best': [],
            'worst': [],
            'median': []
        }
        
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if len(all_results) >= num_samples * 3:
                    break
                    
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Get predictions and confidence
                predictions = model(images)
                confidence = torch.sigmoid(predictions)
                
                # Create visualizations
                viz_batch = create_enhanced_mask_overlay(
                    images,
                    predictions,
                    masks,
                    confidence
                )
                
                # Sort by performance
                viz_batch.sort(key=lambda x: x['metrics']['dice'])
                
                # Add to appropriate categories
                if len(predictions_by_performance['worst']) < num_samples:
                    predictions_by_performance['worst'].extend(viz_batch[:3])
                if len(predictions_by_performance['best']) < num_samples:
                    predictions_by_performance['best'].extend(viz_batch[-3:])
                if len(predictions_by_performance['median']) < num_samples:
                    mid = len(viz_batch) // 2
                    predictions_by_performance['median'].extend(viz_batch[mid-1:mid+2])
        
        # Log to wandb
        wandb.log({
            f"predictions/{dataset_name}/best_cases": [p['image'] for p in predictions_by_performance['best']],
            f"predictions/{dataset_name}/worst_cases": [p['image'] for p in predictions_by_performance['worst']],
            f"predictions/{dataset_name}/median_cases": [p['image'] for p in predictions_by_performance['median']],
            f"predictions/{dataset_name}/metrics": {
                'best_dice': np.mean([p['metrics']['dice'] for p in predictions_by_performance['best']]),
                'worst_dice': np.mean([p['metrics']['dice'] for p in predictions_by_performance['worst']]),
                'median_dice': np.mean([p['metrics']['dice'] for p in predictions_by_performance['median']])
            }
        })

    def _log_predictions_from_stored(self, stored_predictions: List[Dict], dataset_name: str, num_samples: int = 4):
        """Log predictions using stored test step data."""
        all_predictions = []
        
        try:
            # Process stored predictions
            all_predictions = []
            for batch_idx, batch in enumerate(stored_predictions):
                visualizations = create_enhanced_mask_overlay(
                    batch['images'],
                    batch['predictions'],
                    batch['masks'],
                    confidence_maps=torch.sigmoid(batch['predictions'])
                )
                
                # Store raw image arrays instead of wandb.Image objects
                for idx, viz in enumerate(visualizations):
                    all_predictions.append({
                        'image_id': f"{dataset_name}_{batch_idx}_{idx}",
                        'visualization': viz['image'], 
                        'metrics': viz['metrics'],
                        'dice_score': viz['metrics']['dice']
                    })

            prediction_groups = self._group_predictions(all_predictions, num_samples)
            self._log_prediction_groups(prediction_groups, dataset_name)
            
            logger.info(f"Successfully logged visualizations for {dataset_name}")
        
        except Exception as e:
            logger.error(f"Error logging predictions: {str(e)}")
            logger.error(f"Error details:", exc_info=True)  # Add detailed error logging
            raise

    def _group_predictions(self, predictions, num_samples: int):
        """Group predictions into best, worst, and median categories."""
        # Sort by Dice score
        sorted_preds = sorted(predictions, key=lambda x: x['metrics']['dice'])
        total_samples = len(sorted_preds)
        
        return {
            'worst': sorted_preds[:num_samples],
            'median': sorted_preds[total_samples//2 - num_samples//2:total_samples//2 + num_samples//2],
            'best': sorted_preds[-num_samples:]
        }

    def _log_prediction_groups(self, groups, dataset_name: str):
        """Log prediction groups to wandb with organized structure."""
        for category, samples in groups.items():
            # Create category logs
            category_logs = {}
            
            # 1. Overview grid
            grid_summary = create_grid_summary(
                images=[s['visualization'] for s in samples],  # Already numpy arrays
                title=f"{dataset_name} - {category.capitalize()} Predictions",
                metrics=[s['metrics'] for s in samples]
            )
            
            # 2. Log grid and metrics
            wandb.log({
                f"predictions/{dataset_name}/{category}/overview/grid": wandb.Image(grid_summary),
                f"predictions/{dataset_name}/{category}/overview/metrics": {
                    "avg_dice": np.mean([s['metrics']['dice'] for s in samples]),
                    "avg_iou": np.mean([s['metrics']['iou'] for s in samples])
                }
            })
            
            # 3. Log individual samples
            for sample in samples:
                wandb.log({
                    f"predictions/{dataset_name}/{category}/samples/{sample['image_id']}": {
                        "image": wandb.Image(sample['visualization']),  # Convert to wandb.Image here
                        "dice_score": sample['metrics']['dice'],
                        "iou_score": sample['metrics']['iou']
                    }
                })


    def train(self):
        """Run training experiment."""
        # Setup data
        train_loader, val_loader = self.setup_data()
        
        # Get model class from registry
        model_class = MODEL_REGISTRY[self.config.model.name]
        
        # Prepare complete config for model
        model_config = OmegaConf.to_container(self.config.model)
        model_config['mask_type'] = self.config.data.mask_type
        
        logger.info(f"Creating model with config:\n{model_config}")
        logger.info(f"Full config:\n{OmegaConf.to_yaml(self.config)}")
        logger.info(f"Model config after preparation:\n{model_config}")
    
        
        # Create model instance
        model = model_class.from_config(model_config)

        
        # Setup trainer
        trainer = self.setup_trainer()
        
        # Train
        trainer.fit(model, train_loader, val_loader)
        
        # Log model artifact
        self._log_model_artifact(model)
        
        return model
    
    def _log_model_artifact(self, model):
        """Log model artifact to wandb."""
        model_artifact = wandb.Artifact(
            name=f"model-{self.experiment_name}",
            type="model",
            description=f"Trained {self.config.model.name} model"
        )
        model_artifact.add_dir(f"models/checkpoints/{self.experiment_name}")
        self.wandb_logger.experiment.log_artifact(model_artifact)
        
    def evaluate(self, model, dataset_name: Optional[str] = None):
        """Evaluate model on test set with enhanced metrics and visualizations."""
        logger.info(f"Evaluating on dataset: {dataset_name if dataset_name else 'all'}")
        
        # Create test dataset with proper configuration
        test_dataset = NucleiDataset(
            root_dir=self.config.data.root_dir,
            split='test',
            mask_type=self.config.data.mask_type,
            dataset_name=dataset_name if dataset_name else 'all'
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.hardware.memory.batch_size,
            shuffle=False,
            num_workers=self.config.hardware.workers.num_workers,
            pin_memory=self.config.hardware.memory.pin_memory,
            persistent_workers=self.config.hardware.workers.persistent_workers
        )
        
        # Reset test predictions store
        if hasattr(model, 'test_predictions'):
            model.test_predictions = []
        
        # Setup trainer and run test
        trainer = self.setup_trainer(inference=True)
        results = trainer.test(model, test_loader)[0]
        logger.info(f"Raw test results: {results}")
        
        # Process results
        processed_results = self._process_test_results(results)
        logger.info(f"Processed results: {processed_results}")
        
        # Log all metrics
        self.log_enhanced_metrics(processed_results, dataset_name if dataset_name else 'all')
        
        # Get stored predictions and log visualizations
        if hasattr(model, 'test_predictions'):
            self._log_predictions_from_stored(model.test_predictions, dataset_name)
            # Clear predictions to free memory
            model.test_predictions = []
        
        return processed_results

    def _process_test_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process test results to ensure correct format for logging."""
        processed_results = {}
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # Single value tensors
                    processed_results[key] = value.item()
                else:  # Multi-value tensors
                    processed_results[key] = value.detach()
            else:
                processed_results[key] = value
        return processed_results

    def log_enhanced_metrics(self, results: Dict[str, float], dataset_name: str):
        """Log enhanced metrics with basic statistical analysis."""
        logger.info(f"Logging metrics for dataset: {dataset_name}")
        logger.info(f"Available keys in results: {list(results.keys())}")
        
        try:
            # Log basic metrics
            metrics_by_category = {
                'primary_metrics': {
                    'dice': results['test_dice'],
                    'iou': results['test_iou'],
                },
                'statistical_analysis': {
                    'dice_std': results['test_dice_std'],
                    'iou_std': results['test_iou_std'],
                    'dice_min': results['test_dice_min'],
                    'dice_max': results['test_dice_max'],
                    'iou_min': results['test_iou_min'],
                    'iou_max': results['test_iou_max']
                },
                'basic_metrics': {
                    'loss': results['test_loss'],
                }
            }

            # Log metrics to wandb
            for category, metrics in metrics_by_category.items():
                for metric_name, value in metrics.items():
                    wandb.log({
                        f"metrics/{dataset_name}/{category}/{metric_name}": value
                    })

            # Create distributions plot only if per-sample metrics are available
            if 'test_dice_per_sample' in results and 'test_iou_per_sample' in results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Dice distribution
                if isinstance(results['test_dice_per_sample'], torch.Tensor):
                    dice_scores = results['test_dice_per_sample'].cpu().numpy()
                else:
                    dice_scores = results['test_dice_per_sample']
                    
                ax1.hist(dice_scores, bins=20)
                ax1.set_title(f'Dice Score Distribution - {dataset_name}')
                ax1.set_xlabel('Dice Score')
                ax1.set_ylabel('Count')
                ax1.axvline(dice_scores.mean(), color='r', linestyle='--', label=f'Mean: {dice_scores.mean():.3f}')
                ax1.legend()
                
                # IoU distribution
                if isinstance(results['test_iou_per_sample'], torch.Tensor):
                    iou_scores = results['test_iou_per_sample'].cpu().numpy()
                else:
                    iou_scores = results['test_iou_per_sample']
                    
                ax2.hist(iou_scores, bins=20)
                ax2.set_title(f'IoU Score Distribution - {dataset_name}')
                ax2.set_xlabel('IoU Score')
                ax2.set_ylabel('Count')
                ax2.axvline(iou_scores.mean(), color='r', linestyle='--', label=f'Mean: {iou_scores.mean():.3f}')
                ax2.legend()
                
                plt.tight_layout()
                
                # Log distribution plot
                wandb.log({
                    f"metrics/{dataset_name}/distributions/plot": wandb.Image(fig)
                })
                plt.close(fig)
                
            # Create summary table
            stats_table = wandb.Table(columns=["Metric", "Mean", "Std", "Min", "Max"])
            
            for metric in ['dice', 'iou']:
                stats_table.add_data(
                    metric.upper(),
                    f"{results[f'test_{metric}']:.3f}",
                    f"{results[f'test_{metric}_std']:.3f}",
                    f"{results[f'test_{metric}_min']:.3f}",
                    f"{results[f'test_{metric}_max']:.3f}"
                )
                
            wandb.log({
                f"metrics/{dataset_name}/summary_table": stats_table
            })

            logger.info(f"Successfully logged metrics for {dataset_name}")
                
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            logger.error(f"Results dictionary: {results}")
            raise