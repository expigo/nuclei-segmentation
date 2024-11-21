import wandb
from typing import Dict, Any, Optional

class WandBLogger:
    def __init__(self, project: str, config: Dict[str, Any]):
        self.run = wandb.init(project=project, config=config)
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        wandb.log(metrics, step=step)
        
    def log_image(self, image_name: str, image):
        wandb.log({image_name: wandb.Image(image)})
        
    def finish(self):
        wandb.finish()