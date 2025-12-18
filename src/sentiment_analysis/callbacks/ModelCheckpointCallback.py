import os
from pathlib import Path
from sentiment_analysis.components.callbacks import BaseCallback
from sentiment_analysis.entity.config_entity import ModelCheckpointCallbackConfig
from sentiment_analysis.utils.logging_setup import logger

class ModelCheckpointCallback(BaseCallback):
    """Saves the model weights at the end of every epoch or when improvement is seen."""
    def __init__(self, config: ModelCheckpointCallbackConfig):
        self.config = config
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        
        self.best_val_loss = float('inf')

    def on_epoch_end(self, trainer, epoch, val_loss):
        """
        Saves the model. You can choose to save every epoch or only the best.
        """
        # Logic: Save only if it's the best model so far (Standard MLOps approach)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            # Save the "Best" model
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
            trainer.save_checkpoint(path=checkpoint_path)
            
            logger.info(f"New best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
        
        # Optional: Save every epoch as a backup (uncomment if needed)
        # epoch_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        # trainer.save_checkpoint(path=epoch_path)