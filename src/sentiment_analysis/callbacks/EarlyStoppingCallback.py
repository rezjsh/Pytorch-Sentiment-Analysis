from sentiment_analysis.components.callbacks import BaseCallback
from sentiment_analysis.entity.config_entity import EarlyStoppingCallbackConfig
from sentiment_analysis.utils.logging_setup import logger

class EarlyStoppingCallback(BaseCallback):
    """Stops training when a monitored metric (e.g., val_loss) stops improving."""
    def __init__(self, config: EarlyStoppingCallbackConfig):
        self.config = config
        self.best_value = float('inf') if self.config.mode == 'min' else float('-inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, trainer, epoch, val_loss):
        # Logic to check improvement
        if self.config.mode == 'min':
            improved = val_loss < (self.best_value - self.config.min_delta)
        else:
            improved = val_loss > (self.best_value + self.config.min_delta)

        if improved:
            self.best_value = val_loss
            self.wait = 0
            # Optional: Save checkpoint here via trainer
            # trainer.save_checkpoint(is_best=True)
        else:
            self.wait += 1
            if self.wait >= self.config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}. No improvement for {self.wait} epochs.")
                self.stop_training = True