from sentiment_analysis.callbacks import BaseCallback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentiment_analysis.entity.config_entity import LRSchedulerCallbackConfig
from sentiment_analysis.utils.logging_setup import logger

class LRSchedulerCallback(BaseCallback):
    """Adjusts learning rate when validation loss plateaus."""
    def __init__(self, config: LRSchedulerCallbackConfig, optimizer=None):
        self.config = config
        self.optimizer = optimizer
        self.scheduler = None

    def initialize_scheduler(self, optimizer):
        """Late binding of the optimizer."""
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode=self.config.mode, patience=self.config.patience, factor=self.config.factor
        )

    def on_epoch_end(self, trainer, epoch, val_loss):
        if self.scheduler is not None:
            self.scheduler.step(val_loss)