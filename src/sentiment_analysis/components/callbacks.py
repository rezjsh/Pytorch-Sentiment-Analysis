from sentiment_analysis.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
from sentiment_analysis.callbacks.GradientClipCallback import GradientClipCallback
from sentiment_analysis.callbacks.LRSchedulerCallback import LRSchedulerCallback
from sentiment_analysis.entity.config_entity import CallbacksConfig
from sentiment_analysis.utils.logging_setup import logger

class CallabcksManager:
    def __init__(self, config: CallbacksConfig) -> None:
        self.config = config
        self.callbacks = []

    def build_callbacks(self) -> list:
        """Build and return a list of callback instances based on the configuration."""
        if self.config.early_stopping_callback_config is not None:
            early_stopping_callback = EarlyStoppingCallback(config=self.config.early_stopping_callback_config)
            self.callbacks.append(early_stopping_callback)

        if self.config.lr_scheduler_callback_config is not None:
            lr_scheduler_callback = LRSchedulerCallback(config=self.config.lr_scheduler_callback_config)
            self.callbacks.append(lr_scheduler_callback)

        if self.config.model_checkpoint_callback_config is not None:
            model_checkpoint_callback = ModelCheckpointCallback(config=self.config.model_checkpoint_callback_config)
            self.callbacks.append(model_checkpoint_callback)

        if self.config.gradient_clip_callback_config is not None:
            gradient_clip_callback = GradientClipCallback(config=self.config.gradient_clip_callback_config)
            self.callbacks.append(gradient_clip_callback)

        logger.info(f"Total callbacks created: {len(self.callbacks)}")

        return self.callbacks


    @staticmethod
    def attach_optimizer_to_callbacks(callbacks, optimizer):
        """Finds callbacks that require an optimizer and injects it."""
        for cb in callbacks:
            if isinstance(cb, LRSchedulerCallback):
                # Initialize the actual scheduler inside the callback now
                cb.initialize_scheduler(optimizer)
                logger.info(f"Optimizer attached to {cb.__class__.__name__}")