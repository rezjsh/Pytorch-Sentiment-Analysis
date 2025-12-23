from sentiment_analysis.callbacks.BaseCallback import BaseCallback
from sentiment_analysis.entity.config_entity import GradientClipCallbackConfig

class GradientClipCallback(BaseCallback):
    def __init__(self, config: GradientClipCallbackConfig) -> None:
        self.config = config
        self.max_norm = self.config.max_norm

    def on_batch_end(self, trainer, batch_index, loss):
        # Clip gradients *before* the optimizer step (in the trainer)
        pass # Actual clipping is handled inside the Trainer's training_step for convenience