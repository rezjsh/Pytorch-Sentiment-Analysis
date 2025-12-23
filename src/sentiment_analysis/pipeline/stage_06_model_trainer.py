from sentiment_analysis.components.model_trainer import Trainer
from sentiment_analysis.config.configuration import ConfigurationManager
import torch.nn as nn
from sentiment_analysis.utils.logging_setup import logger
class ModelTrainerPipeline:
    def __init__(self, config: ConfigurationManager, model: nn.Module, callbacks: list = None):
        self.config = config
        self.callbacks = callbacks
        self.model = model

    def run_pipeline(self, train_loader, val_loader) -> Trainer:
        """Executes the model training pipeline stage."""
        # Initialize the Trainer with configuration, model, and callbacks
        logger.info("Starting model training pipeline stage.")
        trainer_config = self.config.get_model_trainer_config()
        self.trainer = Trainer(
            config=trainer_config,
            model=self.model,
            callbacks=self.callbacks
        )

        # Train the model
        self.trainer.train(train_loader, val_loader)

        # Plot and save history
        self.trainer.plot_history()

        # Save history to CSV
        self.trainer.save_history_to_csv()
        logger.info("Model training pipeline stage completed.")
        return self.trainer
