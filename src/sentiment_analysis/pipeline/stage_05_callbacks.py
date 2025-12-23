from sentiment_analysis.components.callbacks import CallabcksManager
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.utils.logging_setup import logger

class CallbacksPipeline:
    """
    Pipeline to create and manage callbacks based on configuration.
    """

    def __init__(self, config: ConfigurationManager):
        self.config = config

    def run_pipeline(self) -> list:
        """Creates and returns a list of callback instances."""
        logger.info("Starting callbacks pipeline...")
        callbacks_config = self.config.get_callbacks_config()
        callbacks_manager = CallabcksManager(config=callbacks_config)
        callbacks = callbacks_manager.build_callbacks()
        logger.info("Callbacks pipeline executed successfully.")
        return callbacks
