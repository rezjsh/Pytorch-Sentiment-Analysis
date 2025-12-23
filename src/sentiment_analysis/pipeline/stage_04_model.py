
from sentiment_analysis.entity.config_entity import ModelConfig
from sentiment_analysis.components.model_creator import ModelCreator
from sentiment_analysis.utils.logging_setup import logger
import torch.nn as nn

class ModelPipeline:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def run_pipeline(self, vocab_size: int) -> nn.Module:
        '''Creates and returns the model based on the provided vocabulary size.'''
        logger.info("Starting model pipeline...")
        model_creator = ModelCreator(config=self.model_config)
        model = model_creator.create_model(vocab_size=vocab_size)
        logger.info("Model created successfully.")
        return model