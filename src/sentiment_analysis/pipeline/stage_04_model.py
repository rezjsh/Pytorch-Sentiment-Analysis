
from sentiment_analysis.entity.config_entity import ModelConfig
from sentiment_analysis.components.model_creator import ModelCreator
import torch.nn as nn
from sentiment_analysis.utils.logging_setup import logger

class ModelPipeline:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config.get_model_config()

    def run_pipeline(self) -> nn.Module:
        logger.info("Creating model...")
        model_creator = ModelCreator(config=self.model_config)
        model = model_creator.create_model()
        logger.info("Model created successfully.")
        return model