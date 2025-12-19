# --- sentiment_analysis/components/create_model.py ---

import torch
from sentiment_analysis.models import CNNClassifier, BERTClassifier, LSTMClassifier
from sentiment_analysis.entity.config_entity import ModelConfig
from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.utils.helpers import get_device

class ModelCreator:
    """
    Component responsible for creating the selected model and moving it to the DEVICE.
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = get_device()
        
    def create_model(self, vocab_size: int = None) -> torch.nn.Module:
        """
        Instantiates the model specified in the configuration.
        
        Args:
            vocab_size (int): Required for models using token embeddings (LSTM, CNN).
            
        Returns:
            torch.nn.Module: The instantiated model ready for training.
        """
        model_type = self.config.model_type
        
        logger.info(f"Attempting to create model: {model_type}")

        if model_type == 'LSTM':
            model = LSTMClassifier(config=self.config.LSTM, vocab_size=vocab_size)
            
        elif model_type == 'CNN':
            model = CNNClassifier(vocab_size=vocab_size, config=self.config.CNN)

        elif model_type == 'LSTMATTENTION':
            model = LSTMClassifier(config=self.config.LSTMATTENTION, vocab_size=vocab_size)
        
        elif model_type == "BERT" or model_type == "SBERT":
            model = BERTClassifier(config=self.config)

        elif model_type == 'SBERT':
            model = BERTClassifier(config=self.config.SBERT, vocab_size=vocab_size)
        
        elif model_type == 'LOGREG':
            model = BERTClassifier(config=self.config.LOGREG, vocab_size=vocab_size)
        
        elif model_type == 'SVM':
            model = BERTClassifier(config=self.config.SVM, vocab_size=vocab_size)
        
        else:
            logger.error(f"Unknown model name specified in config: {model_type}")
            raise ValueError(f"Model {model_type} not implemented in ModelCreator.")
            
        # Move the model to the target device (CPU/CUDA)
        if model_type not in ['LOGREG', 'SVM']:
            model.to(self.device)
            logger.info(f"Model {model_type} moved to {self.device}.")
        logger.info(f"Model {model_type} created successfully.")
        logger.info(f"Model architecture:\n{model}")
        
        return model