from typing import Dict, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentiment_analysis.entity.config_entity import BERTClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class BERTClassifier(BaseClassifier):
    """
    A classifier that fine-tunes a pre-trained Transformer model (BERT, DistilBERT, etc.)
    for sentiment analysis using the [CLS] token representation.
    """
    def __init__(self, config: BERTClassifierConfig):
        # Initialize Base with output_dim=1 for binary classification
        super().__init__(output_dim=1)

        self.config = config

        # 1. Load Pre-trained Transformer Backbone
        self.bert = AutoModel.from_pretrained(self.config.model_name)

        # 2. Extract Hidden Dimension dynamically from config
        transformer_config = AutoConfig.from_pretrained(self.config.model_name)
        hidden_size = getattr(transformer_config, "hidden_size", getattr(transformer_config, "dim", 768))

        # 3. Classification Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(hidden_size, self.output_dim)

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> torch.Tensor:
        """
        Forward pass modified to accept unpacked keyword arguments from the Trainer.
        """
        # 1. Extraction logic for flexibility
        if input_ids is None:
            # Fallback if positional batch is passed in kwargs
            batch = kwargs.get('batch', kwargs)
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')

        # 2. Transformer Forward Pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 3. Extract [CLS] Token Representation
        # last_hidden_state shape: [batch_size, seq_len, hidden_size]
        # Index 0 is the [CLS] token, used as a summary of the whole sentence
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 4. Dropout and Linear Projection
        prediction = self.fc(self.dropout(cls_output))

        # Return [batch_size] to match BCEWithLogitsLoss
        return prediction.view(-1)