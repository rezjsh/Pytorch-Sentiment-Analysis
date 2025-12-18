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
        # DistilBERT uses 'dim', standard BERT/RoBERTa uses 'hidden_size'
        transformer_config = AutoConfig.from_pretrained(self.config.model_name)
        hidden_size = getattr(transformer_config, "hidden_size", getattr(transformer_config, "dim", 768))

        # 3. Classification Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(hidden_size, self.output_dim)

    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass accepting the Trainer's batch dictionary.
        """
        # 1. Extract inputs (handling both dict and raw tensor for flexibility)
        if isinstance(batch, dict):
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
        else:
            input_ids = batch
            attention_mask = None

        # 2. Transformer Forward Pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 3. Extract [CLS] Token Representation
        # last_hidden_state shape: [batch_size, seq_len, hidden_size]
        # We take index 0 (the first token) across the sequence
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 4. Dropout and Linear Projection
        prediction = self.fc(self.dropout(cls_output))

        # Return [batch_size] to match BCEWithLogitsLoss
        return prediction.view(-1)