from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from sentiment_analysis.entity.config_entity import BiLSTMAttentionConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class BiLSTMAttentionClassifier(BaseClassifier):
    """
    Bidirectional LSTM with an additive self-attention mechanism.
    Weights hidden states dynamically to capture the most salient features
    of a sentence regardless of their position.
    """
    def __init__(self, config: BiLSTMAttentionConfig, vocab_size: int):
        # Initialize Base with output_dim=1 for binary classification
        super().__init__(output_dim=1)

        self.config = config

        # 1. Embedding Layer
        self.embedding = nn.Embedding(int(vocab_size), int(self.config.embedding_dim))

        # 2. BiLSTM Layer
        # Processes sequence in both directions to capture forward and backward context
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            batch_first=True
        )

        # 3. Additive Self-Attention Mechanism
        # Maps the BiLSTM output (hidden_size * 2) to a scoring space
        self.attention_weight_layer = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * 2)
        # Calculates the scalar importance score for each token
        self.attention_score_layer = nn.Linear(self.config.hidden_size * 2, 1, bias=False)

        # 4. Final Classification Head
        self.fc = nn.Linear(self.config.hidden_size * 2, self.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, input_ids=None, **kwargs) -> torch.Tensor:
        """
        Forward pass with attention scoring.
        Supports keyword argument unpacking from the Trainer.
        """
        # 1. Handle inputs from Trainer unpacking
        if input_ids is None:
            batch = kwargs.get('batch', kwargs)
            input_ids = batch['input_ids']

        # 2. Embedding & BiLSTM
        # Shape: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(input_ids))

        # lstm_out: [batch_size, seq_len, hidden_size * 2]
        lstm_out, _ = self.lstm(embedded)

        # 3. Attention Calculation
        # a) Transform hidden states into a 'contextual energy' representation
        u_prime = torch.tanh(self.attention_weight_layer(lstm_out))

        # b) Calculate raw scores (energy) per time step
        # scores: [batch_size, seq_len, 1]
        scores = self.attention_score_layer(u_prime)

        # c) Apply Softmax to get importance weights (alpha)
        # Each alpha value is between 0 and 1, and the sum over seq_len is 1
        alpha = F.softmax(scores, dim=1)

        # d) Create the Context Vector (weighted sum of LSTM outputs)
        # context_vector: [batch_size, hidden_size * 2]
        context_vector = torch.sum(lstm_out * alpha, dim=1)

        # 4. Final Projection
        prediction = self.fc(self.dropout(context_vector))

        # Return [batch_size] for BCEWithLogitsLoss compatibility
        return prediction.view(-1)