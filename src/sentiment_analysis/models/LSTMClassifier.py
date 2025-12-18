# # --- src/models/lstm_classifier.py ---

from typing import Dict
import torch
import torch.nn as nn

from sentiment_analysis.entity.config_entity import ModelConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class LSTMClassifier(BaseClassifier):
    """
    A standard Bidirectional LSTM model for sequence classification.
    """
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__(output_dim=1)
        self.config = config

        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim)

        # Bidirectional LSTM (Source 4.1)
        self.lstm = nn.LSTM(self.config.embedding_dim,
                            self.config.hidden_dim,
                            num_layers=self.config.num_layers,
                            bidirectional=self.config.bidirectional,
                            dropout=self.config.dropout,
                            batch_first=self.config.batch_first)

        # Classifier layer (2 * hidden_dim because of bidirectional concatenation)
        self.fc = nn.Linear(self.config.hidden_dim * 2, self.output_dim)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1. Embedding
        embedded = self.dropout(self.embedding(batch['input_ids']))
        # 2. LSTM
        # lstm_out: [batch size, seq len, hidden_dim * 2]
        # hidden: [num_layers*2, batch size, hidden_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 3. Use the final hidden state of the *last* LSTM layer (forward and backward)
        # hidden[-2, :, :] is the final forward state
        # hidden[-1, :, :] is the final backward state
        hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # 4. Classification
        prediction = self.fc(self.dropout(hidden_final))

        return prediction.squeeze(1)