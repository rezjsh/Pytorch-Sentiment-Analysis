from typing import Dict, Union
import torch
import torch.nn as nn
from sentiment_analysis.entity.config_entity import LSTMClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class LSTMClassifier(BaseClassifier):
    """
    A standard Bidirectional LSTM model for sequence classification.
    Captures long-range dependencies by processing sequences in both directions.
    """
    def __init__(self, config: LSTMClassifierConfig, vocab_size: int):
        # Initialize Base with output_dim=1 for binary classification
        super().__init__(output_dim=1)
        
        self.config = config
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim)

        # 1. BiLSTM Layer
        # bidirectional=True means hidden_dim is doubled at the output
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            batch_first=self.config.batch_first
        )

        # 2. Classifier head (hidden_dim * 2 because of bidirectional concatenation)
        self.fc = nn.Linear(self.config.hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass accepting the Trainer's batch dictionary or raw input_ids.
        """
        # 1. Extract input_ids
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch

        # 2. Embedding & Dropout
        # embedded shape: [batch size, seq len, embedding_dim]
        embedded = self.dropout(self.embedding(input_ids))

        # 3. LSTM Pass
        # lstm_out: [batch size, seq len, hidden_dim * 2]
        # hidden: [num_layers * 2, batch size, hidden_dim]
        # cell: [num_layers * 2, batch size, hidden_dim]
        _, (hidden, _) = self.lstm(embedded)

        # 4. Extract and Concatenate final states
        # For BiLSTM, the 'hidden' tensor contains states for both directions.
        # hidden[-2, :, :] is the final forward state of the last layer
        # hidden[-1, :, :] is the final backward state of the last layer
        hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # 5. Classification
        prediction = self.fc(self.dropout(hidden_final))

        # Return [batch_size] to match BCEWithLogitsLoss expectations
        return prediction.view(-1)