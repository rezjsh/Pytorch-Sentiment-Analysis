from sentiment_analysis.entity.config_entity import LSTMClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier
import torch
import torch.nn as nn

class LSTMClassifier(BaseClassifier):
    def __init__(self, config: LSTMClassifierConfig, vocab_size: int):
        super().__init__(output_dim=1)

        self.config = config

        # Ensure vocab_size and embedding_dim are integers
        self.embedding = nn.Embedding(int(vocab_size), int(self.config.embedding_dim))

        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            batch_first=self.config.batch_first
        )

        # We multiply by 2 because it is bidirectional
        multiplier = 2 if self.config.bidirectional else 1
        self.fc = nn.Linear(self.config.hidden_size * multiplier, self.output_dim)

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, input_ids=None, **kwargs) -> torch.Tensor:
        """
        Modified to accept unpacked 'input_ids' from the trainer.
        """
        # 1. Handle input from trainer unpacking
        if input_ids is None:
            # Fallback for dictionary input
            batch = kwargs.get('batch', kwargs)
            input_ids = batch['input_ids']

        # 2. Embedding & Dropout
        embedded = self.dropout(self.embedding(input_ids))

        # 3. LSTM Pass
        _, (hidden, _) = self.lstm(embedded)

        # 4. Extract and Concatenate final states
        if self.config.bidirectional:
            # Concatenate the last forward (hidden[-2]) and backward (hidden[-1]) states
            hidden_final = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_final = hidden[-1, :, :]

        # 5. Classification
        prediction = self.fc(self.dropout(hidden_final))

        return prediction.view(-1)