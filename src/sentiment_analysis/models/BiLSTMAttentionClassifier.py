from typing import Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment_analysis.entity.config_entity import BiLSTMAttentionConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class BiLSTMAttentionClassifier(BaseClassifier):
    """
    Bidirectional LSTM with an additive self-attention mechanism.
    Weights hidden states dynamically to capture the most salient features.
    """
    def __init__(self, config: BiLSTMAttentionConfig, vocab_size: int):
        super().__init__(output_dim=1)
        
        self.config = config
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim)

        # 1. BiLSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            bidirectional=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            batch_first=True
        )

        # 2. Attention Layer Components
        # Maps BiLSTM output (2 * hidden_dim) to a context vector space
        self.attention_weight_layer = nn.Linear(self.config.hidden_dim * 2, self.config.hidden_dim * 2)
        # Maps the context space to a single importance score
        self.attention_score_layer = nn.Linear(self.config.hidden_dim * 2, 1, bias=False)

        # 3. Final Classification Head
        self.fc = nn.Linear(self.config.hidden_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with attention scoring.
        """
        # 1. Extract inputs
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch

        # 2. Embedding & BiLSTM
        # embedded shape: [batch size, seq len, embedding_dim]
        embedded = self.dropout(self.embedding(input_ids))
        
        # lstm_out shape: [batch size, seq len, hidden_dim * 2]
        lstm_out, _ = self.lstm(embedded)

        # 3. Self-Attention Mechanism
        # a) Calculate hidden representation for each time step
        # u_prime shape: [batch size, seq len, hidden_dim * 2]
        u_prime = torch.tanh(self.attention_weight_layer(lstm_out))
        
        # b) Calculate raw scores (energy) for each time step
        # scores shape: [batch size, seq len, 1]
        scores = self.attention_score_layer(u_prime)
        
        # c) Softmax to get normalized attention weights (alpha)
        # alpha shape: [batch size, seq len, 1]
        alpha = F.softmax(scores, dim=1)

        # d) Construct Context Vector (weighted sum of hidden states)
        # context_vector shape: [batch size, hidden_dim * 2]
        context_vector = torch.sum(lstm_out * alpha, dim=1)

        # 4. Classification
        prediction = self.fc(self.dropout(context_vector))

        # Return [batch_size] for consistency with BCE loss
        return prediction.view(-1)