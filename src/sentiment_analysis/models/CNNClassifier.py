from typing import Dict, Union
from sentiment_analysis.entity.config_entity import CNNClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(BaseClassifier):
    '''A 1D Convolutional Neural Network (CNN) for text classification.'''
    def __init__(self, config: CNNClassifierConfig, vocab_size: int):
        super(CNNClassifier, self).__init__(output_dim = 1)
        
        self.config = config
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.config.embedding_dim, out_channels=self.config.num_filters, kernel_size=fs)
            for fs in self.config.filter_sizes
        ])

        self.fc = nn.Linear(len(self.config.filter_sizes) * self.config.num_filters, self.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)
        
    def forward(self, batch: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Forward pass handling both Trainer dicts and raw tensors.
        """
        # 1. Extract input_ids
        input_ids = batch['input_ids'] if isinstance(batch, dict) else batch

        # 2. Embedding & Dropout
        # Shape: [batch size, seq len, emb dim]
        embedded = self.dropout(self.embedding(input_ids))
        
        # 3. Permute for Conv1D (expects channels first)
        # Shape: [batch size, emb dim, seq len]
        embedded = embedded.permute(0, 2, 1)

        # 4. Convolution & ReLU activation
        # Each conved[n] shape: [batch size, num_filters, seq_len - fs + 1]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        # 5. Global Max Pooling over time
        # This reduces the sequence dimension to 1, picking the strongest feature
        # Each pooled[n] shape: [batch size, num_filters]
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]

        # 6. Concatenate all pooled features and apply Dropout
        # Shape: [batch size, num_filters * num_filter_sizes]
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # 7. Final Linear layer (logits)
        prediction = self.fc(cat)

        # Return a 1D tensor [batch_size] for BCEWithLogitsLoss
        return prediction.view(-1)