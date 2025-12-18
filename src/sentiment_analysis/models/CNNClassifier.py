from typing import Dict
from sentiment_analysis.entity.config_entity import CNNClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier
import torch
import torch.nn as nn

class CNNClassifier(BaseClassifier):
    '''A 1D Convolutional Neural Network (CNN) for text classification.'''
    def __init__(self, config: CNNClassifierConfig, vocab_size: int):
        super(CNNClassifier, self).__init__(output_dim = 1)
        
        self.config = config
        self.embedding = nn.Embedding(vocab_size, self.config.embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.config.num_filters, kernel_size=(fs, self.config.embedding_dim))
            for fs in self.config.filter_sizes
        ])

        self.fc = nn.Linear(len(self.config.filter_sizes) * self.config.num_filters, 1)
        self.dropout = nn.Dropout(self.config.dropout)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        text = batch['text']  # Assuming batch contains 'text' key with input tensor
        embedded = self.embedding(text).unsqueeze(1)  # Add channel dimension
        
        conved = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        return self.fc(cat)