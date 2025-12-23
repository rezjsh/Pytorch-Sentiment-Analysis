import torch
import torch.nn as nn
import torch.nn.functional as F
from sentiment_analysis.entity.config_entity import CNNClassifierConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class CNNClassifier(BaseClassifier):
    """
    A 1D Convolutional Neural Network (CNN) for text classification.
    Uses multiple filter sizes to capture different n-gram patterns.
    """
    def __init__(self, config: CNNClassifierConfig, vocab_size: int):
        # Initialize Base with output_dim=1 for binary classification
        super(CNNClassifier, self).__init__(output_dim=1)

        self.config = config

        # 1. Embedding Layer
        self.embedding = nn.Embedding(int(vocab_size), int(self.config.embedding_dim))

        # 2. Convolutional Layers
        # We create a list of convolutions with different kernel sizes (filter_sizes)
        # kernel_size=3 looks at trigrams, 4 looks at 4-grams, etc.
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.config.embedding_dim,
                out_channels=self.config.num_filters,
                kernel_size=fs
            )
            for fs in self.config.filter_sizes
        ])

        # 3. Classifier Head
        # The input size is (number of filters * number of filter sizes)
        self.fc = nn.Linear(len(self.config.filter_sizes) * self.config.num_filters, self.output_dim)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, input_ids=None, **kwargs) -> torch.Tensor:
        """
        Forward pass accepting unpacked keyword arguments from the Trainer.
        """
        # 1. Extract input_ids from keyword arguments if not provided positionally
        if input_ids is None:
            batch = kwargs.get('batch', kwargs)
            input_ids = batch['input_ids']

        # 2. Embedding & Dropout
        # Shape: [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(self.embedding(input_ids))

        # 3. Permute for Conv1D
        # Conv1D expects [batch_size, channels, seq_len]
        # Shape: [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # 4. Convolution & Activation
        # Each conved[n] shape: [batch_size, num_filters, seq_len - filter_size + 1]
        conved = [F.relu(conv(embedded)) for conv in self.convs]

        # 5. Global Max Pooling over time
        # This picks the most significant feature found by each filter across the whole sentence
        #
        pooled = [F.max_pool1d(conv, conv.shape[-1]).squeeze(-1) for conv in conved]

        # 6. Concatenate & Final Classification
        # Combine all features from different filter sizes
        cat = self.dropout(torch.cat(pooled, dim=1))

        # Final projection to a single logit
        prediction = self.fc(cat)

        # Return [batch_size] to match BCEWithLogitsLoss
        return prediction.view(-1)