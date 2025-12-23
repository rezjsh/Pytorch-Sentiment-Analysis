import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from sentiment_analysis.entity.config_entity import SBERTConfig
from sentiment_analysis.models.BaseClassifier import BaseClassifier

class SBERTClassifier(BaseClassifier):
    """
    A classifier using a pre-trained Sentence Transformer model.
    Utilizes Mean Pooling to derive fixed-size sentence embeddings.
    """
    def __init__(self, config: SBERTConfig):
        # Initialize Base with output_dim=1 for binary classification
        super().__init__(output_dim=1)

        self.config = config

        # 1. Load Pre-trained Encoder
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        # 2. Extract Hidden Dimension dynamically
        transformer_config = AutoConfig.from_pretrained(self.config.model_name)
        self.hidden_size = getattr(transformer_config, "hidden_size", getattr(transformer_config, "dim", 384))

        # 3. Classification Head
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.hidden_size, self.output_dim)

    def _mean_pooling(self, model_output, attention_mask):
        """
        Derives the sentence embedding by averaging token embeddings,
        accounting for padding via the attention mask.
        """
        # last_hidden_state shape: [batch size, seq len, hidden size]
        token_embeddings = model_output.last_hidden_state

        # Expand mask to match token embeddings: [batch size, seq len, hidden size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings, ignoring padded tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        # Count non-padded tokens per sentence
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Calculate Mean
        return sum_embeddings / sum_mask

    # Update the method signature here
    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> torch.Tensor:
        """
        Forward pass modified to accept unpacked keyword arguments from the Trainer.
        """
        # 1. Handle inputs (keeping your logic for flexibility)
        # If the trainer passed 'batch' as a single positional argument
        if input_ids is None and 'batch' in kwargs:
            batch = kwargs['batch']
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')

        # Ensure we have a mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # 2. Transformer Forward Pass
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 3. Mean pooling
        sentence_embedding = self._mean_pooling(outputs, attention_mask)

        # 4. Classification
        prediction = self.fc(self.dropout(sentence_embedding))

        # Return [batch_size]
        return prediction.view(-1)