from typing import Dict
import torch
import torch.nn as nn


class BaseClassifier(nn.Module):
    '''A base class for all classifiers in the sentiment analysis module.'''
    def __init__(self, output_dim: int = 1):
        super(BaseClassifier, self).__init__()
        self.output_dim = output_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method")