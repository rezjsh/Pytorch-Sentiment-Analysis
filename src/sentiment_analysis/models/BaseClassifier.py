import torch.nn as nn

class BaseClassifier(nn.Module):
    '''A base class for all classifiers in the sentiment analysis module.'''
    def __init__(self, output_dim: int = 1):
        super().__init__() 
        self.output_dim = output_dim