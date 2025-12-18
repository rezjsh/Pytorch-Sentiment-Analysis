from torch.utils.data import Dataset
import torch
from typing import List, Dict
from src.utils.logging_setup import logger

from sentiment_analysis.entity.config_entity import DatasetConfig

class SentimentDataset(Dataset):
    """
    A custom PyTorch Dataset for the tokenized and labeled data from the dataset. (Remains the same)
    """
    def __init__(self, config: DatasetConfig, encodings: Dict[str, List[int]], labels: List[int]):
        logger.info("Initializing SentimentDataset...")    
        self.config = config
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns a single sample as input tensors and a label tensor."""
        logger.info(f"Fetching item at index {idx} from SentimentDataset.")
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        logger.info("Calculating length of SentimentDataset...")
        return len(self.labels)


