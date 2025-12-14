from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path    

@dataclass(frozen=True)
class PreprocessingConfig:
    dataset_name: str
    batch_size: int
    max_length: int
    test_split_ratio: float
    seed: int
    tokenizer_name: str

@dataclass(frozen=True)
class DatasetConfig: # NEW CONFIG ENTITY
    """
    Configuration for creating the final PyTorch DataLoaders.
    """
    batch_size: int
    num_workers: int
    pin_memory: bool

@dataclass(frozen=True)
class EDAConfig:
    """
    Configuration for the SentimentEDA component.
    """
    report_dir: Path        # Directory to save figures and markdown reports
    max_words_to_plot: int  # Max number of words to show in word clouds/bar plots