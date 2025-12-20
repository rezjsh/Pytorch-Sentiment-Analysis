from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path    
from enum import Enum

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
    report_dir: Path
    max_words_to_plot: int  # Max number of words to show in word clouds/bar plots
    min_word_len: int  # Minimum word length to consider in analysis

@dataclass(frozen=True)
class MLClassifierConfig:
    classifier_name: str
    max_features: int
    max_iter: int
    ngram_range: List[int]
    C: float
    model_path: Path
    report_path: Path


class ModelType(str, Enum):
    """Enum for all supported model types."""
    LSTM = "LSTM"
    CNN = "CNN"
    LSTMATTENTION = "LSTMATTENTION"
    BERT = "BERT"
    SBERT = "SBERT"

@dataclass(frozen=True)
class LSTMClassifierConfig:
    embedding_dim: int
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    batch_first: bool

@dataclass(frozen=True)
class CNNClassifierConfig:
    num_filters: int
    filter_sizes: List[int]
    dropout: float
    embedding_dim: int


@dataclass(frozen=True)
class LSTMATTENTIONConfig:
    hidden_size: int
    num_layers: int
    dropout: float
    bidirectional: bool
    attention_size: int


@dataclass(frozen=True)
class BERTClassifierConfig:
    pretrained_model_name: str
    dropout: float
    fine_tune: bool


@dataclass(frozen=True)
class SBERTConfig:
    pretrained_model_name: str
    dropout: float
    fine_tune: bool


@dataclass(frozen=True)
class ModelConfig:
    model_type: ModelType
    LSTM: Optional[LSTMClassifierConfig] = None
    CNN: Optional[CNNClassifierConfig] = None
    LSTMATTENTION: Optional[LSTMATTENTIONConfig] = None
    BERT: Optional[BERTClassifierConfig] = None
    SBERT: Optional[SBERTConfig] = None

@dataclass(frozen=True)
class EarlyStoppingCallbackConfig:
    patience: int
    mode: str
    min_delta: float

@dataclass(frozen=True)
class LRSchedulerCallbackConfig:
    patience: int
    factor: float
    mode: str

@dataclass(frozen=True)
class ModelCheckpointCallbackConfig:
    checkpoint_dir: Path

@dataclass
class GradientClipCallbackConfig:
    max_norm: float

@dataclass(frozen=True)
class CallbacksConfig:
    early_stopping_callback_config: EarlyStoppingCallbackConfig
    lr_scheduler_callback_config: LRSchedulerCallbackConfig
    model_checkpoint_callback_config: ModelCheckpointCallbackConfig
    gradient_clip_callback_config: GradientClipCallbackConfig


@dataclass(frozen=True)
class ModelTrainerConfig:
    max_epochs: int
    learning_rate: float
    gradient_clip_norm: float
    report_dir: Path   

@dataclass(frozen=True)
class ModelEvaluationConfig:
    model_path: Path
    report_dir: Path