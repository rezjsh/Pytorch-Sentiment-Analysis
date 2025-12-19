from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List
from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.entity.config_entity import PreprocessingConfig


class Preprocessing:
    """
    Handles data loading, splitting (Train/Val/Test), tokenization, 
    and manages the dataset/dataloader objects.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.dataset_name = config.dataset_name
        
        # Use a BERT-compatible tokenizer for consistency across models
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
        self.vocab_size = len(self.tokenizer)
        
        # # Datasets (will be populated in setup)
        # self.train_dataset = None
        # self.val_dataset = None
        # self.test_dataset = None
        
        # Variables to store raw data for ML models (LogReg, SVM)
        self.raw_train_texts: List[str] = []
        self.raw_train_labels: np.ndarray = np.array([])
        self.raw_test_texts: List[str] = []
        self.raw_test_labels: np.ndarray = np.array([])
        
        logger.info(f"SentimentDataModule initialized for dataset: {self.dataset_name}")
        logger.info(f"Tokenizer type: {type(self.tokenizer).__name__} | Vocab size: {self.vocab_size}")
    

    def get_raw_data(self):
        """Returns raw text data and labels for ML models."""
        logger.info("Returning raw data for ML models...")
        if not self.raw_train_texts or not self.raw_test_texts:
            logger.error("Raw data not available. Ensure 'setup()' has been called.")
            raise ValueError("Raw data not available.")
        return (self.raw_train_texts, self.raw_train_labels,
                self.raw_test_texts, self.raw_test_labels)

    def load_full_dataset(self):
        """Helper to load the raw HuggingFace dataset."""
        try:
            return load_dataset(self.dataset_name)
        except Exception as e:
            logger.error(f"Failed to load dataset '{self.dataset_name}': {e}")
            raise

    def prepare_data(self):
        """Downloads the data (no state changes)."""
        logger.info(f"Starting data download for {self.dataset_name}...")
        try:
            # HuggingFace datasets often manage download and cache automatically
            load_dataset(self.dataset_name, split='train') 
            logger.info("Data download successful or verified from cache.")
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
        
    def setup(self):
        """
        Performs data splitting, tokenization, and dataset creation.
        """
        if self.train_dataset is not None:
             logger.info("Data setup already performed. Skipping.")
             return

        logger.info("Starting data setup: loading, splitting, and tokenizing.")
        
        dataset = self.load_full_dataset()
        
        # 1. Concatenate and Prepare for Splitting
        train_texts = list(dataset['train']['text'])
        train_labels = list(dataset['train']['label'])
        test_texts = list(dataset['test']['text'])
        test_labels = list(dataset['test']['label'])

        full_texts = train_texts + test_texts
        full_labels = train_labels + test_labels

        test_val_size = self.config.test_split_ratio * 2
        
        # 2. Splitting (Train | Val | Test)
        seed = self.config.seed

        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            full_texts, full_labels, test_size=test_val_size, random_state=seed)
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=seed)
        
        # Logging split information
        logger.info(f"Data split sizes: Train={len(train_texts)}, Validation={len(val_texts)}, Test={len(test_texts)}")

        # 3. Store raw data for ML model (TF-IDF path)
        self.raw_train_texts = train_texts
        self.raw_train_labels = np.array(train_labels) 
        self.raw_test_texts = test_texts
        self.raw_test_labels = np.array(test_labels) 

        # 4. Tokenization (DL model setup)
        max_len = self.config.max_length
        logger.info(f"Tokenizing data with max_length={max_len}")
        
        self.train_encodings = self.tokenizer(train_texts, truncation=True, padding='max_length', max_length=max_len)
        self.val_encodings = self.tokenizer(val_texts, truncation=True, padding='max_length', max_length=max_len)
        self.test_encodings = self.tokenizer(test_texts, truncation=True, padding='max_length', max_length=max_len)

        

        return (self.train_encodings, train_labels,
            self.val_encodings, val_labels,
            self.test_encodings, test_labels)
       