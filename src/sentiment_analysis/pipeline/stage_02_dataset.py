# --- sentiment_analysis/components/dataset_pipeline.py ---
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.entity.config_entity import DatasetConfig
from sentiment_analysis.components.dataset import SentimentDataset

class DatasetPipeline:
    """
    Creates PyTorch Dataset and DataLoader objects from preprocessed encodings.
    """
    def __init__(self, config: DatasetConfig):
        self.config = config
        logger.info(f"DatasetPipeline initialized with batch_size: {config.batch_size}")

    def run_pipline(self, preprocessed_data: Tuple) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Takes the tuple output from the Preprocessing component and creates DataLoaders.
        
        Args:
            preprocessed_data (Tuple): (train_encodings, train_labels, 
                                        val_encodings, val_labels, 
                                        test_encodings, test_labels)
        
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, Validation, and Test DataLoaders.
        """
        try:
            (train_encodings, train_labels,
             val_encodings, val_labels,
             test_encodings, test_labels) = preprocessed_data

            # 1. Create Datasets
            train_dataset = SentimentDataset(train_encodings, train_labels)
            val_dataset = SentimentDataset(val_encodings, val_labels)
            test_dataset = SentimentDataset(test_encodings, test_labels)
            logger.info("Created Train, Validation, and Test Datasets.")
            
            # 2. Create DataLoaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True, # Shuffle training data only
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            logger.info(f"Created DataLoaders (Train={len(train_loader)} batches) successfully.")
            
            return train_loader, val_loader, test_loader

        except Exception as e:
            logger.error(f"Error during DatasetPipeline run: {e}")
            raise