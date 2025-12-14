# src/config/configuration.py
from pathlib import Path
from src.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.utils.helpers import create_directory, read_yaml_file
from src.utils.logging_setup import logger
from src.core.singleton import SingletonMeta
import torch
from entity.config_entity import DatasetConfig, PreprocessingConfig

class ConfigurationManager(metaclass=SingletonMeta):
    def __init__(self, config_file_path: str = CONFIG_FILE_PATH, params_file_path: str = PARAMS_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        self.params = read_yaml_file(params_file_path)
    
    def get_preprocessing_config(self) -> PreprocessingConfig:
        logger.info("Getting text preprocessing config")
        config = self.config.data
        logger.info(f"Text preprocessing config params: {config}")

        preprocessing_config = PreprocessingConfig(
            dataset_name=config.dataset_name,
            batch_size=config.batch_size,
            max_length=config.max_length,
            test_split_ratio=config.test_split_ratio,
            seed=config.seed,
            tokenizer_name=config.tokenizer_name
        )
        logger.info(f"PreprocessingConfig config created: {preprocessing_config}")
        return preprocessing_config
    
    def get_dataset_config(self) -> DatasetConfig:
        """
        Retrieves DataLoader specific configuration parameters.
        """
        logger.info("Getting Dataset and DataLoader configuration.")
        
        # Get data parameters from the static config file
        data_config = self.config.data 
        
        # Get general training parameters from the tunable params file (optional, but good for num_workers)
        # Assuming num_workers and pin_memory are defined in params.yaml under 'training_settings' or similar
        
        # FALLBACK / IMPROVEMENT: Define defaults if not present in YAML
        batch_size = data_config.batch_size
        num_workers = data_config.get('num_workers', 0) 
        pin_memory = torch.cuda.is_available() # True if CUDA is available

        dataset_config = DatasetConfig(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        logger.info(f"DatasetConfig created: {dataset_config}")
        return dataset_config