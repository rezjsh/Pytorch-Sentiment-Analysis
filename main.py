# --- main.py ---
import torch
import sys
from pathlib import Path

from sentiment_analysis.pipeline.stage_02_dataset import DatasetPipeline
# Fix path issues if running from root
sys.path.append(str(Path(__file__).parent.resolve()))
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.pipeline.stage_01_data_preprocessing import PreprocessingPipeline
from sentiment_analysis.utils.logging_setup import logger

# Clear CUDA memory (good practice)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    logger.info("CUDA cache cleared.")

def main():
    """
    Main execution function to orchestrate the MLOps pipeline stages.
    """
    try:
        config_manager = ConfigurationManager()
        
        # --- Stage 1: Data Preprocessing (Splitting & Tokenization) ---
        STAGE_NAME = "Stage 01: Data Preprocessing"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        data_preprocessing_pipeline = PreprocessingPipeline(config_manager)
        
        # This returns a tuple: (train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels)
        preprocessed_data_tuple = data_preprocessing_pipeline.run_pipeline() 
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 2: Dataset & DataLoader Creation ---
        STAGE_NAME = "Stage 02: Dataset and DataLoader Creation"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # 1. Get configuration for DataLoaders
        dataset_config = config_manager.get_dataset_config()
        
        # 2. Initialize and run the DataLoader component
        data_loader_creator = DatasetPipeline(dataset_config)
        train_loader, val_loader, test_loader = data_loader_creator.run_pipline(preprocessed_data_tuple)
        
        logger.info(f"Final DataLoaders created. Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # --- Stage 3: Model Training (To be added later) ---
        # STAGE_NAME = "Stage 03: Model Training"
        # logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        # model_trainer_pipeline = ModelTrainerPipeline(config_manager)
        # model_trainer_pipeline.run(train_loader, val_loader)
        # logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline execution: {e}")
        raise e

if __name__ == '__main__':
    main()