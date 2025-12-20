# --- main.py ---
import torch
import sys
from pathlib import Path
from sentiment_analysis.pipeline.stage_02_dataset import DatasetPipeline
from sentiment_analysis.pipeline.stage_03_EDA import DataEDAPipeline
from sentiment_analysis.pipeline.stage_04_model import ModelPipeline
from sentiment_analysis.pipeline.stage_05_callbacks import CallbacksPipeline
from sentiment_analysis.pipeline.stage_06_model_trainer import ModelTrainerPipeline
from sentiment_analysis.pipeline.stage_07_model_evaluation import ModelEvaluationPipeline
from sentiment_analysis.pipeline.stage_02_ml_classifier import MLClassifierPipeline
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
        preprocessed_data_tuple, raw_data = data_preprocessing_pipeline.run_pipeline() 
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

        # Decide which pipeline to run based on model type
        if config_manager.get_mode() == 'ML':
            run_machine_learning_pipeline(raw_data)
        elif config_manager.get_mode() == 'DL':
            run_deep_learning_pipeline(config_manager.get_model_config(), preprocessed_data_tuple)
        else:
            logger.error("Invalid mode specified in configuration. Choose 'ML' or 'DL'.")
            raise ValueError("Invalid mode specified in configuration. Choose 'ML' or 'DL'.")
        
        

       
    except Exception as e:
        logger.error(f"FATAL ERROR in pipeline execution: {e}")
        raise e

def run_dl_branch(config_manager, model, preprocessed_data_tuple):
    """Executes the Deep Learning Path."""
    # --- Stage 2: Dataset & DataLoader Creation ---
    STAGE_NAME = "Stage 02: Dataset and DataLoader Creation"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    dataset_config = config_manager.get_dataset_config()
    data_loader_creator = DatasetPipeline(dataset_config)
    train_loader, val_loader, test_loader = data_loader_creator.run_pipline(preprocessed_data_tuple)
    logger.info(f"Final DataLoaders created. Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


    # --- Stage 3: Exploratory Data Analysis (EDA) ---
    STAGE_NAME = "Stage 03: Exploratory Data Analysis"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    
    eda_pipeline = DataEDAPipeline(config_manager)
    eda_pipeline.run_pipeline(preprocessed_data=preprocessed_data_tuple)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

     # --- Stage 4: Model Creation ---
    STAGE_NAME = "Stage 04: Model Creation"
    model_config = config_manager.get_model_config()
    model_pipeline = ModelPipeline(model_config)
    model = model_pipeline.run_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")



    # --- Stage 5: Callbacks Creation ---
    STAGE_NAME = "Stage 05: Callbacks Creation"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    callbacks_pipeline = CallbacksPipeline(config_manager)
    callbacks = callbacks_pipeline.run_pipeline()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")


    # --- Stage 6: Model Training ---
    STAGE_NAME = "Stage 05: Model Training"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer_pipeline = ModelTrainerPipeline(config_manager, model, callbacks)
    model_trainer_pipeline.run_pipeline(train_loader, val_loader)
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")

    # --- Stage 7: Model Evaluation ---
    STAGE_NAME = "Stage 06: Model Evaluation"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation_pipeline = ModelEvaluationPipeline(config_manager, model)
    model_evaluation_pipeline.run_pipeline(test_loader) 
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")



def run_ml_branch(config_manager, raw_data):
    """Executes the Machine Learning Baseline Path."""

    # --- Stage 2: ML Classifier Training And Evaluation ---
    STAGE_NAME = "Stage 02: ML Classifier Training And Evaluation"
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    logger.info(">>>>>> Starting ML Baseline Training & Evaluation <<<<<<")
    X_train, y_train, X_test, y_test = raw_data
    ml_pipeline = MLClassifierPipeline(config_manager)
    ml_pipeline.run_pipeline(X_train, y_train)
    logger.info(">>>>>> ML Baseline execution completed <<<<<<")

if __name__ == '__main__':
    main()