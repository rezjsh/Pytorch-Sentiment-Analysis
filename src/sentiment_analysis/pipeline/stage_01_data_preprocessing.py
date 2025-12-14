
from sentiment_analysis.components.preprocessing import Preprocessing
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.utils.logging_setup import logger

class PreprocessingPipeline:
    def __init__(self, config: ConfigurationManager):
        self.config = config
    def run_pipeline(self) -> Preprocessing:
        try:
            logger.info("Starting data preprocessing pipeline...")
            preprocessing_config = self.config.get_preprocessing_config()
            preprocessing = Preprocessing(preprocessing_config)
            preprocessing.prepare_data()
            preprocessed_data = preprocessing.setup()
            logger.info("Data preprocessing pipeline completed.")
            return preprocessed_data
        except Exception as e:
            logger.error(f"Error occurred during data preprocessing pipeline: {e}")
            raise
        