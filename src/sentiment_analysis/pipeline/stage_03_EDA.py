from sentiment_analysis.components.EDA import SentimentEDA
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.utils.logging_setup import logger

class DataEDAPipeline:
    """
    Pipeline stage for Exploratory Data Analysis (EDA).
    """
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager = config_manager

    def run_pipeline(self, raw_data):
        """
        Executes the EDA component.

        Args:
            raw_data: Tuple containing raw texts and labels for train, val, and test sets.
        """
        try:
            logger.info("Starting Data EDA Pipeline...")
            # 1. Get configuration
            eda_config = self.config_manager.get_eda_config()

            # 2. Get raw data by unpacking the tuple
            raw_train_texts = raw_data[0]
            raw_train_labels = raw_data[1]
            raw_val_texts = raw_data[2] 
            raw_val_labels = raw_data[3] 


            # 2. Initialize and run component
            eda_component = SentimentEDA(eda_config)
            results = eda_component.run_full_eda(
                raw_texts=raw_train_texts,
                raw_labels=raw_train_labels.tolist(),
                val_texts=raw_val_texts,
                val_labels=raw_val_labels.tolist()
            )

            logger.info(f"Data EDA Pipeline completed successfully. Results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error occurred during Data EDA Pipeline: {e}")
            raise