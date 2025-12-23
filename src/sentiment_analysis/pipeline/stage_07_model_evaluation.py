
from sentiment_analysis.components.model_evaluation import ModelEvaluation
from sentiment_analysis.utils.logging_setup import logger    

class ModelEvaluationPipeline:
    def __init__(self, config_manager, model):
        self.config = config_manager
        self.model = model

    def run_pipeline(self, test_loader) -> None:
        """Runs the model evaluation pipeline."""
        logger.info("Starting model evaluation pipeline.")
        evaluation_config = self.config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config = evaluation_config, model=self.model)
        model_evaluator.evaluate(test_loader)
        logger.info("Model evaluation pipeline completed.")