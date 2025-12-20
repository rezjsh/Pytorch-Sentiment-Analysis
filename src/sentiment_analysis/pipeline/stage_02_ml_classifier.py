from typing import Any, List, Union
from sentiment_analysis.components import MLClassifier
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.utils.logging_setup import logger
import pandas as pd
import json
from pathlib import Path

class MLClassifierPipeline:
    def __init__(self, config_manager: ConfigurationManager):
        self.config_manager  = config_manager

    def run_pipeline(self,
                     X_train: Union[List[str], pd.Series], y_train: Any, X_test: Union[List[str], pd.Series], y_test: Any):
        '''Orchestrates the ML Classifier Pipeline.'''
        logger.info("Starting ML Baseline Pipeline...")
        
        # 1. Load Configurations
        config = self.config_manager.get_ml_classifier_config()
        # 2. Model Initialization
        model = MLClassifier(config=config)

        # 3. Train Model
        model.train(X_train, y_train)
        logger.info("Model training complete.")
       
        # 4. Evaluate
        logger.info("Evaluating model...")
        accuracy, report = model.evaluate(X_test, y_test)
        
        # 5. Interpret (Feature Importance)
        logger.info("Interpreting model...")
        importance = model.get_feature_importance(n_top=10)

        # 6. Save Artifacts
        model.save()
        logger.info("Model saved.")

        results = {
            "accuracy": accuracy,
            "classification_report": report,
            "top_positive": importance['top_positive'],
            "top_negative": importance['top_negative']
        }
        
        with open(config.report_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Classification Report saved to {config.report_path}")
        
        # Plot Feature Importance
        model.plot_feature_importance()
            
        logger.info(f"Baseline Pipeline complete. Accuracy: {accuracy:.4f}")