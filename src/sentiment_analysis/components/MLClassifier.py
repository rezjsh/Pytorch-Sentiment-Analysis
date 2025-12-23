import json
import joblib
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.entity.config_entity import MLClassifierConfig

class MLClassifier:
    """
    A traditional Machine Learning baseline using TF-IDF vectorization.
    Supports Logistic Regression and Linear SVM.
    """
    def __init__(self, config: MLClassifierConfig):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Constructs a scikit-learn pipeline for text vectorization and classification."""

        # 1. Feature Extraction
        tfidf = TfidfVectorizer(
            ngram_range=tuple(self.config.ngram_range),
            max_features=self.config.max_features,
            stop_words='english',
            sublinear_tf=True # Applies 1 + log(tf) to scale word counts
        )

        # 2. Classifier Selection Registry
        classifiers = {
            "LOGREG": LogisticRegression(
                solver='liblinear',
                C=getattr(self.config, 'C', 1.0),
                random_state=42
            ),
            "SVM": LinearSVC(
                C=getattr(self.config, 'C', 1.0),
                random_state=42,
                max_iter=getattr(self.config, 'max_iter', 2000),
                dual="auto" # Automatically chooses dual/primal based on n_samples
            )
        }

        name = self.config.classifier_name.upper()
        if name not in classifiers:
            raise ValueError(f"Unsupported ML classifier: {name}. Available: {list(classifiers.keys())}")

        return Pipeline([
            ('tfidf', tfidf),
            ('classifier', classifiers[name])
        ])

    def train(self, X_train: Union[List[str], pd.Series], y_train: Any):
        """Fits the pipeline on raw text data."""
        logger.info(f"Training {self.config.classifier_name} baseline...")
        self.pipeline.fit(X_train, y_train)
        logger.info("ML Model training completed.")

    def evaluate(self, X_test: Union[List[str], pd.Series], y_test: Any) -> Tuple[float, Dict]:
        """Evaluates the model and returns metrics."""
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"ML Baseline ({self.config.classifier_name}) Accuracy: {acc:.4f}")
        return acc, report

    def get_feature_importance(self, n_top: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extracts the words most indicative of positive and negative sentiment.
        Only works for linear models (LogReg/SVM).
        """
        tfidf = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['classifier']

        feature_names = tfidf.get_feature_names_out()
        # For binary classification, coef_ has shape (1, n_features)
        coefficients = clf.coef_[0]

        # Zip features with their weights
        feature_importance = sorted(zip(feature_names, coefficients), key=lambda x: x[1])

        return {
            "top_negative": feature_importance[:n_top],
            "top_positive": feature_importance[-n_top:][::-1]
        }

    def plot_feature_importance(self):
        with open(self.config.report_path, "r") as f:
            data = json.load(f)

        pos_data = data["top_positive"]
        neg_data = data["top_negative"]

        # Extract words and weights from the list of tuples
        pos_words = [item[0] for item in pos_data]
        pos_weights = [item[1] for item in pos_data]

        neg_words = [item[0] for item in neg_data]
        neg_weights = [item[1] for item in neg_data]

        save_path = Path(self.config.report_path).parent / "feature_importance.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # Plot Positive Words
        ax1.barh(pos_words, pos_weights, color='green')
        ax1.set_title("Top Words for Positive Sentiment")
        ax1.invert_yaxis()

        # Plot Negative Words
        ax2.barh(neg_words, neg_weights, color='red')
        ax2.set_title("Top Words for Negative Sentiment")
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(save_path)

    def save(self):
        """Serializes the entire pipeline."""
        save_path = self.config.model_path
        joblib.dump(self.pipeline, save_path)
        logger.info(f"ML pipeline saved to {save_path}")

    def load(self):
        """Loads a serialized pipeline."""
        load_path = self.config.model_path
        self.pipeline = joblib.load(load_path)
        logger.info(f"ML pipeline loaded from {load_path}")
