import joblib
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.entity.config_entity import MLClassifierConfig

class MLClassifier:
    """
    A traditional Machine Learning baseline using TF-IDF vectorization
    and linear classifiers (Logistic Regression or SVM).
    """
    def __init__(self, config: MLClassifierConfig):
        self.config = config
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Constructs a scikit-learn pipeline for text vectorization and classification."""
        tfidf = TfidfVectorizer(
            ngram_range=tuple(self.config.ngram_range), 
            max_features=self.config.max_features, 
            stop_words='english'
        )

        name = self.config.classifier_name.upper()
        if name == "LOGREG":
            classifier = LogisticRegression(solver='liblinear', random_state=42)
        elif name == "SVM":
            classifier = LinearSVC(random_state=42, max_iter=2000)
        else:
            raise ValueError(f"Unsupported ML classifier: {name}")

        return Pipeline([
            ('tfidf', tfidf),
            ('classifier', classifier)
        ])

    def train(self, X_train: Any, y_train: Any):
        """Fits the pipeline on raw text data."""
        logger.info(f"Training {self.config.classifier_name} with {self.config.max_features} TF-IDF features...")
        self.pipeline.fit(X_train, y_train)
        logger.info("ML Model training completed.")

    def evaluate(self, X_test: Any, y_test: Any) -> Tuple[float, Dict]:
        """Evaluates the model and returns metrics."""
        y_pred = self.pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info(f"ML Baseline ({self.config.classifier_name}) Accuracy: {acc:.4f}")
        return acc, report

    def save(self, path: str):
        """Serializes the entire pipeline (TF-IDF + Model)."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, save_path)
        logger.info(f"ML pipeline saved to {save_path}")

    def load(self, path: str):
        """Loads a serialized pipeline."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"No model found at {load_path}")
        self.pipeline = joblib.load(load_path)
        logger.info(f"ML pipeline loaded from {load_path}")