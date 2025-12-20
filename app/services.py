from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from sentiment_analysis.components.MLClassifier import MLClassifier
from sentiment_analysis.components.preprocessing import Preprocessing
from sentiment_analysis.entity.config_entity import MLClassifierConfig, PreprocessingConfig
from sentiment_analysis.utils.logging_setup import logger


class BaselineService:
    def __init__(self, checkpoints_dir: Path, reports_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.reports_dir = reports_dir

    def get_model(self, model_name: str, dataset_name: str, on_info=None, on_success=None) -> MLClassifier:
        """Load a cached baseline or train quickly if missing."""
        model_path = self.checkpoints_dir / f"{model_name.lower()}_baseline.joblib"
        cfg = MLClassifierConfig(
            classifier_name=model_name,
            ngram_range=[1, 2],
            max_features=50000,
            max_iter=2000,
            C=1.0,
            model_path=str(model_path),
            report_path=str(self.reports_dir / f"{model_name.lower()}_report.json"),
        )
        clf = MLClassifier(cfg)
        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            clf.load()
            return clf

        if on_info:
            on_info("No saved model found. Training a quick baseline model now (first run only)...")
        # prepare data
        prep_cfg = PreprocessingConfig(
            dataset_name=dataset_name,
            batch_size=32,
            max_length=128,
            test_split_ratio=0.2,
            seed=42,
            tokenizer_name="bert-base-uncased",
        )
        prep = Preprocessing(prep_cfg)
        prep.prepare_data()
        prep.setup()

        X_train, y_train = prep.raw_train_texts, prep.raw_train_labels
        X_test, y_test = prep.raw_test_texts, prep.raw_test_labels

        clf.train(X_train, y_train)
        acc, _ = clf.evaluate(X_test, y_test)

        clf.save()
        try:
            fi = clf.get_feature_importance(15)
            rep = {
                "accuracy": float(acc),
                "top_positive": {"word": [w for w, _ in fi["top_positive"]], "weight": [float(v) for _, v in fi["top_positive"]]},
                "top_negative": {"word": [w for w, _ in fi["top_negative"]], "weight": [float(v) for _, v in fi["top_negative"]]},
            }
            with open(cfg.report_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not compute/save feature importance report: {e}")

        if on_success:
            on_success(f"Baseline {model_name} trained. Accuracy: {acc:.4f}. Model cached for next runs.")
        return clf

    @staticmethod
    def predict(clf: MLClassifier, texts: List[str]) -> np.ndarray:
        return clf.pipeline.predict(texts)

    @staticmethod
    def predict_with_scores(clf: MLClassifier, texts: List[str]):
        """Return predictions and a positive-class score if available.
        - For LogisticRegression: use predict_proba
        - For LinearSVC: use decision_function and min-max scale to [0,1]
        """
        clf_step = clf.pipeline.named_steps.get('classifier')
        preds = clf.pipeline.predict(texts)
        scores = None
        try:
            # LogisticRegression
            if hasattr(clf_step, 'predict_proba'):
                proba = clf.pipeline.predict_proba(texts)
                scores = proba[:, 1]
            elif hasattr(clf_step, 'decision_function'):
                # LinearSVC decision function -> scale to [0,1]
                dec = clf.pipeline.decision_function(texts)
                # Avoid division by zero on constant vectors
                dmin, dmax = float(np.min(dec)), float(np.max(dec))
                if dmax - dmin > 1e-12:
                    scores = (dec - dmin) / (dmax - dmin)
                else:
                    scores = np.zeros_like(dec)
        except Exception:
            pass
        return preds, scores
