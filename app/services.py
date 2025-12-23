from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Any
import torch.nn as nn
import torch
import numpy as np
from sentiment_analysis.components.MLClassifier import MLClassifier
from sentiment_analysis.components.model_creator import ModelCreator
from sentiment_analysis.components.preprocessing import Preprocessing
from sentiment_analysis.config.configuration import ConfigurationManager
from sentiment_analysis.entity.config_entity import MLClassifierConfig, PreprocessingConfig
from sentiment_analysis.utils.helpers import get_device
from sentiment_analysis.utils.logging_setup import logger


class ModelService:
    def __init__(self, paths: Tuple[Path, Path, Path]):
        self.paths = paths
        self.config_manager = ConfigurationManager()
        self.devcie = get_device()
        self.prep = Preprocessing(self.config_manager.get_preprocessing_config())
        self.tokenizer = self.prep.get_tokenizer()
        self.dl_model = None
        self.clf = None


    def get_ml_model(self, model_name: str) -> MLClassifier:
        """Loads or trains a Scikit-Learn Baseline."""
        ml_classifier_config = MLClassifierConfig(**self.config_manager.get_ml_classifier_config())
        model_path = self.paths.checkpoints / f"{model_name.lower()}_baseline.joblib"
        self.clf = MLClassifier(ml_classifier_config)
        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            self.clf.load()
            return self.clf
        

    @staticmethod
    def predict(self, text: List[str]) -> np.ndarray:
        return self.clf.pipeline.predict(text)

    @staticmethod
    def predict_ml(self, text: List[str]) -> Tuple[int, float]:
        preds = self.clf.pipeline.predict([text])
        try:
            scores = self.clf.pipeline.predict_proba([text])[:, 1]
        except:
            dec = self.clf.pipeline.decision_function([text])
            scores = [1 / (1 + np.exp(-dec[0]))] # Sigmoid scaling
        return int(preds[0]), float(scores[0])


    def get_dl_model(self, model_type: str) -> Tuple[nn.Module, Any]:
        """Loads PyTorch model architecture and pre-trained weights."""

        self.dl_model = ModelCreator(config=self.config_manager.get_model_config())
        self.dl_model = self.dl_model.create_model(vocab_size=len(self.tokenizer)).to(self.device)
        weight_path = self.paths.checkpoints / f"{model_type.lower()}_best.pth"
        if weight_path.exists():
            self.dl_model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.dl_model.eval()
        return self.dl_model, self.tokenizer

    @staticmethod
    def predict_dl(self, text: str):
        """Unified inference for PyTorch models."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.dl_model(**inputs)
            # Support for models returning dicts (like BERT) or raw tensors
            if isinstance(logits, dict): logits = logits.get('logits', logits)
            
            probs = torch.sigmoid(logits).cpu().item()
            return 1 if probs > 0.5 else 0, probs