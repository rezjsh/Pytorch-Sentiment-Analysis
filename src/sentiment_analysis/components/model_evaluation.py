import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sentiment_analysis.entity.config_entity import ModelEvaluationConfig
from sentiment_analysis.utils.helpers import get_device
from sentiment_analysis.utils.logging_setup import logger
import torch.nn as nn
class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig, model: nn.Module):
        self.config = config
        self.device = get_device()
        self.model = model.to(self.device)

    def evaluate(self, test_loader):
        """Runs the test loop and generates reports."""

        # 1. Load best weights
        logger.info(f"Loading best model from {self.config.model_path}")
        self.model.load_state_dict(torch.load(self.config.model_path, map_location=self.device))
        self.model.eval()

        all_preds = []
        all_labels = []

        # 2. Test Loop
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Reuse forward pass logic from your trainer or implement here
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                if 'attention_mask' in batch:
                    logits = self.model(input_ids=input_ids, attention_mask=batch['attention_mask'].to(self.device))
                else:
                    logits = self.model(input_ids=input_ids)

                preds = (torch.sigmoid(logits) > 0.5).int()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 3. Generate Reports
        self._generate_reports(all_labels, all_preds)

    def _generate_reports(self, y_true, y_pred):
        """Calculates metrics and saves artifacts."""
        report_dir = Path(self.config.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # Classification Report (Precision, Recall, F1)
        report = classification_report(y_true, y_pred, output_dict=True)
        with open(report_dir / "metrics.json", "w") as f:
            json.dump(report, f, indent=4)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(report_dir / "confusion_matrix.png")
        plt.close()

        logger.info(f"Evaluation reports saved to {report_dir}")
        logger.info(f"Final Accuracy: {report['accuracy']:.4f}")