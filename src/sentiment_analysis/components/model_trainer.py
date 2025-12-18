import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from sentiment_analysis.entity.config_entity import ModelTrainerConfig
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from sentiment_analysis.utils.logging_setup import logger
from sentiment_analysis.utils.helpers import get_device
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
    def __init__(self, config: ModelTrainerConfig, model: nn.Module, callbacks: list = None):
        self.config = config
        self.device = get_device()
        
        self.model = model.to(self.device)
        self.model_name = self.model.__class__.__name__
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Optimizer initiated inside Trainer as requested
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )

        self.callbacks = callbacks if callbacks is not None else []
        
        # Late binding: Attach optimizer to callbacks that need it (e.g., LRScheduler)
        self._register_optimizer_to_callbacks()

        # Metrics setup
        self.train_accuracy = BinaryAccuracy().to(self.device)
        self.val_accuracy = BinaryAccuracy().to(self.device)

        # Training State
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.gradient_clip_norm = self.config.gradient_clip_norm

        # Trigger on_train_start for callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)

    def _register_optimizer_to_callbacks(self):
        """Passes the internal optimizer to any callback that requires it."""
        for cb in self.callbacks:
            if hasattr(cb, 'initialize_scheduler'):
                cb.initialize_scheduler(self.optimizer)
                logger.info(f"Optimizer linked to {cb.__class__.__name__}")

    def save_checkpoint(self, path):
        """Helper used by callbacks to save model weights."""
        torch.save(self.model.state_dict(), path)
        # logger.info(f"Weights saved to {path}")

    def _forward_pass(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device).float() # BCE expects float labels
        
        # Determine if we need attention_mask (for BERT-like models)
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask'].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            logits = self.model(input_ids=input_ids)

        loss = self.loss_fn(logits, labels)
        preds = (torch.sigmoid(logits) > 0.5).int()
        return loss, preds, labels

    def train_epoch(self, train_dataloader):
        self.model.train()
        total_loss = 0
        self.train_accuracy.reset()

        for batch_index, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.optimizer.zero_grad()
            loss, preds, labels = self._forward_pass(batch)
            loss.backward()

            if self.gradient_clip_norm:
                 utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.optimizer.step()
            total_loss += loss.item()
            self.train_accuracy.update(preds, labels.int())

            for callback in self.callbacks:
                callback.on_batch_end(self, batch_index, loss.item())

        avg_loss = total_loss / len(train_dataloader)
        avg_acc = self.train_accuracy.compute().item()
        self.history['train_loss'].append(avg_loss)
        
        logger.info(f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc:.4f}")
        return avg_loss, avg_acc

    def validate(self, val_dataloader):
        self.model.eval()
        total_loss = 0
        self.val_accuracy.reset()

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                loss, preds, labels = self._forward_pass(batch)
                total_loss += loss.item()
                self.val_accuracy.update(preds, labels.int())

        avg_loss = total_loss / len(val_dataloader)
        avg_acc = self.val_accuracy.compute().item()
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(avg_acc)

        logger.info(f"Val Loss: {avg_loss:.4f} | Val Acc: {avg_acc:.4f}")
        return avg_loss, avg_acc

    def train(self, train_dataloader, val_dataloader):
        logger.info(f"--- Starting Training for {self.config.max_epochs} Epochs ---")

        for epoch in range(self.config.max_epochs):
            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)

            logger.info(f"Epoch {epoch + 1}/{self.config.max_epochs}")
            self.train_epoch(train_dataloader)
            val_loss, val_acc = self.validate(val_dataloader)

            # All checkpointing and early stopping logic happens here via callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, val_loss)

            # Check if any callback signaled to stop training
            if any(getattr(cb, 'stop_training', False) for cb in self.callbacks):
                logger.info("Early stopping signal received.")
                break

        logger.info("--- Training Finished ---")
        for callback in self.callbacks:
            callback.on_train_end(self)

    
    def plot_history(self):
        """
        Generates and saves Training vs Validation plots for Loss and Accuracy.
        """
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))

        # 1. Loss Subplot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, self.history['val_loss'], label='Val Loss', marker='o')
        plt.title('Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 2. Accuracy Subplot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], label='Train Acc', marker='s')
        plt.plot(epochs, self.history['val_acc'], label='Val Acc', marker='s')
        plt.title('Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        
        plot_file = self.config.report_dir / f"{self.model_name}_learning_curves.png"
        plt.savefig(plot_file)
        logger.info(f"Learning curves saved to {plot_file}")
        plt.show()

    def save_history_to_csv(self):
        """Saves the history dictionary to a CSV file for future reference."""

        df = pd.DataFrame(self.history)
        csv_file = self.config.report_dir / f"{self.model_name}_history.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Training history CSV saved to {csv_file}")