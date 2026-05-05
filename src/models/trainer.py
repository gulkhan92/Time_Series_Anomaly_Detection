"""
PyTorch Trainer for LSTM Autoencoder anomaly detection.
"""
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from ..utils.logger import setup_logger
from ..utils.metrics import reconstruction_mse, calc_threshold, anomaly_metrics
from ..data.dataset import KDD99Dataset
logger = setup_logger(__name__)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        device: torch.device = torch.device("cpu"),
        patience: int = 5,
        min_delta: float = 0.001,
        log_every: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.criterion = nn.MSELoss(reduction='none')
        
        self.patience = patience
        self.min_delta = min_delta
        self.log_every = log_every
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.mlflow_run = mlflow.active_run() if mlflow.active_run() else mlflow.start_run()

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, list) and len(batch) == 2:
                x = batch[0].to(self.device)
            else:
                x = batch.to(self.device)
            
            self.optimizer.zero_grad()
            recon = self.model(x)
            loss = self.model.calculate_reconstruction_loss(x, recon).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % self.log_every == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("train_loss", avg_loss)
        return avg_loss

    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, list) and len(batch) == 2:
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                recon = self.model(x)
                loss = self.model.calculate_reconstruction_loss(x, recon).mean()
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("val_loss", avg_loss)
        self.scheduler.step(avg_loss)
        return avg_loss

    def fit(
        self,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        epochs: int,
        save_path: Optional[Path] = None
    ) -> Dict[str, float]:
        logger.info("Starting training...")
        metrics = {}
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dl)
            val_loss = self.validate_epoch(val_dl)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    mlflow.pytorch.log_model(self.model, "best_model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info("Early stopping")
                    break
        
        # Train errors for threshold
        train_errors = reconstruction_mse(self.model, train_dl, self.device)
        metrics['train_mse_mean'] = train_errors.mean()
        
        mlflow.log_params({
            'epochs': epochs,
            'best_val_loss': self.best_val_loss,
            'final_lr': self.optimizer.param_groups[0]['lr']
        })
        
        logger.info("Training complete.")
        return metrics

    def compute_errors(self, dataloader: torch.utils.data.DataLoader) -> np.ndarray:
        """Compute reconstruction errors on full dataloader."""
        return reconstruction_mse(self.model, dataloader, self.device)
