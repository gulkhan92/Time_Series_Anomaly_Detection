"""Anomaly detection metrics: reconstruction MSE, threshold."""
import torch
import numpy as np
from typing import Tuple, Optional
from ..logger import setup_logger

logger = setup_logger(__name__)

def reconstruction_mse(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
                      device: torch.device) -> np.ndarray:
    """Compute per-sample MSE errors on dataloader."""
    model.eval()
    mses = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            recon = model(x)
            mse = torch.mean((x - recon)**2, dim=(1,2)).cpu().numpy()
            mses.extend(mse)
    mses = np.array(mses)
    logger.info(f"MSE stats - mean: {mses.mean():.4f}, std: {mses.std():.4f}")
    return mses

def calc_threshold(train_errors: np.ndarray, percentile: float = 95.0) -> float:
    """Calculate anomaly threshold from training errors."""
    threshold = np.percentile(train_errors, percentile)
    logger.info(f"Threshold ({percentile}th percentile): {threshold:.4f}")
    return threshold

def anomaly_metrics(errors: np.ndarray, true_labels: np.ndarray, threshold: float) -> dict:
    """Compute PRF1 for binary anomaly detection."""
    from sklearn.metrics import precision_recall_fscore_support
    
    preds = (errors > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='macro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold
    }
