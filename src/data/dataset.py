"""
KDD99 dataset loader.
"""
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .schemas import KDD99Row, validate_dataframe
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class KDD99Dataset(Dataset):
    """PyTorch Dataset for KDD99 sequences."""
    def __init__(self, sequences: torch.Tensor, labels: Optional[torch.Tensor] = None):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]

def load_kdd99_data(
    root_dir: Path,
    train_file: str,
    test_file: str,
    normal_label: str = "normal",
    val_split: float = 0.2,
    sequence_length: int = 10
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Load and split KDD99 data."""
    logger.info("Loading KDD99 dataset...")

    # Load raw
    train_path = root_dir / train_file
    test_path = root_dir / test_file

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Validate
    train_df = validate_dataframe(train_df)
    test_df = validate_dataframe(test_df)

    # Filter normal for unsupervised train
    normal_train = train_df[train_df['connection_type'] == normal_label].drop('connection_type', axis=1)
    all_test = test_df.drop('connection_type', axis=1)
    test_labels = pd.get_dummies(test_df['connection_type']).values  # Multi-class labels

    logger.info(f"Normal train samples: {len(normal_train)}")

    # Note: Sequences created in preprocessor
    # Here return raw DataFrames for now
    return normal_train, train_df, test_df, {'normal_train': normal_train, 'test': all_test, 'test_labels': test_labels}

# Placeholder for full loader with sequences
