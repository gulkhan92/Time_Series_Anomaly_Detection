"""
Data preprocessing: scaling, encoding, sequence creation.
"""
import logging
import numpy as np
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from ..utils.logger import setup_logger
from .schemas import validate_dataframe
from .dataset import KDD99Dataset

logger = setup_logger(__name__)

class KDD99Preprocessor:
    """Preprocessor for KDD99 features (assumes cleaned numeric CSV)."""
    FEATURE_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    CATEGORICAL_COLS = ['protocol_type', 'flag', 'service']
    NUMERIC_COLS = [col for col in FEATURE_NAMES if col not in CATEGORICAL_COLS]

    def __init__(self, service_top_k: int = 10, sequence_length: int = 10):
        self.service_top_k = service_top_k
        self.sequence_length = sequence_length
        self.preprocessor = None
        self.service_encoder = None

    def fit(self, df: pd.DataFrame):
        """Fit preprocessors on normal training data."""
        logger.info("Fitting preprocessor...")
        
        # Top-K services
        top_services = df['service'].value_counts().head(self.service_top_k).index
        self.service_encoder = OneHotEncoder(categories=[['other']] + [list(top_services)], 
                                           sparse_output=False, handle_unknown='ignore')
        
        ct = ColumnTransformer([
            ('num', StandardScaler(), self.NUMERIC_COLS),
            ('proto', OneHotEncoder(sparse_output=False, drop='first'), ['protocol_type']),
            ('flag', OneHotEncoder(sparse_output=False, drop='first'), ['flag']),
            ('service', self.service_encoder, ['service'])
        ], remainder='passthrough')
        
        self.preprocessor = ct.fit(df[self.FEATURE_NAMES])
        self.output_dim = self.preprocessor.transform(df[self.FEATURE_NAMES]).shape[1]
        logger.info(f"Preprocessor fitted. Output feature shape: {self.output_dim}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform to static features."""
        if self.preprocessor is None:
            raise ValueError("Fit preprocessor first!")
        return self.preprocessor.transform(df[self.FEATURE_NAMES])

    def create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for time series."""
        if len(X) < self.sequence_length:
            raise ValueError("Not enough data for sequences")
        
        seq_X, seq_y = [], []
        for i in range(len(X) - self.sequence_length + 1):
            seq_X.append(X[i:i + self.sequence_length])
            if y is not None:
                seq_y.append(y[i + self.sequence_length - 1])  # Label at end of window
        
        seq_X = np.array(seq_X)
        seq_y = np.array(seq_y) if seq_y else None
        logger.info(f"Created {len(seq_X)} sequences of length {self.sequence_length}")
        return seq_X, seq_y

    def get_dataloaders(self, normal_df: pd.DataFrame, anomalous_df: pd.DataFrame = None, 
                       val_split: float = 0.2, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
        """Full pipeline: preprocess -> sequences -> dataloaders."""
        # Fit on normal
        self.fit(normal_df)
        
        # Transform
        X_normal = self.transform(normal_df)
        X_test = self.transform(anomalous_df) if anomalous_df is not None else None
        
        # Sequences (train normal only for unsupervised)
        train_seq, _ = self.create_sequences(X_normal)
        
        # Train/val split (normal only)
        X_train, X_val, _, _ = train_test_split(train_seq, np.zeros(len(train_seq)), test_size=val_split, shuffle=False)
        
        # Test sequences if provided
        test_seq, test_labels = None, None
        if X_test is not None:
            test_seq, _ = self.create_sequences(X_test)
            # Align labels: end-of-window, argmax multi-hot to class index
            full_labels = pd.get_dummies(anomalous_df['connection_type']).values
            test_labels = np.argmax(full_labels[self.sequence_length-1:], axis=1)
        
        # Datasets
        train_ds = KDD99Dataset(torch.FloatTensor(X_train))
        val_ds = KDD99Dataset(torch.FloatTensor(X_val))
        test_ds = KDD99Dataset(torch.FloatTensor(test_seq), torch.LongTensor(test_labels)) if test_seq is not None else None
        
        # Dataloaders
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
        
        return train_dl, val_dl, test_dl, test_labels
