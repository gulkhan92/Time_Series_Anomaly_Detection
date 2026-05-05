"""Pytest fixtures for anomaly detection project."""
import pytest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def dummy_normal_df():
    """Dummy normal training data."""
    return pd.DataFrame({
        'duration': np.random.rand(100),
        'protocol_type': np.random.choice(['tcp', 'udp'], 100),
        'service': np.random.choice(['http', 'ftp'], 100),
        'flag': np.random.choice(['SF', 'REJ'], 100),
        'src_bytes': np.random.rand(100) * 1000
    })

@pytest.fixture
def dummy_anomalous_df(dummy_normal_df):
    """Dummy anomalous data."""
    df = dummy_normal_df.copy()
    df['src_bytes'] *= 10  # High anomaly feature
    return df
