"""Unit tests for data preprocessor."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import KDD99Preprocessor
from src.data.dataset import KDD99Dataset

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'duration': [1, 2, 3],
        'protocol_type': ['tcp', 'udp', 'tcp'],
        'service': ['http', 'ftp', 'http'],
        'flag': ['SF', 'SF', 'REJ'],
        'src_bytes': [100, 200, 300]
    })

def test_preprocessor_fit_transform(sample_df):
    pp = KDD99Preprocessor(service_top_k=2, sequence_length=2)
    pp.fit(sample_df)
    X = pp.transform(sample_df)
    assert X.shape[1] > 0
    assert len(X) == len(sample_df)

def test_sequences():
    pp = KDD99Preprocessor(sequence_length=2)
    X_dummy = np.random.rand(3, 10)
    seq_X, _ = pp.create_sequences(X_dummy)
    assert seq_X.shape == (2, 2, 10)

def test_dataloader(sample_df):
    pp = KDD99Preprocessor()
    pp.fit(sample_df)
    X = pp.transform(sample_df)
    seq_X, _ = pp.create_sequences(X)
    ds = KDD99Dataset(seq_X)
    assert len(ds) == len(seq_X)
