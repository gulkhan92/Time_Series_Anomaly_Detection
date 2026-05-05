"""Unit tests for LSTM Autoencoder model."""
import pytest
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.autoencoder import LSTMAutoencoder

@pytest.fixture
def model():
    return LSTMAutoencoder(input_dim=41, hidden_dim=64, latent_dim=32)

def test_model_forward(model):
    x = torch.randn(8, 10, 41)  # batch, seq, features
    recon = model(x)
    assert recon.shape == x.shape
    assert not torch.isnan(recon).any()

def test_reconstruction_loss(model):
    x = torch.randn(4, 5, 41)
    recon = model(x)
    loss = model.calculate_reconstruction_loss(x, recon)
    assert loss.shape == (x.size(0),)
    assert (loss > 0).all()

def test_model_serialization(model, tmp_path):
    path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), path)
    model.load_state_dict(torch.load(path))
    assert True  # No errors
