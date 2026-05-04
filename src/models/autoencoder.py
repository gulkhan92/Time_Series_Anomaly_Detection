"""
LSTM Autoencoder for anomaly detection (PyTorch).
"""
import torch
import torch.nn as nn
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for time series reconstruction."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)  # Bi-directional

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim * 2)
        self.decoder = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.decoder_fc_out = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Encoder
        _, (hidden, cell) = self.encoder(x)
        # Concat bi-dir hidden
        hidden = torch.cat((hidden[0::2], hidden[1::2]), dim=1)
        cell = torch.cat((cell[0::2], cell[1::2]), dim=1)
        latent = self.encoder_fc(hidden)

        # Repeat latent for decoder seq
        latent = latent.unsqueeze(1).repeat(1, seq_len, 1)
        decoder_hidden = self.decoder_fc(latent)

        # Decoder
        decoder_hidden = decoder_hidden.view(batch_size, seq_len, self.hidden_dim * 2)
        _, decoder_hidden = self.decoder(decoder_hidden)
        decoder_hidden = torch.cat((decoder_hidden[0::2], decoder_hidden[1::2]), dim=1)
        out = self.decoder_fc_out(decoder_hidden)

        return out

    def calculate_reconstruction_loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """MSE loss."""
        return nn.functional.mse_loss(recon, x, reduction='none').mean(dim=[1,2])

def test_model():
    """Test instantiation."""
    model = LSTMAutoencoder(input_dim=100)
    x = torch.randn(32, 10, 100)
    recon = model(x)
    print(recon.shape)
    logger.info("Model test passed.")
