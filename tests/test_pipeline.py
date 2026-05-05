"""Integration tests for pipelines."""
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessor import KDD99Preprocessor
from src.models.autoencoder import LSTMAutoencoder
from src.models.trainer import Trainer
from src.utils.metrics import reconstruction_mse

@pytest.mark.integration
def test_train_pipeline_flow(tmp_path):
    """Test end-to-end train flow w/ dummy data."""
    # Dummy data
    import pandas as pd
    import numpy as np
    dummy_df = pd.DataFrame(np.random.rand(100, 5), columns=['f1','f2','f3','f4','f5'])
    
    # Components
    pp = KDD99Preprocessor(sequence_length=5, service_top_k=1)
    pp.fit(dummy_df)
    X = pp.transform(dummy_df)
    seq_X, _ = pp.create_sequences(X)
    
    model = LSTMAutoencoder(input_dim=pp.output_dim)
    trainer = Trainer(model)
    
    # Quick train check
    dummy_dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.FloatTensor(seq_X[:10])), 
        batch_size=4
    )
    metrics = trainer.fit(dummy_dl, dummy_dl, epochs=1)
    assert 'train_mse_mean' in metrics
    
    errors = trainer.compute_errors(dummy_dl)
    assert len(errors) == 10

@pytest.mark.skip("Requires dataset - run manually")
def test_full_pipeline():
    """Full pipeline smoke test."""
    from src.pipelines.train_pipeline import main as train_main
    # Would require mocking Hydra/config
    pytest.skip("Manual E2E test")
