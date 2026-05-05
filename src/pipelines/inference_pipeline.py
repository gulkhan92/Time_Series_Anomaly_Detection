"""
Hydra inference pipeline for anomaly detection.
"""
import hydra
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
import mlflow

from src.models import LSTMAutoencoder, Trainer
from src.data.preprocessor import KDD99Preprocessor
from src.data.dataset import load_kdd99_data, KDD99Dataset
from src.utils.metrics import reconstruction_mse, calc_threshold, anomaly_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Paths
    models_dir = Path.cwd() / cfg.paths.models
    model_path = models_dir / "autoencoder.pt"
    output_dir = Path.cwd() / cfg.paths.artifacts / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow
    mlflow.set_experiment("anomaly_detection")
    with mlflow.start_run(run_name="inference"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Load data (new/test data)
        data_cfg = cfg.data
        root_dir = Path(data_cfg.root_dir)
        _, _, test_df_full, splits = load_kdd99_data(
            root_dir=root_dir,
            train_file=data_cfg.train_file,  # Not used
            test_file=data_cfg.test_file,
            normal_label=data_cfg.normal_label
        )
        anomalous_df = splits['test']
        true_labels = splits['test_labels']
        
        # Preprocessor (load fitted? For inference, refit on normal or load fitted)
        pp = KDD99Preprocessor(sequence_length=data_cfg.sequence_length)
        # Fit preprocessor on normal data (load small subset for demo)
        normal_df, _, _, _ = load_kdd99_data(
            root_dir=root_dir,
            train_file=data_cfg.train_file,
            test_file=data_cfg.test_file,
            normal_label=data_cfg.normal_label
        )
        pp.fit(normal_df.head(10000))  # Small fit for demo
        
        # Create test sequences
        X_test = pp.transform(anomalous_df)
        test_seq, _ = pp.create_sequences(X_test, None)
        test_dataset = KDD99Dataset(torch.FloatTensor(test_seq))
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=data_cfg.batch_size, shuffle=False)
        
        logger.info(f"Inference on {len(test_dataset)} test sequences")
        
        # Load model
        input_dim = pp.output_dim
        model = LSTMAutoencoder(input_dim=input_dim, **cfg.model)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # Predict errors
        device = torch.device(cfg.training.device)
        model.to(device)
        trainer = Trainer(model, device=device)  # Minimal trainer for compute_errors
        test_errors = trainer.compute_errors(test_dl)
        
        # Threshold from config percentile on test errors (in prod load trained threshold)
        percentile = cfg.training.threshold_percentile
        threshold = np.percentile(test_errors, percentile)
        
        # Predictions
        predictions = (test_errors > threshold).astype(int)
        
        # Metrics
        eval_metrics = anomaly_metrics(test_errors, true_labels[:len(test_errors)], threshold)
        logger.info(f"Inference PRF1: {eval_metrics['f1']:.4f}")
        mlflow.log_metrics(eval_metrics)
        
        # Save results
        results_df = pd.DataFrame({
            'reconstruction_error': test_errors,
            'prediction': predictions,
            'true_label': true_labels[:len(test_errors)].argmax(axis=1)
        })
        results_df.to_csv(output_dir / "inference_results.csv", index=False)
        logger.info(f"Results saved to {output_dir / 'inference_results.csv'}")
        
        mlflow.log_artifact(str(output_dir / "inference_results.csv"))
