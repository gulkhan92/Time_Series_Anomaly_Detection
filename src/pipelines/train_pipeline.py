"""
Hydra training pipeline for anomaly detection.
"""
import hydra
from hydra.core.config_store import ConfigStore
from pathlib import Path
import torch
import mlflow
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.models import LSTMAutoencoder, Trainer
from src.data.preprocessor import KDD99Preprocessor
from src.utils.metrics import calc_threshold, anomaly_metrics
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

CS = ConfigStore.instance()
# Register config (if needed for overrides)

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Paths
    models_dir = Path.cwd() / cfg.paths.models
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "autoencoder.pt"
    
    # MLflow
    mlflow.set_experiment("anomaly_detection")
    with mlflow.start_run(run_name="train"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Data
        data_cfg = cfg.data
        root_dir = Path(data_cfg.root_dir)
        pp = KDD99Preprocessor(sequence_length=data_cfg.sequence_length)
        
        # Load raw data (use dataset loader for normal_df)
        from src.data.dataset import load_kdd99_data
        normal_df, _, test_df_full, splits = load_kdd99_data(
            root_dir=root_dir,
            train_file=data_cfg.train_file,
            test_file=data_cfg.test_file,
            normal_label=data_cfg.normal_label
        )
        anomalous_df = splits['test']
        
        # Dataloaders
        train_dl, val_dl, test_dl, test_labels_np = pp.get_dataloaders(
            normal_df, anomalous_df,
            val_split=data_cfg.val_split,
            batch_size=data_cfg.batch_size
        )
        
        logger.info(f"Data ready: train {len(train_dl.dataset)}, val {len(val_dl.dataset)}, test {len(test_dl.dataset) if test_dl else 0}")
        logger.info(f"Model input_dim: {pp.output_dim}")
        
        # Model
        model = LSTMAutoencoder(
            input_dim=pp.output_dim,
            hidden_dim=cfg.model.hidden_dim,
            latent_dim=cfg.model.latent_dim,
            num_layers=cfg.model.num_layers
        )
        
        # Trainer
        trainer = Trainer(
            model,
            lr=cfg.model.lr,
            device=torch.device(cfg.training.device),
            patience=cfg.training.patience,
            min_delta=cfg.training.min_delta,
            log_every=cfg.training.log_every
        )
        
        # Train
        metrics = trainer.fit(
            train_dl, val_dl,
            epochs=cfg.model.epochs,
            save_path=model_path
        )
        
        # Evaluate test
        if test_dl:
            test_errors = trainer.compute_errors(test_dl)
            train_errors = trainer.compute_errors(train_dl)
            threshold = calc_threshold(train_errors)
            eval_metrics = anomaly_metrics(test_errors, test_labels_np, threshold)
            
            logger.info(f"Test PRF1: {eval_metrics['f1']:.4f}")
            mlflow.log_metrics(eval_metrics)
            mlflow.log_metric("test_mse_mean", test_errors.mean())
        
        logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
