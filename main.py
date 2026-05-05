#!/usr/bin/env python
"""CLI entrypoint for KDD99 anomaly detection with Hydra + Click."""
import click
import subprocess
import sys
from pathlib import Path
import os

# Ensure src in path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

@click.group()
@click.version_option("1.0.0")
def cli():
    """Time Series Anomaly Detection CLI (LSTM Autoencoder)."""
    pass

@cli.command()
@click.option('--config-path', default='conf', help='Hydra config path')
@click.option('--config-name', default='config', help='Hydra config name')
def train(config_path, config_name):
    """Run training pipeline with Hydra."""
    cmd = [
        sys.executable, '-m', 'src.pipelines.train_pipeline',
        f'+data.root_dir={os.environ.get("PROJECT_ROOT", "./dataset")}/reduced_multiclass/reduced_multiclass/cleaned',
        f'hydra.run.dir=models',
        f'hydra.job.chdir=False'
    ]
    click.echo("🚀 Starting training...")
    result = subprocess.run(cmd, check=True)
    click.echo("✅ Training complete!")

@cli.command()
@click.option('--model-path', default='models/autoencoder.pt', help='Path to trained model')
@click.option('--output-dir', default='artifacts/inference', help='Output directory')
def predict(model_path, output_dir):
    """Run inference pipeline."""
    cmd = [
        sys.executable, '-m', 'src.pipelines.inference_pipeline',
        f'paths.models={Path(model_path).parent}',
        f'paths.artifacts={output_dir}',
        f'hydra.run.dir={output_dir}'
    ]
    click.echo("🔮 Running inference...")
    result = subprocess.run(cmd, check=True)
    click.echo(f"✅ Predictions saved to {output_dir}/inference_results.csv")

@cli.command()
def evaluate():
    """Evaluate model performance (PRF1, ROC)."""
    click.echo("📊 Running evaluation...")
    results_path = Path('artifacts/inference/inference_results.csv')
    if results_path.exists():
        click.echo(f"✅ Results: {results_path}")
        click.echo("📈 View MLflow UI: mlflow ui")
    else:
        click.echo("❌ Run `python main.py predict` first!")

if __name__ == '__main__':
    cli()
