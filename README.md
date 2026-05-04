# Time Series Anomaly Detection with KDD Cup 1999 Dataset

Production-ready modular codebase for unsupervised anomaly detection using LSTM Autoencoder on network intrusion data (KDD99).

## Features
- **Modular architecture**: Data, models, pipelines, utils
- **PyTorch LSTM Autoencoder**: Train on normal traffic, detect anomalies via reconstruction error
- **Hydra configs**: Flexible hparams
- **Click CLI**: `python main.py train`, `predict`, `evaluate`
- **Data validation**: Pydantic schemas
- **Docker-ready**

## Quick Start
```bash
# Activate venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Train model
python main.py train

# Predict on test data
python main.py predict --model-path models/autoencoder.pt --data dataset/.../test.csv --threshold 0.02

# Evaluate
python main.py evaluate --model-path models/autoencoder.pt --test-data dataset/.../test.csv
```

## Dataset
- **Source**: KDD Cup 1999 (reduced multiclass cleaned)
- **Train on**: `normal` connections only
- **Anomalies**: dos, probe, r2l, u2r

## Architecture
See [TODO.md](./TODO.md) for progress.

## Development
```bash
poetry install  # Alternative dep mgmt
pytest tests/
```

## Docker
```bash
docker build -t anomaly-detection .
docker run -v $(pwd)/models:/app/models anomaly-detection python main.py train
