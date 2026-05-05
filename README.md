![Time Series Anomaly Detection](https://img.shields.io/badge/Time_Series-Anomaly_Detection-brightgreen?style=for-the-badge)

# Time Series Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-green.svg)](https://pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gulkhan92/Time_Series_Anomaly_Detection/blob/main/LICENSE)

This repository is intended to provide a baseline for hands-on practice with modular codebase for unsupervised anomaly detection using LSTM Autoencoder on network intrusion data (KDD99).

## Features

| Feature | Description |
|---------|-------------|
| **LSTM Autoencoder** | Bi-LSTM encoder-decoder for sequence reconstruction |
| **Hydra Configs** | Flexible experiment management |
| **MLflow Integration** | Logging, tracking, UI |
| **Pydantic** | Strict data validation/schemas |
| **Click CLI** | `main.py train/infer` ready |
| **Docker** | Containerized deployment |
| **Pytest** | 80%+ coverage tests |
| **Modular** | data/models/pipelines/utils |

## Results Summary (KDD99 5-Class)
| Metric | Value |
|--------|-------|
| Test F1 (macro) | **0.82** |
| Precision | 0.79 |
| Recall | 0.85 |
| Test MSE | 0.028 |
| Threshold | 0.031 |

Terminal: Epoch losses, final F1 printed. `mlflow ui`.

## Mermaid Flow Diagrams

```mermaid
flowchart TD
    A[Raw Train CSV normal] --> B[Pydantic Validate]
    B --> C[Preprocessor fit<br/>Scale OHE Seq=10]
    C --> D[Train DL Shuffle]
    C --> E[Val DL NoShuffle]
    D --> F[LSTM Autoencoder]
    E --> F
    F --> G[MSE Loss Adam<br/>Early Stop pat=5]
    G --> H[Save model.pt<br/>train_errors.npy<br/>MLflow]
```

**Training Flow** (above): Normal data → unsupervised train/val → model.

```mermaid
flowchart TD
    I[Raw Test CSV 5class] --> J[Preprocessor transform]
    J --> K[Test DL]
    K --> L[Load model.pt]
    L --> M[Recon MSE per seq]
    TrainErrors[train_errors.npy<br/>95th percentile] --> N[Threshold = 0.031]
    M --> N
    N --> O[Preds MSE > th]
    O --> P[PRF1 vs labels<br/>MLflow npy save]
```

**Inference Flow** (above): Test → MSE > th → metrics.

(Render in Markdown viewers supporting Mermaid like GitHub/VSCode.)

## Installation

### Virtual Environment (Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Poetry (Alternative)
```bash
poetry install --with dev
```

## Quick Start

```bash
python main.py train  # Trains and logs to MLflow
python main.py infer  # Inference + metrics
mlflow ui             # View experiments
```

## Dataset

dataset/reduced_multiclass/.../*.csv (41 features post-preprocessing: scaled, OHE, seq_len=10)

## CLI Reference

```bash
python main.py train --help
python main.py infer --model-path models/autoencoder.pt --threshold 0.031
```

## Project Structure

```
.
├── conf/config.yaml
├── dataset/           # KDD99 cleaned CSVs
├── src/
│   ├── data/          # Preprocessor, Dataset, Schemas
│   ├── models/        # Autoencoder, Trainer
│   ├── pipelines/     # Train/Inference Pipeline
│   └── utils/         # Logger, Metrics
├── tests/             # Pytest suite
├── main.py            # CLI Entry
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t ts-anomaly-detection .
docker-compose up
```

## Contributing

1. Fork repo & clone
2. `git checkout -b feat/your-feature`
3. Commit/PR to `main`

PRs welcome!

## License

MIT © gulkhan92
