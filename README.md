# Time Series Anomaly Detection with KDD Cup 1999 Dataset

This repository is intended to provide a baseline for hands-on practice with modular codebase for unsupervised anomaly detection using LSTM Autoencoder on network intrusion data (KDD99).

## Features
- PyTorch LSTM Autoencoder (bi-LSTM recon)
- Hydra/MLflow/Hydra CLI
- Pydantic data validation
- Pipelines + CLI ready

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

## Quick Start
```
pip install -r requirements.txt
python main.py train  # See epoch logs
python main.py infer  # See F1 results
mlflow ui
```

## Dataset
dataset/reduced_multiclass/reduced_multiclass/cleaned/*.csv (41 feat post-preproc)

## CLI
```
python main.py train
python main.py infer --model-path models/autoencoder.pt
```

Full project ready. Visual flows in Mermaid diagrams above.
