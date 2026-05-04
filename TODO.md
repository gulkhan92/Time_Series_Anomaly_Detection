# Time Series Anomaly Detection Project - Implementation TODO

## Phase 1: Project Setup [In Progress]
- [x] Create TODO.md ✅
- [x] Create requirements.txt ✅
- [x] Create pyproject.toml (Poetry) ✅
- [x] Setup virtual environment (venv) - installing... ✅
- [x] Create project skeleton directories (src/data, src/models, etc.) ✅
- [x] Create basic files: conf/config.yaml, .env.example, README.md, Dockerfile ✅

## Phase 1: Project Setup [Complete] ✅

## Phase 2: Data Module [In Progress]
- [x] src/data/schemas.py (Pydantic models) ✅
- [x] src/data/dataset.py (KDD99 loader) ✅
- [ ] src/data/preprocessor.py (scaling, encoding, sequencing)
- [x] src/utils/logger.py ✅

## Phase 3: Model & Training
- [ ] src/models/trainer.py
- [ ] src/utils/metrics.py

## Phase 4: Pipelines & CLI
- [ ] src/pipelines/train_pipeline.py
- [ ] src/pipelines/inference_pipeline.py
- [ ] main.py (Click CLI)
- [ ] config/config.yaml (Hydra)

## Phase 5: Testing & Deployment
- [ ] tests/ (pytest unit tests)
- [ ] Docker & docker-compose.yml
- [ ] Train baseline model
- [ ] Documentation

**Dataset**: reduced_multiclass cleaned (5-class)
**Framework**: TensorFlow/Keras LSTM Autoencoder
