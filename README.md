# Multimodal Biometric Recognition (Iris + Fingerprint) — MLOps-ready template

This repository is structured to demonstrate **clean, modular Python**, **typing**, **testing**, **CI**, and a **reproducible ML training + inference pipeline** for a multimodal biometric dataset.

**Dataset download (required):**
```python
import kagglehub

path = kagglehub.dataset_download("ninadmehendale/multimodal-iris-fingerprint-biometric-data")
print("Path to dataset files:", path)
```

## Quickstart

1) Create env + install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

2) Train from CLI (creates a manifest by scanning the dataset folder)
```bash
mmbiometric-train --dataset-dir /path/from/kagglehub --output-dir runs/exp1
```

3) Or run the notebook:
- `notebooks/01_training.ipynb`

## Why this structure?

- **src/**: installable package (`mmbiometric`) with typed modules
- **tests/**: pytest unit + smoke tests
- **configs/**: config-driven training (YAML)
- **.github/workflows/**: CI pipeline (lint + typecheck + tests)
- **docs/**: lightweight docs + architecture notes

## Notes

- The Kaggle notebook link you provided is treated as a reference implementation. This repo ports the approach into reusable modules (dataset abstraction, model, training loop, inference).
