# Architecture

## Goals

- Modular codebase (`src/mmbiometric`)
- Config-driven training (`configs/`)
- Reproducible train/eval split (by `subject_id`)
- CI to enforce style + typing + tests

## Pipeline

1. Download dataset (kagglehub)
2. Build a manifest by scanning folders (paths to iris + fingerprint pairs)
3. Split by identity (subject_id)
4. Train model (two-tower backbone + fusion head)
5. Save checkpoint + label mapping
6. Load for inference

## Notes

The manifest builder is heuristic to tolerate dataset layout differences. For your submission,
document any adjustments you make to pairing logic to match the Kaggle dataset structure.
