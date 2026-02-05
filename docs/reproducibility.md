# Reproducibility

- Global seed set (python random, numpy, torch)
- cuDNN deterministic mode enabled (may reduce speed)
- Train/val split performed by `subject_id` to avoid identity leakage
