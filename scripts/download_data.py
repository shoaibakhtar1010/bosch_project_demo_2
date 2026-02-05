from __future__ import annotations

import kagglehub

if __name__ == "__main__":
    path = kagglehub.dataset_download("ninadmehendale/multimodal-iris-fingerprint-biometric-data")
    print("Path to dataset files:", path)
