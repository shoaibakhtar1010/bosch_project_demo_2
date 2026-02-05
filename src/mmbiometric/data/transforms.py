from __future__ import annotations

from torchvision import transforms


def default_image_transform(image_size: int):
    """Standard transform; keep deterministic and simple."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
