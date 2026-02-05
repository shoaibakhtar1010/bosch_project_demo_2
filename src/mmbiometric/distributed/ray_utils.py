from __future__ import annotations

import os
from typing import Optional

import ray


def init_ray(num_cpus: int | None = None, num_gpus: int | None = None) -> None:
    """
    Connect to an existing Ray cluster if RAY_ADDRESS is set,
    otherwise start a local Ray runtime.
    """
    # Avoid noisy "Calling ray.init() again" logs.
    if ray.is_initialized():
        return

    address = os.environ.get("RAY_ADDRESS")
    if address:
        ray.init(address=address, ignore_reinit_error=True)
        return

    ray.init(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=True,
    )
