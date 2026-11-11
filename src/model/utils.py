"""Reproducibility helpers: fixed random seed and device selection."""
import os
import random

import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix all random number generators so that runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA / Apple MPS / CPU, whichever is available."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
