"""Dataset for the mel spectrograms produced by src/preprocessing/preprocess.py.

Input:  data/spectrograms/*.npy   (n_mels x frames, log-mel in dB)
        data/spectrograms/index.csv   file|class|speaker_id|clip_idx

Every clip belongs to a *recording* ``{class}_{speaker_id}`` (e.g. clips
``0_3_c000.npy`` and ``0_3_c001.npy`` both come from recording ``0_3``).
Splits are always made on recordings, never on single clips, so clips of one
recording cannot end up in both train and test (data leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPEC_DIR = Path("data/spectrograms")


def load_index(spec_dir=SPEC_DIR) -> pd.DataFrame:
    """Read index.csv and add a ``recording`` column (``{class}_{speaker_id}``)."""
    df = pd.read_csv(Path(spec_dir) / "index.csv", sep="|", dtype={"speaker_id": str})
    df["recording"] = df["class"].astype(str) + "_" + df["speaker_id"]
    return df


class SpectrogramDataset(Dataset):
    """Returns ``(x, y)`` with x of shape (1, n_mels, frames) and y in {0, 1}.

    Each spectrogram is standardized on its own (zero mean, unit variance),
    which removes differences in overall loudness between recordings.
    """

    def __init__(self, index: pd.DataFrame, spec_dir=SPEC_DIR):
        self.files = [Path(spec_dir) / f for f in index["file"]]
        self.labels = index["class"].astype(int).to_numpy()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = np.load(self.files[i]).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return torch.from_numpy(x).unsqueeze(0), int(self.labels[i])
