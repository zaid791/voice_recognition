"""Train/val/test split by recording (temporary, until Person 4's split is ready).

The split is stored as three text files, one recording id per line:
    data/splits/train.txt, data/splits/val.txt, data/splits/test.txt
e.g.
    0_3
    1_12

If these files already exist they are used as-is, so Person 4 can replace
them with the final split (by real speaker) without touching the training code.

    python -m src.model.split            # create the split if missing
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.dataset import load_index
from src.model.utils import SEED

SPLIT_DIR = Path("data/splits")
PARTS = ("train", "val", "test")


def make_split(recordings: pd.DataFrame, val=0.15, test=0.15, seed=SEED) -> dict:
    """Stratified (per class) random split of recording ids."""
    rng = np.random.default_rng(seed)
    out = {p: [] for p in PARTS}
    for _, group in recordings.groupby("class"):
        ids = sorted(group["recording"].unique())
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * test))
        n_val = max(1, round(len(ids) * val))
        out["test"] += ids[:n_test]
        out["val"] += ids[n_test:n_test + n_val]
        out["train"] += ids[n_test + n_val:]
    return out


def load_split(split_dir=SPLIT_DIR) -> dict:
    return {p: (Path(split_dir) / f"{p}.txt").read_text().split() for p in PARTS}


def get_split(index: pd.DataFrame, split_dir=SPLIT_DIR) -> dict:
    """Load the split from disk, or create and save it if it does not exist."""
    split_dir = Path(split_dir)
    if all((split_dir / f"{p}.txt").exists() for p in PARTS):
        return load_split(split_dir)
    split = make_split(index)
    split_dir.mkdir(parents=True, exist_ok=True)
    for p in PARTS:
        (split_dir / f"{p}.txt").write_text("\n".join(sorted(split[p])) + "\n")
    print(f"created new split -> {split_dir}")
    return split


if __name__ == "__main__":
    idx = load_index()
    s = get_split(idx)
    for p in PARTS:
        part = idx[idx["recording"].isin(s[p])]
        print(f"{p:5s}: {len(s[p]):3d} recordings, {len(part):4d} clips, "
              f"class counts {part['class'].value_counts().sort_index().to_dict()}")
