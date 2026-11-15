"""Train the simple CNN on mel spectrograms and evaluate it.

Prerequisite: spectrograms exist
    python -m src.preprocessing.preprocess all

Run from the repo root:
    python -m src.model.train
    python -m src.model.train --epochs 30 --lr 1e-3

Output:
    models/simple_cnn.pt     weights of the epoch with the best val macro F1
    models/history.csv       loss and metrics per epoch
    printed table: macro F1, FAR, FRR on train / val / test
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.cnn import SimpleCNN
from src.model.dataset import SPEC_DIR, SpectrogramDataset, load_index
from src.model.metrics import compute_metrics
from src.model.split import PARTS, get_split
from src.model.utils import SEED, set_seed

MODEL_DIR = Path("models")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def evaluate(model, loaders, device) -> pd.DataFrame:
    return pd.DataFrame(
        {p: compute_metrics(*predict(model, loaders[p], device)) for p in loaders}
    ).T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu", help="cpu is fully reproducible")
    a = ap.parse_args()

    set_seed(SEED)
    device = torch.device(a.device)

    index = load_index()
    split = get_split(index)
    datasets = {p: SpectrogramDataset(index[index["recording"].isin(split[p])]) for p in PARTS}
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(datasets["train"], batch_size=a.batch_size, shuffle=True, generator=g)
    eval_loaders = {p: DataLoader(datasets[p], batch_size=64) for p in PARTS}
    for p in PARTS:
        print(f"{p:5s}: {len(datasets[p]):4d} clips, class counts {np.bincount(datasets[p].labels, minlength=2)}")

    input_shape = datasets["train"][0][0].shape
    model = SimpleCNN(input_shape=tuple(input_shape)).to(device)

    # class weights against imbalance (fewer class-1 clips)
    counts = np.bincount(datasets["train"].labels, minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)

    MODEL_DIR.mkdir(exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        train_loss = total / len(datasets["train"])
        tr = compute_metrics(*predict(model, eval_loaders["train"], device))
        va = compute_metrics(*predict(model, eval_loaders["val"], device))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"val_{k}": v for k, v in va.items()}})
        print(f"epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"train F1 {tr['macro_f1']:.3f}  val F1 {va['macro_f1']:.3f}  "
              f"val FAR {va['FAR']:.3f}  val FRR {va['FRR']:.3f}")
        if va["macro_f1"] > best_f1:
            best_f1 = va["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "input_shape": tuple(input_shape),
                        "epoch": epoch}, MODEL_DIR / "simple_cnn.pt")

    pd.DataFrame(history).to_csv(MODEL_DIR / "history.csv", index=False)

    ckpt = torch.load(MODEL_DIR / "simple_cnn.pt")
    model.load_state_dict(ckpt["state_dict"])
    print(f"\nbest model (epoch {ckpt['epoch']}) -> {MODEL_DIR / 'simple_cnn.pt'}")
    print(evaluate(model, eval_loaders, device).round(3).to_string())


if __name__ == "__main__":
    main()
