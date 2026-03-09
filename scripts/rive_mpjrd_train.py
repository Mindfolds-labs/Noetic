from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from noetic_pawp.rive_mpjrd import RIVEDepthLoss, RIVEDepthNet, compute_depth_metrics


def make_synthetic_depth(x: torch.Tensor) -> torch.Tensor:
    """Create dense pseudo-depth from image intensity + smooth prior."""
    b, _, h, w = x.shape
    ys = torch.linspace(-1, 1, h, device=x.device)[None, None, :, None]
    xs = torch.linspace(-1, 1, w, device=x.device)[None, None, None, :]
    radial = torch.sqrt(xs.square() + ys.square())
    depth = 6.0 * (1.0 - x) + 2.0 * radial
    return depth.clamp(0.1, 10.0)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, metrics = [], []
    with torch.no_grad():
        for img, gt in loader:
            img, gt = img.to(device), gt.to(device)
            pred, rive = model(img)
            l = criterion(pred, gt, rive)
            losses.append(l["total"].item())
            m = compute_depth_metrics(pred, gt, max_depth=10.0)
            if m:
                metrics.append(m)
    avg = {k: sum(x[k] for x in metrics) / len(metrics) for k in metrics[0]} if metrics else {}
    return sum(losses) / max(1, len(losses)), avg


def main() -> None:
    ap = argparse.ArgumentParser(description="RIVE+MPJRD training (demo with dense pseudo-depth)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-prefix", type=str, default="rive_mpjrd_digits")
    args = ap.parse_args()

    d = load_digits()
    x = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0
    x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
    y = make_synthetic_depth(x)

    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)
    tr = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    te = DataLoader(TensorDataset(xte, yte), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RIVEDepthNet(output_h=32, output_w=32).to(device)
    criterion = RIVEDepthLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for img, gt in tr:
            img, gt = img.to(device), gt.to(device)
            pred, rive = model(img)
            loss = criterion(pred, gt, rive)["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss, val_metrics = evaluate(model, te, criterion, device)
        row = {"epoch": epoch, "val_loss": round(val_loss, 4), **{k: round(v, 4) for k, v in val_metrics.items()}}
        history.append(row)
        print(row)

    out = Path("docs")
    out.mkdir(exist_ok=True)
    (out / f"{args.out_prefix}.json").write_text(json.dumps({"history": history}, indent=2))


if __name__ == "__main__":
    main()
