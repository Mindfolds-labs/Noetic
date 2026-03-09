from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import json
from pathlib import Path

from noetic_pawp.feature_flags import add_feature_flag_arguments, feature_flags_from_args

import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from noetic_pawp.mmrn_prototype import MMRNPrototype, ProjectiveOCRConfig, ProjectiveOCRLoss


class DigitsProjectiveDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, augment: bool) -> None:
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return self.images.size(0)

    def _degrade(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment:
            return x
        a = (torch.rand(2) - 0.5) * 0.4
        t = (torch.rand(2) - 0.5) * 0.2
        theta = torch.tensor([[1.0 + a[0], 0.12 * a[1], t[0]], [0.12 * a[1], 1.0 + a[1], t[1]]], dtype=x.dtype)
        grid = F.affine_grid(theta.unsqueeze(0), x.unsqueeze(0).size(), align_corners=False)
        out = F.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)
        out = F.avg_pool2d(out.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        out = (out + 0.05 * torch.randn_like(out)).clamp(0.0, 1.0)
        return out

    def __getitem__(self, idx: int):
        clean = self.images[idx]
        label = self.labels[idx]
        degraded = self._degrade(clean)
        ipa_tokens = torch.tensor([int(label.item()) + 1, 0, 0], dtype=torch.long)
        return degraded, clean, ipa_tokens, label


def contour_target(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-6)
    return (mag / (mag.amax(dim=(2, 3), keepdim=True) + 1e-6)).clamp(0.0, 1.0)


def evaluate(model: MMRNPrototype, loader: DataLoader, criterion: ProjectiveOCRLoss, device: torch.device):
    model.eval()
    total = 0
    correct = 0
    losses = []
    with torch.no_grad():
        for x_deg, x_clean, ipa, y in loader:
            x_deg, x_clean, ipa, y = x_deg.to(device), x_clean.to(device), ipa.to(device), y.to(device)
            out = model(x_deg, ipa)
            target_c = contour_target(x_clean)
            ldict = criterion(out, y, target_c)
            losses.append(ldict["total"].item())
            pred = out["y_hat"].argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return sum(losses) / max(1, len(losses)), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino real MMRN + Projective OCR em sklearn digits")
    add_feature_flag_arguments(parser)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-prefix", type=str, default="mmrn_projective_ocr_training")
    args = parser.parse_args()
    feature_flags = feature_flags_from_args(args)

    digits = load_digits()
    x = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
    x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    y = torch.tensor(digits.target, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    train_loader = DataLoader(DigitsProjectiveDataset(x_train, y_train, augment=True), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(DigitsProjectiveDataset(x_test, y_test, augment=False), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMRNPrototype(ProjectiveOCRConfig(image_size=16, num_classes=10)).to(device)
    criterion = ProjectiveOCRLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x_deg, x_clean, ipa, yb in train_loader:
            x_deg, x_clean, ipa, yb = x_deg.to(device), x_clean.to(device), ipa.to(device), yb.to(device)
            out = model(x_deg, ipa)
            target_c = contour_target(x_clean)
            ldict = criterion(out, yb, target_c)
            opt.zero_grad()
            ldict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        row = {"epoch": epoch, "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
        history.append(row)
        print({**row, "feature_flags": feature_flags.to_dict()})

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.out_prefix}.json").write_text(json.dumps({"epochs": args.epochs, "feature_flags": feature_flags.to_dict(), "history": history}, indent=2))


if __name__ == "__main__":
    main()
