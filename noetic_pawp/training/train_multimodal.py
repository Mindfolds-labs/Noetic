from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge


class MultimodalDataset(Dataset):
    def __init__(self, data_path: str, max_samples: Optional[int] = None) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if max_samples is not None:
            items = items[:max_samples]
        self.data = [
            {"text": item["text"], "image_path": item.get("image_path", ""), "concept": item["concept"]}
            for item in items
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        out: Dict[str, Any] = {"text": item["text"], "concept": item["concept"]}
        img_path = item.get("image_path")
        if img_path and Path(img_path).exists():
            try:
                from PIL import Image
                from torchvision import transforms

                img = Image.open(img_path).convert("RGB")
                out["image"] = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])(img)
            except Exception:
                pass
        return out


class MultimodalTrainer:
    def __init__(
        self,
        bridge: NoeticLeibregBridge,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.bridge = bridge.to(device)
        self.optimizer = torch.optim.AdamW(
            self.bridge.parameters(), lr=learning_rate, weight_decay=self.bridge.config.weight_decay
        )
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
            if val_dataset
            else None
        )
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"texts": [x["text"] for x in batch], "concepts": [x["concept"] for x in batch]}
        imgs = [x.get("image") for x in batch if x.get("image") is not None]
        if imgs:
            out["images"] = torch.stack(imgs)
        return out

    def train_epoch(self) -> float:
        self.bridge.train()
        total = 0.0
        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss_acc = torch.tensor(0.0, device=self.device)
            count = 0
            for i, text in enumerate(batch["texts"]):
                kwargs: Dict[str, Any] = {"text": text, "memory_keys": batch["concepts"][i]}
                if "images" in batch and i < len(batch["images"]):
                    kwargs["image"] = batch["images"][i].to(self.device)
                out = self.bridge(**kwargs)
                sample_loss = 0.1 * F.mse_loss(out["reg_output"], out["imagination"])
                if out["q_text_4d"] is not None and out["q_memory_4d"] is not None:
                    sample_loss = sample_loss + F.mse_loss(out["q_text_4d"], out["q_memory_4d"])
                if out["q_text_4d"] is not None and out["q_image_4d"] is not None:
                    sample_loss = sample_loss + F.mse_loss(out["q_text_4d"], out["q_image_4d"])
                loss_acc = loss_acc + sample_loss
                count += 1
            if count == 0:
                continue
            batch_loss = loss_acc / count
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bridge.parameters(), 1.0)
            self.optimizer.step()
            total += float(batch_loss.item())
        return total / max(len(self.train_loader), 1)

    def validate(self) -> float:
        if self.val_loader is None:
            return 0.0
        self.bridge.eval()
        total = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                cur = 0.0
                n = 0
                for i, text in enumerate(batch["texts"]):
                    out = self.bridge(text=text, memory_keys=batch["concepts"][i])
                    if out["q_text_4d"] is not None and out["q_memory_4d"] is not None:
                        cur += float(F.mse_loss(out["q_text_4d"], out["q_memory_4d"]).item())
                        n += 1
                total += cur / max(n, 1)
        return total / max(len(self.val_loader), 1)

    def train(self, epochs: int = 10, save_path: Optional[str] = None) -> Dict[str, List[float]]:
        for epoch in range(epochs):
            _ = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epochs"].append(epoch + 1)
            if save_path and (epoch + 1) % 5 == 0:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"epoch": epoch + 1, "history": self.history}, f"{save_path}_epoch{epoch+1}.pt")
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"history": self.history, "model": self.bridge.state_dict()}, f"{save_path}_final.pt")
        return self.history
