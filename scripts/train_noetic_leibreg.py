#!/usr/bin/env python
from __future__ import annotations

import argparse

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch não instalado; execute `pip install torch` para rodar treino.") from exc

from noetic_pawp.leibreg_bridge import NoeticLeibregBridge, NoeticLeibregConfig
from noetic_pawp.training.train_multimodal import MultimodalDataset, MultimodalTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save", type=str, default="./checkpoints/noetic_leibreg")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    bridge = NoeticLeibregBridge(config=NoeticLeibregConfig(learning_rate=args.lr, fusion_mode="mean"))
    dataset = MultimodalDataset(args.data)
    train_size = int(0.8 * len(dataset))
    val_size = max(0, len(dataset) - train_size)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    trainer = MultimodalTrainer(
        bridge=bridge,
        train_dataset=train_ds,
        val_dataset=val_ds,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )
    history = trainer.train(epochs=args.epochs, save_path=args.save)
    print(f"Treinamento concluído. Loss final: {history['train_loss'][-1]:.6f}")


if __name__ == "__main__":
    main()
