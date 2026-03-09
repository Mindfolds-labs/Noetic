from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse

from noetic_pawp.feature_flags import add_feature_flag_arguments, feature_flags_from_args

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
except ImportError as exc:  # pragma: no cover
    raise SystemExit("train_mnist.py requires torch + torchvision") from exc


class SmallCNN(nn.Module):
    def __init__(self, out_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_dataset():
    tfm = transforms.ToTensor()
    try:
        return datasets.MNIST(root="data", train=True, download=True, transform=tfm), "mnist"
    except Exception as exc:
        print({"warning": "mnist_download_failed", "reason": str(exc)[:180]})
        fake = datasets.FakeData(size=10_000, image_size=(1, 28, 28), num_classes=10, transform=tfm)
        return fake, "fake_mnist"


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino CNN baseline para MNIST.")
    add_feature_flag_arguments(parser)
    args = parser.parse_args()
    feature_flags = feature_flags_from_args(args)

    ds, source = build_dataset()
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    model = SmallCNN(out_dim=10)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print({"dataset": source, "size": len(ds), "feature_flags": feature_flags.to_dict()})
    model.train()
    for step, (x, y) in enumerate(dl):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print({"step": step, "loss": loss.detach().item()})
        if step == 300:
            break


if __name__ == "__main__":
    main()
