from __future__ import annotations

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

import argparse
import contextlib
import io
import json
from pathlib import Path

from noetic_pawp.feature_flags import add_feature_flag_arguments, feature_flags_from_args


def _import_or_raise() -> tuple:
    """Importa dependências pesadas com mensagem explícita de ambiente.

    Mantém o script importável mesmo sem torch/pyfolds/sklearn e falha apenas no runtime.
    """

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, TensorDataset
        from pyfolds.core import MPJRDConfig
        from pyfolds.network import NetworkBuilder
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Dependências ausentes. Instale: torch torchvision scikit-learn pyfolds"
        ) from exc

    return torch, nn, F, load_digits, train_test_split, DataLoader, TensorDataset, MPJRDConfig, NetworkBuilder


def _silent_forward(net, x, reward: float, mode: str = "online"):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return net(x, reward=reward, mode=mode)["output"].float()


def _prepare_pyfolds_input(images, F):
    # Mapeia imagem 8x8 para 28x28 e reestrutura em 49 patches 4x4.
    # Isso preserva localidade espacial para a camada de entrada do PyFolds.
    x = F.interpolate(images.unsqueeze(1), size=(28, 28), mode="bilinear", align_corners=False)
    x = x.clamp(0.0, 1.0)
    return x.view(x.size(0), 49, 4, 4)


def build_digits_dataloaders(load_digits, train_test_split, DataLoader, TensorDataset, batch_size: int = 64):
    digits = load_digits()
    x = digits.images
    y = digits.target

    import torch

    x = torch.tensor(x, dtype=torch.float32) / 16.0
    y = torch.tensor(y, dtype=torch.long)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        len(train_ds),
        len(test_ds),
    )


def build_pyfolds_network(MPJRDConfig, NetworkBuilder):
    cfg = MPJRDConfig(
        n_dendrites=4,
        n_synapses_per_dendrite=4,
        random_seed=42,
        plastic=True,
        defer_updates=True,
    )
    builder = NetworkBuilder("pyfolds_digits")
    builder.add_layer("input", n_neurons=49, cfg=cfg, connect_from_previous=False)
    builder.add_layer("output", n_neurons=10, cfg=cfg, connect_from_previous=True)
    return builder.build()


def evaluate_pyfolds(net, head, loader, device, F):
    head.eval()
    total = 0
    correct = 0
    import torch

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            inp = _prepare_pyfolds_input(x, F)
            feats = _silent_forward(net, inp, reward=0.0, mode="online")
            logits = head(feats)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def train_pyfolds_head(train_loader, test_loader, device, nn, F, MPJRDConfig, NetworkBuilder, epochs: int = 20):
    import torch

    net = build_pyfolds_network(MPJRDConfig, NetworkBuilder).to(device)
    head = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 10)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            inp = _prepare_pyfolds_input(x, F)

            feats = _silent_forward(net, inp, reward=0.0, mode="online")
            logits = head(feats)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(dim=1)
            batch_acc = (pred == y).float().mean().item()
            # Reward bipolar em [-1,1], alinhado com precisão instantânea.
            reward = (2.0 * batch_acc) - 1.0
            _silent_forward(net, inp, reward=reward, mode="online")

            running_loss += loss.detach().item() * y.size(0)
            correct += (pred == y).sum().item()
            total += y.size(0)

        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, total), 4),
            "train_acc": round(correct / max(1, total), 4),
            "test_acc": round(evaluate_pyfolds(net, head, test_loader, device, F), 4),
        }
        history.append(row)
        print({"pyfolds": row})
    return history


def evaluate_baseline(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    import torch

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x.view(x.size(0), -1))
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def train_baseline(train_loader, test_loader, device, nn, epochs: int = 20):
    import torch

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x.view(x.size(0), -1))
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.detach().item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, total), 4),
            "train_acc": round(correct / max(1, total), 4),
            "test_acc": round(evaluate_baseline(model, test_loader, device), 4),
        }
        history.append(row)
        print({"baseline": row})
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino PyFolds no dataset Digits.")
    add_feature_flag_arguments(parser)
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas (mínimo recomendado: 20).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-prefix", type=str, default="pyfolds_digits_training")
    args = parser.parse_args()
    feature_flags = feature_flags_from_args(args)

    if args.epochs < 1:
        raise ValueError("--epochs deve ser >= 1")

    (
        torch,
        nn,
        F,
        load_digits,
        train_test_split,
        DataLoader,
        TensorDataset,
        MPJRDConfig,
        NetworkBuilder,
    ) = _import_or_raise()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, n_train, n_test = build_digits_dataloaders(
        load_digits, train_test_split, DataLoader, TensorDataset, batch_size=args.batch_size
    )

    pyfolds_history = train_pyfolds_head(
        train_loader, test_loader, device, nn, F, MPJRDConfig, NetworkBuilder, epochs=args.epochs
    )
    baseline_history = train_baseline(train_loader, test_loader, device, nn, epochs=args.epochs)

    out_dir = Path("docs")
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / f"{args.out_prefix}.json"
    md_path = out_dir / f"{args.out_prefix}.md"

    payload = {
        "dataset": "sklearn_digits",
        "train_size": n_train,
        "test_size": n_test,
        "device": str(device),
        "epochs": args.epochs,
        "feature_flags": feature_flags.to_dict(),
        "pyfolds": pyfolds_history,
        "baseline": baseline_history,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Resultado de treinamento real (PyFolds + Digits)",
        "",
        f"- Dataset: sklearn digits (real), train={n_train}, test={n_test}",
        f"- Épocas: {args.epochs}",
        f"- Device: {device}",
        "",
        "## PyFolds + Head linear",
        "",
        "| epoch | train_loss | train_acc | test_acc |",
        "|---:|---:|---:|---:|",
    ]
    for r in pyfolds_history:
        lines.append(f"| {r['epoch']} | {r['train_loss']} | {r['train_acc']} | {r['test_acc']} |")

    lines += [
        "",
        "## Baseline MLP (sem PyFolds)",
        "",
        "| epoch | train_loss | train_acc | test_acc |",
        "|---:|---:|---:|---:|",
    ]
    for r in baseline_history:
        lines.append(f"| {r['epoch']} | {r['train_loss']} | {r['train_acc']} | {r['test_acc']} |")

    lines += [
        "",
        f"PyFolds teste final: **{pyfolds_history[-1]['test_acc']:.4f}**",
        f"Baseline teste final: **{baseline_history[-1]['test_acc']:.4f}**",
        "",
        "Conclusão inicial: baseline MLP ainda pode superar esta configuração de PyFolds;"
        " recomenda-se tuning de input encoding, reward schedule e hiperparâmetros MPJRD.",
    ]
    md_path.write_text("\n".join(lines))
    print({"json": str(json_path), "markdown": str(md_path)})


if __name__ == "__main__":
    main()
