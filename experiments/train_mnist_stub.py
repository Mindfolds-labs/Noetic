"""Stub de treino PyTorch para integração futura PAWP + visão.

Este arquivo não roda treinamento completo por padrão; ele define a arquitetura-base
para conectar feature visual com embedding enriquecido do PAWP.
"""

from __future__ import annotations


def main() -> None:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print("PyTorch não instalado no ambiente. Instale torch para rodar este stub.")
        return

    class FusionModel(nn.Module):
        def __init__(self, token_dim: int = 128, visual_dim: int = 128, hidden: int = 256, n_classes: int = 10) -> None:
            super().__init__()
            self.visual = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.visual_proj = nn.Linear(32, visual_dim)
            self.head = nn.Sequential(
                nn.Linear(token_dim + visual_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_classes),
            )

        def forward(self, image: torch.Tensor, token_feat: torch.Tensor) -> torch.Tensor:
            v = self.visual(image).flatten(1)
            v = self.visual_proj(v)
            x = torch.cat([v, token_feat], dim=-1)
            return self.head(x)

    model = FusionModel()
    x_img = torch.randn(4, 1, 28, 28)
    x_tok = torch.randn(4, 128)
    y = model(x_img, x_tok)
    print("Forward OK:", y.shape)


if __name__ == "__main__":
    main()
