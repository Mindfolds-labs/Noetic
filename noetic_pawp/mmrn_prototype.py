from __future__ import annotations

"""MMRN prototype with Projective OCR and trainable geometric outputs.

Implementação sem placeholders do fluxo:
Imagem -> retificação projetiva (aprox. afim) -> contorno -> Bézier -> embedding geométrico
-> fusão geo-linguística -> espaço PRS.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class ProjectiveOCRConfig:
    image_size: int = 16
    num_classes: int = 10
    num_bezier_curves: int = 2
    geom_dim: int = 64


class PPNProjectivePreNormalizer(nn.Module):
    """Prediz transformação afim 2x3 e aplica retificação com STN.

    Observação matemática: uma homografia completa teria 8 DoF; aqui usamos 6 DoF afins
    por estabilidade numérica e gradientes mais estáveis no treino inicial.
    """

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.localizer = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.theta_head = nn.Linear(32, 6)
        with torch.no_grad():
            self.theta_head.weight.zero_()
            self.theta_head.bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        feat = self.localizer(x).flatten(1)
        theta = self.theta_head(feat).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        rectified = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
        return {"rectified": rectified, "theta": theta}


class CEBContourExtractionBlock(nn.Module):
    """Extrai contornos com Sobel fixo + refinamento aprendível."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = sobel_x.t()
        kernel = torch.stack([sobel_x, sobel_y]).unsqueeze(1)
        self.register_buffer("sobel", kernel)
        self.refiner = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        grad = F.conv2d(x, self.sobel, padding=1)
        return self.refiner(grad)


class BFFBezierFormFitter(nn.Module):
    """Ajusta curvas Bézier cúbicas (4 pontos de controle por curva)."""

    def __init__(self, num_curves: int) -> None:
        super().__init__()
        self.num_curves = num_curves
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.ctrl_head = nn.Linear(32, num_curves * 8)  # (x,y)*4

    def forward(self, contour: Tensor) -> Tensor:
        h = self.backbone(contour).flatten(1)
        ctrl = torch.tanh(self.ctrl_head(h))
        return ctrl.view(-1, self.num_curves, 4, 2)


class GTEGlyphTokenEncoder(nn.Module):
    """Codifica assinatura geométrica g a partir de contorno e Bézier."""

    def __init__(self, num_curves: int, geom_dim: int) -> None:
        super().__init__()
        self.contour_proj = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.bezier_proj = nn.Linear(num_curves * 8, 32)
        self.out = nn.Sequential(
            nn.Linear(16 + 32, geom_dim),
            nn.LayerNorm(geom_dim),
            nn.GELU(),
        )

    def forward(self, contour: Tensor, bezier: Tensor) -> Tensor:
        c = self.contour_proj(contour).flatten(1)
        b = self.bezier_proj(bezier.flatten(1))
        return self.out(torch.cat([c, b], dim=-1))


class SDHSymbolDecodingHead(nn.Module):
    def __init__(self, geom_dim: int, num_classes: int) -> None:
        super().__init__()
        self.cls = nn.Linear(geom_dim, num_classes)
        self.conf = nn.Sequential(nn.Linear(geom_dim, 1), nn.Sigmoid())

    def forward(self, g: Tensor) -> Dict[str, Tensor]:
        return {"logits": self.cls(g), "confidence": self.conf(g)}


class ProjectiveOCR(nn.Module):
    def __init__(self, cfg: ProjectiveOCRConfig) -> None:
        super().__init__()
        self.ppn = PPNProjectivePreNormalizer(cfg.image_size)
        self.ceb = CEBContourExtractionBlock()
        self.bff = BFFBezierFormFitter(cfg.num_bezier_curves)
        self.gte = GTEGlyphTokenEncoder(cfg.num_bezier_curves, cfg.geom_dim)
        self.sdh = SDHSymbolDecodingHead(cfg.geom_dim, cfg.num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pre = self.ppn(x)
        contour = self.ceb(pre["rectified"])
        bezier = self.bff(contour)
        g = self.gte(contour, bezier)
        head = self.sdh(g)
        return {
            "y_hat": head["logits"],
            "C_hat": contour,
            "B_hat": bezier,
            "g": g,
            "q": head["confidence"],
            "rectified": pre["rectified"],
            "theta": pre["theta"],
        }


class GeoLinguisticEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, emb_dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU())

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.proj(self.emb(token_ids).mean(dim=1))


class NoeticFusionEngine(nn.Module):
    def __init__(self, geom_dim: int = 64, lang_dim: int = 64, prs_dim: int = 128) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(geom_dim + lang_dim, prs_dim),
            nn.GELU(),
            nn.LayerNorm(prs_dim),
            nn.Linear(prs_dim, prs_dim),
        )

    def forward(self, geom: Tensor, lang: Tensor) -> Tensor:
        return self.fuse(torch.cat([geom, lang], dim=-1))


class PRSDecoder(nn.Module):
    def __init__(self, prs_dim: int, out_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(prs_dim, prs_dim), nn.GELU(), nn.Linear(prs_dim, out_dim))

    def forward(self, prs: Tensor) -> Tensor:
        return self.net(prs)


class MMRNPrototype(nn.Module):
    def __init__(self, ocr_cfg: ProjectiveOCRConfig | None = None, prs_dim: int = 128) -> None:
        super().__init__()
        cfg = ocr_cfg or ProjectiveOCRConfig()
        self.ocr = ProjectiveOCR(cfg)
        self.geo_ling = GeoLinguisticEncoder(emb_dim=cfg.geom_dim)
        self.fusion = NoeticFusionEngine(geom_dim=cfg.geom_dim, lang_dim=cfg.geom_dim, prs_dim=prs_dim)
        self.prs_decoder = PRSDecoder(prs_dim=prs_dim)

    def forward(self, image: Tensor, ipa_tokens: Tensor) -> Dict[str, Tensor]:
        ocr_out = self.ocr(image)
        ling = self.geo_ling(ipa_tokens)
        prs = self.fusion(ocr_out["g"], ling)
        decoded = self.prs_decoder(prs)
        return {**ocr_out, "ling": ling, "prs": prs, "decoded": decoded}


class ProjectiveOCRLoss(nn.Module):
    """Loss híbrida: cls + proj + contour + bezier(reg) + topologia."""

    def __init__(self, lambda_cls: float = 1.0, lambda_proj: float = 0.2, lambda_contour: float = 0.5, lambda_bezier: float = 0.1, lambda_topo: float = 0.1) -> None:
        super().__init__()
        self.w = (lambda_cls, lambda_proj, lambda_contour, lambda_bezier, lambda_topo)

    def forward(self, out: Dict[str, Tensor], y: Tensor, contour_target: Tensor) -> Dict[str, Tensor]:
        l_cls = F.cross_entropy(out["y_hat"], y)

        eye = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=out["theta"].device).view(1, 2, 3)
        l_proj = F.mse_loss(out["theta"], eye.expand_as(out["theta"]))

        l_contour = F.l1_loss(out["C_hat"], contour_target)

        # Regularização de suavidade nos pontos de controle (evita curvas degeneradas).
        second_diff = out["B_hat"][:, :, 2:, :] - 2 * out["B_hat"][:, :, 1:-1, :] + out["B_hat"][:, :, :-2, :]
        l_bezier = second_diff.square().mean()

        # Topologia aproximada: preservar massa de contorno para evitar quebra de conexões.
        l_topo = (out["C_hat"].sum(dim=(1, 2, 3)) - contour_target.sum(dim=(1, 2, 3))).abs().mean() / contour_target[0].numel()

        wt = self.w
        total = wt[0] * l_cls + wt[1] * l_proj + wt[2] * l_contour + wt[3] * l_bezier + wt[4] * l_topo
        return {
            "total": total,
            "L_cls": l_cls,
            "L_proj": l_proj,
            "L_contour": l_contour,
            "L_bezier": l_bezier,
            "L_topo": l_topo,
        }
