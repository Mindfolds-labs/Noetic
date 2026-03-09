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


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} deve ser >= 0 para manter combinação convexa parcial de perdas.")


@dataclass
class ProjectiveOCRConfig:
    image_size: int = 16
    num_classes: int = 10
    num_bezier_curves: int = 2
    geom_dim: int = 64
    num_concepts: int = 128
    num_attributes: int = 64
    num_relations: int = 64
    num_contexts: int = 32


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


class SemanticNoeticHead(nn.Module):
    """Cabeça semântica para modelar significado (conceito/atributo/relação/contexto).

    Teoria -> implementação:
    - conceito e contexto: classificação categórica (cross-entropy)
    - atributos e relações: multi-rótulo (BCE com logits)
    """

    def __init__(self, prs_dim: int, num_concepts: int, num_attributes: int, num_relations: int, num_contexts: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(prs_dim, prs_dim),
            nn.GELU(),
            nn.LayerNorm(prs_dim),
        )
        self.concept_head = nn.Linear(prs_dim, num_concepts)
        self.attribute_head = nn.Linear(prs_dim, num_attributes)
        self.relation_head = nn.Linear(prs_dim, num_relations)
        self.context_head = nn.Linear(prs_dim, num_contexts)

    def forward(self, prs: Tensor) -> Dict[str, Tensor]:
        h = self.trunk(prs)
        return {
            "concept_logits": self.concept_head(h),
            "attribute_logits": self.attribute_head(h),
            "relation_logits": self.relation_head(h),
            "context_logits": self.context_head(h),
        }


class MMRNPrototype(nn.Module):
    def __init__(self, ocr_cfg: ProjectiveOCRConfig | None = None, prs_dim: int = 128) -> None:
        super().__init__()
        cfg = ocr_cfg or ProjectiveOCRConfig()
        self.ocr = ProjectiveOCR(cfg)
        self.geo_ling = GeoLinguisticEncoder(emb_dim=cfg.geom_dim)
        self.fusion = NoeticFusionEngine(geom_dim=cfg.geom_dim, lang_dim=cfg.geom_dim, prs_dim=prs_dim)
        self.prs_decoder = PRSDecoder(prs_dim=prs_dim)
        self.semantic_head = SemanticNoeticHead(
            prs_dim=prs_dim,
            num_concepts=cfg.num_concepts,
            num_attributes=cfg.num_attributes,
            num_relations=cfg.num_relations,
            num_contexts=cfg.num_contexts,
        )

    def forward(self, image: Tensor, ipa_tokens: Tensor) -> Dict[str, Tensor]:
        ocr_out = self.ocr(image)
        ling = self.geo_ling(ipa_tokens)
        prs = self.fusion(ocr_out["g"], ling)
        decoded = self.prs_decoder(prs)
        semantic = self.semantic_head(prs)
        return {**ocr_out, "ling": ling, "prs": prs, "decoded": decoded, **semantic}


class ProjectiveOCRLoss(nn.Module):
    """Loss híbrida: cls + proj + contour + bezier(reg) + topologia."""

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_proj: float = 0.2,
        lambda_contour: float = 0.5,
        lambda_bezier: float = 0.1,
        lambda_topo: float = 0.1,
        lambda_concept: float = 0.5,
        lambda_attr: float = 0.3,
        lambda_rel: float = 0.3,
        lambda_ctx: float = 0.2,
    ) -> None:
        super().__init__()
        self.w = (lambda_cls, lambda_proj, lambda_contour, lambda_bezier, lambda_topo, lambda_concept, lambda_attr, lambda_rel, lambda_ctx)
        for name, value in zip(("lambda_cls", "lambda_proj", "lambda_contour", "lambda_bezier", "lambda_topo", "lambda_concept", "lambda_attr", "lambda_rel", "lambda_ctx"), self.w):
            _validate_non_negative(name, value)

    def _validate_semantic_targets(self, out: Dict[str, Tensor], semantic_targets: Dict[str, Tensor]) -> None:
        batch = out["y_hat"].shape[0]
        if "concept_id" in semantic_targets and semantic_targets["concept_id"].shape != (batch,):
            raise ValueError("concept_id deve ter shape [B].")
        if "context_id" in semantic_targets and semantic_targets["context_id"].shape != (batch,):
            raise ValueError("context_id deve ter shape [B].")
        if "attributes" in semantic_targets and semantic_targets["attributes"].shape != out["attribute_logits"].shape:
            raise ValueError("attributes deve ter o mesmo shape de attribute_logits [B, A].")
        if "relations" in semantic_targets and semantic_targets["relations"].shape != out["relation_logits"].shape:
            raise ValueError("relations deve ter o mesmo shape de relation_logits [B, R].")

    def forward(self, out: Dict[str, Tensor], y: Tensor, contour_target: Tensor, semantic_targets: Dict[str, Tensor] | None = None) -> Dict[str, Tensor]:
        l_cls = F.cross_entropy(out["y_hat"], y)

        eye = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=out["theta"].device).view(1, 2, 3)
        l_proj = F.mse_loss(out["theta"], eye.expand_as(out["theta"]))

        l_contour = F.l1_loss(out["C_hat"], contour_target)

        # Regularização de suavidade nos pontos de controle (evita curvas degeneradas).
        second_diff = out["B_hat"][:, :, 2:, :] - 2 * out["B_hat"][:, :, 1:-1, :] + out["B_hat"][:, :, :-2, :]
        l_bezier = second_diff.square().mean()

        # Topologia aproximada: preservar massa de contorno para evitar quebra de conexões.
        norm = float(contour_target.shape[1] * contour_target.shape[2] * contour_target.shape[3])
        l_topo = (out["C_hat"].sum(dim=(1, 2, 3)) - contour_target.sum(dim=(1, 2, 3))).abs().mean() / max(norm, 1.0)

        losses = {
            "L_cls": l_cls,
            "L_proj": l_proj,
            "L_contour": l_contour,
            "L_bezier": l_bezier,
            "L_topo": l_topo,
        }

        l_concept = torch.zeros_like(l_cls)
        l_attr = torch.zeros_like(l_cls)
        l_rel = torch.zeros_like(l_cls)
        l_ctx = torch.zeros_like(l_cls)
        if semantic_targets is not None:
            self._validate_semantic_targets(out, semantic_targets)
            # Conceito/contexto são tarefas categóricas: ID único por amostra.
            if "concept_id" in semantic_targets:
                l_concept = F.cross_entropy(out["concept_logits"], semantic_targets["concept_id"])
            if "context_id" in semantic_targets:
                l_ctx = F.cross_entropy(out["context_logits"], semantic_targets["context_id"])

            # Atributos/relações são multi-rótulo: vetor multi-hot por amostra.
            if "attributes" in semantic_targets:
                l_attr = F.binary_cross_entropy_with_logits(out["attribute_logits"], semantic_targets["attributes"].float())
            if "relations" in semantic_targets:
                l_rel = F.binary_cross_entropy_with_logits(out["relation_logits"], semantic_targets["relations"].float())

        wt = self.w
        total = (
            wt[0] * l_cls
            + wt[1] * l_proj
            + wt[2] * l_contour
            + wt[3] * l_bezier
            + wt[4] * l_topo
            + wt[5] * l_concept
            + wt[6] * l_attr
            + wt[7] * l_rel
            + wt[8] * l_ctx
        )
        losses["total"] = total
        losses["L_concept"] = l_concept
        losses["L_attr"] = l_attr
        losses["L_rel"] = l_rel
        losses["L_ctx"] = l_ctx
        return losses
