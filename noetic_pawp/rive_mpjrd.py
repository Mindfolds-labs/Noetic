from __future__ import annotations

"""RIVE + MPJRD depth pipeline with mathematically bounded operations.

Design choices focus on numerical stability and consistency with depth-estimation
literature (Eigen et al. metrics, BerHu, gradient regularization).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RIVEEncoder(nn.Module):
    """Geometry-only encoder: 72 (Legendre) + 8 (APM) + 4 (PDS) = 84 dims."""

    def __init__(self, p: int = 8, max_crops: int = 60) -> None:
        super().__init__()
        self.p = p
        self.max_crops = max_crops
        self.n_channels = 8
        self.out_dim = (p + 1) * self.n_channels + 8 + 4

    @staticmethod
    def _extract_crops(gray: Tensor, max_crops: int) -> Tuple[list, int]:
        h, w = gray.shape
        n = min(h, w)
        rows = []
        for k in range(min(n // 2, max_crops)):
            s = n - 2 * k
            if s < 4:
                break
            crop = gray[k : h - k, k : w - k]
            a = float(s * s)
            z = 1.0 - (s / max(float(n), 1.0))
            rows.append((crop, a, z, s, k))
        return rows, n

    @staticmethod
    def _phi(crop: Tensor, area: float, z: float, s: int, w: int) -> list[float]:
        mu = float(crop.mean().item())
        sigma = float(crop.std().item())
        theta = float(np.arctan2(s, max(w - s, 1)))
        jac = area / max((1.0 - z) ** 2, 1e-4)
        if crop.shape[0] > 2 and crop.shape[1] > 2:
            flux = abs(float(crop[0, :].mean() - crop[-1, :].mean())) + abs(
                float(crop[:, 0].mean() - crop[:, -1].mean())
            )
        else:
            flux = 0.0
        z_cauchy = flux / (area * max(1.0 / max(s, 1), 1e-4) + 1e-6)
        return [mu, sigma, area, z, theta, flux, jac, z_cauchy]

    def _legendre_project(self, phi: np.ndarray) -> np.ndarray:
        m = phi.shape[0]
        t = np.linspace(0.0, 1.0, m)
        l = np.zeros((m, self.p + 1), dtype=np.float64)
        for j in range(self.p + 1):
            coeff = np.zeros(j + 1, dtype=np.float64)
            coeff[-1] = 1.0
            l[:, j] = np.polynomial.legendre.legval(2.0 * t - 1.0, coeff)

        out = np.zeros((self.n_channels, self.p + 1), dtype=np.float64)
        for ch in range(self.n_channels):
            c, _, _, _ = np.linalg.lstsq(l, phi[:, ch], rcond=None)
            out[ch] = c
        return out.astype(np.float32)

    @staticmethod
    def _apm(gray: Tensor, d: int = 32) -> np.ndarray:
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        img = gray.detach().cpu().numpy()
        dirs = {"NE": (-1, 1), "NW": (1, 1), "SW": (1, -1), "SE": (-1, -1)}
        vecs: Dict[str, np.ndarray] = {}
        for name, (sx, sy) in dirs.items():
            vals = []
            for t in range(d):
                f = t / max(d - 1, 1)
                x = int(np.clip(cx + sx * f * cx, 0, w - 1))
                y = int(np.clip(cy + sy * f * cy, 0, h - 1))
                vals.append(float(img[y, x]))
            vecs[name] = np.asarray(vals, dtype=np.float32)

        asym1 = np.abs(vecs["NE"] - vecs["SW"])
        asym2 = np.abs(vecs["NW"] - vecs["SE"])

        def slope(v: np.ndarray) -> float:
            if float(v.std()) < 1e-6:
                return 0.0
            x = np.arange(len(v), dtype=np.float32)
            return float(np.polyfit(x, v, 1)[0])

        return np.array(
            [
                asym1.mean(),
                asym1.std(),
                asym2.mean(),
                asym2.std(),
                slope(vecs["NE"]),
                slope(vecs["NW"]),
                slope(vecs["SW"]),
                slope(vecs["SE"]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _estimate_neff(areas: np.ndarray) -> float:
        s = np.sqrt(np.maximum(areas, 1e-9))
        valid = s > 2.0
        if int(valid.sum()) < 3:
            return 2.0
        log_s = np.log(s[valid])
        log_a = np.log(np.maximum(areas[valid], 1e-9))
        slope = float(np.polyfit(log_s, log_a, 1)[0])
        return float(np.clip(slope, 0.5, 5.0))

    def forward(self, img: Tensor) -> Tensor:
        feats = []
        b = img.shape[0]
        for i in range(b):
            ib = img[i]
            gray = 0.299 * ib[0] + 0.587 * ib[1] + 0.114 * ib[2] if ib.shape[0] == 3 else ib[0]
            h, w = gray.shape
            rows, _ = self._extract_crops(gray, self.max_crops)
            m = len(rows)
            if m < self.p + 2:
                enc72 = np.zeros(((self.p + 1) * self.n_channels,), dtype=np.float32)
                areas = np.array([r[1] for r in rows], dtype=np.float32) if rows else np.array([1.0], dtype=np.float32)
            else:
                phi = np.zeros((m, self.n_channels), dtype=np.float32)
                areas = np.zeros((m,), dtype=np.float32)
                for j, (crop, a, z, s, _) in enumerate(rows):
                    phi[j] = self._phi(crop, a, z, s, w)
                    areas[j] = a
                enc72 = self._legendre_project(phi).reshape(-1)

            apm = self._apm(gray)
            global_neff = self._estimate_neff(areas)
            qvals = []
            for qy, qx in ((0, 0), (0, 1), (1, 0), (1, 1)):
                qh, qw = h // 2, w // 2
                q = gray[qy * qh : (qy + 1) * qh, qx * qw : (qx + 1) * qw]
                qr, _ = self._extract_crops(q, max_crops=30)
                if len(qr) > 3:
                    qa = np.array([x[1] for x in qr], dtype=np.float32)
                    qvals.append(self._estimate_neff(qa))
                else:
                    qvals.append(global_neff)
            feats.append(np.concatenate([enc72, apm, np.asarray(qvals, dtype=np.float32)], axis=0))
        return torch.tensor(np.asarray(feats), dtype=torch.float32, device=img.device)


class MPJRDNeuron(nn.Module):
    def __init__(self, input_dim: int, n_dendrites: int = 4, soma_nonlin: str = "gelu") -> None:
        super().__init__()
        if input_dim < n_dendrites:
            raise ValueError("input_dim deve ser >= n_dendrites")
        self.n_dendrites = n_dendrites
        dsize = input_dim // n_dendrites
        rem = input_dim - dsize * n_dendrites
        self.w_dend = nn.ModuleList([nn.Linear(dsize + (1 if i < rem else 0), 1) for i in range(n_dendrites)])
        self.g = nn.Parameter(torch.ones(n_dendrites))
        self.w_soma = nn.Linear(n_dendrites, 1)
        self.nonlin = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": F.relu, "gelu": F.gelu}[soma_nonlin]

    def forward(self, x: Tensor) -> Tensor:
        idx = 0
        outs = []
        for d in range(self.n_dendrites):
            din = self.w_dend[d].in_features
            xd = x[:, idx : idx + din]
            ud = torch.abs(self.g[d]) * torch.tanh(self.w_dend[d](xd))
            outs.append(ud)
            idx += din
        stack = torch.cat(outs, dim=1)
        return self.nonlin(self.w_soma(stack))


class MPJRDLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_dendrites: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.neurons = nn.ModuleList([MPJRDNeuron(input_dim, n_dendrites) for _ in range(output_dim)])
        self.bn = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.cat([n(x) for n in self.neurons], dim=1)
        return self.dropout(self.bn(y))


class RIVEDepthNet(nn.Module):
    def __init__(self, output_h: int, output_w: int, hidden_dims: Sequence[int] = (128, 256, 128), n_dendrites: int = 4, max_depth: float = 10.0) -> None:
        super().__init__()
        self.output_h, self.output_w, self.max_depth = output_h, output_w, max_depth
        self.encoder = RIVEEncoder(p=8)
        dims = [84, *hidden_dims]
        self.mpjrd_layers = nn.ModuleList([MPJRDLayer(dims[i], dims[i + 1], n_dendrites=n_dendrites) for i in range(len(dims) - 1)])
        self.head = MPJRDLayer(dims[-1], output_h * output_w, n_dendrites=min(n_dendrites, 4), dropout=0.0)

    def forward(self, img: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            rive = self.encoder(img)
        x = rive
        for layer in self.mpjrd_layers:
            x = layer(x)
        depth = self.head(x).view(img.size(0), 1, self.output_h, self.output_w)
        depth = torch.sigmoid(depth) * self.max_depth
        return depth, rive


@dataclass
class RIVEDepthLossConfig:
    lambda_grad: float = 0.5
    lambda_pds: float = 0.1


class RIVEDepthLoss(nn.Module):
    def __init__(self, cfg: Optional[RIVEDepthLossConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or RIVEDepthLossConfig()

    @staticmethod
    def berhu(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
        diff = torch.abs(pred[mask] - gt[mask])
        if diff.numel() == 0:
            return pred.sum() * 0.0
        c = torch.clamp(0.2 * diff.max().detach(), min=1e-6)
        small = diff[diff <= c]
        large = diff[diff > c]
        if large.numel() > 0:
            large = (large.square() + c.square()) / (2.0 * c)
            allv = torch.cat([small, large], dim=0)
        else:
            allv = small
        return allv.mean()

    @staticmethod
    def grad_loss(pred: Tensor, gt: Tensor) -> Tensor:
        pdy, pdx = pred[:, :, 1:, :] - pred[:, :, :-1, :], pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gdy, gdx = gt[:, :, 1:, :] - gt[:, :, :-1, :], gt[:, :, :, 1:] - gt[:, :, :, :-1]
        return 0.5 * (F.l1_loss(pdy, gdy) + F.l1_loss(pdx, gdx))

    @staticmethod
    def pds_loss(pred: Tensor, rive: Tensor) -> Tensor:
        target = rive[:, -4:].mean(dim=1)  # (B,)
        _, _, h, w = pred.shape
        ys = torch.arange(h, device=pred.device).float() - (h // 2)
        xs = torch.arange(w, device=pred.device).float() - (w // 2)
        r = torch.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
        r = r / (r.max() + 1e-6)
        radial = (pred[:, 0] * r).mean(dim=(1, 2)) / (pred[:, 0].mean(dim=(1, 2)) + 1e-6)
        est = 1.0 + radial.clamp(0, 3)
        return F.mse_loss(est, target)

    def forward(self, pred: Tensor, gt: Tensor, rive: Optional[Tensor]) -> Dict[str, Tensor]:
        mask = gt > 0.1
        if mask.sum() < 10:
            z = pred.sum() * 0.0
            return {"total": z, "berhu": z, "gradient": z, "pds_consistency": z}
        l_berhu = self.berhu(pred, gt, mask)
        l_grad = self.grad_loss(pred * mask, gt * mask)
        l_pds = self.pds_loss(pred, rive) if rive is not None else pred.sum() * 0.0
        total = l_berhu + self.cfg.lambda_grad * l_grad + self.cfg.lambda_pds * l_pds
        return {"total": total, "berhu": l_berhu, "gradient": l_grad, "pds_consistency": l_pds}


def compute_depth_metrics(pred: Tensor, gt: Tensor, max_depth: float) -> Dict[str, float]:
    mask = (gt > 0.1) & (gt < max_depth)
    if mask.sum() < 10:
        return {}
    d = pred[mask].clamp(1e-3, max_depth)
    ds = gt[mask].clamp(1e-3, max_depth)
    ratio = torch.max(d / ds, ds / d)
    return {
        "AbsRel": (torch.abs(d - ds) / ds).mean().item(),
        "SqRel": (((d - ds).square()) / ds).mean().item(),
        "RMSE": torch.sqrt((d - ds).square().mean()).item(),
        "RMSElog": torch.sqrt((torch.log(d) - torch.log(ds)).square().mean()).item(),
        "d1": (ratio < 1.25).float().mean().item(),
        "d2": (ratio < (1.25**2)).float().mean().item(),
        "d3": (ratio < (1.25**3)).float().mean().item(),
    }
