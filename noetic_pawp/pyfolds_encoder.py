from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import math
import random


class IntentionState(str, Enum):
    CURIOSITY = "CURIOSITY"
    FOCUS = "FOCUS"
    VIGILANCE = "VIGILANCE"
    CONSOLIDATION = "CONSOLIDATION"


@dataclass
class PyFoldsConfig:
    n_legendre: int = 72
    radial_steps: int = 32
    temporal_buffer: int = 8
    alpha_rhat: float = 0.95
    intention_smoothing: float = 0.9


def _shape3(x: List[List[List[float]]]) -> Tuple[int, int, int]:
    return (len(x), len(x[0]) if x else 0, len(x[0][0]) if x and x[0] else 0)


class RIVEEncoder:
    def __init__(self, n_coeffs: int = 72, p: int = 8) -> None:
        if n_coeffs % p != 0:
            raise ValueError("n_coeffs deve ser divisível por p")
        self.n_coeffs = n_coeffs
        self.p = p

    @staticmethod
    def _gray(image: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
        if not image or not image[0]:
            raise ValueError("imagem vazia")
        if isinstance(image[0][0], list):
            rgb = image  # type: ignore[assignment]
            return [[sum(px) / len(px) for px in row] for row in rgb]  # type: ignore[arg-type]
        return [list(map(float, row)) for row in image]  # type: ignore[arg-type]

    @staticmethod
    def _crop(img: List[List[float]], k: int) -> List[List[float]]:
        h, w = len(img), len(img[0])
        return [row[k : w - k] for row in img[k : h - k]]

    @staticmethod
    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        m = sum(vals) / max(len(vals), 1)
        v = sum((x - m) ** 2 for x in vals) / max(len(vals), 1)
        return m, math.sqrt(v)

    def _crop_features(self, img: List[List[float]], k: int, max_side: int) -> List[float]:
        crop = self._crop(img, k)
        if not crop or not crop[0]:
            return [0.0] * 8
        h, w = len(crop), len(crop[0])
        flat = [v for row in crop for v in row]
        mu, sigma = self._mean_std(flat)
        area = float(h * w)
        sk = max_side - 2 * k
        z = max(0.0, min(1.0, 1.0 - (sk / max_side)))

        top = crop[0]
        bottom = crop[-1]
        left = [row[0] for row in crop]
        right = [row[-1] for row in crop]
        perimeter = top + bottom + left + right
        flux = sum(perimeter) - 2 * (len(img) + len(img[0]) - 4 * k)

        gx = []
        gy = []
        for y in range(h):
            for x in range(w):
                x0, x1 = max(0, x - 1), min(w - 1, x + 1)
                y0, y1 = max(0, y - 1), min(h - 1, y + 1)
                gx.append(crop[y][x1] - crop[y][x0])
                gy.append(crop[y1][x] - crop[y0][x])
        gmx = sum(gx) / max(len(gx), 1)
        gmy = sum(gy) / max(len(gy), 1)
        theta = math.atan2(gmy, gmx)

        jac_area = area / max((1.0 - z) ** 2, 1e-6)
        grad_mag = [math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]) for i in range(len(gx))]
        kappa = sum(grad_mag) / max(len(grad_mag), 1)
        cauchy = sum(abs(v) for v in perimeter) / max(area * (kappa + 1e-6), 1e-6)
        return [mu, sigma, area, z, theta, flux, jac_area, cauchy]

    @staticmethod
    def _legendre(n: int, x: float) -> float:
        if n == 0:
            return 1.0
        if n == 1:
            return x
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        return p1

    def __call__(self, image: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]) -> List[float]:
        img = self._gray(image)
        max_side = min(len(img), len(img[0]))
        n_crops = self.n_coeffs // self.p
        max_k = max((max_side // 2) - 1, 1)
        ks = [round(i * max_k / max(n_crops - 1, 1)) for i in range(n_crops)]
        phi = [self._crop_features(img, k, max_side) for k in ks]

        coeffs = [0.0] * self.n_coeffs
        # Projeção simples em base de Legendre para manter estabilidade sem dependência externa.
        for n in range(self.n_coeffs):
            acc = 0.0
            for i, row in enumerate(phi):
                x = -1.0 + 2.0 * (i / max(len(phi) - 1, 1))
                acc += sum(row) * self._legendre(n, x)
            coeffs[n] = acc / max(len(phi) * len(phi[0]), 1)
        return coeffs


class RadialExtractor:
    def __init__(self, D: int = 32) -> None:
        self.D = D
        self.directions = [(0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, -1), (-1, -1), (1, 1)]

    def _gray(self, image: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]) -> List[List[float]]:
        return RIVEEncoder._gray(image)

    def _ray(self, img: List[List[float]], center: Tuple[int, int], dy: int, dx: int) -> List[float]:
        h, w = len(img), len(img[0])
        cy, cx = center
        out = []
        for i in range(1, self.D + 1):
            y = min(max(cy + i * dy, 0), h - 1)
            x = min(max(cx + i * dx, 0), w - 1)
            out.append(img[y][x])
        return out

    def __call__(self, image: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]], center: Optional[Tuple[int, int]] = None) -> List[List[List[float]]]:
        img = self._gray(image)
        h, w = len(img), len(img[0])
        ctr = center or (h // 2, w // 2)
        rays = [self._ray(img, ctr, dy, dx) for dy, dx in self.directions]
        axes = [rays[0] + rays[1], rays[2] + rays[3], rays[4] + rays[5], rays[6] + rays[7]]
        return [axes]


class TemporalBuffer:
    def __init__(self, T: int = 8, n_coeffs: int = 72) -> None:
        self.buffer: Deque[List[float]] = deque(maxlen=T)
        self.prev_vel = [0.0] * n_coeffs
        self.n_coeffs = n_coeffs

    def update(self, cn: Sequence[float], dt: float = 1.0) -> Tuple[List[float], List[float]]:
        if len(cn) != self.n_coeffs:
            raise ValueError("dimensão de cn inválida")
        cur = list(cn)
        if self.buffer:
            vel = [(cur[i] - self.buffer[-1][i]) / max(dt, 1e-6) for i in range(self.n_coeffs)]
        else:
            vel = [0.0] * self.n_coeffs
        acc = [(vel[i] - self.prev_vel[i]) / max(dt, 1e-6) for i in range(self.n_coeffs)]
        self.buffer.append(cur)
        self.prev_vel = vel
        return vel, acc


class DendriticFuser:
    def __init__(self, replicate_global: bool = True) -> None:
        self.replicate_global = replicate_global

    def __call__(self, cn: Sequence[float], x_rad: List[List[List[float]]]) -> List[List[List[float]]]:
        if len(cn) != 72:
            raise ValueError("cn deve ter 72 coeficientes")
        b, d, _ = _shape3(x_rad)
        if d != 4:
            raise ValueError("x_rad precisa de 4 dendritos")
        sectors = [list(cn[i * 18 : (i + 1) * 18]) for i in range(4)]
        if self.replicate_global:
            g = list(cn[:3])
            sectors = [s + g for s in sectors]

        out: List[List[List[float]]] = []
        for bi in range(b):
            dendrites = []
            for di in range(4):
                dendrites.append(sectors[di] + list(x_rad[bi][di]))
            out.append(dendrites)
        return out


class SurpriseField:
    def __init__(self, alpha: float = 0.95) -> None:
        self.alpha = alpha
        self.r_hat: Optional[List[List[float]]] = None

    def update(self, activation: List[List[float]]) -> List[List[float]]:
        if self.r_hat is None:
            self.r_hat = [row[:] for row in activation]
        surprise = [[abs(activation[i][j] - self.r_hat[i][j]) for j in range(len(activation[i]))] for i in range(len(activation))]
        self.r_hat = [
            [self.alpha * self.r_hat[i][j] + (1.0 - self.alpha) * activation[i][j] for j in range(len(activation[i]))]
            for i in range(len(activation))
        ]
        return surprise


@dataclass
class IntentionProfile:
    r_threshold: float
    stdp_scale: float
    theta_gain: float


class IntentionCtrl:
    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.r_global = 0.0
        self.state = IntentionState.VIGILANCE
        self.profiles = {
            IntentionState.CURIOSITY: IntentionProfile(0.1, 1.0, 0.8),
            IntentionState.FOCUS: IntentionProfile(0.8, 0.5, 1.2),
            IntentionState.VIGILANCE: IntentionProfile(0.3, 0.2, 1.0),
            IntentionState.CONSOLIDATION: IntentionProfile(0.9, 0.0, 1.3),
        }

    def update(self, surprise: List[List[float]]) -> IntentionState:
        values = [v for row in surprise for v in row]
        cur = sum(values) / max(len(values), 1)
        self.r_global = self.smoothing * self.r_global + (1.0 - self.smoothing) * cur
        if self.r_global < 0.08:
            self.state = IntentionState.CONSOLIDATION
        elif self.r_global < 0.2:
            self.state = IntentionState.FOCUS
        elif self.r_global < 0.5:
            self.state = IntentionState.VIGILANCE
        else:
            self.state = IntentionState.CURIOSITY
        return self.state

    def profile(self) -> IntentionProfile:
        return self.profiles[self.state]


class GeoTokenizer:
    def __call__(self, cn: Sequence[float], cn_dot: Sequence[float], cn_ipa: Optional[Sequence[float]] = None) -> List[float]:
        return list(cn) + list(cn_dot) + (list(cn_ipa) if cn_ipa is not None else list(cn))


class MPJRDLayer:
    def __init__(self, n_neurons: int, input_size: int, alpha_rhat: float = 0.95, target_rate: float = 0.1) -> None:
        self.n_neurons = n_neurons
        self.input_size = input_size
        rnd = random.Random(42)
        self.W = [[[rnd.uniform(-0.05, 0.05) for _ in range(input_size)] for _ in range(n_neurons)] for _ in range(4)]
        self.theta_soma = [0.4] * n_neurons
        self.target_rate = target_rate
        self.surprise_field = SurpriseField(alpha=alpha_rhat)

    def forward(self, x: List[List[List[float]]]) -> Dict[str, List[List[float]]]:
        b, d, s = _shape3(x)
        if d != 4 or s != self.input_size:
            raise ValueError("x com shape inválido")

        dendritic: List[List[List[float]]] = [[[0.0] * self.n_neurons for _ in range(4)] for _ in range(b)]
        for bi in range(b):
            for di in range(4):
                norm = math.sqrt(sum(v * v for v in x[bi][di]))
                for n in range(self.n_neurons):
                    v = sum(x[bi][di][k] * self.W[di][n][k] for k in range(self.input_size))
                    gate = 1.0 / (1.0 + math.exp(-v))
                    shunt = v / (1.0 + norm)
                    dendritic[bi][di][n] = gate * shunt

        soma = [[sum(math.tanh(dendritic[bi][di][n]) for di in range(4)) for n in range(self.n_neurons)] for bi in range(b)]
        spikes = [[1.0 if soma[bi][n] > self.theta_soma[n] else 0.0 for n in range(self.n_neurons)] for bi in range(b)]
        surprise = self.surprise_field.update(soma)

        for n in range(self.n_neurons):
            rate = sum(spikes[bi][n] for bi in range(b)) / max(b, 1)
            self.theta_soma[n] += 0.01 * (rate - self.target_rate)

        return {"dendritic": [row for row in dendritic], "soma": soma, "spikes": spikes, "surprise": surprise}


class UnifiedPyFoldsEncoder:
    def __init__(self, cfg: Optional[PyFoldsConfig] = None, n_neurons: int = 32) -> None:
        self.cfg = cfg or PyFoldsConfig()
        self.rive = RIVEEncoder(n_coeffs=self.cfg.n_legendre)
        self.radial = RadialExtractor(D=self.cfg.radial_steps)
        self.temporal = TemporalBuffer(T=self.cfg.temporal_buffer, n_coeffs=self.cfg.n_legendre)
        self.fuser = DendriticFuser(replicate_global=True)
        self.layer = MPJRDLayer(n_neurons=n_neurons, input_size=18 + 3 + 2 * self.cfg.radial_steps, alpha_rhat=self.cfg.alpha_rhat)
        self.intention = IntentionCtrl(smoothing=self.cfg.intention_smoothing)
        self.geo = GeoTokenizer()

    def step(self, image: Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]], center: Optional[Tuple[int, int]] = None, dt: float = 1.0) -> Dict[str, object]:
        cn = self.rive(image)
        x_rad = self.radial(image, center=center)
        cn_dot, cn_ddot = self.temporal.update(cn, dt=dt)
        x_fused = self.fuser(cn, x_rad)
        neural = self.layer.forward(x_fused)

        state = self.intention.update(neural["surprise"])
        profile = self.intention.profile()
        tau = self.geo(cn, cn_dot, cn_ddot)

        z = [i / max(self.cfg.temporal_buffer - 1, 1) for i in range(self.cfg.temporal_buffer)]
        area = [float(i + 1) for i in range(self.cfg.temporal_buffer)]
        vol = sum(area[i] / max((1 - z[i]) ** 2, 1e-3) for i in range(self.cfg.temporal_buffer))

        return {
            "cn": cn,
            "cn_dot": cn_dot,
            "cn_ddot": cn_ddot,
            "x_rad": x_rad,
            "x_fused": x_fused,
            "spikes": neural["spikes"],
            "surprise": neural["surprise"],
            "state": state.value,
            "r_threshold": [profile.r_threshold],
            "tau_geo": tau,
            "Z": z,
            "V": [vol],
        }
