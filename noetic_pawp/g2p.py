from __future__ import annotations

import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Iterable

_PT_RULES = [
    (r"nh", "ɲ"),
    (r"lh", "ʎ"),
    (r"ch", "ʃ"),
    (r"rr", "ʁ"),
    (r"ss", "s"),
    (r"qu", "k"),
    (r"gu([ei])", r"g\1"),
    (r"ç", "s"),
    (r"á|â|ã", "a"),
    (r"é|ê", "e"),
    (r"í", "i"),
    (r"ó|ô|õ", "o"),
    (r"ú", "u"),
]


class G2PBackend(ABC):
    @abstractmethod
    def to_ipa(self, text: str, lang: str) -> str:
        raise NotImplementedError


class HeuristicFallbackBackend(G2PBackend):
    def to_ipa(self, text: str, lang: str) -> str:
        w = text.lower()
        if lang.startswith("pt"):
            for pattern, repl in _PT_RULES:
                w = re.sub(pattern, repl, w)
            w = w.replace("x", "ʃ").replace("j", "ʒ").replace("c", "k").replace("q", "k")
        return w


class EpitranBackend(G2PBackend):
    def __init__(self) -> None:
        import epitran

        self._epitran = epitran
        self._instances: dict[str, object] = {}

    def to_ipa(self, text: str, lang: str) -> str:
        code = {"pt": "por-Latn", "en": "eng-Latn", "es": "spa-Latn"}.get(lang, f"{lang}-Latn")
        engine = self._instances.get(code)
        if engine is None:
            engine = self._epitran.Epitran(code)
            self._instances[code] = engine
        return engine.transliterate(text)


class EspeakBackend(G2PBackend):
    def __init__(self, binary: str = "espeak-ng") -> None:
        self.binary = binary

    def to_ipa(self, text: str, lang: str) -> str:
        completed = subprocess.run(
            [self.binary, "-v", lang, "--ipa=3", "-q", text],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "espeak-ng failed")
        return completed.stdout.strip().replace(" ", "")


# Backward-compatible alias used by external importers.
FallbackBackend = HeuristicFallbackBackend


@lru_cache(maxsize=32)
def _load_backend(name: str) -> G2PBackend | None:
    name = name.lower()
    if name == "epitran":
        try:
            return EpitranBackend()
        except ModuleNotFoundError:
            return None
    if name in {"espeak", "espeak-ng"}:
        if shutil.which("espeak-ng"):
            return EspeakBackend()
        return None
    if name in {"fallback", "heuristic"}:
        return HeuristicFallbackBackend()
    return None


def _normalize_backend_order(backend: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(backend, str):
        if backend == "auto":
            return ("epitran", "espeak", "fallback")
        return (backend.lower(),)
    normalized = tuple(item.lower() for item in backend)
    return normalized or ("fallback",)


def _to_backend_key(backends: tuple[str, ...]) -> str:
    return "|".join(backends)


def _select_backend(backends: tuple[str, ...]) -> tuple[str, G2PBackend]:
    for name in backends:
        instance = _load_backend(name)
        if instance is not None:
            resolved = "espeak" if name == "espeak-ng" else name
            return resolved, instance
    # Deterministic total fallback ensures convergence of IPA side-channel features.
    return "fallback", HeuristicFallbackBackend()


@lru_cache(maxsize=8192)
def _surface_to_ipa_cached(text: str, lang: str, backend_key: str) -> str:
    backend_order = tuple(item for item in backend_key.split("|") if item)
    _, backend = _select_backend(backend_order)
    return backend.to_ipa(text, lang)


def surface_to_ipa(
    text: str,
    lang: str = "pt",
    backend: str | Iterable[str] = "auto",
) -> str:
    ordered_backends = _normalize_backend_order(backend)
    backend_key = _to_backend_key(ordered_backends)
    return _surface_to_ipa_cached(text, lang, backend_key)


def word_to_ipa(word: str, language: str = "pt") -> str:
    """Backwards-compatible API for IPA generation."""
    return surface_to_ipa(word, lang=language, backend="auto")
