from __future__ import annotations

import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import lru_cache

from .config import PAWPConfig

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


def _try_create_backend(name: str) -> G2PBackend | None:
    if name == "epitran":
        try:
            return EpitranBackend()
        except ModuleNotFoundError:
            return None

    if name == "espeak":
        if shutil.which("espeak-ng"):
            return EspeakBackend()
        return None

    if name == "fallback":
        return HeuristicFallbackBackend()

    return None


def _resolve_priority(priority: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if priority is not None:
        return priority
    return tuple(PAWPConfig().g2p_backend_priority)


@lru_cache(maxsize=256)
def _build_g2p_backend(priority: tuple[str, ...]) -> tuple[str, G2PBackend]:
    for name in priority:
        backend = _try_create_backend(name)
        if backend is not None:
            return name, backend
    return "fallback", HeuristicFallbackBackend()


@lru_cache(maxsize=8192)
def _surface_to_ipa_cached(text: str, lang: str, backend_name: str) -> str:
    resolved = _try_create_backend(backend_name)
    if resolved is None:
        resolved = HeuristicFallbackBackend()
    return resolved.to_ipa(text, lang)


def surface_to_ipa(
    text: str,
    lang: str = "pt",
    backend: str = "auto",
    backend_priority: tuple[str, ...] | None = None,
) -> str:
    resolved_name = backend
    if backend == "auto":
        selected_priority = _resolve_priority(backend_priority)
        resolved_name, _ = _build_g2p_backend(selected_priority)

    return _surface_to_ipa_cached(text, lang, resolved_name)


@lru_cache(maxsize=8192)
def _word_to_ipa_cached(word: str, language: str, backend_priority: tuple[str, ...]) -> str:
    backend_name, _ = _build_g2p_backend(backend_priority)
    # cache key is deterministic by (word, language, backend_name)
    return _surface_to_ipa_cached(word, language, backend_name)


def word_to_ipa(word: str, language: str = "pt") -> str:
    """Backwards-compatible API for IPA generation."""
    priority = tuple(PAWPConfig().g2p_backend_priority)
    return _word_to_ipa_cached(word, language, priority)
