from __future__ import annotations

import re
import shutil
import subprocess
from abc import ABC, abstractmethod
from functools import lru_cache


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


class FallbackBackend(G2PBackend):
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


def _build_g2p_backend(preferred: str = "auto") -> G2PBackend:
    if preferred in {"epitran", "auto"}:
        try:
            return EpitranBackend()
        except ModuleNotFoundError:
            pass

    if preferred in {"espeak-ng", "auto"} and shutil.which("espeak-ng"):
        return EspeakBackend()

    return FallbackBackend()


@lru_cache(maxsize=8192)
def surface_to_ipa(text: str, lang: str = "pt", backend: str = "auto") -> str:
    return _build_g2p_backend(preferred=backend).to_ipa(text, lang)


def word_to_ipa(word: str, language: str = "pt") -> str:
    """Backwards-compatible API for IPA generation."""
    return surface_to_ipa(word, lang=language, backend="auto")
