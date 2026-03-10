from __future__ import annotations

from noetic_pawp import g2p


def test_try_create_backend_handles_missing_epitran(monkeypatch) -> None:
    class MissingEpitran:
        def __init__(self) -> None:
            raise ModuleNotFoundError("epitran")

    monkeypatch.setattr(g2p, "EpitranBackend", MissingEpitran)
    assert g2p._try_create_backend("epitran") is None


def test_surface_to_ipa_cache_key_depends_on_backend(monkeypatch) -> None:
    class DummyBackend(g2p.G2PBackend):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def to_ipa(self, text: str, lang: str) -> str:
            return f"{self.tag}:{lang}:{text}"

    def fake_try_create(name: str):
        if name in {"a", "b"}:
            return DummyBackend(name)
        if name == "fallback":
            return DummyBackend("fallback")
        return None

    monkeypatch.setattr(g2p, "_try_create_backend", fake_try_create)
    g2p._surface_to_ipa_cached.cache_clear()

    out_a = g2p.surface_to_ipa("casa", lang="pt", backend="a")
    out_b = g2p.surface_to_ipa("casa", lang="pt", backend="b")

    assert out_a != out_b
    assert out_a == "a:pt:casa"
    assert out_b == "b:pt:casa"


def test_surface_to_ipa_uses_priority_router(monkeypatch) -> None:
    class DummyBackend(g2p.G2PBackend):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def to_ipa(self, text: str, lang: str) -> str:
            return f"{self.tag}:{text}"

    def fake_try_create(name: str):
        if name == "espeak":
            return DummyBackend("espeak")
        if name == "fallback":
            return DummyBackend("fallback")
        return None

    monkeypatch.setattr(g2p, "_try_create_backend", fake_try_create)
    g2p._build_g2p_backend.cache_clear()
    g2p._surface_to_ipa_cached.cache_clear()

    out = g2p.surface_to_ipa(
        "ola",
        lang="pt",
        backend="auto",
        backend_priority=("epitran", "espeak", "fallback"),
    )

    assert out == "espeak:ola"


def test_word_to_ipa_keeps_backward_compatible_facade() -> None:
    out = g2p.word_to_ipa("casa", language="pt")
    assert isinstance(out, str)
    assert out
