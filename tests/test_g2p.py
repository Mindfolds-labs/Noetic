from noetic_pawp import g2p


def test_word_to_ipa_facade_matches_surface_auto() -> None:
    word = "linguística"
    assert g2p.word_to_ipa(word, language="pt") == g2p.surface_to_ipa(word, lang="pt", backend="auto")


def test_surface_to_ipa_cache_is_deterministic_by_backend_order() -> None:
    g2p._surface_to_ipa_cached.cache_clear()
    g2p.surface_to_ipa("casa", lang="pt", backend=["fallback"])
    g2p.surface_to_ipa("casa", lang="pt", backend=["fallback"])
    info = g2p._surface_to_ipa_cached.cache_info()
    assert info.hits >= 1


def test_backend_routing_uses_first_available_backend() -> None:
    class _DummyBackend(g2p.G2PBackend):
        def to_ipa(self, text: str, lang: str) -> str:
            return f"dummy:{lang}:{text}"

    original = g2p._load_backend

    def _fake_loader(name: str):
        if name == "epitran":
            return None
        if name == "espeak":
            return _DummyBackend()
        if name == "fallback":
            return g2p.HeuristicFallbackBackend()
        return None

    try:
        g2p._load_backend.cache_clear()
        g2p._load_backend = _fake_loader  # type: ignore[assignment]
        resolved, backend = g2p._select_backend(("epitran", "espeak", "fallback"))
        assert resolved == "espeak"
        assert backend.to_ipa("ola", "pt") == "dummy:pt:ola"
    finally:
        g2p._load_backend = original  # type: ignore[assignment]
        g2p._load_backend.cache_clear()
