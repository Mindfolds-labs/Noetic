from __future__ import annotations

import re


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


def word_to_ipa(word: str, language: str = "pt") -> str:
    """Heurística simples de G2P para protótipo PAWP v0.3."""
    w = word.lower()
    if language == "pt":
        for pattern, repl in _PT_RULES:
            w = re.sub(pattern, repl, w)
        w = w.replace("x", "ʃ")
        w = w.replace("j", "ʒ")
        w = w.replace("c", "k")
        w = w.replace("q", "k")
    return w
