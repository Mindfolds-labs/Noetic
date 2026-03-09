from __future__ import annotations

from typing import List, Sequence, Tuple

_VOWELS = set("aeiouáàâãéêíóôõúüɐɑɔəɛɜɪʊ")
_SOFT_EQUIV = {
    ("c", "k"),
    ("c", "s"),
    ("q", "k"),
    ("x", "ʃ"),
    ("x", "s"),
    ("j", "ʒ"),
    ("r", "ʁ"),
    ("n", "ŋ"),
    ("l", "ʎ"),
}


def _clean_subword(piece: str) -> str:
    return piece.replace("##", "")


def _subst_cost(char: str, phoneme: str) -> float:
    """Approximate grapheme/phoneme substitution cost.

    This keeps alignment monotonic while allowing common allophones
    (e.g., c↔k/s, x↔ʃ/s) to remain cheaper than arbitrary substitutions.
    """
    if char == phoneme:
        return 0.0
    if (char, phoneme) in _SOFT_EQUIV:
        return 0.2
    if char in _VOWELS and phoneme in _VOWELS:
        return 0.35
    return 1.0


def align_subwords_to_ipa(subwords: Sequence[str], ipa_units: Sequence[str]) -> List[Tuple[int, int]]:
    """Monotonic grapheme-to-IPA alignment using dynamic programming.

    The algorithm computes an edit lattice over characters vs IPA units, then
    projects matched character indices back to subword spans.
    """
    if not subwords:
        return []
    if not ipa_units:
        return [(0, 0) for _ in subwords]

    pieces = [_clean_subword(piece) for piece in subwords]
    chars = [ch for p in pieces for ch in p]
    n = len(chars)
    m = len(ipa_units)
    if n == 0:
        return [(0, 0) for _ in subwords]

    ins_del = 0.7
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up, 2 left

    for i in range(1, n + 1):
        dp[i][0] = i * ins_del
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j * ins_del
        bt[0][j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + _subst_cost(chars[i - 1], ipa_units[j - 1])
            up = dp[i - 1][j] + ins_del
            left = dp[i][j - 1] + ins_del
            if diag <= up and diag <= left:
                dp[i][j] = diag
                bt[i][j] = 0
            elif up <= left:
                dp[i][j] = up
                bt[i][j] = 1
            else:
                dp[i][j] = left
                bt[i][j] = 2

    char_to_ipa: List[List[int]] = [[] for _ in range(n)]
    i, j = n, m
    while i > 0 or j > 0:
        move = bt[i][j]
        if i > 0 and j > 0 and move == 0:
            char_to_ipa[i - 1].append(j - 1)
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or move == 1):
            i -= 1
        else:
            j -= 1

    spans: List[Tuple[int, int]] = []
    cursor = 0
    c0 = 0
    for idx, piece in enumerate(pieces):
        c1 = c0 + len(piece)
        indices = [k for ci in range(c0, c1) for k in char_to_ipa[ci]]
        if indices:
            start = max(cursor, min(indices))
            end = max(start, max(indices) + 1)
        else:
            # Silent segment fallback: preserve monotonicity.
            start = cursor
            end = cursor

        if idx == len(pieces) - 1:
            end = m
        spans.append((start, end))
        cursor = end
        c0 = c1

    return spans
