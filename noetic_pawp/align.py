from __future__ import annotations

from typing import List, Sequence, Tuple


def align_subwords_to_ipa(subwords: Sequence[str], ipa_units: Sequence[str]) -> List[Tuple[int, int]]:
    """Alinhamento heurístico proporcional entre subwords e sequência IPA."""
    if not subwords:
        return []
    if not ipa_units:
        return [(0, 0) for _ in subwords]

    lengths = [max(len(piece.replace("##", "")), 1) for piece in subwords]
    total_len = sum(lengths)
    total_ipa = len(ipa_units)

    spans: List[Tuple[int, int]] = []
    cursor = 0

    for i, piece_len in enumerate(lengths):
        if i == len(lengths) - 1:
            end = total_ipa
        else:
            alloc = max(1, round((piece_len / total_len) * total_ipa))
            end = min(total_ipa, cursor + alloc)
            remaining_slots = len(lengths) - (i + 1)
            if total_ipa - end < remaining_slots:
                end = total_ipa - remaining_slots

        spans.append((cursor, end))
        cursor = end

    return spans
