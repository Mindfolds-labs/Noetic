from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set


class ConceptNormalizer:
    """Resolves multilingual aliases into canonical concept identifiers."""

    def __init__(self, seed_path: Optional[Path] = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "data" / "concepts" / "seed_concepts.json"
        self.seed_path = Path(seed_path) if seed_path else default_path
        self._alias_to_concept: Dict[str, str] = {}
        self._concept_to_aliases: Dict[str, List[str]] = {}
        self._load_seed()

    def _load_seed(self) -> None:
        entries = self._read_entries(self.seed_path)
        for entry in entries:
            concept_id = entry.get("concept_id")
            if not concept_id:
                continue

            alias_set: Set[str] = set()

            aliases_by_lang = entry.get("aliases", {})
            for aliases in aliases_by_lang.values():
                for alias in aliases:
                    key = self._normalize_alias(alias)
                    if key:
                        self._alias_to_concept[key] = concept_id
                        alias_set.add(alias)

            for variant in entry.get("unicode_variations", []):
                key = self._normalize_alias(variant)
                if key:
                    self._alias_to_concept[key] = concept_id
                    alias_set.add(variant)

            ipa = entry.get("ipa")
            if ipa:
                key = self._normalize_alias(ipa)
                if key:
                    self._alias_to_concept[key] = concept_id
                    alias_set.add(ipa)

            self._concept_to_aliases[concept_id] = sorted(alias_set)

    def _read_entries(self, path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            return []
        if path.suffix == ".jsonl":
            entries: List[Dict[str, object]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
            return entries

        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("concepts"), list):
            return data["concepts"]
        return []

    def _normalize_alias(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold().strip()
        return " ".join(normalized.split())

    def resolve_concept(self, token_text: str, language: str) -> Optional[str]:
        del language  # reserved for language-prioritized lookup in a future iteration.
        return self._alias_to_concept.get(self._normalize_alias(token_text))

    def resolve_aliases(self, concept_id: str) -> List[str]:
        return list(self._concept_to_aliases.get(concept_id, []))


def resolve_concept(token_text: str, language: str, normalizer: Optional[ConceptNormalizer] = None) -> Optional[str]:
    resolver = normalizer or ConceptNormalizer()
    return resolver.resolve_concept(token_text=token_text, language=language)


def resolve_aliases(concept_id: str, normalizer: Optional[ConceptNormalizer] = None) -> List[str]:
    resolver = normalizer or ConceptNormalizer()
    return resolver.resolve_aliases(concept_id=concept_id)
