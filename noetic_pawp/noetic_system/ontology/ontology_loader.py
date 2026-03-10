from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class OntologyLoader:
    """Ingests ontology sources and normalizes to a shared concept schema."""

    def __init__(self, bootstrap_path: str | Path | None = None) -> None:
        self.bootstrap_path = Path(bootstrap_path) if bootstrap_path else Path(__file__).with_name("seed_concepts.json")

    def load_bootstrap(self) -> list[dict[str, str]]:
        with self.bootstrap_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [self._normalize_record(item, source="seed") for item in data]

    def ingest_wordnet(self, records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        return [self._normalize_record(r, source="wordnet") for r in records]

    def ingest_conceptnet(self, records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        return [self._normalize_record(r, source="conceptnet") for r in records]

    def ingest_wikidata(self, records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        return [self._normalize_record(r, source="wikidata") for r in records]

    def ingest_ipa_dictionary(self, records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        return [self._normalize_record(r, source="ipa_dict") for r in records]

    def bootstrap_wordspace(self, *sources: Iterable[dict[str, str]]) -> list[dict[str, str]]:
        merged: dict[str, dict[str, str]] = {}
        for source in sources:
            for concept in source:
                merged[concept["concept_id"]] = concept
        return list(merged.values())

    @staticmethod
    def _normalize_record(record: dict[str, str], source: str) -> dict[str, str]:
        label = (record.get("label") or record.get("concept") or record.get("id") or "").strip().lower()
        cid = (record.get("concept_id") or record.get("id") or label).strip().lower().replace(" ", "_")
        return {"concept_id": cid, "label": label, "source": source}
