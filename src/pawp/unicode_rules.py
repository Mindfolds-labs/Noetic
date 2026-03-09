from __future__ import annotations

import re
import unicodedata
from typing import List


class UnicodeNormalizer:
    def __init__(self, form: str = "NFC", lowercase: bool = True) -> None:
        self.form = form
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        text = unicodedata.normalize(self.form, text)
        return text.lower() if self.lowercase else text


class BasicUnicodeRules:
    @staticmethod
    def char_class(ch: str) -> str:
        category = unicodedata.category(ch)
        if category.startswith("L"):
            return "LETTER"
        if category.startswith("N"):
            return "NUMBER"
        if category.startswith("P"):
            return "PUNCT"
        if category.startswith("Z"):
            return "SPACE"
        return "OTHER"


class PreTokenizer:
    _token_pattern = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def split_words(self, text: str) -> List[str]:
        return self._token_pattern.findall(text)
