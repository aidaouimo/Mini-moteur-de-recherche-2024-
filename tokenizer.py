"""
tokenizer.py — Text processing pipeline.

Steps:
    1. Lowercase
    2. Unicode normalization (NFD → ASCII)
    3. Punctuation / special character removal
    4. Tokenization
    5. Stop-word removal
    6. (Optional) Stemming via Porter algorithm
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Set


# ─── Minimal English stop-word list ──────────────────────────────────────────
ENGLISH_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "be", "are", "was",
    "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "as", "if", "then", "that", "this", "these", "those", "which",
    "who", "whom", "whose", "when", "where", "why", "how", "all", "any",
    "both", "each", "every", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "they", "his", "her", "their", "what", "there", "here",
    "into", "about", "up", "out", "over", "after", "before", "also",
}

# ─── Minimal French stop-word list ───────────────────────────────────────────
FRENCH_STOPWORDS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "en", "au",
    "aux", "est", "sont", "avec", "sur", "pour", "par", "que", "qui",
    "dans", "ne", "pas", "se", "ce", "il", "elle", "ils", "elles", "on",
    "nous", "vous", "je", "tu", "me", "te", "lui", "leur", "y", "ou",
    "mais", "donc", "ni", "car", "si", "plus", "très", "tout", "tous",
    "cette", "ces", "mon", "ton", "son", "ma", "ta", "sa", "nos", "vos",
    "leurs", "être", "avoir", "faire", "dire", "aller", "voir", "venir",
    "même", "comme", "aussi", "bien", "ainsi",
}


class PorterStemmer:
    """
    Lightweight implementation of the Porter stemming algorithm (English).
    Handles the 5 major reduction steps.
    """

    _vowels = frozenset("aeiou")

    def _has_vowel(self, stem: str) -> bool:
        return any(c in self._vowels for c in stem)

    def _ends_double_consonant(self, word: str) -> bool:
        return len(word) >= 2 and word[-1] == word[-2] and word[-1] not in self._vowels

    def _ends_cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        return (
            word[-1] not in self._vowels
            and word[-2] in self._vowels
            and word[-3] not in self._vowels
            and word[-1] not in "wxy"
        )

    def _measure(self, stem: str) -> int:
        """Count VC sequences (measure m)."""
        pattern = re.sub(r"[aeiou]+", "V", re.sub(r"[^aeiou]+", "C", stem))
        return pattern.count("VC")

    def stem(self, word: str) -> str:
        if len(word) <= 2:
            return word

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        return word

    def _step1a(self, w: str) -> str:
        if w.endswith("sses"):
            return w[:-2]
        if w.endswith("ies"):
            return w[:-2]
        if w.endswith("ss"):
            return w
        if w.endswith("s"):
            return w[:-1]
        return w

    def _step1b(self, w: str) -> str:
        if w.endswith("eed"):
            if self._measure(w[:-3]) > 0:
                return w[:-1]
            return w
        for suffix in ("ed", "ing"):
            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if self._has_vowel(stem):
                    if stem.endswith(("at", "bl", "iz")):
                        return stem + "e"
                    if self._ends_double_consonant(stem) and not stem.endswith(("l", "s", "z")):
                        return stem[:-1]
                    if self._measure(stem) == 1 and self._ends_cvc(stem):
                        return stem + "e"
                    return stem
        return w

    def _step1c(self, w: str) -> str:
        if w.endswith("y") and self._has_vowel(w[:-1]):
            return w[:-1] + "i"
        return w

    def _step2(self, w: str) -> str:
        replacements = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, replacement in replacements:
            if w.endswith(suffix) and self._measure(w[: -len(suffix)]) > 0:
                return w[: -len(suffix)] + replacement
        return w

    def _step3(self, w: str) -> str:
        replacements = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, replacement in replacements:
            if w.endswith(suffix) and self._measure(w[: -len(suffix)]) > 0:
                return w[: -len(suffix)] + replacement
        return w

    def _step4(self, w: str) -> str:
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
            "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize",
        ]
        for suffix in suffixes:
            if w.endswith(suffix):
                stem = w[: -len(suffix)]
                if suffix == "ion" and not stem.endswith(("s", "t")):
                    continue
                if self._measure(stem) > 1:
                    return stem
        return w

    def _step5(self, w: str) -> str:
        if w.endswith("e"):
            stem = w[:-1]
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._ends_cvc(stem):
                return stem
        if w.endswith("ll") and self._measure(w[:-1]) > 1:
            return w[:-1]
        return w


class Tokenizer:
    """
    Configurable text tokenizer.

    Args:
        language:        'en' or 'fr' (affects stop-word list)
        use_stemming:    Apply Porter stemmer
        remove_stopwords: Remove stop words
        min_token_length: Discard tokens shorter than this
    """

    def __init__(
        self,
        language: str = "en",
        use_stemming: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 2,
    ):
        self.language = language
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        self._stopwords = FRENCH_STOPWORDS if language == "fr" else ENGLISH_STOPWORDS
        self._stemmer = PorterStemmer() if use_stemming else None

    def normalize(self, text: str) -> str:
        """Normalize a single token (lowercase + unicode stripping)."""
        text = text.lower()
        text = unicodedata.normalize("NFD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        return text

    def tokenize(self, text: str) -> List[str]:
        """Full tokenization pipeline."""
        text = self.normalize(text)
        # Keep only alphanumeric + apostrophes
        text = re.sub(r"[^a-z0-9\s']", " ", text)
        tokens = text.split()

        result = []
        for token in tokens:
            token = token.strip("'")
            if len(token) < self.min_token_length:
                continue
            if self.remove_stopwords and token in self._stopwords:
                continue
            if self._stemmer:
                token = self._stemmer.stem(token)
            result.append(token)

        return result
