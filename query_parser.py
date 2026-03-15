"""
query_parser.py — Query parsing and execution.

Supported syntax:
    Simple       : machine learning
    AND          : python AND algorithm
    OR           : java OR python
    NOT          : machine NOT learning
    Phrase       : "inverted index"
    Mixed        : "neural network" AND python NOT java
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Set, Tuple

from .index import InvertedIndex
from .tokenizer import Tokenizer


class QueryType(Enum):
    SIMPLE = auto()
    BOOLEAN = auto()
    PHRASE = auto()


@dataclass
class ParsedQuery:
    raw: str
    query_type: QueryType
    tokens: List[str]
    phrases: List[str]
    must_tokens: List[str]      # AND
    should_tokens: List[str]    # OR
    must_not_tokens: List[str]  # NOT


class QueryParser:
    """
    Parse a raw query string into a structured ParsedQuery.
    Supports phrase, boolean (AND / OR / NOT), and simple keyword queries.
    """

    _PHRASE_RE = re.compile(r'"([^"]+)"')
    _BOOLEAN_KEYWORDS = {"AND", "OR", "NOT"}

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()

    def parse(self, raw: str) -> ParsedQuery:
        raw = raw.strip()

        # Extract phrases
        phrases = self._PHRASE_RE.findall(raw)
        stripped = self._PHRASE_RE.sub("", raw).strip()

        has_boolean = any(kw in stripped.split() for kw in self._BOOLEAN_KEYWORDS)

        if has_boolean:
            return self._parse_boolean(raw, stripped, phrases)

        # Simple query
        tokens = self.tokenizer.tokenize(stripped)
        phrase_tokens = []
        for p in phrases:
            phrase_tokens.extend(self.tokenizer.tokenize(p))

        all_tokens = tokens + phrase_tokens

        return ParsedQuery(
            raw=raw,
            query_type=QueryType.PHRASE if phrases else QueryType.SIMPLE,
            tokens=all_tokens,
            phrases=phrases,
            must_tokens=all_tokens,
            should_tokens=[],
            must_not_tokens=[],
        )

    def _parse_boolean(self, raw: str, stripped: str, phrases: List[str]) -> ParsedQuery:
        must, should, must_not = [], [], []

        # Split on AND / OR / NOT (simple left-to-right parse)
        parts = re.split(r"\b(AND|OR|NOT)\b", stripped)
        current_op = "OR"  # default

        for part in parts:
            part = part.strip()
            if part in self._BOOLEAN_KEYWORDS:
                current_op = part
                continue
            tokens = self.tokenizer.tokenize(part)
            if not tokens:
                continue
            if current_op == "AND":
                must.extend(tokens)
            elif current_op == "NOT":
                must_not.extend(tokens)
            else:
                should.extend(tokens)

        # Phrase tokens → must
        for p in phrases:
            must.extend(self.tokenizer.tokenize(p))

        all_tokens = list(set(must + should))

        return ParsedQuery(
            raw=raw,
            query_type=QueryType.BOOLEAN,
            tokens=all_tokens,
            phrases=phrases,
            must_tokens=must,
            should_tokens=should,
            must_not_tokens=must_not,
        )


class QueryExecutor:
    """
    Execute a ParsedQuery against an InvertedIndex and return candidate doc IDs.
    Filtering is applied here; scoring is left to the Ranker.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    def execute(self, query: ParsedQuery) -> Set[str]:
        """Return the set of candidate document IDs matching the query."""

        if query.query_type == QueryType.SIMPLE:
            return self.index.boolean_or(query.tokens)

        if query.query_type == QueryType.PHRASE:
            candidates: Set[str] = set()
            for phrase in query.phrases:
                candidates |= self.index.phrase_search(phrase)
            # Also include simple keyword matches
            if query.tokens:
                candidates |= self.index.boolean_or(query.tokens)
            return candidates

        # BOOLEAN
        candidates = set(self.index._documents.keys())

        if query.must_tokens:
            must_results = self.index.boolean_and(query.must_tokens)
            candidates &= must_results

        if query.should_tokens:
            should_results = self.index.boolean_or(query.should_tokens)
            if query.must_tokens:
                candidates |= should_results
            else:
                candidates = should_results

        if query.must_not_tokens:
            for term in query.must_not_tokens:
                candidates -= set(self.index.get_postings(term).keys())

        # Phrase filtering
        for phrase in query.phrases:
            phrase_matches = self.index.phrase_search(phrase)
            candidates &= phrase_matches

        return candidates
