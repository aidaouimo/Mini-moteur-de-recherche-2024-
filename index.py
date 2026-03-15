"""
index.py — Inverted Index with positional information.

Structure:
    {
        "term": {
            "doc_freq": int,
            "postings": {
                "doc_id": {
                    "tf": int,          # raw term frequency
                    "positions": [int]  # token positions (for phrase queries)
                }
            }
        }
    }
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .tokenizer import Tokenizer


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    url: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return f"{self.title} {self.body}"


class InvertedIndex:
    """
    Memory-efficient inverted index with positional postings.

    Supports:
      - Boolean retrieval (AND / OR / NOT)
      - Phrase queries
      - TF-IDF scoring (delegated to the Ranker)
      - Serialization (pickle + JSON)
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()

        # term → { "doc_freq": int, "postings": { doc_id: {"tf": int, "positions": [...]} } }
        self._index: Dict[str, dict] = defaultdict(lambda: {"doc_freq": 0, "postings": {}})

        # doc_id → Document
        self._documents: Dict[str, Document] = {}

        # doc_id → number of tokens  (needed for TF-IDF normalisation)
        self._doc_lengths: Dict[str, int] = {}

        self._total_docs = 0
        self._build_time: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────────

    def add_document(self, doc: Document) -> None:
        """Index a single document."""
        if doc.doc_id in self._documents:
            raise ValueError(f"Document '{doc.doc_id}' already indexed.")

        tokens = self.tokenizer.tokenize(doc.full_text)
        self._documents[doc.doc_id] = doc
        self._doc_lengths[doc.doc_id] = len(tokens)
        self._total_docs += 1

        # Build term → positions map for this document
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        for term, positions in term_positions.items():
            entry = self._index[term]
            entry["doc_freq"] += 1
            entry["postings"][doc.doc_id] = {
                "tf": len(positions),
                "positions": positions,
            }

    def build_from_documents(self, documents: List[Document], verbose: bool = True) -> None:
        """Batch-index a list of documents."""
        t0 = time.perf_counter()
        for i, doc in enumerate(documents):
            self.add_document(doc)
            if verbose and (i + 1) % 500 == 0:
                print(f"  Indexed {i + 1}/{len(documents)} documents…")
        self._build_time = time.perf_counter() - t0
        if verbose:
            print(f"  ✓ {len(documents)} documents indexed in {self._build_time:.3f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # Retrieval helpers
    # ──────────────────────────────────────────────────────────────────────────

    def get_postings(self, term: str) -> Dict[str, dict]:
        """Return postings list for a term (empty dict if term not found)."""
        norm = self.tokenizer.normalize(term)
        return self._index.get(norm, {}).get("postings", {})

    def get_doc_freq(self, term: str) -> int:
        norm = self.tokenizer.normalize(term)
        return self._index.get(norm, {}).get("doc_freq", 0)

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    def get_doc_length(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return sum(self._doc_lengths.values()) / len(self._doc_lengths)

    # ──────────────────────────────────────────────────────────────────────────
    # Boolean retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def boolean_and(self, terms: List[str]) -> Set[str]:
        """Return doc IDs containing ALL terms."""
        if not terms:
            return set()
        sets = [set(self.get_postings(t).keys()) for t in terms]
        return set.intersection(*sets)

    def boolean_or(self, terms: List[str]) -> Set[str]:
        """Return doc IDs containing ANY term."""
        if not terms:
            return set()
        sets = [set(self.get_postings(t).keys()) for t in terms]
        return set.union(*sets)

    def boolean_not(self, term: str) -> Set[str]:
        """Return doc IDs NOT containing the term."""
        matching = set(self.get_postings(term).keys())
        return set(self._documents.keys()) - matching

    def phrase_search(self, phrase: str) -> Set[str]:
        """
        Return doc IDs where all phrase tokens appear consecutively
        (exact phrase match using positional index).
        """
        tokens = self.tokenizer.tokenize(phrase)
        if not tokens:
            return set()

        candidates = self.boolean_and(tokens)
        results = set()

        for doc_id in candidates:
            positions = [self._index[t]["postings"][doc_id]["positions"] for t in tokens]
            # Check if any starting position of tokens[0] leads to a valid phrase
            for start in positions[0]:
                if all(
                    (start + offset) in positions[offset]
                    for offset in range(1, len(tokens))
                ):
                    results.add(doc_id)
                    break

        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._index)

    @property
    def total_docs(self) -> int:
        return self._total_docs

    def stats(self) -> dict:
        return {
            "total_documents": self._total_docs,
            "vocabulary_size": self.vocab_size,
            "avg_doc_length": round(self.avg_doc_length(), 2),
            "build_time_seconds": round(self._build_time or 0, 4),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Index saved → {path}")

    @staticmethod
    def load(path: str) -> "InvertedIndex":
        with open(path, "rb") as f:
            idx = pickle.load(f)
        print(f"Index loaded ← {path}")
        return idx

    def export_json(self, path: str, max_terms: int = 1000) -> None:
        """Export a subset of the index to JSON (for debugging / inspection)."""
        subset = {
            term: {
                "doc_freq": data["doc_freq"],
                "postings": {
                    doc_id: {"tf": p["tf"]}
                    for doc_id, p in list(data["postings"].items())[:5]
                },
            }
            for term, data in list(self._index.items())[:max_terms]
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"Index exported (subset) → {path}")
