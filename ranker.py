"""
ranker.py — TF-IDF and BM25 scoring algorithms.

TF-IDF (classic):
    tf(t,d)  = count(t in d) / |d|          (normalized TF)
    idf(t)   = log( N / (1 + df(t)) ) + 1   (smoothed IDF)
    score    = tf * idf

BM25 (Okapi BM25):
    score = Σ IDF(t) · [ tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl)) ]

    Default parameters: k1=1.5, b=0.75
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .index import InvertedIndex
from .tokenizer import Tokenizer


@dataclass
class SearchResult:
    doc_id: str
    title: str
    body_snippet: str
    url: str
    score: float
    rank: int = 0

    def __repr__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.score:.4f}, title={self.title!r})"


class TFIDFRanker:
    """Classic TF-IDF ranker with optional BM25 mode."""

    def __init__(
        self,
        index: InvertedIndex,
        mode: str = "tfidf",   # "tfidf" | "bm25"
        k1: float = 1.5,
        b: float = 0.75,
        snippet_length: int = 200,
    ):
        if mode not in ("tfidf", "bm25"):
            raise ValueError("mode must be 'tfidf' or 'bm25'")
        self.index = index
        self.mode = mode
        self.k1 = k1
        self.b = b
        self.snippet_length = snippet_length

    # ──────────────────────────────────────────────────────────────────────────
    # IDF
    # ──────────────────────────────────────────────────────────────────────────

    def _idf(self, term: str) -> float:
        N = self.index.total_docs
        df = self.index.get_doc_freq(term)
        if df == 0:
            return 0.0
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Per-document scores
    # ──────────────────────────────────────────────────────────────────────────

    def _score_tfidf(self, tf: int, doc_length: int, idf: float) -> float:
        norm_tf = tf / doc_length if doc_length > 0 else 0
        return norm_tf * idf

    def _score_bm25(self, tf: int, doc_length: int, idf: float) -> float:
        avgdl = self.index.avg_doc_length()
        denom = tf + self.k1 * (1 - self.b + self.b * doc_length / max(avgdl, 1))
        return idf * (tf * (self.k1 + 1)) / denom

    # ──────────────────────────────────────────────────────────────────────────
    # Public scoring API
    # ──────────────────────────────────────────────────────────────────────────

    def score_documents(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Compute relevance scores for all candidate documents.

        Returns:
            dict mapping doc_id → cumulative score
        """
        scores: Dict[str, float] = {}

        for term in query_tokens:
            idf = self._idf(term)
            if idf == 0:
                continue
            postings = self.index.get_postings(term)

            for doc_id, data in postings.items():
                tf = data["tf"]
                doc_length = self.index.get_doc_length(doc_id)

                if self.mode == "bm25":
                    s = self._score_bm25(tf, doc_length, idf)
                else:
                    s = self._score_tfidf(tf, doc_length, idf)

                scores[doc_id] = scores.get(doc_id, 0.0) + s

        return scores

    # ──────────────────────────────────────────────────────────────────────────
    # Snippet generation
    # ──────────────────────────────────────────────────────────────────────────

    def _make_snippet(self, body: str, query_tokens: List[str]) -> str:
        """Extract the most relevant passage from the body text."""
        words = body.split()
        if len(words) <= self.snippet_length // 5:
            return body[: self.snippet_length]

        best_start, best_score = 0, 0
        window = self.snippet_length // 5

        for i in range(0, max(1, len(words) - window), max(1, window // 2)):
            chunk = " ".join(words[i : i + window]).lower()
            hits = sum(1 for t in query_tokens if t in chunk)
            if hits > best_score:
                best_score = hits
                best_start = i

        snippet = " ".join(words[best_start : best_start + window])
        return snippet[: self.snippet_length] + ("…" if len(snippet) > self.snippet_length else "")

    # ──────────────────────────────────────────────────────────────────────────
    # Main search entry point
    # ──────────────────────────────────────────────────────────────────────────

    def rank(self, query_tokens: List[str], top_k: int = 10) -> List[SearchResult]:
        """
        Score, sort and return the top-k results.

        Args:
            query_tokens: Pre-tokenized query terms.
            top_k:        Max results to return.

        Returns:
            List of SearchResult sorted by descending score.
        """
        scores = self.score_documents(query_tokens)
        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []

        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc = self.index.get_document(doc_id)
            if doc is None:
                continue
            snippet = self._make_snippet(doc.body, query_tokens)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=doc.title,
                    body_snippet=snippet,
                    url=doc.url,
                    score=score,
                    rank=rank,
                )
            )

        return results
