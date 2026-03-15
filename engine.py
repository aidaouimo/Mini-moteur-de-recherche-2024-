"""
engine.py — High-level SearchEngine facade.

Combines Index + Tokenizer + QueryParser + Ranker into a single interface.
"""

from __future__ import annotations

import time
from typing import List, Optional

from .index import Document, InvertedIndex
from .tokenizer import Tokenizer
from .ranker import TFIDFRanker, SearchResult
from .query_parser import QueryParser, QueryExecutor


class SearchEngine:
    """
    All-in-one search engine.

    Usage:
        engine = SearchEngine(mode="bm25")
        engine.index_documents(docs)
        results = engine.search("inverted index python", top_k=10)
    """

    def __init__(
        self,
        language: str = "en",
        mode: str = "bm25",          # "tfidf" | "bm25"
        use_stemming: bool = True,
        remove_stopwords: bool = True,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenizer = Tokenizer(
            language=language,
            use_stemming=use_stemming,
            remove_stopwords=remove_stopwords,
        )
        self.index = InvertedIndex(tokenizer=self.tokenizer)
        self.ranker = TFIDFRanker(self.index, mode=mode, k1=k1, b=b)
        self.parser = QueryParser(tokenizer=self.tokenizer)
        self.executor = QueryExecutor(self.index)
        self._mode = mode
        self._query_log: list = []

    # ──────────────────────────────────────────────────────────────────────────
    # Indexing
    # ──────────────────────────────────────────────────────────────────────────

    def index_documents(self, documents: List[Document], verbose: bool = True) -> None:
        self.index.build_from_documents(documents, verbose=verbose)

    def index_text_files(self, paths: List[str]) -> None:
        """Convenience: index plain-text files."""
        docs = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            doc_id = path
            title = path.split("/")[-1]
            docs.append(Document(doc_id=doc_id, title=title, body=content))
        self.index_documents(docs)

    # ──────────────────────────────────────────────────────────────────────────
    # Search
    # ──────────────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Execute a query and return ranked results.

        Args:
            query:  Raw query string (supports boolean + phrase syntax)
            top_k:  Maximum number of results to return

        Returns:
            List of SearchResult sorted by descending relevance score.
        """
        t0 = time.perf_counter()

        parsed = self.parser.parse(query)
        candidates = self.executor.execute(parsed)

        # Score only the candidates (not the whole index)
        scores = self.ranker.score_documents(parsed.tokens)
        filtered_scores = {doc_id: s for doc_id, s in scores.items() if doc_id in candidates}

        sorted_docs = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc = self.index.get_document(doc_id)
            if doc is None:
                continue
            snippet = self.ranker._make_snippet(doc.body, parsed.tokens)
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

        elapsed = time.perf_counter() - t0
        self._query_log.append({"query": query, "results": len(results), "time_ms": elapsed * 1000})

        return results

    def print_results(self, results: List[SearchResult], query: str = "") -> None:
        """Pretty-print search results."""
        if query:
            print(f"\n🔍 Query: «{query}»")
        print(f"   {len(results)} result(s)\n")
        if not results:
            print("   No results found.")
            return
        for r in results:
            print(f"  [{r.rank}] {r.title}")
            print(f"      Score  : {r.score:.4f}")
            if r.url:
                print(f"      URL    : {r.url}")
            print(f"      Snippet: {r.body_snippet}")
            print()

    # ──────────────────────────────────────────────────────────────────────────
    # Stats
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        s = self.index.stats()
        s["scoring_mode"] = self._mode
        if self._query_log:
            avg_ms = sum(q["time_ms"] for q in self._query_log) / len(self._query_log)
            s["queries_run"] = len(self._query_log)
            s["avg_query_time_ms"] = round(avg_ms, 3)
        return s

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_index(self, path: str) -> None:
        self.index.save(path)

    def load_index(self, path: str) -> None:
        self.index = InvertedIndex.load(path)
        self.ranker.index = self.index
        self.executor.index = self.index
