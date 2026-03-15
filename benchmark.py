"""
benchmarks/benchmark.py — Performance benchmarks on synthetic large-scale data.

Tests:
  - Index build time vs. corpus size
  - Query throughput (queries/second)
  - Memory footprint estimation
"""

from __future__ import annotations

import os
import random
import string
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.index import Document, InvertedIndex
from src.engine import SearchEngine
from src.tokenizer import Tokenizer


# ─── Synthetic corpus generator ───────────────────────────────────────────────

SAMPLE_WORDS = [
    "algorithm", "data", "structure", "search", "index", "query", "python",
    "machine", "learning", "neural", "network", "graph", "tree", "hash",
    "sort", "binary", "stack", "queue", "heap", "dynamic", "programming",
    "complexity", "performance", "memory", "cache", "database", "retrieval",
    "information", "text", "document", "term", "frequency", "inverse",
    "weight", "score", "rank", "precision", "recall", "vector", "space",
    "model", "language", "natural", "processing", "tokenize", "stem",
    "filter", "boolean", "phrase", "cosine", "similarity", "cluster",
]


def generate_corpus(n_docs: int, avg_words: int = 150, seed: int = 42) -> list[Document]:
    rng = random.Random(seed)
    docs = []
    for i in range(n_docs):
        n_words = rng.randint(avg_words // 2, avg_words * 2)
        body = " ".join(rng.choices(SAMPLE_WORDS, k=n_words))
        title = " ".join(rng.choices(SAMPLE_WORDS, k=rng.randint(3, 8))).title()
        docs.append(Document(doc_id=f"doc_{i}", title=title, body=body))
    return docs


def generate_queries(n: int = 100, seed: int = 99) -> list[str]:
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        k = rng.randint(1, 4)
        queries.append(" ".join(rng.choices(SAMPLE_WORDS, k=k)))
    return queries


# ─── Benchmark helpers ────────────────────────────────────────────────────────

def bench_index_build(sizes: list[int]) -> None:
    print("\n" + "═" * 60)
    print("  BENCHMARK 1 — Index build time vs corpus size")
    print("═" * 60)
    print(f"  {'Docs':>8}  {'Tokens (approx)':>16}  {'Time (s)':>10}  {'Docs/s':>10}")
    print("  " + "-" * 54)

    for n in sizes:
        corpus = generate_corpus(n, avg_words=150)
        idx = InvertedIndex()
        t0 = time.perf_counter()
        idx.build_from_documents(corpus, verbose=False)
        elapsed = time.perf_counter() - t0
        total_tokens = sum(idx.get_doc_length(f"doc_{i}") for i in range(n))
        print(f"  {n:>8,}  {total_tokens:>16,}  {elapsed:>10.3f}  {n/elapsed:>10.0f}")


def bench_query_throughput(n_docs: int = 10_000, n_queries: int = 500) -> None:
    print("\n" + "═" * 60)
    print(f"  BENCHMARK 2 — Query throughput ({n_docs:,} docs, {n_queries} queries)")
    print("═" * 60)

    corpus = generate_corpus(n_docs)
    queries = generate_queries(n_queries)

    for mode in ("tfidf", "bm25"):
        engine = SearchEngine(mode=mode)
        engine.index_documents(corpus, verbose=False)

        t0 = time.perf_counter()
        for q in queries:
            engine.search(q, top_k=10)
        elapsed = time.perf_counter() - t0

        qps = n_queries / elapsed
        avg_ms = (elapsed / n_queries) * 1000
        print(f"  [{mode.upper():5s}]  {qps:>8.1f} queries/s  avg={avg_ms:.2f}ms/query")


def bench_vocab_growth(sizes: list[int]) -> None:
    print("\n" + "═" * 60)
    print("  BENCHMARK 3 — Vocabulary growth vs corpus size")
    print("═" * 60)
    print(f"  {'Docs':>8}  {'Vocab size':>12}  {'Avg doc len':>12}")
    print("  " + "-" * 38)

    for n in sizes:
        corpus = generate_corpus(n)
        idx = InvertedIndex()
        idx.build_from_documents(corpus, verbose=False)
        stats = idx.stats()
        print(f"  {n:>8,}  {stats['vocabulary_size']:>12,}  {stats['avg_doc_length']:>12.1f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Mini Search Engine — Performance Benchmarks")
    print("=" * 60)

    sizes = [500, 1_000, 5_000, 10_000, 50_000]
    bench_index_build(sizes)
    bench_query_throughput(n_docs=10_000, n_queries=500)
    bench_vocab_growth([500, 2_000, 10_000, 50_000])

    print("\n✓ Benchmarks complete.")
