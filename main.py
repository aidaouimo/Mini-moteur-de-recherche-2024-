"""
main.py — Interactive demo of the Mini Search Engine.

Usage:
    python main.py                          # demo with built-in sample corpus
    python main.py --interactive            # interactive query mode
    python main.py --benchmark              # run performance benchmarks
    python main.py --files path/to/*.txt    # index your own text files
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.index import Document, InvertedIndex
from src.engine import SearchEngine


# ─── Sample corpus ────────────────────────────────────────────────────────────

SAMPLE_CORPUS = [
    Document("d01", "Introduction to Algorithms",
        "An algorithm is a step-by-step procedure for solving a problem. Common algorithms include sorting, searching, and graph traversal. Binary search runs in O(log n) time.",
        url="https://example.com/algorithms"),
    Document("d02", "Inverted Index in Information Retrieval",
        "An inverted index maps terms to the documents in which they occur. It is the core data structure used by search engines. The index enables efficient full-text search over large document collections.",
        url="https://example.com/inverted-index"),
    Document("d03", "TF-IDF Weighting Scheme",
        "Term Frequency-Inverse Document Frequency (TF-IDF) is a numerical statistic reflecting how important a word is to a document in a corpus. High TF-IDF means the term is frequent in the document but rare across the corpus.",
        url="https://example.com/tfidf"),
    Document("d04", "Python for Data Science",
        "Python is widely used in data science, machine learning, and natural language processing. Libraries like NumPy, Pandas, and scikit-learn make Python ideal for data analysis.",
        url="https://example.com/python-ds"),
    Document("d05", "BM25 Ranking Algorithm",
        "Okapi BM25 is a ranking function used by search engines. It improves on TF-IDF by adding document length normalization and a saturation factor for term frequency.",
        url="https://example.com/bm25"),
    Document("d06", "Graph Data Structures",
        "Graphs consist of vertices and edges. Common graph algorithms include depth-first search, breadth-first search, Dijkstra shortest path, and minimum spanning tree.",
        url="https://example.com/graphs"),
    Document("d07", "Machine Learning Fundamentals",
        "Machine learning is a subset of artificial intelligence. Supervised learning, unsupervised learning, and reinforcement learning are the main paradigms. Neural networks have revolutionized the field.",
        url="https://example.com/ml"),
    Document("d08", "Hash Tables and Dictionaries",
        "A hash table provides O(1) average-case lookup, insertion, and deletion. Collision resolution strategies include chaining and open addressing. Python dicts are implemented as hash tables.",
        url="https://example.com/hash"),
    Document("d09", "Natural Language Processing",
        "NLP enables computers to understand, interpret, and generate human language. Key tasks include tokenization, named entity recognition, sentiment analysis, and machine translation.",
        url="https://example.com/nlp"),
    Document("d10", "Database Indexing Strategies",
        "Database indexes speed up query execution by reducing the number of rows scanned. B-tree indexes are the most common. Full-text search indexes use inverted index structures similar to search engines.",
        url="https://example.com/db-index"),
]


# ─── Demo queries ────────────────────────────────────────────────────────────

DEMO_QUERIES = [
    "inverted index search engine",
    "machine learning neural network",
    '"TF-IDF"',
    "python data science",
    "algorithm AND search NOT graph",
    '"hash table"',
    "ranking algorithm BM25",
]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mini Search Engine Demo")
    p.add_argument("--interactive", action="store_true", help="Interactive query mode")
    p.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    p.add_argument("--files", nargs="+", help="Index custom .txt files")
    p.add_argument("--mode", choices=["tfidf", "bm25"], default="bm25", help="Scoring mode")
    p.add_argument("--top-k", type=int, default=5, help="Results per query")
    p.add_argument("--no-stem", action="store_true", help="Disable stemming")
    return p.parse_args()


def section(title: str):
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print("═" * 62)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.benchmark:
        from benchmarks.benchmark import bench_index_build, bench_query_throughput, bench_vocab_growth
        bench_index_build([500, 1_000, 5_000, 10_000])
        bench_query_throughput(n_docs=10_000, n_queries=300)
        bench_vocab_growth([500, 2_000, 10_000])
        return

    # ── Build engine ──────────────────────────────────────────────────────────
    section("1. Building search engine")
    engine = SearchEngine(mode=args.mode, use_stemming=not args.no_stem)

    if args.files:
        docs = []
        for pattern in args.files:
            for path in glob.glob(pattern):
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                docs.append(Document(doc_id=path, title=os.path.basename(path), body=content))
        print(f"  Indexing {len(docs)} file(s)…")
        engine.index_documents(docs)
    else:
        print(f"  Indexing {len(SAMPLE_CORPUS)} sample documents…")
        engine.index_documents(SAMPLE_CORPUS)

    stats = engine.stats()
    print(f"\n  Documents    : {stats['total_documents']}")
    print(f"  Vocabulary   : {stats['vocabulary_size']:,} terms")
    print(f"  Avg doc len  : {stats['avg_doc_length']} tokens")
    print(f"  Index built  : {stats['build_time_seconds']}s")
    print(f"  Scoring mode : {stats['scoring_mode'].upper()}")

    # ── Demo queries ──────────────────────────────────────────────────────────
    if not args.interactive:
        section("2. Demo queries")
        for q in DEMO_QUERIES:
            results = engine.search(q, top_k=args.top_k)
            engine.print_results(results, query=q)

    # ── Interactive mode ──────────────────────────────────────────────────────
    section("3. Interactive mode" if args.interactive else "3. Interactive mode (--interactive to enable)")
    if not args.interactive:
        print("  Tip: run  python main.py --interactive  to search interactively.")
        print("\nSyntax guide:")
        print("  Simple  : python algorithm")
        print('  Phrase  : "inverted index"')
        print("  AND     : python AND search")
        print("  NOT     : search NOT graph")
        print("  Mixed   : \"inverted index\" AND python NOT java")
        return

    print("  Type your query (or 'quit' to exit)\n")
    while True:
        try:
            query = input("  🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = engine.search(query, top_k=args.top_k)
        engine.print_results(results, query=query)

    print("\n  Bye!")


if __name__ == "__main__":
    main()
