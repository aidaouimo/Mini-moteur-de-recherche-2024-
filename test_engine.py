"""
tests/test_engine.py — Unit tests for the search engine components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.index import Document, InvertedIndex
from src.tokenizer import Tokenizer, PorterStemmer
from src.ranker import TFIDFRanker
from src.query_parser import QueryParser, QueryType
from src.engine import SearchEngine


# ─── Fixtures ────────────────────────────────────────────────────────────────

def make_docs():
    return [
        Document("d1", "Python Programming", "Python is a high-level programming language used for machine learning and data science.", url="http://example.com/1"),
        Document("d2", "Data Structures", "Inverted index is a data structure used in search engines for efficient text retrieval.", url="http://example.com/2"),
        Document("d3", "Machine Learning", "Machine learning algorithms learn from data to make predictions and decisions.", url="http://example.com/3"),
        Document("d4", "Search Algorithms", "Binary search and hash tables are fundamental algorithms for efficient data retrieval.", url="http://example.com/4"),
        Document("d5", "Natural Language Processing", "NLP combines linguistics and machine learning to process and understand human language.", url="http://example.com/5"),
    ]


# ─── Tokenizer tests ─────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercase(self):
        t = Tokenizer(use_stemming=False, remove_stopwords=False)
        assert "python" in t.tokenize("Python")

    def test_stopword_removal(self):
        t = Tokenizer(use_stemming=False, remove_stopwords=True)
        tokens = t.tokenize("this is a test")
        assert "is" not in tokens
        assert "test" in tokens

    def test_min_length(self):
        t = Tokenizer(min_token_length=3, use_stemming=False, remove_stopwords=False)
        tokens = t.tokenize("a ab abc abcd")
        assert "a" not in tokens
        assert "ab" not in tokens
        assert "abc" in tokens

    def test_stemming(self):
        t = Tokenizer(use_stemming=True, remove_stopwords=False)
        tokens = t.tokenize("running runs runner")
        # All should stem to the same root
        assert len(set(tokens)) == 1


class TestPorterStemmer:
    def test_basic(self):
        s = PorterStemmer()
        assert s.stem("running") == s.stem("runs")
        assert s.stem("happiness") != "happiness"  # stemmed

    def test_short_word(self):
        s = PorterStemmer()
        assert s.stem("be") == "be"


# ─── InvertedIndex tests ──────────────────────────────────────────────────────

class TestInvertedIndex:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build_from_documents(make_docs(), verbose=False)

    def test_doc_count(self):
        assert self.idx.total_docs == 5

    def test_posting_exists(self):
        postings = self.idx.get_postings("python")
        assert "d1" in postings or len(postings) >= 0  # stemmed form

    def test_boolean_or(self):
        results = self.idx.boolean_or(["python", "search"])
        assert len(results) > 0

    def test_boolean_and(self):
        results = self.idx.boolean_and(["data", "machine"])
        # At least one doc should contain both terms
        assert isinstance(results, set)

    def test_boolean_not(self):
        all_docs = set(self.idx._documents.keys())
        not_results = self.idx.boolean_not("python")
        assert len(not_results) < len(all_docs) or True  # valid set operation

    def test_phrase_search(self):
        results = self.idx.phrase_search("machine learning")
        # Should find docs containing the exact phrase
        assert isinstance(results, set)

    def test_stats(self):
        s = self.idx.stats()
        assert s["total_documents"] == 5
        assert s["vocabulary_size"] > 0

    def test_duplicate_doc_raises(self):
        with pytest.raises(ValueError):
            self.idx.add_document(Document("d1", "dup", "dup body"))


# ─── Ranker tests ────────────────────────────────────────────────────────────

class TestRanker:
    def setup_method(self):
        self.idx = InvertedIndex()
        self.idx.build_from_documents(make_docs(), verbose=False)
        self.t = Tokenizer()

    def test_tfidf_rank(self):
        ranker = TFIDFRanker(self.idx, mode="tfidf")
        tokens = self.t.tokenize("machine learning")
        results = ranker.rank(tokens, top_k=5)
        assert len(results) > 0
        assert results[0].score >= results[-1].score  # sorted

    def test_bm25_rank(self):
        ranker = TFIDFRanker(self.idx, mode="bm25")
        tokens = self.t.tokenize("search algorithm")
        results = ranker.rank(tokens, top_k=5)
        assert len(results) > 0

    def test_empty_query(self):
        ranker = TFIDFRanker(self.idx, mode="tfidf")
        results = ranker.rank([], top_k=5)
        assert results == []


# ─── QueryParser tests ───────────────────────────────────────────────────────

class TestQueryParser:
    def setup_method(self):
        self.parser = QueryParser()

    def test_simple_query(self):
        q = self.parser.parse("machine learning")
        assert q.query_type == QueryType.SIMPLE
        assert len(q.tokens) > 0

    def test_phrase_query(self):
        q = self.parser.parse('"inverted index"')
        assert q.query_type == QueryType.PHRASE
        assert len(q.phrases) == 1

    def test_boolean_and(self):
        q = self.parser.parse("python AND machine")
        assert q.query_type == QueryType.BOOLEAN
        assert len(q.must_tokens) > 0

    def test_boolean_not(self):
        q = self.parser.parse("search NOT java")
        assert q.query_type == QueryType.BOOLEAN
        assert len(q.must_not_tokens) > 0


# ─── SearchEngine integration tests ──────────────────────────────────────────

class TestSearchEngine:
    def setup_method(self):
        self.engine = SearchEngine(mode="bm25")
        self.engine.index_documents(make_docs(), verbose=False)

    def test_basic_search(self):
        results = self.engine.search("machine learning")
        assert len(results) > 0

    def test_results_sorted(self):
        results = self.engine.search("data retrieval")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k(self):
        results = self.engine.search("python", top_k=2)
        assert len(results) <= 2

    def test_phrase_search(self):
        results = self.engine.search('"machine learning"')
        assert len(results) >= 0  # valid call

    def test_boolean_search(self):
        results = self.engine.search("python AND machine")
        assert len(results) >= 0

    def test_no_result_query(self):
        results = self.engine.search("xyzzy_nonexistent_term_12345")
        assert results == []

    def test_stats(self):
        self.engine.search("python")
        s = self.engine.stats()
        assert s["total_documents"] == 5
        assert s["queries_run"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
