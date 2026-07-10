"""BM25 keyword index with character bigram tokenizer"""
from __future__ import annotations

from typing import Optional

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 keyword search index using character bigram tokenization.

    Usage::

        idx = BM25Index("my_collection")
        idx.build(corpus)
        results = idx.search("query text", top_k=20)
    """

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._corpus: list[str] = []
        self._index: Optional[BM25Okapi] = None

    def build(self, texts: list[str] | None) -> None:
        """Build BM25Okapi index from a corpus of texts.

        Args:
            texts: List of text documents. None or empty list clears the index.
        """
        if not texts:
            self._corpus = []
            self._index = None
            return

        self._corpus = list(texts)
        tokenized = [self._tokenize(t) for t in self._corpus]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search the index and return top_k (corpus_index, score) pairs.

        Args:
            query: Raw query string.
            top_k: Maximum number of results.

        Returns:
            List of (corpus_index, bm25_score) sorted by score descending.
            Empty list if the index is not built.
        """
        if self._index is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Pair (index, score) and sort by score descending
        scored = [(i, float(s)) for i, s in enumerate(scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored[:top_k]

    def get_text(self, idx: int) -> str:
        """Get original corpus text by index.

        Args:
            idx: Corpus index.

        Returns:
            Original text string, or empty string if index is out of bounds.
        """
        if 0 <= idx < len(self._corpus):
            return self._corpus[idx]
        return ""

    @property
    def corpus_size(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self._corpus)

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into character bigrams.

        For text shorter than 2 characters, returns the characters as-is
        to avoid producing an empty token list that would break BM25Okapi.

        Args:
            text: Raw text to tokenize.

        Returns:
            List of character bigram tokens.
        """
        chars = list(text)
        if len(chars) < 2:
            return chars
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
