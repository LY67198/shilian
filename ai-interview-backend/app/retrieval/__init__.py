"""Hybrid retrieval pipeline — vector + BM25 + RRF + rerank

Usage:
    from app.retrieval import SearchResult
    from app.retrieval.pipeline import RetrievalPipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """Normalized search result across all retrieval stages."""

    id: int
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "vector"  # "vector" | "bm25" | "both"
