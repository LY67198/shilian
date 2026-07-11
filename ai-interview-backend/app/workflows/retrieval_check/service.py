"""RetrievalCheckService — wraps self-check graph for interview integration"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import StateGraph

from app.retrieval.pipeline import RetrievalPipeline
from app.workflows.retrieval_check.graph import build_retrieval_check_graph

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCheckResult:
    """Output from self-check retrieval."""
    final_context: list[str] = field(default_factory=list)
    debug_info: dict[str, Any] = field(default_factory=dict)


class RetrievalCheckService:
    """Wraps the retrieval self-check graph.

    Usage:
        service = RetrievalCheckService(pipeline, max_retries=2)
        result = await service.check_and_retrieve("Python GIL explained")
        context_strings = result.final_context
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        max_retries: int = 2,
    ):
        self.pipeline = pipeline
        self.max_retries = max_retries
        self._graph: StateGraph | None = None

    def _get_graph(self) -> StateGraph:
        if self._graph is None:
            builder = build_retrieval_check_graph()
            self._graph = builder.compile()
        return self._graph

    async def check_and_retrieve(
        self,
        query: str,
        max_retries: int | None = None,
        retrieval_filters: dict | None = None,
    ) -> RetrievalCheckResult:
        """Run retrieval with self-check loop.

        Args:
            query: The search query (interview question text).
            max_retries: Override default max retries.
            retrieval_filters: Optional filters forwarded to the pipeline.

        Returns:
            RetrievalCheckResult with final_context and debug_info.
        """
        max_retries = max_retries if max_retries is not None else self.max_retries
        graph = self._get_graph()

        initial_state = {
            "query": query,
            "current_query": query,
            "retry_count": 0,
            "custom": {
                "pipeline": self.pipeline,
                "max_retries": max_retries,
                "retrieval_filters": retrieval_filters,
            },
        }

        try:
            result = await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"Self-check graph failed: {e}")
            return RetrievalCheckResult(
                final_context=[],
                debug_info={"error": str(e)},
            )

        return RetrievalCheckResult(
            final_context=result.get("final_context", []),
            debug_info=result.get("debug_info", {}),
        )
