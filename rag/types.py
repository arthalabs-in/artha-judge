"""Type definitions extracted from rag_core.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RAGResult:
    """
    Container for a single RAG result.

    Field names are kept compatible with the existing `rag_core.RAGResult`
    usage in `main.py` and related integrations.
    """

    response: str
    source_documents: List[str]
    retrieval_time_s: float
    rerank_time_s: float
    llm_generation_time_s: float
    total_response_time_s: float

    # Agentic execution details (optional).
    execution_plan: Optional[List[Dict[str, Any]]] = None
    steps_executed: Optional[int] = None
    scratchpad_variables: Optional[List[str]] = None
    complexity_score: Optional[int] = None

