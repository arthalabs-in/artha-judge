"""
rag/ - Modularized implementation extracted from rag_core.py.

This package is intentionally NOT wired into rag_core.py yet.
It exists to enable an incremental refactor while keeping current
call sites stable until we switch the facade.
"""

from .config import DEFAULT_LLM_MODEL, PLANNER_LLM_MODEL
from .types import RAGResult

__all__ = [
    "DEFAULT_LLM_MODEL",
    "PLANNER_LLM_MODEL",
    "RAGResult",
]

