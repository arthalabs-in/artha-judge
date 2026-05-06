"""Configuration/constants extracted from rag_core.py."""

from __future__ import annotations

import os

# Keep env var names aligned with current rag_core.py behavior.
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL") or os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
PLANNER_LLM_MODEL = os.getenv("PLANNER_MODEL_NAME", "llama-3.1-8b-instant")

# LLM rate limiting defaults (mirrors rag_core.py).
DEFAULT_REQUESTS_PER_MINUTE = int(os.getenv("RAG_LLM_RPM", "30"))
DEFAULT_TOKENS_PER_MINUTE = int(os.getenv("RAG_LLM_TPM", "14000"))
