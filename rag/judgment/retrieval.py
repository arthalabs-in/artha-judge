from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.documents import Document


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


@dataclass
class EvidenceSearchResult:
    page: int | None
    chunk_id: str | None
    snippet: str
    retrieval_score: float
    rerank_score: float
    match_strategy: str
    document: Document


class JudgmentEvidenceIndex:
    """Small hybrid evidence index for a single judgment."""

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._tokens = [_tokens(doc.page_content or "") for doc in documents]
        self._idf = _idf(self._tokens)

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> "JudgmentEvidenceIndex":
        return cls([doc for doc in documents if (doc.page_content or "").strip()])

    def search(self, query: str, *, top_k: int = 5) -> list[EvidenceSearchResult]:
        query_tokens = _tokens(query)
        scored: list[EvidenceSearchResult] = []
        for document, doc_tokens in zip(self.documents, self._tokens):
            lexical = _bm25_like(query_tokens, doc_tokens, self._idf)
            dense = _token_overlap(query_tokens, doc_tokens)
            score = (0.7 * lexical) + (0.3 * dense)
            if score <= 0:
                continue
            metadata = document.metadata or {}
            strategy = "hybrid_fused"
            if lexical > dense * 1.5:
                strategy = "hybrid_lexical"
            elif dense > lexical * 1.5:
                strategy = "hybrid_dense"
            scored.append(
                EvidenceSearchResult(
                    page=metadata.get("page"),
                    chunk_id=metadata.get("chunk_id"),
                    snippet=_best_snippet(document.page_content or "", query_tokens),
                    retrieval_score=round(score, 4),
                    rerank_score=round(score + _legal_boost(document.page_content or ""), 4),
                    match_strategy=strategy,
                    document=document,
                )
            )
        return sorted(scored, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def _tokens(text: str) -> list[str]:
    return [item.lower() for item in _TOKEN_RE.findall(text)]


def _idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    doc_count = max(1, len(tokenized_docs))
    seen: dict[str, int] = {}
    for tokens in tokenized_docs:
        for token in set(tokens):
            seen[token] = seen.get(token, 0) + 1
    return {token: math.log((doc_count + 1) / (count + 0.5)) + 1 for token, count in seen.items()}


def _bm25_like(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    frequencies: dict[str, int] = {}
    for token in doc_tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
    score = 0.0
    doc_len = len(doc_tokens)
    for token in query_tokens:
        freq = frequencies.get(token, 0)
        if not freq:
            continue
        score += idf.get(token, 1.0) * ((freq * 2.2) / (freq + 1.2 * (0.25 + 0.75 * doc_len / 120)))
    return score


def _token_overlap(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    return len(query_set & doc_set) / len(query_set)


def _legal_boost(text: str) -> float:
    return 0.25 if re.search(r"\b(?:directed|shall|within|compliance|appeal|report)\b", text, re.I) else 0.0


def _best_snippet(text: str, query_tokens: list[str], *, limit: int = 260) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    lowered = clean.lower()
    positions = [lowered.find(token) for token in query_tokens if lowered.find(token) >= 0]
    start = max(0, min(positions) - 70) if positions else 0
    return clean[start : start + limit].strip()
