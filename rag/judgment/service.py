from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .action_plan import build_action_plan
from .extractor import extract_judgment_details
from .retrieval import JudgmentEvidenceIndex
from .types import JudgmentReviewPackage


def build_judgment_review_package(
    documents: list[Document],
    case_metadata: dict[str, Any] | None = None,
    llm_client: Any | None = None,
) -> JudgmentReviewPackage:
    del llm_client  # Reserved for the later strict-JSON extraction adapter.

    extraction = extract_judgment_details(documents)
    action_items = build_action_plan(extraction)
    _attach_retrieval_scores(extraction, action_items, documents)
    risk_flags = list(extraction.risk_flags)

    if not action_items:
        risk_flags.append("missing_action_items")

    return JudgmentReviewPackage(
        extraction=extraction,
        action_items=action_items,
        source_metadata=dict(case_metadata or {}),
        overall_confidence=_overall_confidence(extraction, action_items),
        risk_flags=risk_flags,
    )


def _overall_confidence(extraction, action_items) -> float:
    scores = [
        extraction.case_number.confidence,
        extraction.court.confidence,
        extraction.judgment_date.confidence,
        extraction.parties.confidence,
    ]
    scores.extend(item.confidence for item in action_items)
    non_zero_scores = [score for score in scores if score]
    if not non_zero_scores:
        return 0.0
    return round(sum(non_zero_scores) / len(non_zero_scores), 2)


def _attach_retrieval_scores(extraction, action_items, documents: list[Document]) -> None:
    index = JudgmentEvidenceIndex.from_documents(documents)
    fields = [
        extraction.case_number,
        extraction.case_type,
        extraction.court,
        extraction.bench,
        extraction.judgment_date,
        extraction.parties,
        extraction.petitioners,
        extraction.respondents,
        extraction.departments,
        extraction.advocates,
        extraction.disposition,
        extraction.legal_phrases,
        *extraction.directions,
    ]
    for field in fields:
        _tag_evidence(index, str(field.raw_value or field.value or ""), field.evidence, "layered_field")
    for action in action_items:
        _tag_evidence(index, str(action.legal_basis or action.title or ""), action.evidence, "action_plan")


def _tag_evidence(index: JudgmentEvidenceIndex, query: str, evidence_items, layer: str) -> None:
    if not query.strip() or not evidence_items:
        return
    result = next(iter(index.search(query, top_k=1)), None)
    for evidence in evidence_items:
        evidence.layer = evidence.layer or layer
        if result:
            evidence.retrieval_score = result.retrieval_score
            evidence.rerank_score = result.rerank_score
            evidence.match_strategy = evidence.match_strategy or result.match_strategy
            evidence.page_role = evidence.page_role or str((result.document.metadata or {}).get("page_role") or "")
