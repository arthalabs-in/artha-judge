from __future__ import annotations

import inspect
import json
import re
from copy import deepcopy
from typing import Any, Awaitable, Callable

from rag.config import DEFAULT_LLM_MODEL
from rag.judgment.retrieval import JudgmentEvidenceIndex
from rag.judgment.types import ActionItem, SourceEvidence, Timeline
from rag.llm import llm_complete

from .serialization import serialize_review_package

LOW_CONFIDENCE_THRESHOLD = 0.5
MAX_RESEARCH_SNIPPETS_PER_QUERY = 3


async def enrich_review_package_with_llm(
    review_package,
    *,
    documents: list[Any] | None = None,
    llm_callable: Callable[[str], str | Awaitable[str]] | None = None,
    model_name: str = DEFAULT_LLM_MODEL,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    logger: Any | None = None,
):
    research_context = _build_research_context(review_package, documents or [], confidence_threshold)
    prompt = _build_prompt(review_package, research_context=research_context, confidence_threshold=confidence_threshold)
    try:
        if llm_callable is None:
            response = await llm_complete(
                model_name=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON only. Do not add unsupported facts.",
                    },
                    {"role": "user", "content": prompt},
                ],
                logger=logger,
            )
        else:
            result = llm_callable(prompt)
            response = await result if inspect.isawaitable(result) else result
        payload = json.loads(response)
    except Exception as exc:
        if logger:
            logger.warning(f"LLM enrichment skipped: {exc}")
        return review_package

    enriched = deepcopy(review_package)
    context_evidence = _context_evidence_by_snippet(research_context)
    extraction_updates = payload.get("extraction", {})
    for field_name, field_payload in extraction_updates.items():
        field = getattr(enriched.extraction, field_name, None)
        if field is None or not isinstance(field_payload, dict):
            continue
        evidence = _supported_evidence(field_payload, field, context_evidence)
        if evidence is None:
            enriched.risk_flags.append(f"llm_discarded_unsupported_{field_name}")
            continue
        if "value" in field_payload:
            field.value = field_payload["value"]
        if "confidence" in field_payload:
            field.confidence = float(field_payload["confidence"])
        if "raw_value" in field_payload:
            field.raw_value = field_payload["raw_value"]
        if evidence and not _has_duplicate_evidence(field.evidence, evidence):
            field.evidence.append(evidence)
        if field_payload.get("reason") and field_payload["reason"] not in field.notes:
            field.notes.append(str(field_payload["reason"]))

    for index, action_payload in enumerate(payload.get("action_items", [])):
        if not isinstance(action_payload, dict):
            continue
        action_item = enriched.action_items[index] if index < len(enriched.action_items) else None
        evidence = _supported_evidence(action_payload, action_item, context_evidence)
        if evidence is None:
            enriched.risk_flags.append(f"llm_discarded_unsupported_action_{index}")
            continue
        if action_item is None:
            action_item = _action_item_from_payload(action_payload, index, evidence)
            enriched.action_items.append(action_item)
            continue
        for key in (
            "category",
            "priority",
            "responsible_department",
            "title",
            "legal_basis",
            "escalation_recommendation",
        ):
            if key in action_payload and action_payload[key] is not None:
                setattr(action_item, key, action_payload[key])
        if "ambiguity_flags" in action_payload and isinstance(action_payload["ambiguity_flags"], list):
            action_item.ambiguity_flags = list(dict.fromkeys(action_item.ambiguity_flags + action_payload["ambiguity_flags"]))
        if "confidence" in action_payload:
            action_item.confidence = float(action_payload["confidence"])
        if evidence and not _has_duplicate_evidence(action_item.evidence, evidence):
            action_item.evidence.append(evidence)

    return enriched


def _build_prompt(
    review_package,
    *,
    research_context: list[dict[str, Any]] | None = None,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> str:
    serialized = serialize_review_package(review_package)
    return (
        "You are filling a court judgment review packet for a human reviewer.\n"
        "Return strict JSON with optional keys 'extraction' and 'action_items'.\n"
        "When LLM enrichment is enabled, fill or correct review values using only supplied source evidence.\n"
        f"Any field/action with confidence below {confidence_threshold} is weak and should be re-checked against "
        "the retrieval_context before you answer.\n"
        "Every changed or newly added field/action must include evidence_snippet copied exactly from either the "
        "current package evidence or retrieval_context snippets.\n"
        "Do not include legal_phrases unless explicitly needed. Do not invent parties, courts, dates, owners, "
        "deadlines, or actions.\n"
        "For action_items, you may add missing action items if the judgment contains a clear direction. Use keys: "
        "title, responsible_department, category, priority, legal_basis, confidence, ambiguity_flags, "
        "timeline, evidence_snippet.\n"
        f"Current package:\n{json.dumps(serialized, ensure_ascii=True)}"
        f"\nRetrieval context:\n{json.dumps(research_context or [], ensure_ascii=True)}"
    )


def _build_research_context(review_package, documents: list[Any], confidence_threshold: float) -> list[dict[str, Any]]:
    if not documents:
        return []
    index = JudgmentEvidenceIndex.from_documents(documents)
    context: list[dict[str, Any]] = []
    queries = _research_queries(review_package, confidence_threshold)
    seen: set[tuple[str, str]] = set()
    for purpose, query in queries:
        for result in index.search(query, top_k=MAX_RESEARCH_SNIPPETS_PER_QUERY):
            key = (purpose, result.snippet)
            if key in seen:
                continue
            seen.add(key)
            context.append(
                {
                    "purpose": purpose,
                    "query": query,
                    "page": result.page,
                    "chunk_id": result.chunk_id,
                    "snippet": result.snippet,
                    "retrieval_score": result.retrieval_score,
                    "rerank_score": result.rerank_score,
                    "match_strategy": result.match_strategy,
                    "source_id": (result.document.metadata or {}).get("source") or (result.chunk_id or "judgment"),
                }
            )
    return context


def _research_queries(review_package, confidence_threshold: float) -> list[tuple[str, str]]:
    extraction = review_package.extraction
    field_queries = {
        "case_number": "case number writ petition civil appeal criminal appeal number",
        "case_type": "case type writ petition civil appeal criminal appeal",
        "court": "court name high court supreme court tribunal",
        "bench": "bench coram judges before justice",
        "judgment_date": "judgment date dated pronounced on",
        "parties": "between petitioner respondent appellant versus",
        "petitioners": "petitioner appellant applicant",
        "respondents": "respondent state government department",
        "departments": "department board authority corporation government responsible",
        "disposition": "allowed dismissed disposed quashed remanded partly allowed order",
    }
    queries: list[tuple[str, str]] = []
    for field_name, query in field_queries.items():
        field = getattr(extraction, field_name, None)
        if field is None:
            continue
        queries.append((field_name, query))
        if _needs_research(getattr(field, "value", None), getattr(field, "confidence", 0.0), confidence_threshold):
            queries.append((f"{field_name}_low_confidence", query))
    queries.append(("directions", "directed shall ordered compliance report within weeks months days"))
    for index, direction in enumerate(extraction.directions):
        if getattr(direction, "confidence", 0.0) < confidence_threshold:
            queries.append((f"direction_{index}", "directed shall ordered compliance report deadline responsible department"))
    queries.append(("action_items", "directed shall ordered compliance report deadline responsible department action"))
    return queries


def _needs_research(value: Any, confidence: float, confidence_threshold: float) -> bool:
    if confidence < confidence_threshold:
        return True
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "unknown", "not mentioned", "none"}:
        return True
    if isinstance(value, list) and not value:
        return True
    return False


def _context_evidence_by_snippet(research_context: list[dict[str, Any]]) -> dict[str, SourceEvidence]:
    evidence: dict[str, SourceEvidence] = {}
    for item in research_context:
        snippet = str(item.get("snippet") or "").strip()
        if not snippet:
            continue
        evidence[snippet.lower()] = SourceEvidence(
            source_id=str(item.get("source_id") or item.get("chunk_id") or "judgment"),
            page=item.get("page"),
            chunk_id=item.get("chunk_id"),
            snippet=snippet,
            confidence=0.86,
            extraction_method="llm_research",
            retrieval_score=item.get("retrieval_score"),
            rerank_score=item.get("rerank_score"),
            match_strategy=item.get("match_strategy"),
        )
    return evidence


def _supported_evidence(
    payload: dict[str, Any],
    target: Any | None,
    context_evidence: dict[str, SourceEvidence],
) -> SourceEvidence | None | bool:
    snippet = str(payload.get("evidence_snippet") or "").strip()
    existing_evidence = getattr(target, "evidence", []) or [] if target is not None else []
    if not snippet:
        return True if existing_evidence else None
    lowered = snippet.lower()
    for item in existing_evidence:
        if _evidence_matches(lowered, item.snippet or ""):
            return item
    for context_snippet, evidence in context_evidence.items():
        if _evidence_matches(lowered, context_snippet):
            return evidence
    return None


def _action_item_from_payload(payload: dict[str, Any], index: int, evidence: SourceEvidence | bool | None) -> ActionItem:
    evidence_items = [evidence] if isinstance(evidence, SourceEvidence) else []
    timeline_payload = payload.get("timeline")
    timeline = Timeline()
    if isinstance(timeline_payload, dict):
        timeline = Timeline(
            raw_text=timeline_payload.get("raw_text"),
            confidence=float(timeline_payload.get("confidence") or 0.0),
            timeline_type=str(timeline_payload.get("timeline_type") or "missing"),
        )
    elif isinstance(timeline_payload, str) and timeline_payload.strip():
        timeline = Timeline(raw_text=timeline_payload.strip(), confidence=0.65, timeline_type="explicit")
    return ActionItem(
        title=str(payload.get("title") or "Review judgment direction"),
        responsible_department=payload.get("responsible_department"),
        timeline=timeline,
        category=str(payload.get("category") or "compliance"),
        priority=str(payload.get("priority") or "medium"),
        legal_basis=payload.get("legal_basis") or payload.get("evidence_snippet"),
        confidence=float(payload.get("confidence") or 0.7),
        evidence=evidence_items,
        notes=["LLM-filled from re-searched evidence."],
        action_id=str(payload.get("action_id") or f"action-{index}"),
        direction_summary=payload.get("direction_summary") or payload.get("legal_basis"),
        owner_source="llm_research" if payload.get("responsible_department") else None,
        timeline_type=timeline.timeline_type,
        ambiguity_flags=list(payload.get("ambiguity_flags") or []),
        escalation_recommendation=payload.get("escalation_recommendation"),
    )


def _has_duplicate_evidence(existing: list[SourceEvidence], candidate: SourceEvidence | bool | None) -> bool:
    if not isinstance(candidate, SourceEvidence):
        return True
    return any((item.snippet or "").strip().lower() == candidate.snippet.strip().lower() for item in existing)


def _evidence_matches(requested_snippet: str, source_snippet: str) -> bool:
    source = source_snippet.lower()
    if requested_snippet in source or source in requested_snippet:
        return True
    requested_tokens = _match_tokens(requested_snippet)
    source_tokens = _match_tokens(source)
    if len(requested_tokens) < 4 or len(source_tokens) < 4:
        return False
    overlap = len(requested_tokens & source_tokens) / len(requested_tokens)
    return overlap >= 0.55


def _match_tokens(value: str) -> set[str]:
    stopwords = {"the", "and", "or", "of", "in", "to", "a", "an", "is", "are", "was", "were", "for", "on"}
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stopwords}
