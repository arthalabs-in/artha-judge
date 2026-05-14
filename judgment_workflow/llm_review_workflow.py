from __future__ import annotations

import inspect
import json
import re
from datetime import date
from time import perf_counter
from typing import Any, Awaitable, Callable

from langchain_core.documents import Document

from rag.config import DEFAULT_LLM_MODEL
from rag.judgment import build_judgment_review_package
from rag.judgment.action_classifier import classify_action, priority_for_action
from rag.judgment.evidence import normalize_evidence
from rag.judgment.owner_resolution import apply_inferred_action_owner, remand_owner_from_text
from rag.judgment.retrieval import JudgmentEvidenceIndex
from rag.judgment.types import (
    ActionItem,
    ExtractedField,
    JudgmentExtraction,
    JudgmentReviewPackage,
    SourceEvidence,
    Timeline,
)
from rag.llm import llm_complete

from .serialization import serialize_review_package


LLMCallable = Callable[[str], str | Awaitable[str]]
CASE_DETAIL_QUERY = (
    "case number court bench parties petitioner respondent appellant date judgment order "
    "directions ordered directed shall within timeline compliance"
)


async def build_llm_first_review_package(
    documents: list[Document],
    case_metadata: dict[str, Any] | None = None,
    *,
    pdf_profile: dict[str, Any] | None = None,
    llm_callable: LLMCallable | None = None,
    model_name: str = DEFAULT_LLM_MODEL,
    logger: Any | None = None,
) -> JudgmentReviewPackage:
    """Build the reviewer package with LLM output as the primary source of truth."""

    stage_timings: list[dict[str, Any]] = []

    def record_stage(stage: str, stage_started_at: float, **metadata: Any) -> None:
        entry: dict[str, Any] = {
            "stage": stage,
            "duration_ms": round((perf_counter() - stage_started_at) * 1000),
        }
        if metadata:
            entry["metadata"] = metadata
        stage_timings.append(entry)

    async def timed_llm_json(stage: str, prompt: str) -> dict[str, Any]:
        stage_started_at = perf_counter()
        payload = await _call_llm_json(
            prompt,
            stage_name=stage,
            stage_events=stage_timings,
            llm_callable=llm_callable,
            model_name=model_name,
            logger=logger,
        )
        record_stage(stage, stage_started_at, returned_keys=sorted(payload.keys()) if payload else [])
        return payload

    stage_started_at = perf_counter()
    deterministic_package = build_judgment_review_package(documents, case_metadata)
    page_count = _page_count(documents, pdf_profile)
    precontext = _documents_to_context(_select_pages(documents, start=1, end=1), label="first_page")
    deterministic_hints = _safe_package_hints(deterministic_package)
    context_plan = _default_context_plan(page_count)
    context_plan_source = "deterministic_default"
    agentic_context = _agentic_pdf_context(documents, context_plan, page_count)
    record_stage("llm_context_prep", stage_started_at, page_count=page_count, agentic_context_count=len(agentic_context))

    stage_started_at = perf_counter()
    detail_context = _top_reranked_context(documents, CASE_DETAIL_QUERY, top_k=10)
    record_stage("llm_reranked_context", stage_started_at, top_k=len(detail_context))

    last_pages = _select_last_pages(documents, page_count, count=3)
    action_context = _documents_to_context(last_pages, label="last_3_pages", limit=None)
    action_supporting_context = _action_supporting_context(agentic_context)
    llm_context = {
        "first_page": precontext,
        "evidence_context": detail_context,
        "agentic_context": agentic_context,
        "action_context": action_context,
        "supporting_context": action_supporting_context,
        "deterministic_hints": deterministic_hints,
    }
    llm_outputs: dict[str, Any] = {}

    case_details_source = "llm"
    if _can_reuse_vision_case_details(deterministic_package, documents, case_metadata or {}):
        case_details_source = "deterministic_vision_ocr"
        details_payload = _details_payload_from_package(deterministic_package)
        llm_outputs["case_details_skipped"] = {
            "reason": "strong_vision_ocr_fields",
            "source": "deterministic_vision_ocr",
        }
    else:
        details_payload = await timed_llm_json(
            "llm_case_details",
            _case_details_prompt(
                precontext=precontext,
                evidence_context=detail_context,
                agentic_context=agentic_context,
                deterministic_hints=deterministic_hints,
                source_metadata=case_metadata or {},
                pdf_profile=pdf_profile or {},
            ),
        )
        if not details_payload:
            details_payload = await timed_llm_json(
                "llm_case_details_repair",
                _case_details_repair_prompt(
                    precontext=precontext,
                    agentic_context=agentic_context,
                    deterministic_hints=deterministic_hints,
                    source_metadata=case_metadata or {},
                    pdf_profile=pdf_profile or {},
                ),
            )
    llm_outputs["case_details_initial"] = json.loads(json.dumps(details_payload)) if details_payload else {}

    for field_name in _missing_fields_to_repair(deterministic_package, details_payload):
        repair_context = _field_repair_context(field_name, deterministic_package, llm_context)
        llm_context.setdefault("field_repair_context", {})[field_name] = repair_context
        repair_payload = await timed_llm_json(
            f"llm_case_detail_repair_{field_name}",
            _case_detail_field_repair_prompt(
                field_name=field_name,
                current_details=details_payload,
                deterministic_value=getattr(deterministic_package.extraction, field_name).value,
                repair_context=repair_context,
            ),
        )
        llm_outputs.setdefault("field_repairs", {})[field_name] = repair_payload
        _apply_field_repair(details_payload, field_name, repair_payload, deterministic_package)

    llm_outputs["case_details"] = details_payload

    action_payload = await timed_llm_json(
        "llm_action_plan",
        _action_plan_prompt(
            action_context=action_context,
            supporting_context=action_supporting_context,
            case_details=details_payload,
        ),
    )
    llm_outputs["action_plan"] = action_payload

    if _needs_more_action_context(action_payload):
        previous_pages = _select_previous_pages(documents, page_count, before_pages=3, count=3)
        if previous_pages:
            previous_context = _documents_to_context(previous_pages, label="previous_3_pages", limit=None)
            llm_context["previous_action_context"] = previous_context
            action_payload = await timed_llm_json(
                "llm_action_plan_second_pass",
                _action_plan_prompt(
                    action_context=previous_context,
                    supporting_context=action_supporting_context,
                    case_details=details_payload,
                    previous_context_summary=str(action_payload.get("context_summary") or ""),
                    second_pass=True,
                ),
            )
            llm_outputs["second_pass_action_plan"] = action_payload

    forced_action_pass = False
    if not action_payload.get("action_items"):
        forced_action_pass = True
        forced_context = _documents_to_context(_select_final_pages(documents, page_count, count=6), label="final_6_pages", limit=None)
        llm_context["forced_action_context"] = forced_context
        action_payload = await timed_llm_json(
            "llm_action_plan_forced_decision",
            _forced_action_plan_prompt(
                action_context=forced_context,
                supporting_context=action_supporting_context,
                case_details=details_payload,
                previous_payload=action_payload,
            ),
        )
        llm_outputs["forced_action_plan"] = action_payload

    if not action_payload.get("action_items"):
        action_payload = _action_payload_from_package(deterministic_package)
        llm_outputs["deterministic_action_fallback"] = action_payload
        llm_outputs["action_plan"] = action_payload

    if _action_payload_needs_repair(action_payload):
        repair_context = _action_repair_context(documents, page_count)
        llm_context["action_owner_repair_context"] = repair_context
        repair_payload = await timed_llm_json(
            "llm_action_owner_timeline_repair",
            _action_owner_repair_prompt(
                current_action_payload=action_payload,
                case_details=details_payload,
                repair_context=repair_context,
            ),
        )
        llm_outputs["action_owner_timeline_repair"] = repair_payload
        action_payload = _merge_action_repair_payload(action_payload, repair_payload)
        llm_outputs["action_plan"] = action_payload

    if not _details_payload_has_case_signal(details_payload):
        return _llm_failure_fallback_package(
            deterministic_package,
            source_metadata=case_metadata or {},
            reason="llm_case_details_failed",
            review_mode="deterministic_after_llm_case_detail_failure",
            stage_timings=stage_timings,
            llm_context=llm_context,
            llm_outputs=llm_outputs,
            model_name=model_name,
        )
    if not action_payload:
        return _llm_failure_fallback_package(
            deterministic_package,
            source_metadata=case_metadata or {},
            reason="llm_action_plan_failed",
            review_mode="deterministic_after_llm_action_plan_failure",
            stage_timings=stage_timings,
            llm_context=llm_context,
            llm_outputs=llm_outputs,
            model_name=model_name,
        )

    stage_started_at = perf_counter()
    package = _package_from_llm_payloads(
        details_payload=details_payload,
        action_payload=action_payload,
        documents=documents,
        fallback_package=deterministic_package,
        source_metadata=case_metadata or {},
    )
    record_stage("llm_package_assembly", stage_started_at)
    package.source_metadata.update(
        {
            "llm_review_mode": "llm_first",
            "llm_enabled": True,
            "llm_used": True,
            "llm_model": model_name,
            "llm_agentic_query_count": len(_context_plan_queries(context_plan)),
            "llm_agentic_context_count": len(agentic_context),
            "llm_context_plan_source": context_plan_source,
            "llm_case_details_source": case_details_source,
            "llm_case_detail_top_k": 10,
            "llm_action_pages": [doc.metadata.get("page") for doc in last_pages],
            "llm_action_second_pass": bool(action_payload.get("second_pass_used")),
            "llm_action_forced_pass": forced_action_pass,
            "llm_stage_timings": stage_timings,
        }
    )
    package.source_metadata["extraction_debug"] = _build_extraction_debug_trace(
        deterministic_package=deterministic_package,
        final_package=package,
        llm_context=llm_context,
        llm_outputs=llm_outputs,
    )
    return package


def _details_payload_has_case_signal(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    case_details = payload.get("case_details")
    if isinstance(case_details, dict):
        for key in ("case_number", "court", "bench", "disposition"):
            if case_details.get(key) not in (None, "", []):
                return True
    date_payload = payload.get("date_of_order")
    if isinstance(date_payload, dict) and date_payload.get("value"):
        return True
    parties = payload.get("parties_involved")
    if isinstance(parties, dict):
        for key in ("petitioners", "respondents", "other_parties"):
            if parties.get(key):
                return True
    return bool(payload.get("key_directions_orders"))


def _can_reuse_vision_case_details(
    package: JudgmentReviewPackage,
    documents: list[Document],
    source_metadata: dict[str, Any],
) -> bool:
    if not str(source_metadata.get("ocr_routing") or "").startswith("vision_ocr"):
        return False
    if not any(
        str((doc.metadata or {}).get("extraction_method") or "").startswith("vision")
        or (doc.metadata or {}).get("source_quality") == "vision_ocr"
        for doc in documents
    ):
        return False
    extraction = package.extraction
    required_fields = (
        extraction.case_number,
        extraction.court,
        extraction.bench,
        extraction.judgment_date,
        extraction.disposition,
    )
    if any(_is_empty_value(field.value) for field in required_fields):
        return False
    if str(extraction.disposition.value or "").strip().lower() == "unknown":
        return False
    return bool(extraction.directions)


def _details_payload_from_package(package: JudgmentReviewPackage) -> dict[str, Any]:
    extraction = package.extraction
    case_evidence = _best_field_snippet(
        extraction.case_number,
        extraction.court,
        extraction.bench,
        extraction.disposition,
    )
    parties_evidence = _best_field_snippet(extraction.petitioners, extraction.respondents, extraction.parties)
    date_evidence = _best_field_snippet(extraction.judgment_date)
    return {
        "case_details": {
            "case_number": serialize_value_for_debug(extraction.case_number.value),
            "case_type": serialize_value_for_debug(extraction.case_type.value),
            "court": serialize_value_for_debug(extraction.court.value),
            "bench": _as_list(extraction.bench.value),
            "departments": _as_list(extraction.departments.value),
            "advocates": _as_list(extraction.advocates.value),
            "disposition": serialize_value_for_debug(extraction.disposition.value),
            "evidence_snippet": case_evidence,
            "bench_evidence_snippet": _best_field_snippet(extraction.bench),
            "disposition_evidence_snippet": _best_field_snippet(extraction.disposition),
        },
        "date_of_order": {
            "value": serialize_value_for_debug(extraction.judgment_date.value),
            "raw_text": serialize_value_for_debug(extraction.judgment_date.raw_value or extraction.judgment_date.value),
            "confidence": extraction.judgment_date.confidence,
            "evidence_snippet": date_evidence,
        },
        "parties_involved": {
            "petitioners": _as_list(extraction.petitioners.value),
            "respondents": _as_list(extraction.respondents.value),
            "other_parties": [],
            "evidence_snippet": parties_evidence,
        },
        "key_directions_orders": [
            {
                "text": str(direction.value),
                "confidence": direction.confidence,
                "evidence_snippet": _best_field_snippet(direction),
                "source_page": direction.evidence[0].page if direction.evidence else None,
            }
            for direction in extraction.directions
            if not _is_empty_value(direction.value)
        ],
        "relevant_timelines": [
            {"text": str(item.get("text")), "confidence": 0.72, "evidence_snippet": str(item.get("text"))}
            for item in _as_list(extraction.legal_phrases.value)
            if isinstance(item, dict) and item.get("type") == "timeline" and item.get("text")
        ],
        "confidence": max(
            _safe_float(extraction.case_number.confidence),
            _safe_float(extraction.court.confidence),
            _safe_float(extraction.disposition.confidence),
        ),
    }


def _best_field_snippet(*fields: ExtractedField) -> str | None:
    for field in fields:
        for evidence in getattr(field, "evidence", []) or []:
            if evidence.snippet:
                return evidence.snippet
    return None


def _llm_failure_fallback_package(
    package: JudgmentReviewPackage,
    *,
    source_metadata: dict[str, Any],
    reason: str,
    review_mode: str,
    stage_timings: list[dict[str, Any]],
    llm_context: dict[str, Any],
    llm_outputs: dict[str, Any],
    model_name: str,
) -> JudgmentReviewPackage:
    package.source_metadata.update(source_metadata)
    package.source_metadata.update(
        {
            "llm_review_mode": review_mode,
            "llm_enabled": True,
            "llm_used": False,
            "llm_model": model_name,
            "llm_fallback_reason": reason,
            "llm_stage_timings": stage_timings,
            "extraction_debug": {
                "llm_context": llm_context,
                "llm_outputs": llm_outputs,
                "deterministic_package": serialize_review_package(package),
            },
        }
    )
    if reason not in package.risk_flags:
        package.risk_flags.append(reason)
    return package


async def _call_llm_json(
    prompt: str,
    *,
    stage_name: str = "llm_json",
    stage_events: list[dict[str, Any]] | None = None,
    llm_callable: LLMCallable | None,
    model_name: str,
    logger: Any | None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        request_prompt = prompt
        if attempt:
            request_prompt = (
                "Your previous response was not valid JSON. Return exactly one JSON object, "
                "with no markdown, no explanation, and no trailing text.\n\n"
                + prompt
            )
        try:
            if llm_callable is None:
                transport_events: list[dict[str, Any]] = []
                response = await llm_complete(
                    model_name=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return strict JSON only. Use only supplied judgment text.",
                        },
                        {"role": "user", "content": request_prompt},
                    ],
                    temperature=0.1,
                    telemetry_callback=transport_events.append,
                    logger=logger,
                )
                if stage_events is not None:
                    for event in transport_events:
                        event = dict(event)
                        event["logical_stage_name"] = stage_name
                        event["json_attempt_index"] = attempt
                        stage_events.append(event)
            else:
                result = llm_callable(request_prompt)
                response = await result if inspect.isawaitable(result) else result
            parse_started_at = perf_counter()
            payload = _parse_llm_json_object(response)
            parse_s = perf_counter() - parse_started_at
            if stage_events is not None:
                stage_events.append(
                    {
                        "stage": "llm_json_parse",
                        "logical_stage_name": stage_name,
                        "duration_ms": round(parse_s * 1000),
                        "json_attempt_index": attempt,
                        "json_parse_s": parse_s,
                        "json_valid_dict": isinstance(payload, dict),
                        "response_chars": len(response) if isinstance(response, str) else 0,
                    }
                )
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            last_error = exc
            if stage_events is not None:
                stage_events.append(
                    {
                        "stage": "llm_json_parse",
                        "logical_stage_name": stage_name,
                        "duration_ms": 0,
                        "json_attempt_index": attempt,
                        "json_parse_s": 0.0,
                        "json_valid_dict": False,
                        "exception_type": exc.__class__.__name__,
                        "timeout": "timeout" in exc.__class__.__name__.lower() or "timed out" in str(exc).lower(),
                    }
                )
            if logger:
                logger.warning(f"LLM-first review step returned invalid JSON: {exc}")
    if logger and last_error:
        logger.warning(f"LLM-first review step skipped: {last_error}")
    return {}


def _parse_llm_json_object(response: Any) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("empty LLM response")
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("LLM response JSON root was not an object")
    return payload


def _context_request_prompt(
    *,
    precontext: list[dict[str, Any]],
    deterministic_hints: dict[str, Any],
    source_metadata: dict[str, Any],
    pdf_profile: dict[str, Any],
) -> str:
    return (
        "You are planning what additional text to retrieve from this uploaded judgment PDF before extraction.\n"
        "You may only request text from this PDF. Ask for narrow searches/pages that improve extraction accuracy, "
        "action-plan quality, and source verification.\n"
        "Return strict JSON with this shape:\n"
        "{\n"
        '  "queries": [{"query": "...", "reason": "..."}],\n'
        '  "page_requests": [{"pages": "first_2|last_3|page:5|pages:3-5", "reason": "..."}]\n'
        "}\n"
        "Rules:\n"
        "- Use at most 5 queries and at most 4 page_requests.\n"
        "- Include searches for final operative order/disposition, directions/deadlines, parties/date, "
        "and government owner/department when useful.\n"
        "- Do not request the full PDF.\n"
        f"source_metadata={json.dumps(source_metadata, ensure_ascii=True)}\n"
        f"pdf_profile={json.dumps(pdf_profile, ensure_ascii=True)}\n"
        f"first_page={json.dumps(precontext, ensure_ascii=True)}\n"
        f"deterministic_hints={json.dumps(deterministic_hints, ensure_ascii=True)}"
    )


def _default_context_plan(page_count: int) -> dict[str, Any]:
    """Cheap in-PDF search plan that avoids a latency-heavy planning LLM call."""

    return {
        "queries": [
            {
                "query": "case number court bench petitioner respondent appellant judgment date order",
                "reason": "Find stable case identity and parties.",
            },
            {
                "query": "final operative order allowed dismissed disposed quashed set aside remanded",
                "reason": "Find the final court outcome.",
            },
            {
                "query": "directed ordered shall within weeks days compliance report consider application",
                "reason": "Find directions, timelines, and conditional duties.",
            },
            {
                "query": "state government department officer authority police municipal corporation tribunal respondent",
                "reason": "Find possible public owners and departments.",
            },
        ],
        "page_requests": [
            {"pages": "first_2", "reason": "Header and cause-title context."},
            {"pages": "last_3", "reason": "Operative order and relief context."},
        ],
    }


def _case_details_prompt(
    *,
    precontext: list[dict[str, Any]],
    evidence_context: list[dict[str, Any]],
    agentic_context: list[dict[str, Any]],
    deterministic_hints: dict[str, Any],
    source_metadata: dict[str, Any],
    pdf_profile: dict[str, Any],
) -> str:
    return (
        "You are extracting only high-confidence case metadata from an Indian court judgment.\n"
        "The first_page context is for orientation only. The evidence_context contains the top 10 reranked "
        "source chunks. agentic_context contains extra in-PDF text retrieved after deterministic search planning. "
        "Deterministic hints are untrusted hints, not facts.\n"
        "Return strict JSON with this shape:\n"
        "{\n"
        '  "case_details": {"case_number": null, "case_type": null, "court": null, "bench": [], '
        '"departments": [], "responsible_entities": [], "advocates": [], "disposition": null, "evidence_snippet": null},\n'
        '  "date_of_order": {"value": null, "raw_text": null, "confidence": 0.0, "evidence_snippet": null},\n'
        '  "parties_involved": {"petitioners": [], "respondents": [], "other_parties": [], "evidence_snippet": null},\n'
        '  "key_directions_orders": [{"text": "...", "confidence": 0.0, "evidence_snippet": "..."}],\n'
        '  "relevant_timelines": [{"text": "...", "confidence": 0.0, "evidence_snippet": "..."}],\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "- Prefer the actual cause title/header for case number, court, bench, parties, and date.\n"
        "- Do not treat issue summaries, headnotes, submissions, or cited cases as the court/date/parties.\n"
        "- departments/responsible_entities must be actual government/public bodies, tribunals, authorities, "
        "or officers involved in the case/order; "
        "never extract random legal phrases or cited authorities as departments.\n"
        "- If no public owner is identifiable, return departments=[] and responsible_entities=[] with low confidence; "
        "do not borrow a private party or cited authority.\n"
        "- disposition should be the final result such as allowed, dismissed, disposed, quashed, remanded, "
        "partly_allowed, leave_granted, or unknown.\n"
        "- Every non-null/non-empty value must be traceable to an evidence_snippet copied from supplied context.\n"
        "- If evidence is weak, return null/[] with low confidence instead of guessing.\n"
        f"source_metadata={json.dumps(source_metadata, ensure_ascii=True)}\n"
        f"pdf_profile={json.dumps(pdf_profile, ensure_ascii=True)}\n"
        f"first_page={json.dumps(precontext, ensure_ascii=True)}\n"
        f"evidence_context={json.dumps(evidence_context, ensure_ascii=True)}\n"
        f"agentic_context={json.dumps(agentic_context, ensure_ascii=True)}\n"
        f"deterministic_hints={json.dumps(deterministic_hints, ensure_ascii=True)}"
    )


def _case_details_repair_prompt(
    *,
    precontext: list[dict[str, Any]],
    agentic_context: list[dict[str, Any]],
    deterministic_hints: dict[str, Any],
    source_metadata: dict[str, Any],
    pdf_profile: dict[str, Any],
) -> str:
    return (
        "The previous case-detail extraction failed. Use this smaller, safer context and return only strict JSON.\n"
        "Extract the best available case details from first_page and agentic_context. If uncertain, return null/[] "
        "but still return a valid JSON object.\n"
        "Return exactly this shape:\n"
        "{\n"
        '  "case_details": {"case_number": null, "case_type": null, "court": null, "bench": [], '
        '"departments": [], "advocates": [], "disposition": null, "evidence_snippet": null},\n'
        '  "date_of_order": {"value": null, "raw_text": null, "confidence": 0.0, "evidence_snippet": null},\n'
        '  "parties_involved": {"petitioners": [], "respondents": [], "other_parties": [], "evidence_snippet": null},\n'
        '  "key_directions_orders": [],\n'
        '  "relevant_timelines": [],\n'
        '  "confidence": 0.0\n'
        "}\n"
        f"source_metadata={json.dumps(source_metadata, ensure_ascii=True)}\n"
        f"pdf_profile={json.dumps(pdf_profile, ensure_ascii=True)}\n"
        f"first_page={json.dumps(precontext, ensure_ascii=True)}\n"
        f"agentic_context={json.dumps(agentic_context[:12], ensure_ascii=True)}\n"
        f"deterministic_hints={json.dumps(deterministic_hints, ensure_ascii=True)}"
    )


def _missing_fields_to_repair(
    deterministic_package: JudgmentReviewPackage,
    details_payload: dict[str, Any],
) -> list[str]:
    repairable_fields = ["bench", "judgment_date", "advocates", "disposition"]
    case_details = details_payload.get("case_details") if isinstance(details_payload, dict) else {}
    missing = []
    for field_name in repairable_fields:
        deterministic_field = getattr(deterministic_package.extraction, field_name)
        if _is_empty_value(deterministic_field.value):
            continue
        llm_value = _llm_field_value(field_name, details_payload)
        if field_name == "bench" and _bench_needs_repair(deterministic_field.value, llm_value):
            missing.append(field_name)
            continue
        if field_name == "disposition" and _disposition_needs_repair(deterministic_field.value, llm_value):
            missing.append(field_name)
            continue
        if not _is_empty_value(llm_value):
            continue
        missing.append(field_name)
    return missing


def _bench_needs_repair(deterministic_value: Any, llm_value: Any) -> bool:
    deterministic_bench = [str(item).strip() for item in _as_list(deterministic_value) if str(item or "").strip()]
    llm_bench = [str(item).strip() for item in _as_list(llm_value) if str(item or "").strip()]
    if len(deterministic_bench) < 2 or not llm_bench:
        return False
    if len(llm_bench) >= len(deterministic_bench):
        return False
    deterministic_names = {_person_name_key(item) for item in deterministic_bench}
    llm_names = {_person_name_key(item) for item in llm_bench}
    return bool(deterministic_names - llm_names)


def _person_name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower().replace("chief justice", "").replace("justice", ""))


def _disposition_needs_repair(deterministic_value: Any, llm_value: Any) -> bool:
    deterministic = str(deterministic_value or "").strip().lower()
    llm = str(llm_value or "").strip().lower()
    if not deterministic or deterministic == "unknown" or not llm or llm == "unknown":
        return False
    return deterministic != llm


def _field_repair_context(
    field_name: str,
    deterministic_package: JudgmentReviewPackage,
    llm_context: dict[str, Any],
) -> list[dict[str, Any]]:
    keywords = TRACE_KEYWORDS.get(field_name, [])
    context: list[dict[str, Any]] = []
    field = getattr(deterministic_package.extraction, field_name, None)
    for evidence in getattr(field, "evidence", [])[:3]:
        context.append(
            {
                "label": "deterministic_evidence",
                "page": evidence.page,
                "chunk_id": evidence.chunk_id,
                "text": _compact_text(evidence.snippet or "", limit=1000),
            }
        )
    for section, items in llm_context.items():
        if not isinstance(items, list):
            continue
        for item in items:
            text = str((item or {}).get("text") or (item or {}).get("snippet") or "")
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                context.append(
                    {
                        "label": f"{section}:{(item or {}).get('label') or 'context'}",
                        "page": (item or {}).get("page"),
                        "chunk_id": (item or {}).get("chunk_id"),
                        "text": _compact_text(text, limit=1000),
                    }
                )
    return _dedupe_context(context)[:8]


def _apply_field_repair(
    details_payload: dict[str, Any],
    field_name: str,
    repair_payload: dict[str, Any],
    deterministic_package: JudgmentReviewPackage,
) -> None:
    if not isinstance(details_payload, dict) or not isinstance(repair_payload, dict):
        return
    case_details = details_payload.setdefault("case_details", {})
    if not isinstance(case_details, dict):
        return
    if field_name == "judgment_date":
        value = str(repair_payload.get("value") or "").strip()
        if not value:
            return
        date_payload = details_payload.setdefault("date_of_order", {})
        if not isinstance(date_payload, dict):
            return
        date_payload["value"] = value
        date_payload["raw_text"] = repair_payload.get("raw_text") or value
        date_payload["confidence"] = _safe_float(repair_payload.get("confidence"))
        date_payload["evidence_snippet"] = (
            repair_payload.get("evidence_snippet")
            or _first_evidence_snippet(deterministic_package.extraction.judgment_date)
        )
        return
    if field_name == "disposition":
        value = str(repair_payload.get("value") or "").strip().lower()
        if not value:
            return
        case_details["disposition"] = value
        case_details["disposition_confidence"] = _safe_float(repair_payload.get("confidence"))
        case_details["disposition_evidence_snippet"] = (
            repair_payload.get("evidence_snippet")
            or _first_evidence_snippet(deterministic_package.extraction.disposition)
        )
        return
    repaired_values = [
        str(item).strip()
        for item in _as_list(repair_payload.get("value"))
        if str(item or "").strip()
    ]
    if not repaired_values:
        return
    if field_name == "advocates":
        repaired_values = _prefer_richer_deterministic_advocates(
            repaired_values,
            getattr(deterministic_package.extraction, field_name).value,
        )
    if field_name == "bench":
        repaired_values = _prefer_richer_deterministic_bench(
            repaired_values,
            getattr(deterministic_package.extraction, field_name).value,
        )
    case_details[field_name] = list(dict.fromkeys(repaired_values))
    confidence = _safe_float(repair_payload.get("confidence"))
    if field_name == "advocates" and getattr(deterministic_package.extraction, field_name).requires_review:
        confidence = min(confidence, 0.74)
    case_details[f"{field_name}_confidence"] = confidence
    case_details[f"{field_name}_evidence_snippet"] = (
        repair_payload.get("evidence_snippet")
        or _first_evidence_snippet(getattr(deterministic_package.extraction, field_name))
    )


def _first_evidence_snippet(field: ExtractedField) -> str | None:
    return field.evidence[0].snippet if field.evidence else None


def _prefer_richer_deterministic_advocates(repaired_values: list[str], deterministic_value: Any) -> list[str]:
    deterministic_values = [str(item).strip() for item in _as_list(deterministic_value) if str(item or "").strip()]
    if not deterministic_values:
        return repaired_values
    enriched: list[str] = []
    for repaired in repaired_values:
        repaired_key = _person_name_key(repaired)
        richer = next(
            (
                deterministic
                for deterministic in deterministic_values
                if repaired_key and repaired_key in _person_name_key(deterministic) and len(deterministic) > len(repaired)
            ),
            None,
        )
        enriched.append(richer or repaired)
    return enriched


def _prefer_richer_deterministic_bench(repaired_values: list[str], deterministic_value: Any) -> list[str]:
    deterministic_values = [str(item).strip() for item in _as_list(deterministic_value) if str(item or "").strip()]
    if len(deterministic_values) <= len(repaired_values):
        return repaired_values
    repaired_keys = {_person_name_key(item) for item in repaired_values}
    deterministic_keys = {_person_name_key(item) for item in deterministic_values}
    if repaired_keys and repaired_keys.issubset(deterministic_keys):
        return deterministic_values
    return repaired_values


def _action_plan_prompt(
    *,
    action_context: list[dict[str, Any]],
    supporting_context: list[dict[str, Any]],
    case_details: dict[str, Any],
    previous_context_summary: str | None = None,
    second_pass: bool = False,
) -> str:
    return (
        "You are creating a decision-support action plan from the OPERATIVE ORDER portion of a court judgment.\n"
        "Be strict, but do not dodge. For every final direction/order/outcome, create one reviewable decision item. "
        "The item may be a direct compliance task, a conditional follow-up, a legal-review/record-update task, "
        "or a no-operational-action decision with a clear reason.\n"
        "Ignore headnotes, factual background, party arguments, counsel submissions, case-law quotations, and "
        "generic legal principles unless the final order itself makes them actionable.\n"
        "Return strict JSON with this shape:\n"
        "{\n"
        '  "context_summary": "short summary of the provided action context",\n'
        '  "needs_more_context": false,\n'
        '  "missing_context_reason": null,\n'
        '  "action_items": [\n'
        '    {"title": "...", "responsible_department": null, "category": "compliance", '
        '"priority": "medium", "timeline": {"raw_text": null, "timeline_type": "missing", '
        '"confidence": 0.0}, "legal_basis": "...", "decision_reason": "...", '
        '"review_recommendation": "...", "requires_human_review": true, "confidence": 0.0, '
        '"ambiguity_flags": [], "evidence_snippet": "..."}\n'
        "  ]\n"
        "}\n"
        "Tough rules:\n"
        "- If the pages do not contain the final/operative order, set needs_more_context=true and action_items=[].\n"
        "- Use the supplied context fully. Do not stop at the first dismissed/allowed/disposed sentence if the same "
        "operative paragraph also orders deposit, payment, cross-examination, remand, transmission of records, "
        "set-aside/modification of an order, or any dated listing.\n"
        "- Do not return an empty action plan when you have any final outcome, direction, dismissal, allowance, "
        "quashing, remand, bail liberty, or disposed application. Convert that outcome into a reviewable decision.\n"
        "- If the operative text says an appeal/application is allowed and then states what follows, the action item "
        "must be about the concrete consequence, not a generic appeal review.\n"
        "- If the order says deposit/pay/compensation/enhanced compensation/interest/within weeks, create a "
        "payment_release or direct_compliance item with the payer as responsible_department when visible.\n"
        "- If the order says post/list/cross-examination/recall witness/transmit records, create a direct_compliance "
        "or conditional_follow_up item with the date/timeline copied exactly.\n"
        "- If the order says set aside/quash/remand/restore/modify, create a record_update or legal_review item that "
        "names the affected order/case when visible.\n"
        "- Valid categories: direct_compliance, conditional_follow_up, legal_review, record_update, "
        "appeal_consideration, compliance, affidavit_report_filing, payment_release, reconsideration, "
        "no_operational_action.\n"
        "- Use no_operational_action only when there is truly no government/public-authority task; still explain "
        "why in decision_reason and recommend human verification in review_recommendation.\n"
        "- If the outcome affects an existing government order, tribunal order, prosecution, bail position, "
        "department record, compliance posture, or public authority response, prefer legal_review, record_update, "
        "conditional_follow_up, or appeal_consideration instead of no_operational_action.\n"
        "- If the order creates a conditional future duty, such as considering an application if filed, create "
        "a reviewable action item with the condition in timeline.raw_text or ambiguity_flags.\n"
        "- Clauses saying parties bear their own/respective costs or no order as to costs are not payment, "
        "notice, or direct-compliance tasks. Only record them as no_operational_action if no substantive "
        "order action exists.\n"
        "- Set responsible_department from the direction first, then from public/respondent entities. If owner is "
        "unclear, keep it null and add ambiguity flag owner_unclear.\n"
        "- Each action must have an evidence_snippet copied from supplied context.\n"
        "- confidence must reflect evidence quality. Use <=0.55 for unclear owner/timeline/evidence, "
        "0.56-0.80 for partial support, and >0.80 only when owner/action/timeline are clearly supported.\n"
        "- Keep titles short and executable: e.g. 'File compliance report', 'Release arrears', 'Reconsider application'.\n"
        "- Titles must be judgment-specific. Include the case number, affected order, beneficiary, amount, or concrete "
        "next step when visible. Avoid generic titles such as 'Review judgment outcome for appeal or compliance decision' "
        "unless the context truly contains only a bare dismissal/allowance with no other operative consequence.\n"
        "- If owner or timeline is unclear, keep it null/missing and add ambiguity flags; do not guess.\n"
        f"- This is {'the second and final pass' if second_pass else 'the first pass over the last 3 pages'}.\n"
        f"case_details={json.dumps(case_details, ensure_ascii=True)}\n"
        f"previous_action_context_summary={json.dumps(previous_context_summary or '', ensure_ascii=True)}\n"
        f"supporting_context={json.dumps(supporting_context, ensure_ascii=True)}\n"
        f"action_context={json.dumps(action_context, ensure_ascii=True)}"
    )


def _forced_action_plan_prompt(
    *,
    action_context: list[dict[str, Any]],
    supporting_context: list[dict[str, Any]],
    case_details: dict[str, Any],
    previous_payload: dict[str, Any],
) -> str:
    return (
        "Your previous action-plan attempt produced no reviewable action items. That is not acceptable for this "
        "decision-support system unless the PDF has no final outcome at all.\n"
        "Return strict JSON with at least one action_items entry. If the order has no operational government task, "
        "create a no_operational_action decision item. If the order affects legal posture, records, bail, tribunal "
        "orders, compliance posture, or public-authority response, create legal_review, record_update, "
        "conditional_follow_up, appeal_consideration, or direct_compliance as appropriate.\n"
        "Use the supplied action_context fully; do not stop at the first dismissed/allowed/disposed sentence when "
        "the same final-order passage contains deposit, payment, compensation, cross-examination, recall witness, "
        "remand, transmission of records, set-aside/modification, or a dated listing.\n"
        "If the order allows an appeal/application and then states a concrete consequence, the item must describe "
        "that consequence. Do not fall back to a generic review title when amount, case number, affected order, "
        "beneficiary, hearing date, or next step is visible.\n"
        "For payment/deposit/enhanced compensation/interest/within weeks, create payment_release or "
        "direct_compliance with the payer as owner when visible. For posting/listing/cross-examination/recall "
        "witness/transmit records, create direct_compliance or conditional_follow_up and copy the exact date/timeline. "
        "Titles must be judgment-specific.\n"
        "Do not convert own-cost/no-cost clauses into notification, payment, or direct-compliance tasks.\n"
        "Each item must include title, responsible_department or null, category, priority, timeline, legal_basis, "
        "decision_reason, review_recommendation, requires_human_review, confidence, ambiguity_flags, and "
        "evidence_snippet copied from context.\n"
        "Return exactly this JSON shape:\n"
        "{\n"
        '  "context_summary": "short summary",\n'
        '  "needs_more_context": false,\n'
        '  "forced_decision_pass": true,\n'
        '  "action_items": [\n'
        '    {"title": "...", "responsible_department": null, "category": "legal_review", '
        '"priority": "medium", "timeline": {"raw_text": null, "timeline_type": "missing", '
        '"confidence": 0.0}, "legal_basis": "...", "decision_reason": "...", '
        '"review_recommendation": "...", "requires_human_review": true, "confidence": 0.0, '
        '"ambiguity_flags": [], "evidence_snippet": "..."}\n'
        "  ]\n"
        "}\n"
        f"case_details={json.dumps(case_details, ensure_ascii=True)}\n"
        f"previous_payload={json.dumps(previous_payload, ensure_ascii=True)}\n"
        f"supporting_context={json.dumps(supporting_context, ensure_ascii=True)}\n"
        f"action_context={json.dumps(action_context, ensure_ascii=True)}"
    )


def _action_owner_repair_prompt(
    *,
    current_action_payload: dict[str, Any],
    case_details: dict[str, Any],
    repair_context: list[dict[str, Any]],
) -> str:
    return (
        "Repair action ownership and timelines using the broader judgment context.\n"
        "You are not extracting metadata. You are correcting the action plan only when the current actions have "
        "missing, weak, suspicious, or internally-mixed owners/timelines.\n"
        "Return strict JSON with this shape:\n"
        "{\n"
        '  "context_summary": "short reason for repairs",\n'
        '  "action_items": [\n'
        '    {"title": "...", "responsible_department": null, "category": "direct_compliance", '
        '"priority": "medium", "timeline": {"raw_text": null, "timeline_type": "not_specified", '
        '"confidence": 0.0}, "legal_basis": "...", "decision_reason": "...", '
        '"review_recommendation": "...", "requires_human_review": true, "confidence": 0.0, '
        '"ambiguity_flags": [], "evidence_snippet": "..."}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Use the final/operative order only; ignore lower-court history, applications list text, footers, and staff names.\n"
        "- Legal actions and internal workflow actions must be separate items.\n"
        "- If the order remands/remits/restores a matter to a court, tribunal, board, or authority, that receiving entity is the legal owner.\n"
        "- Do not assign a respondent or department as owner merely because it is a party or subject-matter entity.\n"
        "- Only assign a respondent/department if the operative text directs that entity to act.\n"
        "- Record/case/dashboard updates are internal_review actions with responsible_department='Case reviewer' and timeline_type='not_configured'.\n"
        "- If the court gives no deadline for a legal action, use timeline_type='not_specified', not 'missing'.\n"
        "- Do not use Court Master, PS to Registrar, advocates, or footer/signature staff as owners unless the order expressly directs registry staff.\n"
        "- Every action must cite an evidence_snippet copied from repair_context.\n"
        "- If no reliable repair is possible, return the current actions but mark unsupported owners null.\n"
        f"case_details={json.dumps(case_details, ensure_ascii=True)}\n"
        f"current_action_payload={json.dumps(current_action_payload, ensure_ascii=True)}\n"
        f"repair_context={json.dumps(repair_context, ensure_ascii=True)}"
    )


def _case_detail_field_repair_prompt(
    *,
    field_name: str,
    current_details: dict[str, Any],
    deterministic_value: Any,
    repair_context: list[dict[str, Any]],
) -> str:
    return (
        "Repair one missing case-detail field from the supplied judgment context.\n"
        "The first extraction left this field empty, but deterministic extraction found a possible value. "
        "Use the deterministic value only as a hint; write a clean value only if the context supports it.\n"
        "Return strict JSON with this exact shape:\n"
        "{\n"
        f'  "field": "{field_name}",\n'
        '  "value": null,\n'
        '  "raw_text": null,\n'
        '  "confidence": 0.0,\n'
        '  "evidence_snippet": null,\n'
        '  "reason": "..."\n'
        "}\n"
        "Rules:\n"
        "- For bench, extract only judges listed under CORAM, BEFORE, PRESENT, signature blocks like CJI./J. "
        "(NAME), or equivalent bench headers. A single 'NAME, J.' below JUDGMENT is the authoring judge, not "
        "necessarily the full bench.\n"
        "- For judgment_date, return ISO YYYY-MM-DD from the court's own Date/Order/Judgment date, not an appealed-from date.\n"
        "- For advocates, return a list of advocate/counsel names under For Petitioner(s), For Appellant(s), or For Respondent(s).\n"
        "- For disposition, return one of allowed, dismissed, disposed, quashed, remanded, partly_allowed, leave_granted, or unknown from the final order.\n"
        "- Normalize judge names neatly, e.g. Justice S.H. Kapadia. Do not include CORAM labels or issue text.\n"
        "- If the context does not support the field, return value=null or [] with low confidence.\n"
        "- evidence_snippet must be copied from the supplied context and must contain the supporting lines.\n"
        f'field={json.dumps({"field": field_name}, ensure_ascii=True)}\n'
        f"current_case_details={json.dumps(current_details, ensure_ascii=True)}\n"
        f"deterministic_value={json.dumps(serialize_value_for_debug(deterministic_value), ensure_ascii=True)}\n"
        f"repair_context={json.dumps(repair_context, ensure_ascii=True)}"
    )


def _package_from_llm_payloads(
    *,
    details_payload: dict[str, Any],
    action_payload: dict[str, Any],
    documents: list[Document],
    fallback_package: JudgmentReviewPackage,
    source_metadata: dict[str, Any],
) -> JudgmentReviewPackage:
    extraction = JudgmentExtraction()
    case_details = details_payload.get("case_details") or {}
    parties = details_payload.get("parties_involved") or {}

    extraction.case_number = _field(
        "case_number",
        case_details.get("case_number"),
        case_details.get("case_number"),
        details_payload.get("confidence", 0.0),
        case_details.get("evidence_snippet") or parties.get("evidence_snippet"),
        documents,
    )
    extraction.case_type = _field(
        "case_type",
        case_details.get("case_type"),
        case_details.get("case_type"),
        details_payload.get("confidence", 0.0),
        case_details.get("evidence_snippet"),
        documents,
    )
    extraction.court = _field(
        "court",
        case_details.get("court"),
        case_details.get("court"),
        details_payload.get("confidence", 0.0),
        case_details.get("evidence_snippet"),
        documents,
    )
    extraction.bench = _field(
        "bench",
        _as_list(case_details.get("bench")),
        None,
        case_details.get("bench_confidence", details_payload.get("confidence", 0.0)),
        case_details.get("bench_evidence_snippet") or case_details.get("evidence_snippet"),
        documents,
    )
    date_payload = details_payload.get("date_of_order") or {}
    extraction.judgment_date = _field(
        "judgment_date",
        _parse_iso_date(date_payload.get("value")) or date_payload.get("value"),
        date_payload.get("raw_text") or date_payload.get("value"),
        date_payload.get("confidence", 0.0),
        date_payload.get("evidence_snippet"),
        documents,
    )
    petitioners = _as_list(parties.get("petitioners"))
    respondents = _as_list(parties.get("respondents"))
    other_parties = _as_list(parties.get("other_parties"))
    extraction.petitioners = _field("petitioners", petitioners, None, details_payload.get("confidence", 0.0), parties.get("evidence_snippet"), documents)
    extraction.respondents = _field("respondents", respondents, None, details_payload.get("confidence", 0.0), parties.get("evidence_snippet"), documents)
    extraction.parties = _field("parties", petitioners + respondents + other_parties, None, details_payload.get("confidence", 0.0), parties.get("evidence_snippet"), documents)
    extraction.departments = _field(
        "departments",
        _as_list(case_details.get("departments")) or _as_list(case_details.get("responsible_entities")),
        None,
        details_payload.get("confidence", 0.0),
        case_details.get("evidence_snippet"),
        documents,
    )
    extraction.advocates = _field(
        "advocates",
        _as_list(case_details.get("advocates")),
        None,
        case_details.get("advocates_confidence", details_payload.get("confidence", 0.0)),
        case_details.get("advocates_evidence_snippet") or case_details.get("evidence_snippet"),
        documents,
    )
    extraction.disposition = _field(
        "disposition",
        case_details.get("disposition"),
        case_details.get("disposition"),
        case_details.get("disposition_confidence", details_payload.get("confidence", 0.0)),
        case_details.get("disposition_evidence_snippet") or case_details.get("evidence_snippet"),
        documents,
    )
    extraction.legal_phrases = ExtractedField("legal_phrases", value=[], confidence=0.0)

    extraction.directions = [
        _field(
            "direction",
            item.get("text"),
            item.get("text"),
            item.get("confidence", 0.0),
            item.get("evidence_snippet"),
            documents,
        )
        for item in details_payload.get("key_directions_orders", [])
        if isinstance(item, dict) and item.get("text")
    ]

    action_items = _action_items_from_payload(action_payload, documents, extraction)
    action_items, action_quality_trace = _apply_action_quality_gate(action_items, extraction.disposition.value)
    action_items = _ensure_remand_action(action_items, documents)
    action_items = _prune_background_actions_when_remand_found(action_items)
    action_payload["quality_trace"] = action_quality_trace
    if _payload_declares_no_action(action_payload) and not action_items:
        action_items = [_no_operational_action_item(action_payload, documents)]
    risk_flags = _llm_risk_flags(extraction, action_items)
    if not action_items and "missing_action_items" not in risk_flags:
        risk_flags.append("missing_action_items")
    if action_payload.get("needs_more_context"):
        risk_flags.append("action_context_insufficient")
    if _has_no_operational_action(action_items):
        risk_flags = [flag for flag in risk_flags if flag != "missing_action_items"]

    return JudgmentReviewPackage(
        extraction=extraction,
        action_items=action_items,
        source_metadata=dict(source_metadata),
        overall_confidence=_overall_confidence(extraction, action_items, details_payload, action_payload),
        risk_flags=list(dict.fromkeys(risk_flags)),
    )


TRACE_FIELDS = [
    "case_number",
    "case_type",
    "court",
    "bench",
    "judgment_date",
    "parties",
    "petitioners",
    "respondents",
    "departments",
    "advocates",
    "disposition",
]

TRACE_KEYWORDS = {
    "case_number": ["case", "petition", "appeal", "slp"],
    "case_type": ["petition", "appeal", "slp"],
    "court": ["court"],
    "bench": ["bench", "coram", "hon", "honble", "justice", "cji"],
    "judgment_date": ["judgment", "order", "dated", "date"],
    "parties": ["versus", "v.", "petitioner", "respondent", "appellant"],
    "petitioners": ["petitioner", "appellant"],
    "respondents": ["respondent", "versus"],
    "departments": ["state", "government", "department", "authority", "officer"],
    "advocates": ["advocate", "counsel", "solicitor", "attorney"],
    "disposition": ["allowed", "dismissed", "disposed", "quashed", "remanded", "grant leave", "tag this appeal"],
}


def _build_extraction_debug_trace(
    *,
    deterministic_package: JudgmentReviewPackage,
    final_package: JudgmentReviewPackage,
    llm_context: dict[str, Any],
    llm_outputs: dict[str, Any],
) -> dict[str, Any]:
    deterministic = serialize_review_package(deterministic_package)
    final = serialize_review_package(final_package)
    deterministic_extraction = deterministic.get("extraction", {})
    final_extraction = final.get("extraction", {})
    details_output = llm_outputs.get("case_details") or {}
    field_repairs = llm_outputs.get("field_repairs") or {}
    field_repairs_attempted = sorted(field_repairs.keys()) if isinstance(field_repairs, dict) else []
    field_trace = {}
    missing_final_fields = []
    deterministic_found_but_final_missing = []
    llm_returned_empty_fields = []
    field_repairs_applied = []

    for field_name in TRACE_FIELDS:
        deterministic_field = _debug_field_payload(deterministic_extraction.get(field_name) or {})
        final_field = _debug_field_payload(final_extraction.get(field_name) or {})
        llm_value = _llm_field_value(field_name, details_output)
        context_presence = _context_presence_for_field(field_name, deterministic_field, llm_context)
        diagnosis = _field_diagnosis(deterministic_field.get("value"), llm_value, final_field.get("value"))
        if _is_empty_value(final_field.get("value")):
            missing_final_fields.append(field_name)
        if not _is_empty_value(deterministic_field.get("value")) and _is_empty_value(final_field.get("value")):
            deterministic_found_but_final_missing.append(field_name)
        if _is_empty_value(llm_value) and not _is_empty_value(deterministic_field.get("value")):
            llm_returned_empty_fields.append(field_name)
        if (
            field_name in field_repairs_attempted
            and not _is_empty_value((field_repairs.get(field_name) or {}).get("value"))
            and not _is_empty_value(final_field.get("value"))
        ):
            field_repairs_applied.append(field_name)
        field_trace[field_name] = {
            "deterministic": deterministic_field,
            "llm_context_presence": context_presence,
            "llm_output": {"value": llm_value},
            "final": final_field,
            "diagnosis": diagnosis,
        }

    return {
        "summary": {
            "llm_enabled": True,
            "llm_review_mode": "llm_first",
            "missing_final_fields": missing_final_fields,
            "deterministic_found_but_final_missing": deterministic_found_but_final_missing,
            "llm_returned_empty_fields": llm_returned_empty_fields,
            "field_repairs_attempted": field_repairs_attempted,
            "field_repairs_applied": field_repairs_applied,
            "context_contains_keywords": _keyword_presence(llm_context),
        },
        "field_trace": field_trace,
        "deterministic": {
            "extraction": _debug_extraction_snapshot(deterministic_extraction),
            "action_items": _debug_action_items(deterministic.get("action_items", [])),
        },
        "llm_context": _debug_context_payload(llm_context),
        "llm_outputs": serialize_value_for_debug(llm_outputs),
        "action_trace": _action_trace(llm_outputs),
        "final": {
            "extraction": _debug_extraction_snapshot(final_extraction),
            "action_items": _debug_action_items(final.get("action_items", [])),
        },
    }


def _debug_extraction_snapshot(extraction: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _debug_field_payload(value)
        for key, value in extraction.items()
        if key not in {"directions", "risk_flags"} and isinstance(value, dict)
    } | {
        "directions": [_debug_field_payload(item) for item in extraction.get("directions", [])[:12] if isinstance(item, dict)],
        "risk_flags": list(extraction.get("risk_flags") or []),
    }


def _debug_action_items(action_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "action_id": item.get("action_id"),
            "title": item.get("title"),
            "responsible_department": item.get("responsible_department"),
            "category": item.get("category"),
            "priority": item.get("priority"),
            "confidence": item.get("confidence"),
            "timeline": item.get("timeline"),
            "evidence": [
                {
                    "page": evidence.get("page"),
                    "snippet": _compact_text(str(evidence.get("snippet") or ""), limit=500),
                    "confidence": evidence.get("confidence"),
                }
                for evidence in (item.get("evidence") or [])[:3]
                if isinstance(evidence, dict)
            ],
            "ambiguity_flags": list(item.get("ambiguity_flags") or []),
        }
        for item in action_items[:12]
        if isinstance(item, dict)
    ]


def _action_trace(llm_outputs: dict[str, Any]) -> dict[str, Any]:
    action_payload = llm_outputs.get("action_plan") or {}
    if not isinstance(action_payload, dict):
        return {}
    trace = action_payload.get("quality_trace") or {}
    if not isinstance(trace, dict):
        return {}
    return serialize_value_for_debug(trace)


def _debug_field_payload(field: dict[str, Any]) -> dict[str, Any]:
    evidence = field.get("evidence") or []
    return {
        "value": field.get("value"),
        "confidence": field.get("confidence", 0.0),
        "requires_review": bool(field.get("requires_review", False)),
        "evidence": [
            {
                "page": item.get("page"),
                "snippet": _compact_text(str(item.get("snippet") or ""), limit=500),
                "confidence": item.get("confidence"),
                "match_strategy": item.get("match_strategy"),
            }
            for item in evidence[:3]
            if isinstance(item, dict)
        ],
        "notes": list(field.get("notes") or [])[:5],
    }


def _llm_field_value(field_name: str, details_output: dict[str, Any]) -> Any:
    case_details = details_output.get("case_details") or {}
    parties = details_output.get("parties_involved") or {}
    if field_name == "judgment_date":
        return (details_output.get("date_of_order") or {}).get("value")
    if field_name == "parties":
        return _as_list(parties.get("petitioners")) + _as_list(parties.get("respondents")) + _as_list(parties.get("other_parties"))
    if field_name == "petitioners":
        return _as_list(parties.get("petitioners"))
    if field_name == "respondents":
        return _as_list(parties.get("respondents"))
    if field_name == "departments":
        return _as_list(case_details.get("departments")) or _as_list(case_details.get("responsible_entities"))
    if field_name in {"bench", "advocates"}:
        return _as_list(case_details.get(field_name))
    return case_details.get(field_name)


def _field_diagnosis(deterministic_value: Any, llm_value: Any, final_value: Any) -> str:
    deterministic_empty = _is_empty_value(deterministic_value)
    llm_empty = _is_empty_value(llm_value)
    final_empty = _is_empty_value(final_value)
    if not deterministic_empty and llm_empty and final_empty:
        return "deterministic_found_but_llm_final_empty"
    if deterministic_empty and not llm_empty and final_empty:
        return "llm_found_but_final_empty"
    if final_empty:
        return "missing_everywhere"
    if not llm_empty:
        return "llm_value_written"
    if not deterministic_empty:
        return "deterministic_value_written"
    return "present"


def _context_presence_for_field(field_name: str, deterministic_field: dict[str, Any], llm_context: dict[str, Any]) -> dict[str, Any]:
    keywords = list(TRACE_KEYWORDS.get(field_name, []))
    value_tokens = [
        token for token in re.findall(r"[a-z0-9]+", str(deterministic_field.get("value") or "").lower())
        if len(token) >= 4
    ][:6]
    keywords.extend(value_tokens)
    sections = {}
    snippets = []
    for section, items in llm_context.items():
        if section == "deterministic_hints":
            text = json.dumps(items, ensure_ascii=True).lower()
            found = any(keyword in text for keyword in keywords)
            sections[section] = found
            continue
        found = False
        for item in items if isinstance(items, list) else []:
            text = str((item or {}).get("text") or (item or {}).get("snippet") or "")
            lowered = text.lower()
            if any(keyword in lowered for keyword in keywords):
                found = True
                if len(snippets) < 5:
                    snippets.append(
                        {
                            "section": section,
                            "page": (item or {}).get("page"),
                            "text": _compact_text(text, limit=500),
                        }
                    )
        sections[section] = found
    return {**sections, "matching_context_snippets": snippets}


def _keyword_presence(llm_context: dict[str, Any]) -> dict[str, bool]:
    text = json.dumps(llm_context, ensure_ascii=True).lower()
    return {
        "coram": "coram" in text,
        "honble": "honble" in text or "hon'ble" in text or "hon\\u0027ble" in text,
        "justice": "justice" in text,
    }


def _debug_context_payload(llm_context: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _cap_debug_value(value)
        for key, value in llm_context.items()
    }


def _cap_debug_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_cap_debug_value(item) for item in value[:12]]
    if isinstance(value, dict):
        capped = {}
        for key, item in value.items():
            if key in {"text", "snippet"}:
                capped[key] = _compact_text(str(item or ""), limit=800)
            else:
                capped[key] = _cap_debug_value(item)
        return capped
    if isinstance(value, str):
        return _compact_text(value, limit=800)
    return value


def serialize_value_for_debug(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: serialize_value_for_debug(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_value_for_debug(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    return value


def _field(
    name: str,
    value: Any,
    raw_value: str | None,
    confidence: Any,
    evidence_snippet: str | None,
    documents: list[Document],
) -> ExtractedField:
    empty = _is_empty_value(value)
    evidence = None if empty else _evidence_for_snippet(evidence_snippet, documents)
    calibrated_confidence = _calibrated_confidence(confidence, empty=empty, has_evidence=bool(evidence))
    notes: list[str] = []
    if empty:
        notes.append("LLM returned no supported value for this field.")
    elif not evidence:
        notes.append("LLM value lacks source evidence and needs review.")
    return ExtractedField(
        name=name,
        value=value,
        raw_value=raw_value,
        confidence=calibrated_confidence,
        evidence=[evidence] if evidence else [],
        notes=notes,
        requires_review=calibrated_confidence < 0.75 or not evidence,
    )


def _action_items_from_payload(
    action_payload: dict[str, Any],
    documents: list[Document],
    extraction: JudgmentExtraction,
) -> list[ActionItem]:
    items: list[ActionItem] = []
    for index, payload in enumerate(action_payload.get("action_items", [])):
        if not isinstance(payload, dict) or not payload.get("title"):
            continue
        evidence = _evidence_for_snippet(payload.get("evidence_snippet"), documents)
        category = _normalise_action_category(payload.get("category") or classify_action(str(payload.get("title") or ""))[0])
        ambiguity_flags = _normalise_risk_flags(payload.get("ambiguity_flags") or [])
        timeline_payload = payload.get("timeline") if isinstance(payload.get("timeline"), dict) else {}
        timeline = Timeline(
            raw_text=timeline_payload.get("raw_text"),
            confidence=_safe_float(timeline_payload.get("confidence")),
            timeline_type=str(timeline_payload.get("timeline_type") or "missing"),
        )
        priority = _normalise_priority(payload.get("priority") or priority_for_action(category, ambiguity_flags, bool(timeline.raw_text)))
        confidence = _calibrated_confidence(
            payload.get("confidence") or 0.0,
            empty=False,
            has_evidence=bool(evidence),
            has_owner=bool(payload.get("responsible_department")) or category == "no_operational_action",
        )
        requires_human_review = bool(payload.get("requires_human_review", False)) or confidence < 0.75 or bool(ambiguity_flags)
        action = ActionItem(
            title=str(payload["title"]),
            responsible_department=payload.get("responsible_department"),
            timeline=timeline,
            category=category,
            priority=priority,
            legal_basis=payload.get("legal_basis") or payload.get("evidence_snippet"),
            confidence=confidence,
            evidence=[evidence] if evidence else [],
            notes=[] if evidence else ["LLM action lacks source evidence and requires review."],
            action_id=str(payload.get("action_id") or f"action-{index}"),
            direction_summary=payload.get("legal_basis") or payload.get("title"),
            owner_source="llm_action_plan" if payload.get("responsible_department") else None,
            timeline_type=timeline.timeline_type,
            ambiguity_flags=ambiguity_flags,
            decision_reason=payload.get("decision_reason"),
            review_recommendation=payload.get("review_recommendation"),
            requires_human_review=requires_human_review,
        )
        items.append(apply_inferred_action_owner(action, extraction))
    return items


def _apply_action_quality_gate(
    action_items: list[ActionItem],
    disposition: str | None = None,
) -> tuple[list[ActionItem], dict[str, Any]]:
    trace = {
        "original_action_count": len(action_items),
        "removed_cost_only_actions": [],
        "rewritten_cost_only_actions": [],
        "removed_stale_dismissal_actions": [],
        "kept_action_count": len(action_items),
    }
    if not action_items:
        return action_items, trace

    stale_dismissal_items = [
        item
        for item in action_items
        if _is_stale_dismissal_action(item, disposition)
    ]
    if stale_dismissal_items:
        action_items = [item for item in action_items if item not in stale_dismissal_items]
        trace["removed_stale_dismissal_actions"] = [item.title for item in stale_dismissal_items]
        trace["kept_action_count"] = len(action_items)
        if not action_items:
            return action_items, trace

    cost_items = [item for item in action_items if _is_cost_only_action(item)]
    if not cost_items:
        return action_items, trace

    substantive_items = [item for item in action_items if item not in cost_items]
    if substantive_items:
        trace["removed_cost_only_actions"] = [item.title for item in cost_items]
        trace["kept_action_count"] = len(substantive_items)
        return substantive_items, trace

    first = cost_items[0]
    rewritten = _cost_only_no_operational_action(first)
    trace["rewritten_cost_only_actions"] = [first.title]
    trace["kept_action_count"] = 1
    return [rewritten], trace


def _ensure_remand_action(action_items: list[ActionItem], documents: list[Document]) -> list[ActionItem]:
    if any(item.category != "internal_review" and item.owner_source == "remand_destination" for item in action_items):
        return action_items

    remand_source = _remand_text_from_documents(documents)
    if not remand_source:
        return action_items

    text, evidence = remand_source
    owner = remand_owner_from_text(text)
    if not owner:
        return action_items

    if any(item.category != "internal_review" and owner.lower() in str(item.responsible_department or "").lower() for item in action_items):
        return action_items

    action = ActionItem(
        title=_remand_action_title(text, owner),
        responsible_department=owner,
        timeline=Timeline(raw_text=None, due_date=None, confidence=0.0, timeline_type="not_specified"),
        category="direct_compliance",
        priority="medium",
        legal_basis=text,
        confidence=0.82,
        evidence=[evidence],
        notes=["No explicit timeline specified by court."],
        action_id=f"action-{len(action_items)}",
        direction_summary=text[:240],
        owner_source="remand_destination",
        timeline_type="not_specified",
        ambiguity_flags=["timeline_not_specified"],
        review_recommendation="Verify receiving court or authority before publication.",
        requires_human_review=True,
    )
    return [*action_items, action]


def _prune_background_actions_when_remand_found(action_items: list[ActionItem]) -> list[ActionItem]:
    remand_items = [
        item
        for item in action_items
        if item.category != "internal_review" and item.owner_source == "remand_destination"
    ]
    if not remand_items:
        return action_items
    remand_owners = [str(item.responsible_department or "").lower() for item in remand_items if item.responsible_department]

    kept: list[ActionItem] = []
    for item in action_items:
        if item.category == "internal_review" or item in remand_items:
            kept.append(item)
            continue
        if _looks_like_background_action_next_to_remand(item, remand_owners):
            continue
        kept.append(item)
    return kept or remand_items


def _looks_like_background_action_next_to_remand(item: ActionItem, remand_owners: list[str]) -> bool:
    text = " ".join(
        str(value or "")
        for value in [
            item.title,
            item.legal_basis,
            item.direction_summary,
            item.decision_reason,
            item.review_recommendation,
            " ".join(evidence.snippet for evidence in item.evidence),
        ]
    ).lower()
    if re.search(r"\b(?:remit|remand|restore|fresh consideration|high court|trial court|tribunal|board of)\b", text):
        owner = str(item.responsible_department or "").lower()
        title = str(item.title or "").lower()
        if item.confidence < 0.75 and not any(remand_owner and remand_owner in owner for remand_owner in remand_owners):
            return True
        if not re.search(r"\b(?:remit|remand|restore|fresh consideration)\b", title):
            return item.confidence < 0.65
        return False
    if re.search(r"\b(?:shall|must|directed to|required to|ordered to|within)\b", text):
        return False
    if item.confidence < 0.7:
        return True
    return bool(re.search(r"\b(?:notification|background|categorization|cdec|eligibility certificate)\b", text))


def _remand_text_from_documents(documents: list[Document]) -> tuple[str, SourceEvidence] | None:
    for document in reversed(documents):
        text = " ".join(str(document.page_content or "").split())
        for match in re.finditer(
            r"[^.]{0,220}\b(?:remit(?:ted)?|remand(?:ed)?|restore(?:d)?)\b[^.]{0,260}\bto\s+(?:the\s+)?"
            r"(?:High Court|Trial Court|District Court|Tribunal|Board of [A-Z][A-Za-z .&-]+)[^.]{0,220}",
            text,
            re.I,
        ):
            snippet = match.group(0).strip(" .")
            owner = remand_owner_from_text(snippet)
            if not owner:
                continue
            return snippet, normalize_evidence(document, snippet)
    return None


def _remand_action_title(text: str, owner: str) -> str:
    appeal_numbers = re.search(
        r"\b(?:Customs\s+)?Appeal\s+Nos?\.?\s*([0-9,\sand]+of\s+\d{4})",
        text,
        re.I,
    )
    if appeal_numbers:
        return f"Fresh consideration of Appeal Nos. {appeal_numbers.group(1).strip()} by {owner}"
    return f"Fresh consideration of remanded matter by {owner}"


def _is_stale_dismissal_action(action_item: ActionItem, disposition: str | None) -> bool:
    final_disposition = str(disposition or "").strip().lower()
    if final_disposition in {"", "unknown", "dismissed"}:
        return False
    text = " ".join(
        str(value or "")
        for value in [
            action_item.title,
            action_item.legal_basis,
            action_item.direction_summary,
            action_item.decision_reason,
            " ".join(evidence.snippet for evidence in action_item.evidence),
        ]
    ).lower()
    if not re.search(r"\b(?:dismissed|dismissal|liable\s+to\s+be\s+dismissed|case\s+status\s+to\s+dismissed)\b", text):
        return False
    return bool(
        re.search(r"\b(?:high\s+court|tribunal|lower\s+court|impugned|writ\s+petition)\b", text)
        or re.search(r"\bcase\s+status\s+to\s+dismissed\b", text)
    )


def _is_cost_only_action(action_item: ActionItem) -> bool:
    title_basis_text = " ".join(
        str(value or "")
        for value in [
            action_item.title,
            action_item.legal_basis,
            action_item.direction_summary,
            action_item.decision_reason,
        ]
    ).lower()
    text = " ".join(
        str(value or "")
        for value in [
            action_item.title,
            action_item.legal_basis,
            action_item.direction_summary,
            action_item.decision_reason,
            " ".join(evidence.snippet for evidence in action_item.evidence),
        ]
    ).lower()
    has_cost_clause = bool(
        re.search(
            r"\b(?:no\s+order\s+as\s+to\s+costs?|no\s+costs?\s+(?:were\s+)?ordered|"
            r"bear\s+(?:their\s+)?(?:own|respective)\s+costs?)\b",
            text,
        )
    )
    if not has_cost_clause:
        return False
    if action_item.category == "no_operational_action" and re.search(r"\bcosts?\b", title_basis_text):
        return True
    return not bool(
        re.search(
            r"\b(?:remit|remitted|reconsider|set\s+aside|quash|allowed|dismissed|directed|shall|comply|"
            r"file|release|pay|appeal|record\s+update)\b",
            text.replace("notify parties of cost order", ""),
        )
    )


def _cost_only_no_operational_action(action_item: ActionItem) -> ActionItem:
    evidence = list(action_item.evidence)
    basis = action_item.legal_basis or action_item.direction_summary or "Cost clause does not create an operational task."
    return ActionItem(
        title="Record no cost recovery required",
        responsible_department=None,
        timeline=Timeline(raw_text=None, confidence=0.0, timeline_type="missing"),
        category="no_operational_action",
        priority="low",
        legal_basis=basis,
        confidence=min(action_item.confidence, 0.72),
        evidence=evidence,
        notes=["Cost clause records that no party-specific recovery or payment task is required."],
        action_id=action_item.action_id,
        direction_summary=basis,
        owner_source=None,
        timeline_type="missing",
        ambiguity_flags=[],
        decision_reason="The costs clause does not direct a government department to pay, recover, or notify.",
        review_recommendation="Verify the costs clause before marking this as no operational action.",
        requires_human_review=True,
    )


def _no_operational_action_item(action_payload: dict[str, Any], documents: list[Document]) -> ActionItem:
    summary = str(action_payload.get("context_summary") or "The operative order does not create an immediate government task.")
    evidence = _evidence_for_snippet(action_payload.get("evidence_snippet") or summary, documents)
    confidence = _calibrated_confidence(action_payload.get("confidence") or 0.65, empty=False, has_evidence=bool(evidence), has_owner=True)
    return ActionItem(
        title="Record no operational government action",
        responsible_department=None,
        timeline=Timeline(raw_text=None, confidence=0.0, timeline_type="missing"),
        category="no_operational_action",
        priority="low",
        legal_basis=summary,
        confidence=confidence,
        evidence=[evidence] if evidence else [],
        notes=["LLM classified the operative order as not requiring an operational government task."],
        action_id="action-no-operational-action",
        direction_summary=summary,
        owner_source="llm_action_plan",
        timeline_type="missing",
        ambiguity_flags=_normalise_risk_flags(action_payload.get("ambiguity_flags") or []),
        decision_reason=str(action_payload.get("decision_reason") or summary),
        review_recommendation=str(
            action_payload.get("review_recommendation")
            or "Verify the final order before marking this as no operational action."
        ),
        requires_human_review=True,
    )


def _agentic_pdf_context(documents: list[Document], context_plan: dict[str, Any], page_count: int) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    index = JudgmentEvidenceIndex.from_documents(documents)

    for query in _context_plan_queries(context_plan):
        for result in index.search(query, top_k=3):
            item = _result_to_context(result, label="agentic_query")
            item["query"] = query
            context.append(item)

    for request in _context_plan_page_requests(context_plan, page_count):
        context.extend(_documents_to_context(_select_pages(documents, start=request[0], end=request[1]), label="agentic_page_request", limit=None))

    if not context:
        context.extend(_top_reranked_context(documents, CASE_DETAIL_QUERY, top_k=6))
        context.extend(_documents_to_context(_select_last_pages(documents, page_count, count=3), label="agentic_default_last_pages", limit=None))

    return _dedupe_context(context)[:8]


def _context_plan_queries(context_plan: dict[str, Any]) -> list[str]:
    raw_queries = context_plan.get("queries") if isinstance(context_plan, dict) else None
    queries: list[str] = []
    for item in raw_queries or []:
        if isinstance(item, str):
            query = item
        elif isinstance(item, dict):
            query = str(item.get("query") or "")
        else:
            query = ""
        query = " ".join(query.split())
        if len(query) >= 8:
            queries.append(query[:240])
    return list(dict.fromkeys(queries))[:5]


def _context_plan_page_requests(context_plan: dict[str, Any], page_count: int) -> list[tuple[int, int]]:
    raw_requests = context_plan.get("page_requests") if isinstance(context_plan, dict) else None
    requests: list[tuple[int, int]] = []
    for item in raw_requests or []:
        spec = item if isinstance(item, str) else item.get("pages") if isinstance(item, dict) else None
        parsed = _parse_page_request(str(spec or ""), page_count)
        if parsed:
            requests.append(parsed)
    return list(dict.fromkeys(requests))[:4]


def _parse_page_request(spec: str, page_count: int) -> tuple[int, int] | None:
    value = spec.strip().lower().replace(" ", "")
    if value == "first_2":
        return (1, min(2, page_count))
    if value == "last_3":
        return (max(1, page_count - 2), page_count)
    page_match = re.fullmatch(r"page:(\d+)", value)
    if page_match:
        page = _clamp_page(int(page_match.group(1)), page_count)
        return (page, page)
    range_match = re.fullmatch(r"pages:(\d+)-(\d+)", value)
    if range_match:
        start = _clamp_page(int(range_match.group(1)), page_count)
        end = _clamp_page(int(range_match.group(2)), page_count)
        if start > end:
            start, end = end, start
        if end - start > 4:
            end = start + 4
        return (start, end)
    return None


def _clamp_page(page: int, page_count: int) -> int:
    return max(1, min(page, max(1, page_count)))


def _action_supporting_context(agentic_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_terms = ("order", "direct", "shall", "within", "dismiss", "allow", "quash", "dispose", "compliance", "bail")
    selected = [
        item for item in agentic_context
        if any(term in json.dumps(item, ensure_ascii=True).lower() for term in action_terms)
    ]
    return (selected or agentic_context)[:8]


def _dedupe_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in context:
        text = str(item.get("text") or item.get("snippet") or "")
        key = (item.get("page"), item.get("chunk_id"), text[:120])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _llm_risk_flags(extraction: JudgmentExtraction, action_items: list[ActionItem]) -> list[str]:
    flags: list[str] = []
    if not extraction.case_number.value:
        flags.append("missing_case_number")
    if not extraction.court.value:
        flags.append("missing_court")
    if not extraction.judgment_date.value:
        flags.append("missing_judgment_date")
    for item in action_items:
        if item.category not in {"no_operational_action", "no_immediate_action", "internal_review"} and not item.responsible_department:
            flags.append("owner_unclear")
        if item.category not in {"no_operational_action", "no_immediate_action", "internal_review"} and item.timeline.timeline_type == "missing":
            flags.append("missing_timeline")
        flags.extend(_normalise_risk_flags(item.ambiguity_flags))
    return list(dict.fromkeys(flags))


def _top_reranked_context(documents: list[Document], query: str, *, top_k: int) -> list[dict[str, Any]]:
    index = JudgmentEvidenceIndex.from_documents(documents)
    return [_result_to_context(result, label="reranked_evidence") for result in index.search(query, top_k=top_k)]


def _documents_to_context(documents: list[Document], *, label: str, limit: int | None = 1200) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text": _context_text(doc.page_content or "", limit=limit),
        }
        for doc in documents
        if (doc.page_content or "").strip()
    ]


def _result_to_context(result, *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "page": result.page,
        "chunk_id": result.chunk_id,
        "snippet": _compact_text(result.snippet or "", limit=1000),
        "retrieval_score": result.retrieval_score,
        "rerank_score": result.rerank_score,
        "match_strategy": result.match_strategy,
    }


def _select_pages(documents: list[Document], *, start: int, end: int) -> list[Document]:
    selected = [
        doc for doc in documents
        if isinstance((doc.metadata or {}).get("page"), int) and start <= int(doc.metadata["page"]) <= end
    ]
    return selected or documents[: max(1, end - start + 1)]


def _select_last_pages(documents: list[Document], page_count: int, *, count: int) -> list[Document]:
    start = max(1, page_count - count + 1)
    return _select_pages(documents, start=start, end=page_count)


def _select_final_pages(documents: list[Document], page_count: int, *, count: int) -> list[Document]:
    start = max(1, page_count - count + 1)
    return _select_pages(documents, start=start, end=page_count)


def _select_previous_pages(documents: list[Document], page_count: int, *, before_pages: int, count: int) -> list[Document]:
    end = max(1, page_count - before_pages)
    start = max(1, end - count + 1)
    if end >= page_count:
        return []
    return _select_pages(documents, start=start, end=end)


def _page_count(documents: list[Document], pdf_profile: dict[str, Any] | None) -> int:
    if pdf_profile and pdf_profile.get("page_count"):
        return int(pdf_profile["page_count"])
    pages = [int(doc.metadata["page"]) for doc in documents if isinstance((doc.metadata or {}).get("page"), int)]
    return max(pages or [len(documents) or 1])


def _needs_more_action_context(payload: dict[str, Any]) -> bool:
    return bool(payload.get("needs_more_context")) and not payload.get("action_items")


def _action_payload_needs_repair(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    for item in payload.get("action_items", []):
        if not isinstance(item, dict):
            continue
        text = " ".join(
            str(value or "")
            for value in [
                item.get("title"),
                item.get("responsible_department"),
                item.get("category"),
                item.get("legal_basis"),
                item.get("decision_reason"),
                item.get("evidence_snippet"),
                " ".join(str(flag or "") for flag in _as_list(item.get("ambiguity_flags"))),
            ]
        ).lower()
        timeline = item.get("timeline") if isinstance(item.get("timeline"), dict) else {}
        if not item.get("responsible_department") or "owner_unclear" in text or "owner unclear" in text:
            return True
        if timeline.get("timeline_type") == "missing" and re.search(r"\b(remit|remand|direct|shall|within)\b", text):
            return True
        if re.search(r"\b(update|record|reflect|case status|dashboard)\b", text) and item.get("category") != "internal_review":
            return True
        if re.search(r"\b(remit|remand|restore|fresh consideration)\b", text):
            return True
        if re.search(r"\b(court\s+master|p\.?\s*s\.?\s+to\s+registrar|ps\s+to\s+registrar)\b", text):
            return True
    return False


def _action_repair_context(documents: list[Document], page_count: int) -> list[dict[str, Any]]:
    if page_count <= 12:
        return _documents_to_context(documents, label="whole_document_repair", limit=None)
    selected: dict[tuple[int | None, str | None], Document] = {}
    candidate_docs = [
        *_select_pages(documents, start=1, end=2),
        *_select_final_pages(documents, page_count, count=6),
    ]
    operative_terms = re.compile(
        r"\b(?:accordingly|in\s+the\s+result|we\s+(?:direct|allow|dismiss|set\s+aside)|"
        r"remit|remand|restore|fresh\s+consideration|ordered|shall|within)\b",
        re.I,
    )
    candidate_docs.extend(doc for doc in documents if operative_terms.search(doc.page_content or ""))
    for doc in candidate_docs:
        metadata = doc.metadata or {}
        selected[(metadata.get("page"), metadata.get("chunk_id"))] = doc
    return _documents_to_context(list(selected.values()), label="targeted_document_repair", limit=None)


def _merge_action_repair_payload(
    current_payload: dict[str, Any],
    repair_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(repair_payload, dict) or not isinstance(repair_payload.get("action_items"), list):
        return current_payload
    repaired_items = [
        item for item in repair_payload.get("action_items", [])
        if isinstance(item, dict) and item.get("title") and item.get("evidence_snippet")
    ]
    if not repaired_items:
        return current_payload
    merged = dict(current_payload)
    merged["action_items"] = repaired_items
    merged["context_summary"] = repair_payload.get("context_summary") or current_payload.get("context_summary")
    merged["owner_timeline_repaired"] = True
    return merged


def _action_payload_from_package(package: JudgmentReviewPackage) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for item in package.action_items:
        evidence_snippet = _best_action_evidence_snippet(item)
        items.append(
            {
                "title": item.title,
                "responsible_department": item.responsible_department,
                "category": item.category,
                "priority": item.priority,
                "timeline": {
                    "raw_text": item.timeline.raw_text,
                    "due_date": item.timeline.due_date.isoformat() if item.timeline.due_date else None,
                    "timeline_type": item.timeline.timeline_type,
                    "confidence": item.timeline.confidence,
                },
                "legal_basis": item.legal_basis or item.direction_summary or evidence_snippet or item.title,
                "decision_reason": item.decision_reason
                or "Fallback deterministic action draft used because the LLM action pass returned no usable JSON.",
                "review_recommendation": item.review_recommendation
                or "Reviewer should verify this fallback action against the source evidence.",
                "requires_human_review": True,
                "confidence": item.confidence,
                "ambiguity_flags": list(item.ambiguity_flags),
                "evidence_snippet": evidence_snippet or item.legal_basis or item.title,
            }
        )
    return {
        "context_summary": "LLM action extraction returned no usable action items; deterministic action draft used as repair input.",
        "fallback_source": "deterministic_action_plan",
        "action_items": items,
    }


def _best_action_evidence_snippet(item: ActionItem) -> str | None:
    for evidence in item.evidence:
        if evidence.snippet:
            return evidence.snippet
    return item.legal_basis or item.direction_summary


def _payload_declares_no_action(payload: dict[str, Any]) -> bool:
    if payload.get("no_immediate_action") or payload.get("no_operational_action"):
        return True
    return str(payload.get("category") or "").lower() in {"no_immediate_action", "no_operational_action"}


def _has_no_operational_action(action_items: list[ActionItem]) -> bool:
    return any(item.category in {"no_operational_action", "no_immediate_action"} for item in action_items)


def _safe_package_hints(package: JudgmentReviewPackage) -> dict[str, Any]:
    extraction = package.extraction
    return {
        "case_number": extraction.case_number.value,
        "case_type": extraction.case_type.value,
        "court": extraction.court.value,
        "bench": extraction.bench.value,
        "judgment_date": str(extraction.judgment_date.value or ""),
        "parties": extraction.parties.value,
        "advocates": extraction.advocates.value,
        "disposition": extraction.disposition.value,
        "directions": [direction.value for direction in extraction.directions],
        "actions": [item.title for item in package.action_items],
    }


def _evidence_for_snippet(snippet: str | None, documents: list[Document]) -> SourceEvidence | None:
    if not snippet:
        return None
    lowered = snippet.lower()
    best_doc = None
    best_score = 0.0
    for doc in documents:
        text = (doc.page_content or "").lower()
        score = 1.0 if lowered in text else _token_overlap(lowered, text)
        if score > best_score:
            best_doc = doc
            best_score = score
    if not best_doc or best_score < 0.45:
        return None
    metadata = best_doc.metadata or {}
    return SourceEvidence(
        source_id=str(metadata.get("source") or "judgment"),
        page=metadata.get("page"),
        chunk_id=metadata.get("chunk_id"),
        snippet=snippet,
        confidence=round(0.95 if best_score >= 0.99 else min(0.88, max(0.45, best_score)), 2),
        extraction_method="llm_first",
        match_strategy="llm_cited_snippet",
        retrieval_score=round(best_score, 4),
    )


def _token_overlap(query: str, text: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokens(text)
    return len(query_tokens & text_tokens) / len(query_tokens)


def _tokens(value: str) -> set[str]:
    stopwords = {"the", "and", "or", "of", "in", "to", "a", "an", "is", "are", "was", "were", "for", "on"}
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in stopwords}


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"unknown", "none", "null", "not mentioned"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _calibrated_confidence(
    confidence: Any,
    *,
    empty: bool,
    has_evidence: bool,
    has_owner: bool = True,
) -> float:
    score = max(0.0, min(1.0, _safe_float(confidence)))
    if empty:
        return min(score, 0.3)
    if not has_evidence:
        score = min(score, 0.5)
    if not has_owner:
        score = min(score, 0.65)
    return round(score, 2)


def _normalise_action_category(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "no_immediate_action": "no_operational_action",
        "none": "no_operational_action",
        "no_action": "no_operational_action",
        "direct": "direct_compliance",
        "compliance": "direct_compliance",
        "follow_up": "conditional_follow_up",
        "conditional": "conditional_follow_up",
        "information_update": "record_update",
    }
    return mapping.get(raw, raw or "legal_review")


def _normalise_priority(value: Any) -> str:
    raw = str(value or "medium").strip().lower()
    return raw if raw in {"low", "medium", "high"} else "medium"


def _normalise_risk_flags(value: Any) -> list[str]:
    raw_items = value if isinstance(value, list) else [value]
    flags: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered == "timeline_not_specified":
            flags.append("timeline_not_specified")
            continue
        if "owner" in lowered or "department" in lowered or "authority" in lowered:
            flags.append("owner_unclear")
            continue
        if "timeline" in lowered or "deadline" in lowered or "date" in lowered:
            flags.append("missing_timeline")
            continue
        if "condition" in lowered or "if filed" in lowered or "if any" in lowered:
            flags.append("conditional_follow_up")
            continue
        if "evidence" in lowered or "source" in lowered:
            flags.append("source_review_required")
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
        if slug:
            flags.append(slug[:80])
    return list(dict.fromkeys(flags))


def _compact_text(text: str, *, limit: int) -> str:
    clean = " ".join(text.split())
    return clean[:limit]


def _context_text(text: str, *, limit: int | None) -> str:
    clean = " ".join(text.split())
    if limit is None:
        return clean
    return clean[:limit]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _overall_confidence(
    extraction: JudgmentExtraction,
    action_items: list[ActionItem],
    details_payload: dict[str, Any],
    action_payload: dict[str, Any],
) -> float:
    scores = [
        _safe_float(details_payload.get("confidence")),
        extraction.case_number.confidence,
        extraction.court.confidence,
        extraction.judgment_date.confidence,
        extraction.parties.confidence,
        *[item.confidence for item in action_items],
    ]
    if action_payload.get("needs_more_context"):
        scores.append(0.4)
    non_zero = [score for score in scores if score]
    return round(sum(non_zero) / len(non_zero), 2) if non_zero else 0.0
