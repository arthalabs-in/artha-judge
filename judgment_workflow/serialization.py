from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import date, datetime
from typing import Any

from rag.judgment.types import (
    ActionStatus,
    ActionItem,
    DashboardRecord,
    ExtractedField,
    FieldReviewStatus,
    JudgmentExtraction,
    JudgmentReviewPackage,
    ReviewStatus,
    SourceEvidence,
    Timeline,
)


def serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_value(val) for key, val in value.items()}
    return deepcopy(value)


def serialize_source_evidence(evidence: SourceEvidence) -> dict[str, Any]:
    return serialize_value(asdict(evidence))


def serialize_extracted_field(field: ExtractedField, *, field_id: str | None = None) -> dict[str, Any]:
    return {
        "field_id": field_id or field.name,
        "name": field.name,
        "value": serialize_value(field.value),
        "ai_value": serialize_value(field.value),
        "raw_value": field.raw_value,
        "confidence": field.confidence,
        "evidence": [serialize_source_evidence(item) for item in field.evidence],
        "notes": list(field.notes),
        "status": _enum_value(field.status),
        "reason": field.reason,
        "reviewer_notes": field.reviewer_notes,
        "manual_override": field.manual_override,
        "requires_review": field.requires_review,
    }


def serialize_timeline(timeline: Timeline) -> dict[str, Any]:
    return {
        "raw_text": timeline.raw_text,
        "due_date": serialize_value(timeline.due_date),
        "confidence": timeline.confidence,
        "timeline_type": timeline.timeline_type,
    }


def serialize_action_item(action_item: ActionItem, *, action_id: str) -> dict[str, Any]:
    return {
        "action_id": action_id,
        "title": action_item.title,
        "responsible_department": action_item.responsible_department,
        "timeline": serialize_timeline(action_item.timeline),
        "category": action_item.category,
        "priority": action_item.priority,
        "legal_basis": action_item.legal_basis,
        "confidence": action_item.confidence,
        "evidence": [serialize_source_evidence(item) for item in action_item.evidence],
        "notes": list(action_item.notes),
        "status": _enum_value(getattr(action_item, "status", ActionStatus.PENDING)),
        "direction_summary": action_item.direction_summary,
        "owner_source": action_item.owner_source,
        "timeline_type": action_item.timeline_type,
        "ambiguity_flags": list(action_item.ambiguity_flags),
        "escalation_recommendation": action_item.escalation_recommendation,
        "decision_reason": action_item.decision_reason,
        "review_recommendation": action_item.review_recommendation,
        "requires_human_review": action_item.requires_human_review,
        "reviewer_notes": action_item.reviewer_notes,
    }


def serialize_extraction(extraction: JudgmentExtraction) -> dict[str, Any]:
    directions = [
        serialize_extracted_field(direction, field_id=f"direction-{index}")
        for index, direction in enumerate(extraction.directions)
    ]
    return {
        "case_number": serialize_extracted_field(extraction.case_number),
        "case_type": serialize_extracted_field(extraction.case_type),
        "court": serialize_extracted_field(extraction.court),
        "bench": serialize_extracted_field(extraction.bench),
        "judgment_date": serialize_extracted_field(extraction.judgment_date),
        "parties": serialize_extracted_field(extraction.parties),
        "petitioners": serialize_extracted_field(extraction.petitioners),
        "respondents": serialize_extracted_field(extraction.respondents),
        "departments": serialize_extracted_field(extraction.departments),
        "advocates": serialize_extracted_field(extraction.advocates),
        "disposition": serialize_extracted_field(extraction.disposition),
        "legal_phrases": serialize_extracted_field(extraction.legal_phrases),
        "directions": directions,
        "risk_flags": list(extraction.risk_flags),
    }


def serialize_review_package(review_package: JudgmentReviewPackage) -> dict[str, Any]:
    action_items = [
        serialize_action_item(item, action_id=f"action-{index}")
        for index, item in enumerate(review_package.action_items)
    ]
    return {
        "extraction": serialize_extraction(review_package.extraction),
        "action_items": action_items,
        "source_metadata": serialize_value(review_package.source_metadata),
        "review_status": str(review_package.review_status.value),
        "overall_confidence": review_package.overall_confidence,
        "risk_flags": list(review_package.risk_flags),
        "reviewer_id": review_package.reviewer_id,
        "reviewer_notes": review_package.reviewer_notes,
        "created_at": serialize_value(review_package.created_at),
        "reviewed_at": serialize_value(review_package.reviewed_at),
    }


def deserialize_date(value: Any) -> date | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def deserialize_datetime(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def deserialize_source_evidence(payload: dict[str, Any]) -> SourceEvidence:
    return SourceEvidence(
        source_id=str(payload.get("source_id", "unknown")),
        page=payload.get("page"),
        chunk_id=payload.get("chunk_id"),
        snippet=str(payload.get("snippet", "")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        extraction_method=str(payload.get("extraction_method", "deterministic")),
        bbox=payload.get("bbox"),
        locator_confidence=payload.get("locator_confidence"),
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        source_quality=str(payload.get("source_quality", "text")),
        quad_points=payload.get("quad_points"),
        match_strategy=payload.get("match_strategy"),
        retrieval_score=payload.get("retrieval_score"),
        rerank_score=payload.get("rerank_score"),
        layer=payload.get("layer"),
        page_role=payload.get("page_role"),
        ocr_confidence=payload.get("ocr_confidence"),
    )


def deserialize_extracted_field(payload: dict[str, Any], *, default_name: str) -> ExtractedField:
    raw_value = payload.get("value")
    value = raw_value
    if default_name == "judgment_date":
        value = deserialize_date(raw_value)
    return ExtractedField(
        name=str(payload.get("name", default_name)),
        value=value,
        raw_value=payload.get("raw_value"),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        evidence=[deserialize_source_evidence(item) for item in payload.get("evidence", [])],
        notes=list(payload.get("notes", [])),
        field_id=payload.get("field_id", default_name),
        status=_coerce_field_status(payload.get("status")),
        reason=payload.get("reason"),
        reviewer_notes=payload.get("reviewer_notes"),
        manual_override=bool(payload.get("manual_override", False)),
        requires_review=bool(payload.get("requires_review", False)),
    )


def deserialize_timeline(payload: dict[str, Any]) -> Timeline:
    return Timeline(
        raw_text=payload.get("raw_text"),
        due_date=deserialize_date(payload.get("due_date")),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        timeline_type=str(payload.get("timeline_type", "missing")),
    )


def deserialize_action_item(payload: dict[str, Any]) -> ActionItem:
    action_item = ActionItem(
        title=str(payload.get("title", "")),
        responsible_department=payload.get("responsible_department"),
        timeline=deserialize_timeline(payload.get("timeline", {})),
        category=str(payload.get("category", "compliance")),
        priority=str(payload.get("priority", "medium")),
        legal_basis=payload.get("legal_basis"),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        evidence=[deserialize_source_evidence(item) for item in payload.get("evidence", [])],
        notes=list(payload.get("notes", [])),
        action_id=payload.get("action_id"),
        direction_summary=payload.get("direction_summary"),
        owner_source=payload.get("owner_source"),
        timeline_type=str(payload.get("timeline_type", payload.get("timeline", {}).get("timeline_type", "missing"))),
        ambiguity_flags=list(payload.get("ambiguity_flags", [])),
        escalation_recommendation=payload.get("escalation_recommendation"),
        decision_reason=payload.get("decision_reason"),
        review_recommendation=payload.get("review_recommendation"),
        requires_human_review=bool(payload.get("requires_human_review", False)),
        status=_coerce_action_status(payload.get("status")),
        reviewer_notes=payload.get("reviewer_notes"),
    )
    setattr(action_item, "action_id", payload.get("action_id"))
    return action_item


def deserialize_review_package(payload: dict[str, Any]) -> JudgmentReviewPackage:
    extraction_payload = payload.get("extraction", {})
    extraction = JudgmentExtraction(
        case_number=deserialize_extracted_field(
            extraction_payload.get("case_number", {}), default_name="case_number"
        ),
        case_type=deserialize_extracted_field(
            extraction_payload.get("case_type", {"value": "unknown"}), default_name="case_type"
        ),
        court=deserialize_extracted_field(
            extraction_payload.get("court", {}), default_name="court"
        ),
        bench=deserialize_extracted_field(
            extraction_payload.get("bench", {"value": []}), default_name="bench"
        ),
        judgment_date=deserialize_extracted_field(
            extraction_payload.get("judgment_date", {}), default_name="judgment_date"
        ),
        parties=deserialize_extracted_field(
            extraction_payload.get("parties", {}), default_name="parties"
        ),
        petitioners=deserialize_extracted_field(
            extraction_payload.get("petitioners", {"value": []}), default_name="petitioners"
        ),
        respondents=deserialize_extracted_field(
            extraction_payload.get("respondents", {"value": []}), default_name="respondents"
        ),
        departments=deserialize_extracted_field(
            extraction_payload.get("departments", {"value": []}), default_name="departments"
        ),
        advocates=deserialize_extracted_field(
            extraction_payload.get("advocates", {"value": []}), default_name="advocates"
        ),
        disposition=deserialize_extracted_field(
            extraction_payload.get("disposition", {"value": "unknown"}), default_name="disposition"
        ),
        legal_phrases=deserialize_extracted_field(
            extraction_payload.get("legal_phrases", {"value": []}), default_name="legal_phrases"
        ),
        directions=[
            deserialize_extracted_field(item, default_name="direction")
            for item in extraction_payload.get("directions", [])
        ],
        risk_flags=list(extraction_payload.get("risk_flags", [])),
    )
    review_status = payload.get("review_status", ReviewStatus.PENDING.value)
    if review_status not in {item.value for item in ReviewStatus}:
        review_status = ReviewStatus.PENDING.value
    return JudgmentReviewPackage(
        extraction=extraction,
        action_items=[deserialize_action_item(item) for item in payload.get("action_items", [])],
        source_metadata=dict(payload.get("source_metadata", {})),
        review_status=ReviewStatus(review_status),
        overall_confidence=float(payload.get("overall_confidence", 0.0) or 0.0),
        risk_flags=list(payload.get("risk_flags", [])),
        reviewer_id=payload.get("reviewer_id"),
        reviewer_notes=payload.get("reviewer_notes"),
        created_at=deserialize_datetime(payload.get("created_at")) or datetime.utcnow(),
        reviewed_at=deserialize_datetime(payload.get("reviewed_at")),
    )


def serialize_dashboard_record(record: DashboardRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, dict):
        return serialize_value(record)
    return {
        "case_number": record.case_number,
        "court": record.court,
        "judgment_date": serialize_value(record.judgment_date),
        "departments": list(record.departments),
        "pending_actions": list(record.pending_actions),
        "review_status": str(record.review_status.value),
        "highest_priority": record.highest_priority,
        "risk_flags": list(record.risk_flags),
        "action_categories": list(record.action_categories),
        "due_dates": serialize_value(record.due_dates),
        "escalations": list(record.escalations),
        "action_register": serialize_value(record.action_register),
    }


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _coerce_field_status(value: Any) -> FieldReviewStatus:
    try:
        return FieldReviewStatus(value or FieldReviewStatus.PENDING.value)
    except ValueError:
        return FieldReviewStatus.PENDING


def _coerce_action_status(value: Any) -> ActionStatus:
    try:
        return ActionStatus(value or ActionStatus.PENDING.value)
    except ValueError:
        return ActionStatus.PENDING
