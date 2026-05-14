from __future__ import annotations

from typing import Any


def build_record_metrics(record: dict[str, Any], audit_events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    audit_events = audit_events or []
    extraction = record.get("extraction", {})
    fields = [
        value for key, value in extraction.items()
        if key not in {"directions", "risk_flags"}
        and isinstance(value, dict)
        and value.get("value") not in (None, "", [], "unknown")
    ]
    directions = extraction.get("directions") or []
    if isinstance(directions, list):
        fields.extend(item for item in directions if isinstance(item, dict))
    action_items = record.get("action_items", [])

    evidence_targets = fields + [item for item in action_items if isinstance(item, dict)]
    covered = sum(1 for item in evidence_targets if item.get("evidence"))
    evidence_coverage = round((covered / len(evidence_targets)) * 100) if evidence_targets else 0

    extraction_methods = {}
    for item in evidence_targets:
        for evidence in item.get("evidence", []):
            method = evidence.get("extraction_method") or evidence.get("match_strategy") or "unknown"
            extraction_methods[method] = extraction_methods.get(method, 0) + 1

    review_edit_count = sum(
        1 for event in audit_events
        if event.get("event_type") in {"field_update", "action_update"}
    )
    ambiguous_count = sum(1 for item in fields if item.get("status") == "ambiguous")
    ambiguous_count += sum(len(item.get("ambiguity_flags", [])) for item in action_items if isinstance(item, dict))

    profile = record.get("pdf_profile") or {}
    processing = record.get("processing_metrics") or {}
    source_metadata = record.get("source_metadata") or {}
    ocr_routing = str(source_metadata.get("ocr_routing") or "")
    vision_ocr_used = (
        ocr_routing.startswith("vision_ocr")
        or bool(source_metadata.get("vision_fallback_used"))
        or bool(processing.get("extraction_methods", {}).get("vision_ocr"))
        or any(str(method).startswith("vision_") for method in extraction_methods)
    )
    ocr_used = bool(
        profile.get("ocr_used")
        or source_metadata.get("ocr_used")
        or vision_ocr_used
    )
    return {
        "record_id": record.get("record_id"),
        "processing_ms": processing.get("processing_ms"),
        "page_count": profile.get("page_count"),
        "ocr_used": ocr_used,
        "vision_ocr_used": vision_ocr_used,
        "ocr_pages": source_metadata.get("ocr_pages") or source_metadata.get("vision_pages") or [],
        "evidence_coverage_percent": evidence_coverage,
        "ambiguous_count": ambiguous_count,
        "duplicate_count": len(record.get("duplicate_candidates") or []),
        "review_edit_count": review_edit_count,
        "extraction_method_counts": extraction_methods,
        "overall_confidence": record.get("overall_confidence", 0),
    }


def build_dashboard_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "approved_or_completed_count": 0,
            "duplicate_count": 0,
            "average_confidence": 0,
            "evidence_coverage_percent": 0,
            "ocr_used": False,
            "vision_ocr_used": False,
        }
    metrics = [record.get("metrics") or build_record_metrics(record) for record in records]
    return {
        "record_count": len(records),
        "approved_or_completed_count": sum(
            1 for record in records if record.get("review_status") in {"approved", "edited", "completed"}
        ),
        "duplicate_count": sum(item.get("duplicate_count", 0) for item in metrics),
        "average_confidence": round(
            sum(float(record.get("overall_confidence", 0) or 0) for record in records) / len(records),
            2,
        ),
        "evidence_coverage_percent": round(
            sum(item.get("evidence_coverage_percent", 0) for item in metrics) / len(metrics)
        ),
        "ocr_used": any(bool(item.get("ocr_used")) for item in metrics),
        "vision_ocr_used": any(bool(item.get("vision_ocr_used")) for item in metrics),
    }
