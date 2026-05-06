from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .types import SourceEvidence


def normalize_evidence(
    document: Document,
    matched_text: str | None = None,
    *,
    confidence: float = 0.9,
    extraction_method: str = "deterministic",
) -> SourceEvidence:
    metadata: dict[str, Any] = document.metadata or {}
    content = document.page_content or ""
    snippet = _build_snippet(content, matched_text)
    char_start, char_end = _find_offsets(content, matched_text)
    bbox = _to_bbox(metadata.get("approx_bbox") or metadata.get("bbox"))
    is_ocr = bool(metadata.get("ocr_used") or metadata.get("degraded_extraction"))

    return SourceEvidence(
        source_id=str(metadata.get("source") or metadata.get("file_name") or "unknown"),
        page=_to_int(metadata.get("page") or metadata.get("page_number")),
        chunk_id=_to_optional_str(metadata.get("chunk_id") or metadata.get("chunk_index")),
        snippet=snippet,
        confidence=confidence,
        extraction_method=extraction_method,
        bbox=bbox,
        locator_confidence=_to_float(metadata.get("locator_confidence")) or (0.85 if bbox else (0.35 if is_ocr else None)),
        char_start=char_start,
        char_end=char_end,
        source_quality=str(metadata.get("source_quality") or ("ocr" if is_ocr else "text")),
    )


def _build_snippet(content: str, matched_text: str | None, window: int = 120) -> str:
    clean_content = " ".join(content.split())
    if not matched_text:
        return clean_content[:window]

    clean_match = " ".join(matched_text.split())
    idx = clean_content.lower().find(clean_match.lower())
    if idx < 0:
        return clean_content[:window]

    start = max(0, idx - window // 3)
    end = min(len(clean_content), idx + len(clean_match) + window // 3)
    return clean_content[start:end]


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(part) for part in value]
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_offsets(content: str, matched_text: str | None) -> tuple[int | None, int | None]:
    if not content or not matched_text:
        return None, None
    idx = content.lower().find(matched_text.lower())
    if idx < 0:
        clean_match = " ".join(matched_text.split())
        idx = " ".join(content.split()).lower().find(clean_match.lower())
        if idx < 0:
            return None, None
        return idx, idx + len(clean_match)
    return idx, idx + len(matched_text)
