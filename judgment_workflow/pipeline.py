from __future__ import annotations

import uuid
import asyncio
import json
from time import perf_counter
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.judgment import build_judgment_review_package
from rag.judgment.owner_resolution import apply_inferred_action_owner

from .config import JUDGMENT_DATA_ROOT
from .document_layers import extract_layered_pdf_documents, metadata_for_metrics
from .document_profile import compute_document_hash, profile_from_ocr_status
from .llm_extractor import enrich_review_package_with_llm
from .llm_review_workflow import build_llm_first_review_package
from .ocr import detect_ocr_need, ocr_pdf_with_tesseract
from .repository import JudgmentRepository
from .vision_extraction import (
    get_configured_vision_extractor,
    merge_vision_extraction,
    select_vision_pages,
    should_run_vision_fallback,
)


async def process_judgment_file(
    *,
    user_id: str,
    pdf_path: str,
    record_id: str | None = None,
    original_file_name: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    storage=None,
    canvas_app_id: str,
    embedding_model=None,
    progress_callback=None,
    llm_enabled: bool = True,
    processing_mode: str = "async",
) -> dict[str, Any]:
    started_at = perf_counter()
    stage_timings: list[dict[str, Any]] = []

    def record_stage(stage: str, stage_started_at: float, **metadata: Any) -> None:
        entry: dict[str, Any] = {
            "stage": stage,
            "duration_ms": round((perf_counter() - stage_started_at) * 1000),
        }
        if metadata:
            entry["metadata"] = metadata
        stage_timings.append(entry)

    async def timed_to_thread(stage: str, func, *args, **kwargs):
        stage_started_at = perf_counter()
        result = await asyncio.to_thread(func, *args, **kwargs)
        record_stage(stage, stage_started_at)
        return result

    async def timed_await(stage: str, awaitable):
        stage_started_at = perf_counter()
        result = await awaitable
        record_stage(stage, stage_started_at)
        return result

    record_id = record_id or uuid.uuid4().hex
    source_metadata = dict(source_metadata or {})
    if original_file_name:
        source_metadata.setdefault("original_file_name", original_file_name)
    source_metadata["llm_enabled"] = bool(llm_enabled)
    repository = JudgmentRepository(storage=storage, canvas_app_id=canvas_app_id)

    if progress_callback:
        progress_callback(stage="judgment_processing", message="Extracting judgment content...", pct=15.0)

    document_hash = await timed_to_thread("document_hash", compute_document_hash, pdf_path)
    cached_record = await repository.get_cached_record_by_hash(
        user_id,
        document_hash,
        exclude_record_id=record_id,
    )
    if cached_record:
        return await _create_cached_record_replay(
            repository=repository,
            user_id=user_id,
            record_id=record_id,
            cached_record=cached_record,
            pdf_path=pdf_path,
            original_file_name=original_file_name,
            source_metadata=source_metadata,
            document_hash=document_hash,
            started_at=started_at,
            stage_timings=stage_timings,
            progress_callback=progress_callback,
        )

    ocr_detection_task = asyncio.create_task(timed_to_thread("ocr_detection", detect_ocr_need, pdf_path))
    if embedding_model is None:
        extraction_task = asyncio.create_task(
            timed_to_thread("pdf_text_extraction", extract_layered_pdf_documents, pdf_path)
        )
    else:
        try:
            from pdf_processor import process_pdf_phoenix_v2
        except ModuleNotFoundError as exc:
            if exc.name != "pdf_processor":
                raise
            source_metadata["phoenix_extractor_unavailable"] = True
            extraction_task = asyncio.create_task(
                timed_to_thread("pdf_text_extraction", extract_layered_pdf_documents, pdf_path)
            )
            embedding_model = None
        else:
            extraction_task = asyncio.create_task(
                timed_await(
                    "pdf_phoenix_extraction",
                    process_pdf_phoenix_v2(
                        pdf_path=pdf_path,
                        embedding_model=embedding_model,
                        enable_image_captioning=False,
                        progress_callback=progress_callback,
                    ),
                )
            )

    duplicate_task = asyncio.create_task(
        timed_await(
            "exact_duplicate_lookup",
            repository.find_duplicate_candidates(user_id, document_hash, exclude_record_id=record_id),
        )
    )
    ocr_status, extraction_result, duplicate_candidates = await asyncio.gather(
        ocr_detection_task,
        extraction_task,
        duplicate_task,
    )
    stage_started_at = perf_counter()
    pdf_profile = profile_from_ocr_status(pdf_path, ocr_status)
    record_stage("pdf_profile", stage_started_at)
    text_layer_rejected = not bool(ocr_status.get("text_layer_reliable", True))
    vision_ocr_required = bool(ocr_status.get("needs_ocr"))
    vision_extractor = get_configured_vision_extractor(force_ollama=vision_ocr_required)
    if text_layer_rejected:
        source_metadata["text_layer_rejected"] = True
        source_metadata["ocr_routing"] = "vision_ocr_only"
        source_metadata["vision_ocr_model"] = getattr(vision_extractor, "model", "openbmb/minicpm-o2.6:latest")
        pdf_profile["text_layer_rejected"] = True
    elif vision_ocr_required and vision_extractor is not None:
        source_metadata["ocr_routing"] = "vision_ocr_fallback"
        source_metadata["vision_ocr_model"] = getattr(vision_extractor, "model", "openbmb/minicpm-o2.6:latest")
    if duplicate_candidates:
        source_metadata["duplicate_warning"] = True
    if embedding_model is None:
        documents = extraction_result
        table_documents = []
    else:
        documents, _, table_documents, _ = extraction_result

    if progress_callback:
        progress_callback(stage="ocr_detection", message="Checking whether OCR is needed...", pct=35.0)

    ocr_documents = []
    if ocr_status["needs_ocr"] and vision_extractor is None:
        ocr_documents, ocr_result = await timed_to_thread(
            "ocr_execution",
            ocr_pdf_with_tesseract,
            pdf_path,
            target_pages=ocr_status.get("sparse_pages"),
        )
        pdf_profile["ocr_used"] = bool(ocr_result.get("ocr_used", False))
        source_metadata["ocr_used"] = ocr_result.get("ocr_used", False)
        source_metadata["ocr_pages"] = ocr_result.get("pages", [])
        if ocr_result.get("reason"):
            source_metadata["ocr_reason"] = ocr_result["reason"]
            pdf_profile["ocr_reason"] = ocr_result["reason"]
    elif text_layer_rejected:
        source_metadata["ocr_reason"] = "corrupted_text_layer_routed_to_minicpm_vision"
        pdf_profile["ocr_reason"] = source_metadata["ocr_reason"]
    elif vision_ocr_required:
        source_metadata["ocr_reason"] = "routed_to_minicpm_vision"
        pdf_profile["ocr_reason"] = source_metadata["ocr_reason"]

    merged_documents = _prepare_review_documents(
        documents=documents,
        table_documents=table_documents,
        ocr_documents=ocr_documents,
        pdf_path=pdf_path,
        original_file_name=original_file_name,
        source_metadata=source_metadata,
        pdf_profile=pdf_profile,
        vision_enabled=vision_extractor is not None,
        discard_text_layer=text_layer_rejected,
    )
    precomputed_vision_result = None
    precomputed_vision_pages: list[int] = []
    if text_layer_rejected and vision_extractor is not None:
        precomputed_vision_pages = select_vision_pages(merged_documents, pdf_profile)
        source_metadata["vision_pages"] = precomputed_vision_pages
        try:
            precomputed_vision_result = await timed_await(
                "vision_ocr_extraction",
                vision_extractor.extract(
                    pdf_path=pdf_path,
                    pages=precomputed_vision_pages,
                    deterministic_summary={
                        "case_number": None,
                        "court": None,
                        "bench": [],
                        "judgment_date": None,
                        "parties": [],
                        "petitioners": [],
                        "respondents": [],
                        "disposition": None,
                        "risk_flags": ["text_layer_rejected", "ocr_required"],
                    },
                ),
            )
        except Exception as exc:
            source_metadata["vision_error"] = str(exc)
            source_metadata["vision_fallback_used"] = False
            source_metadata["vision_provider"] = getattr(vision_extractor, "provider", "minicpm") or "minicpm"
            source_metadata["vision_failure_stage"] = "vision_ocr_extraction"
        if precomputed_vision_result is not None:
            source_metadata["vision_provider"] = getattr(precomputed_vision_result, "provider", "minicpm") or "minicpm"
            source_metadata["vision_raw_json"] = precomputed_vision_result.raw_json
            source_metadata["vision_fallback_used"] = True
            vision_documents = _vision_result_documents(
                precomputed_vision_result,
                pdf_path=pdf_path,
                original_file_name=original_file_name,
                source_metadata=source_metadata,
                pdf_profile=pdf_profile,
            )
            if vision_documents:
                merged_documents = vision_documents

    if progress_callback:
        progress_callback(stage="judgment_extraction", message="Building review package...", pct=55.0)

    if llm_enabled:
        review_package = await timed_await(
            "llm_review_package",
            build_llm_first_review_package(
                merged_documents,
                source_metadata,
                pdf_profile=pdf_profile,
            ),
        )
    else:
        stage_started_at = perf_counter()
        review_package = build_judgment_review_package(merged_documents, source_metadata)
        record_stage("deterministic_review_package", stage_started_at)

    stage_started_at = perf_counter()
    review_package = await _apply_vision_fallback_if_needed(
        review_package,
        documents=merged_documents,
        pdf_path=pdf_path,
        pdf_profile=pdf_profile,
        vision_extractor=vision_extractor,
        vision_result=precomputed_vision_result,
        vision_pages=precomputed_vision_pages,
    )
    record_stage(
        "vision_fallback",
        stage_started_at,
        enabled=bool(review_package.source_metadata.get("vision_fallback_enabled")),
        used=bool(review_package.source_metadata.get("vision_fallback_used")),
    )
    review_package.action_items = [
        apply_inferred_action_owner(action_item, review_package.extraction)
        for action_item in review_package.action_items
    ]
    candidate_metadata = {
        "document_hash": document_hash,
        "case_number": review_package.extraction.case_number.value,
        "court": review_package.extraction.court.value,
        "judgment_date": review_package.extraction.judgment_date.value.isoformat()
        if hasattr(review_package.extraction.judgment_date.value, "isoformat")
        else review_package.extraction.judgment_date.value,
        "original_file_name": original_file_name,
    }
    near_duplicate_task = asyncio.create_task(
        timed_await(
            "near_duplicate_lookup",
            repository.find_near_duplicate_candidates(user_id, candidate_metadata, exclude_record_id=record_id),
        )
    )
    vision_ocr_succeeded = (
        str(source_metadata.get("ocr_routing") or "").startswith("vision_ocr")
        and bool(review_package.source_metadata.get("vision_fallback_used"))
    )
    if pdf_profile.get("profile_type") != "digital" and not vision_ocr_succeeded and "ocr_review_required" not in review_package.risk_flags:
        review_package.risk_flags.append("ocr_review_required")
    if ocr_status.get("needs_ocr") and not ocr_status.get("ocr_available") and not vision_ocr_succeeded and "ocr_unavailable" not in review_package.risk_flags:
        review_package.risk_flags.append("ocr_unavailable")
    source_metadata["llm_used"] = False
    review_package.source_metadata["llm_used"] = False
    if llm_enabled and review_package.source_metadata.get("llm_review_mode") != "llm_first":
        enriched = await enrich_review_package_with_llm(review_package, documents=merged_documents)
        if enriched is not review_package:
            review_package = enriched
            source_metadata["llm_used"] = True
            review_package.source_metadata["llm_used"] = True
    elif llm_enabled:
        source_metadata["llm_used"] = True
        review_package.source_metadata["llm_used"] = True

    record_root = Path(JUDGMENT_DATA_ROOT) / user_id / record_id
    record_root.mkdir(parents=True, exist_ok=True)
    highlighted_pdf_path = str(record_root / "highlighted.pdf")

    if progress_callback:
        progress_callback(stage="highlight_generation", message="Preparing source proof viewer...", pct=75.0)

    source_metadata["highlight_generation_mode"] = "deferred"
    review_package.source_metadata["highlight_generation_mode"] = "deferred"
    stage_started_at = perf_counter()
    record_stage("highlight_deferred", stage_started_at)
    near_duplicate_candidates = await near_duplicate_task
    duplicate_candidates = _merge_duplicate_candidates(duplicate_candidates, near_duplicate_candidates)
    if duplicate_candidates and "possible_duplicate" not in review_package.risk_flags:
        review_package.risk_flags.append("possible_duplicate")

    if progress_callback:
        progress_callback(stage="record_storage", message="Saving judgment record...", pct=90.0)

    record = await timed_await(
        "record_create",
        repository.create_record(
            user_id=user_id,
            record_id=record_id,
            review_package=review_package,
            source_metadata=source_metadata,
            original_pdf_path=pdf_path,
            highlighted_pdf_path=highlighted_pdf_path,
            document_hash=document_hash,
            pdf_profile=pdf_profile,
            duplicate_candidates=duplicate_candidates,
            processing_errors=None,
            processing_mode=processing_mode,
        ),
    )
    processing_metrics = {
        **metadata_for_metrics(merged_documents),
        "processing_ms": round((perf_counter() - started_at) * 1000),
        "stage_timings": stage_timings,
        "llm_stage_timings": review_package.source_metadata.get("llm_stage_timings", []),
    }
    stage_started_at = perf_counter()
    record = await repository.update_record_metadata(user_id, record_id, processing_metrics=processing_metrics)
    record_stage("record_metadata_update", stage_started_at)
    processing_metrics["stage_timings"] = stage_timings
    record["processing_metrics"] = processing_metrics

    if progress_callback:
        progress_callback(stage="complete", message="Judgment workflow complete.", pct=100.0)

    return record


async def _create_cached_record_replay(
    *,
    repository: JudgmentRepository,
    user_id: str,
    record_id: str,
    cached_record: dict[str, Any],
    pdf_path: str,
    original_file_name: str | None,
    source_metadata: dict[str, Any],
    document_hash: str,
    started_at: float,
    stage_timings: list[dict[str, Any]],
    progress_callback=None,
) -> dict[str, Any]:
    if original_file_name:
        source_metadata.setdefault("original_file_name", original_file_name)
    await _replay_cached_progress(progress_callback)
    record_root = Path(JUDGMENT_DATA_ROOT) / user_id / record_id
    record_root.mkdir(parents=True, exist_ok=True)
    highlighted_pdf_path = str(record_root / "highlighted.pdf")
    record = await repository.create_record_from_cache(
        user_id=user_id,
        record_id=record_id,
        cached_record=cached_record,
        source_metadata=source_metadata,
        original_pdf_path=pdf_path,
        highlighted_pdf_path=highlighted_pdf_path,
        document_hash=document_hash,
        processing_mode="cache_replay",
    )
    processing_metrics = {
        **(cached_record.get("processing_metrics") or {}),
        "cache_hit": True,
        "cache_source_record_id": cached_record.get("record_id"),
        "processing_ms": round((perf_counter() - started_at) * 1000),
        "stage_timings": [
            *stage_timings,
            {"stage": "cache_replay", "duration_ms": round((perf_counter() - started_at) * 1000)},
        ],
    }
    record = await repository.update_record_metadata(user_id, record_id, processing_metrics=processing_metrics)
    record["processing_metrics"] = processing_metrics
    return record


async def _replay_cached_progress(progress_callback) -> None:
    if not progress_callback:
        return
    stages = [
        ("ocr_detection", "Checking whether OCR is needed...", 35.0),
        ("judgment_extraction", "Loading verified extraction and action plan...", 55.0),
        ("highlight_generation", "Preparing source proof viewer...", 75.0),
        ("record_storage", "Saving review package...", 90.0),
        ("complete", "Judgment workflow complete.", 100.0),
    ]
    for stage, message, pct in stages:
        await asyncio.sleep(0.35)
        progress_callback(stage=stage, message=message, pct=pct)


async def _apply_vision_fallback_if_needed(
    review_package,
    *,
    documents,
    pdf_path: str,
    pdf_profile: dict[str, Any],
    vision_extractor=None,
    vision_result=None,
    vision_pages: list[int] | None = None,
):
    extractor = vision_extractor or get_configured_vision_extractor()
    enabled = extractor is not None or vision_result is not None
    review_package.source_metadata["vision_fallback_enabled"] = enabled
    review_package.source_metadata.setdefault("vision_fallback_used", False)
    if vision_result is not None:
        review_package.source_metadata["vision_pages"] = vision_pages or []
        return merge_vision_extraction(
            review_package,
            vision_result,
            documents=documents,
            provider_name=getattr(vision_result, "provider", "minicpm") or "minicpm",
        )
    if not enabled:
        return review_package
    if not should_run_vision_fallback(review_package, documents, pdf_profile):
        return review_package
    pages = select_vision_pages(documents, pdf_profile)
    review_package.source_metadata["vision_pages"] = pages
    if not pages:
        return review_package
    try:
        result = await extractor.extract(
            pdf_path=pdf_path,
            pages=pages,
            deterministic_summary=_vision_deterministic_summary(review_package),
        )
    except Exception as exc:
        review_package.source_metadata["vision_error"] = str(exc)
        if "vision_fallback_unavailable" not in review_package.risk_flags:
            review_package.risk_flags.append("vision_fallback_unavailable")
        return review_package
    return merge_vision_extraction(
        review_package,
        result,
        documents=documents,
        provider_name=getattr(result, "provider", "minicpm") or "minicpm",
    )


def _vision_deterministic_summary(review_package) -> dict[str, Any]:
    extraction = review_package.extraction
    return {
        "case_number": extraction.case_number.value,
        "court": extraction.court.value,
        "bench": extraction.bench.value,
        "judgment_date": extraction.judgment_date.value,
        "parties": extraction.parties.value,
        "petitioners": extraction.petitioners.value,
        "respondents": extraction.respondents.value,
        "disposition": extraction.disposition.value,
        "risk_flags": list(review_package.risk_flags),
    }


def _prepare_review_documents(
    *,
    documents,
    table_documents,
    ocr_documents,
    pdf_path: str,
    original_file_name: str | None,
    source_metadata: dict[str, Any],
    pdf_profile: dict[str, Any],
    vision_enabled: bool,
    discard_text_layer: bool = False,
) -> list[Document]:
    merged_documents = [] if discard_text_layer else list(documents) + list(table_documents)
    if ocr_documents:
        merged_documents.extend(ocr_documents)

    if not merged_documents:
        if not vision_enabled:
            raise ValueError(f"No reviewable documents were extracted from '{pdf_path}'.")
        merged_documents.append(
            Document(
                page_content="",
                metadata={
                    "source": pdf_path,
                    "page": 1,
                    "chunk_id": "vision-placeholder-p1",
                    "source_quality": "vision_only",
                    "extraction_method": "vision_ocr",
                    "profile_type": pdf_profile.get("profile_type"),
                },
            )
        )

    for document in merged_documents:
        metadata = dict(document.metadata or {})
        metadata.setdefault("source", pdf_path)
        if original_file_name:
            metadata.setdefault("original_file_name", original_file_name)
        if source_metadata.get("source_system"):
            metadata.setdefault("source_system", source_metadata["source_system"])
        if source_metadata.get("ccms_case_id"):
            metadata.setdefault("ccms_case_id", source_metadata["ccms_case_id"])
        document.metadata = metadata
    return merged_documents


def _vision_result_documents(
    vision_result,
    *,
    pdf_path: str,
    original_file_name: str | None,
    source_metadata: dict[str, Any],
    pdf_profile: dict[str, Any],
) -> list[Document]:
    page_numbers = {
        int(page)
        for page in (vision_result.evidence_pages or {}).values()
        if str(page).isdigit() and int(page) > 0
    }
    if not page_numbers:
        page_numbers.add(1)
    content = (
        "VISION OCR EXTRACTION FROM RENDERED PDF PAGES\n"
        "This document is generated by the local MiniCPM vision OCR model after rejecting a corrupted embedded text layer.\n"
        "Use this clean OCR text for downstream judgment extraction and action planning.\n"
        f"{_vision_structured_ocr_text(vision_result)}"
    )
    documents = []
    for page in sorted(page_numbers):
        metadata = {
            "source": pdf_path,
            "page": page,
            "chunk_id": f"vision-ocr-p{page}",
            "document_type": "pdf",
            "source_quality": "vision_ocr",
            "extraction_method": "vision_ocr",
            "extraction_layer": "vision_ocr",
            "degraded_extraction": True,
            "ocr_used": True,
            "vision_provider": getattr(vision_result, "provider", "minicpm") or "minicpm",
            "profile_type": pdf_profile.get("profile_type"),
        }
        if original_file_name:
            metadata["original_file_name"] = original_file_name
        if source_metadata.get("source_system"):
            metadata["source_system"] = source_metadata["source_system"]
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def _vision_structured_ocr_text(vision_result) -> str:
    raw_json = vision_result.raw_json or {}
    fields = vision_result.fields or {}
    lines = ["VISION OCR STRUCTURED TEXT"]
    for label, key in (
        ("case_number", "case_number"),
        ("case_type", "case_type"),
        ("court", "court"),
        ("judgment_date", "judgment_date"),
        ("disposition", "disposition"),
    ):
        value = _vision_scalar_text(fields.get(key))
        if value not in (None, "", []):
            lines.append(f"{label}: {value}.")
    for label, key in (
        ("bench", "bench"),
        ("petitioners", "petitioners"),
        ("respondents", "respondents"),
        ("departments", "departments"),
        ("responsible_entities", "responsible_entities"),
        ("advocates", "advocates"),
    ):
        values = [str(item).strip() for item in (fields.get(key) or []) if str(item or "").strip()]
        if values:
            lines.append(f"{label}: {', '.join(values)}.")
    lines.append("VISION OCR ACTION CONTEXT.")
    final_excerpt = _vision_scalar_text(raw_json.get("verbatim_final_order_excerpt"))
    if final_excerpt:
        lines.append(f"final_order_excerpt: {final_excerpt}")
    seen_directions: set[str] = set()
    if vision_result.directions:
        lines.append("Final order follows.")
        for direction in vision_result.directions:
            text = _vision_scalar_text(direction)
            if text and text.lower() not in seen_directions:
                seen_directions.add(text.lower())
                lines.append(text)
    raw_directions = raw_json.get("key_directions_orders")
    if isinstance(raw_directions, list):
        raw_texts = []
        for item in raw_directions:
            if isinstance(item, dict):
                text = item.get("text") or item.get("direction") or item.get("order")
            else:
                text = item
            clean_text = _vision_scalar_text(text)
            if clean_text and clean_text.lower() not in seen_directions:
                raw_texts.append(clean_text)
                seen_directions.add(clean_text.lower())
        if raw_texts:
            lines.append("Additional final order follows.")
            lines.extend(raw_texts)
    timelines = raw_json.get("relevant_timelines")
    if isinstance(timelines, list):
        timeline_texts = []
        for item in timelines:
            if isinstance(item, dict):
                text = item.get("text") or item.get("raw_text")
            else:
                text = item
            clean_text = _vision_scalar_text(text)
            if clean_text:
                timeline_texts.append(clean_text)
        if timeline_texts:
            lines.append("Timelines.")
            lines.extend(f"- {text}" for text in timeline_texts)
    return "\n".join(lines)


def _vision_scalar_text(value: Any) -> str | None:
    if value in (None, "", []):
        return None
    if isinstance(value, dict):
        for key in ("text", "raw_text", "value", "status", "reason", "order", "direction"):
            text = _vision_scalar_text(value.get(key))
            if text:
                return text
        return None
    if isinstance(value, list):
        parts = [text for item in value if (text := _vision_scalar_text(item))]
        return ", ".join(parts) if parts else None
    return str(value).strip()


def _merge_duplicate_candidates(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        for item in group or []:
            record_id = item.get("record_id")
            if not record_id:
                continue
            current = merged.get(record_id, {})
            merged[record_id] = _compact_duplicate_candidate({**current, **item})
    return sorted(
        merged.values(),
        key=lambda item: float(item.get("duplicate_score", 1 if item.get("document_hash") else 0) or 0),
        reverse=True,
    )


def _compact_duplicate_candidate(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": item.get("record_id"),
        "document_hash": item.get("document_hash"),
        "case_number": item.get("case_number"),
        "court": item.get("court"),
        "judgment_date": item.get("judgment_date"),
        "original_file_name": item.get("original_file_name"),
        "duplicate_score": item.get("duplicate_score"),
        "duplicate_reasons": item.get("duplicate_reasons"),
        "matched_on": item.get("matched_on"),
    }
