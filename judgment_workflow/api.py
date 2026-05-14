from __future__ import annotations

import csv
import io
import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from werkzeug.utils import secure_filename

from celery_config import celery_app
from storage import get_storage

from .ccms_client import CCMSClient
from .config import (
    CANVAS_APP_ID,
    CCMS_API_KEY,
    CCMS_BASE_URL,
    JUDGMENT_DATA_ROOT,
    JUDGMENT_SYNC_PROCESSING,
)
from .demo_seed import build_demo_pdf_bytes
from .metrics import build_dashboard_metrics, build_record_metrics
from .pdf_highlights import generate_highlighted_pdf, render_highlighted_page_png
from .pipeline import process_judgment_file
from .progress_jobs import progress_jobs, serialize_job, start_progress_task
from .repository import JudgmentRepository
from .schemas import (
    CCMSFetchRequest,
    DuplicateResolutionRequest,
    JudgmentQuestionRequest,
    JudgmentReviewRequest,
    JudgmentUploadResponse,
)


judgment_router = APIRouter(prefix="/judgments", tags=["judgments"])


def _safe_user_id(user_id: str) -> str:
    safe = secure_filename(str(user_id or "").strip())
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid user_id provided.")
    return safe


def _record_root(user_id: str, record_id: str) -> Path:
    data_root = Path(JUDGMENT_DATA_ROOT).resolve()
    record_root = (data_root / user_id / record_id).resolve()
    try:
        record_root.relative_to(data_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid record path.") from exc
    return record_root


@judgment_router.post("/upload", response_model=JudgmentUploadResponse)
async def upload_judgment(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    sync: bool = Query(False),
):
    user_id = _safe_user_id(user_id)
    sanitized_filename = secure_filename(file.filename or "")
    if not sanitized_filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")

    record_id = uuid.uuid4().hex
    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(await file.read())

    return await _queue_or_process(
        user_id=user_id,
        pdf_path=str(pdf_path),
        record_id=record_id,
        original_file_name=sanitized_filename,
        source_metadata={"source_system": "manual_upload", "original_file_name": sanitized_filename},
        sync=sync,
        message=f"Judgment '{sanitized_filename}' uploaded for processing.",
    )


@judgment_router.post("/upload-progress")
async def upload_judgment_with_progress(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    user_id = _safe_user_id(user_id)
    sanitized_filename = secure_filename(file.filename or "")
    if not sanitized_filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")

    record_id = uuid.uuid4().hex
    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(await file.read())

    job = progress_jobs.create(record_id=record_id)

    async def process(progress_callback):
        return await process_judgment_file(
            user_id=user_id,
            pdf_path=str(pdf_path),
            record_id=record_id,
            original_file_name=sanitized_filename,
            source_metadata={"source_system": "manual_upload", "original_file_name": sanitized_filename},
            storage=get_storage(),
            canvas_app_id=CANVAS_APP_ID,
            processing_mode="progress",
            progress_callback=progress_callback,
        )

    start_progress_task(job=job, process=process)
    return serialize_job(job)


@judgment_router.post("/from-ccms", response_model=JudgmentUploadResponse)
async def upload_judgment_from_ccms(payload: CCMSFetchRequest, sync: bool = Query(False)):
    user_id = _safe_user_id(payload.user_id)
    record_id = uuid.uuid4().hex
    client = CCMSClient(base_url=CCMS_BASE_URL, api_key=CCMS_API_KEY)
    try:
        case_payload = await client.fetch_case(payload.ccms_case_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch CCMS case: {exc}") from exc

    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(case_payload["pdf_bytes"])
    metadata = {
        "source_system": "ccms",
        "ccms_case_id": payload.ccms_case_id,
        **case_payload.get("metadata", {}),
        **payload.metadata,
    }
    if metadata.get("source_system") == "mock_ccms":
        metadata["source_system"] = "mock_ccms"

    return await _queue_or_process(
        user_id=user_id,
        pdf_path=str(pdf_path),
        record_id=record_id,
        original_file_name=f"{payload.ccms_case_id}.pdf",
        source_metadata=metadata,
        sync=sync,
        message=f"CCMS case '{payload.ccms_case_id}' queued for processing.",
    )


@judgment_router.post("/from-ccms-progress")
async def upload_judgment_from_ccms_with_progress(payload: CCMSFetchRequest):
    user_id = _safe_user_id(payload.user_id)
    record_id = uuid.uuid4().hex
    client = CCMSClient(base_url=CCMS_BASE_URL, api_key=CCMS_API_KEY)
    try:
        case_payload = await client.fetch_case(payload.ccms_case_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch CCMS case: {exc}") from exc

    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(case_payload["pdf_bytes"])
    metadata = {
        "source_system": "ccms",
        "ccms_case_id": payload.ccms_case_id,
        **case_payload.get("metadata", {}),
        **payload.metadata,
    }
    if metadata.get("source_system") == "mock_ccms":
        metadata["source_system"] = "mock_ccms"

    job = progress_jobs.create(record_id=record_id)

    async def process(progress_callback):
        return await process_judgment_file(
            user_id=user_id,
            pdf_path=str(pdf_path),
            record_id=record_id,
            original_file_name=f"{payload.ccms_case_id}.pdf",
            source_metadata=metadata,
            storage=get_storage(),
            canvas_app_id=CANVAS_APP_ID,
            processing_mode="progress",
            progress_callback=progress_callback,
        )

    start_progress_task(job=job, process=process)
    return serialize_job(job)


@judgment_router.post("/demo-seed", response_model=JudgmentUploadResponse)
async def create_demo_seed(user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    record_id = uuid.uuid4().hex
    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(build_demo_pdf_bytes())
    return await _queue_or_process(
        user_id=user_id,
        pdf_path=str(pdf_path),
        record_id=record_id,
        original_file_name="34897.pdf",
        source_metadata={"source_system": "demo_seed", "ccms_case_id": "DEMO-34897"},
        sync=True,
        message="Demo judgment 34897.pdf created for review.",
    )


@judgment_router.post("/demo-seed-progress")
async def create_demo_seed_with_progress(user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    record_id = uuid.uuid4().hex
    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(build_demo_pdf_bytes())

    job = progress_jobs.create(record_id=record_id)

    async def process(progress_callback):
        return await process_judgment_file(
            user_id=user_id,
            pdf_path=str(pdf_path),
            record_id=record_id,
            original_file_name="34897.pdf",
            source_metadata={"source_system": "demo_seed", "ccms_case_id": "DEMO-34897"},
            storage=get_storage(),
            canvas_app_id=CANVAS_APP_ID,
            processing_mode="progress",
            progress_callback=progress_callback,
        )

    start_progress_task(job=job, process=process)
    return serialize_job(job)


@judgment_router.get("/jobs/{job_id}")
async def get_progress_job(job_id: str):
    job = progress_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Progress job not found.")
    return serialize_job(job)


@judgment_router.post("/evaluate")
async def evaluate_judgment_pdf(
    user_id: str = Form("evaluation"),
    file: UploadFile = File(...),
    include_full_record: bool = Form(False),
    llm_enabled: bool = Form(True),
    metadata_json: str | None = Form(None),
):
    user_id = _safe_user_id(user_id)
    sanitized_filename = secure_filename(file.filename or "")
    if not sanitized_filename:
        raise HTTPException(status_code=400, detail="Invalid filename provided.")

    source_metadata = {
        "source_system": "evaluation_upload",
        "original_file_name": sanitized_filename,
    }
    if metadata_json:
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata_json must be valid JSON.") from exc
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="metadata_json must be a JSON object.")
        source_metadata.update(metadata)
        source_metadata["source_system"] = "evaluation_upload"
        source_metadata["original_file_name"] = sanitized_filename

    record_id = uuid.uuid4().hex
    record_root = _record_root(user_id, record_id)
    record_root.mkdir(parents=True, exist_ok=True)
    pdf_path = record_root / "original.pdf"
    pdf_path.write_bytes(await file.read())

    try:
        record = await process_judgment_file(
            user_id=user_id,
            pdf_path=str(pdf_path),
            record_id=record_id,
            original_file_name=sanitized_filename,
            source_metadata=source_metadata,
            storage=get_storage(),
            canvas_app_id=CANVAS_APP_ID,
            llm_enabled=llm_enabled,
            processing_mode="evaluation",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _build_evaluation_response(record, include_full_record=include_full_record)


@judgment_router.get("/records/{user_id}")
async def list_judgment_records(
    user_id: str,
    status: str | None = Query(None),
    department: str | None = Query(None),
    role: str | None = Query(None),
    case_query: str | None = Query(None),
    limit: int = Query(100),
):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    return {
        "records": await repository.list_records(
            user_id,
            status=status,
            department=department,
            role=role,
            case_query=case_query,
            limit=limit,
        )
    }


@judgment_router.get("/dashboard/{user_id}")
async def get_dashboard_records(
    user_id: str,
    department: str | None = Query(None),
    action_type: str | None = Query(None),
    review_status: str | None = Query(None),
    deadline_from: str | None = Query(None),
    deadline_to: str | None = Query(None),
    case_query: str | None = Query(None),
    priority: str | None = Query(None),
):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    records = await repository.list_dashboard_records(user_id)
    filtered = repository.filter_dashboard_records(
        records,
        department=department,
        action_type=action_type,
        review_status=review_status,
        deadline_from=deadline_from,
        deadline_to=deadline_to,
        case_query=case_query,
        priority=priority,
    )
    return {
        "records": filtered,
        "metrics": build_dashboard_metrics(filtered),
    }


@judgment_router.get("/dashboard/{user_id}/export")
async def export_dashboard_records(user_id: str, format: str = Query("json")):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    records = await repository.list_dashboard_records(user_id)
    if format.lower() == "json":
        return {"records": records}
    if format.lower() != "csv":
        raise HTTPException(status_code=400, detail="Unsupported export format.")
    fieldnames = [
        "record_id",
        "case_number",
        "court",
        "judgment_date",
        "departments",
        "pending_actions",
        "review_status",
        "highest_priority",
        "action_categories",
        "due_dates",
        "action_register",
        "risk_flags",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({key: _csv_value(record.get(key)) for key in fieldnames})
    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="judgment_dashboard.csv"'},
    )


@judgment_router.get("/{record_id}")
async def get_judgment_record(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")
    return record


@judgment_router.delete("/{record_id}")
async def delete_judgment_record(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    try:
        await repository.delete_record(user_id, record_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"record_id": record_id, "deleted": True}


@judgment_router.get("/{record_id}/metrics")
async def get_judgment_metrics(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")
    audit_events = await repository.list_audit_events(user_id, record_id)
    return {"metrics": build_record_metrics(record, audit_events)}


@judgment_router.get("/{record_id}/duplicates")
async def get_judgment_duplicates(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")
    extraction = record.get("extraction", {})
    candidate = {
        "document_hash": record.get("document_hash"),
        "case_number": (extraction.get("case_number") or {}).get("value"),
        "court": (extraction.get("court") or {}).get("value"),
        "judgment_date": (extraction.get("judgment_date") or {}).get("value"),
        "original_file_name": (record.get("source_metadata") or {}).get("original_file_name"),
    }
    candidates = await repository.find_near_duplicate_candidates(
        user_id, candidate, exclude_record_id=record_id
    )
    return {
        "candidates": candidates,
        "resolution": record.get("duplicate_resolution"),
    }


@judgment_router.post("/{record_id}/duplicates/resolve")
async def resolve_judgment_duplicate(
    record_id: str,
    payload: DuplicateResolutionRequest,
    user_id: str = Query(...),
):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    try:
        return await repository.resolve_duplicate(
            user_id=user_id,
            record_id=record_id,
            reviewer_id=payload.reviewer_id,
            resolution=payload.resolution,
            duplicate_record_id=payload.duplicate_record_id,
            notes=payload.notes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@judgment_router.post("/{record_id}/review")
async def review_judgment_record(
    record_id: str,
    payload: JudgmentReviewRequest,
    user_id: str = Query(...),
):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    extraction_updates = {
        key: value.model_dump(exclude_none=True) if hasattr(value, "model_dump") else value
        for key, value in payload.extraction_updates.items()
    }
    try:
        record = await repository.apply_review_decision(
            user_id=user_id,
            record_id=record_id,
            reviewer_id=payload.reviewer_id,
            reviewer_role=payload.reviewer_role,
            decision=payload.decision,
            notes=payload.notes,
            extraction_updates=extraction_updates,
            action_updates=[item.model_dump(exclude_none=True) for item in payload.action_updates],
        )
    except ValueError as exc:
        status_code = 400 if str(exc).startswith("Cannot approve") else 404
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return record


@judgment_router.post("/{record_id}/ask")
async def ask_judgment_record(record_id: str, payload: JudgmentQuestionRequest):
    user_id = _safe_user_id(payload.user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")
    return _answer_from_record(record, payload.question)


@judgment_router.get("/{record_id}/audit")
async def get_judgment_audit(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    return {"events": await repository.list_audit_events(user_id, record_id)}


@judgment_router.get("/{record_id}/debug-extraction")
async def get_judgment_extraction_debug(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")
    return {
        "record_id": record_id,
        "user_id": user_id,
        "extraction_debug": record.get("extraction_debug")
        or (record.get("source_metadata") or {}).get("extraction_debug")
        or {},
    }


@judgment_router.get("/{record_id}/highlighted-pdf")
async def get_highlighted_pdf(record_id: str, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")

    highlighted_pdf_path = record.get("highlighted_pdf_path")
    original_pdf_path = record.get("original_pdf_path")
    if not original_pdf_path or not os.path.exists(original_pdf_path):
        raise HTTPException(status_code=404, detail="Original PDF not found.")

    if not highlighted_pdf_path:
        highlighted_pdf_path = str(_record_root(user_id, record_id) / "highlighted.pdf")
    if not os.path.exists(highlighted_pdf_path):
        generate_highlighted_pdf(original_pdf_path, highlighted_pdf_path, record)
        await repository.update_record_metadata(
            user_id, record_id, highlighted_pdf_path=highlighted_pdf_path
        )

    return FileResponse(
        highlighted_pdf_path,
        media_type="application/pdf",
        filename="highlighted-judgment.pdf",
        content_disposition_type="inline",
    )


@judgment_router.get("/{record_id}/highlighted-page/{page_number}")
async def get_highlighted_page(record_id: str, page_number: int, user_id: str = Query(...)):
    user_id = _safe_user_id(user_id)
    repository = JudgmentRepository(storage=get_storage(), canvas_app_id=CANVAS_APP_ID)
    record = await repository.get_record(user_id, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Judgment record not found.")

    highlighted_pdf_path = record.get("highlighted_pdf_path")
    original_pdf_path = record.get("original_pdf_path")
    if not original_pdf_path or not os.path.exists(original_pdf_path):
        raise HTTPException(status_code=404, detail="Original PDF not found.")

    if not highlighted_pdf_path:
        highlighted_pdf_path = str(_record_root(user_id, record_id) / "highlighted.pdf")
    if not os.path.exists(highlighted_pdf_path):
        generate_highlighted_pdf(original_pdf_path, highlighted_pdf_path, record)
        await repository.update_record_metadata(
            user_id, record_id, highlighted_pdf_path=highlighted_pdf_path
        )

    page_count = int((record.get("pdf_profile") or {}).get("page_count") or 1)
    safe_page = max(1, min(int(page_number), page_count))
    png = render_highlighted_page_png(highlighted_pdf_path, safe_page)
    return StreamingResponse(io.BytesIO(png), media_type="image/png")


async def _queue_or_process(
    *,
    user_id: str,
    pdf_path: str,
    record_id: str,
    original_file_name: str,
    source_metadata: dict,
    sync: bool,
    message: str,
) -> JudgmentUploadResponse:
    if sync or JUDGMENT_SYNC_PROCESSING:
        record = await process_judgment_file(
            user_id=user_id,
            pdf_path=pdf_path,
            record_id=record_id,
            original_file_name=original_file_name,
            source_metadata=source_metadata,
            storage=get_storage(),
            canvas_app_id=CANVAS_APP_ID,
            processing_mode="sync",
        )
        return JudgmentUploadResponse(
            message=message,
            task_id=None,
            record_id=record_id,
            state="SUCCESS",
            record=record,
        )

    task = celery_app.send_task(
        "process_judgment_task",
        kwargs={
            "user_id": user_id,
            "pdf_path": pdf_path,
            "record_id": record_id,
            "original_file_name": original_file_name,
            "source_metadata": source_metadata,
        },
    )
    return JudgmentUploadResponse(message=message, task_id=task.id, record_id=record_id, state="queued")


def _csv_value(value):
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _build_evaluation_response(record: dict[str, Any], *, include_full_record: bool = False) -> dict[str, Any]:
    extraction = record.get("extraction") or {}
    source_metadata = record.get("source_metadata") or {}
    metrics = record.get("metrics") or build_record_metrics(record)
    response = {
        "schema_version": "judgment-evaluation-v1",
        "record_id": record.get("record_id"),
        "user_id": record.get("user_id"),
        "input": {
            "source_system": record.get("source_system") or source_metadata.get("source_system"),
            "original_file_name": source_metadata.get("original_file_name"),
            "document_hash": record.get("document_hash"),
            "pdf_profile": record.get("pdf_profile") or {},
            "source_metadata": source_metadata,
        },
        "review": {
            "status": record.get("review_status"),
            "overall_confidence": record.get("overall_confidence", 0),
            "reviewer_id": record.get("reviewer_id"),
            "reviewer_notes": record.get("reviewer_notes"),
            "created_at": record.get("created_at"),
            "reviewed_at": record.get("reviewed_at"),
        },
        "metadata": {
            "llm_enabled": bool(source_metadata.get("llm_enabled")),
            "llm_used": bool(source_metadata.get("llm_used")),
            "llm_review_mode": source_metadata.get("llm_review_mode"),
            "llm_action_second_pass": source_metadata.get("llm_action_second_pass"),
            "llm_model": source_metadata.get("llm_model"),
            "source_metadata": source_metadata,
        },
        "extraction": {
            "fields": _evaluation_fields(extraction),
            "directions": [
                _evaluation_field_payload("direction", direction)
                for direction in extraction.get("directions", [])
                if isinstance(direction, dict)
            ],
            "risk_flags": list(extraction.get("risk_flags") or []),
        },
        "action_plan": {
            "items": [_evaluation_action_payload(item) for item in record.get("action_items", [])],
        },
        "quality": {
            "risk_flags": list(record.get("risk_flags") or []),
            "duplicate_candidates": _evaluation_duplicate_candidates(record.get("duplicate_candidates") or []),
            "duplicate_resolution": record.get("duplicate_resolution"),
            "metrics": metrics,
            "processing_metrics": record.get("processing_metrics") or {},
            "processing_errors": record.get("processing_errors") or [],
        },
        "artifacts": {
            "original_pdf_path": record.get("original_pdf_path"),
            "highlighted_pdf_path": record.get("highlighted_pdf_path"),
            "highlighted_pdf_url": f"/judgments/{record.get('record_id')}/highlighted-pdf?user_id={record.get('user_id')}"
            if record.get("record_id") and record.get("user_id")
            else None,
        },
    }
    if include_full_record:
        response["record"] = record
    return response


def _evaluation_fields(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    fields = []
    for field_name, payload in extraction.items():
        if field_name in {"directions", "risk_flags"} or not isinstance(payload, dict):
            continue
        fields.append(_evaluation_field_payload(field_name, payload))
    return fields


def _evaluation_field_payload(field_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "field": field_name,
        "field_id": payload.get("field_id") or field_name,
        "label": _humanize_label(field_name),
        "value": payload.get("value"),
        "ai_value": payload.get("ai_value", payload.get("value")),
        "confidence": payload.get("confidence", 0),
        "status": payload.get("status"),
        "reason": payload.get("reason"),
        "requires_review": bool(payload.get("requires_review", False)),
        "manual_override": bool(payload.get("manual_override", False)),
        "evidence": [_evaluation_evidence_payload(item) for item in payload.get("evidence", [])],
    }


def _evaluation_action_payload(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_id": action.get("action_id"),
        "title": action.get("title"),
        "owner": action.get("responsible_department"),
        "category": action.get("category"),
        "priority": action.get("priority"),
        "status": action.get("status"),
        "timeline": action.get("timeline") or {},
        "legal_basis": action.get("legal_basis"),
        "direction_summary": action.get("direction_summary"),
        "confidence": action.get("confidence", 0),
        "ambiguity_flags": list(action.get("ambiguity_flags") or []),
        "escalation_recommendation": action.get("escalation_recommendation"),
        "decision_reason": action.get("decision_reason"),
        "review_recommendation": action.get("review_recommendation"),
        "requires_human_review": bool(action.get("requires_human_review", False)),
        "reviewer_notes": action.get("reviewer_notes"),
        "evidence": [_evaluation_evidence_payload(item) for item in action.get("evidence", [])],
    }


def _evaluation_duplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in candidates[:10]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "record_id": item.get("record_id"),
                "case_number": item.get("case_number"),
                "court": item.get("court"),
                "judgment_date": item.get("judgment_date"),
                "original_file_name": item.get("original_file_name"),
                "duplicate_score": item.get("duplicate_score"),
                "duplicate_reasons": item.get("duplicate_reasons"),
            }
        )
    return compact


def _evaluation_evidence_payload(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": evidence.get("source_id"),
        "page": evidence.get("page"),
        "chunk_id": evidence.get("chunk_id"),
        "snippet": evidence.get("snippet"),
        "confidence": evidence.get("confidence"),
        "bbox": evidence.get("bbox"),
        "quad_points": evidence.get("quad_points"),
        "char_start": evidence.get("char_start"),
        "char_end": evidence.get("char_end"),
        "extraction_method": evidence.get("extraction_method"),
        "match_strategy": evidence.get("match_strategy"),
        "retrieval_score": evidence.get("retrieval_score"),
        "rerank_score": evidence.get("rerank_score"),
        "layer": evidence.get("layer"),
        "page_role": evidence.get("page_role"),
        "source_quality": evidence.get("source_quality"),
        "ocr_confidence": evidence.get("ocr_confidence"),
    }


def _humanize_label(value: str) -> str:
    return str(value).replace("_", " ").strip().title()


def _answer_from_record(record: dict, question: str) -> dict:
    question_lc = question.lower()
    extraction = record.get("extraction", {})
    actions = record.get("action_items", [])

    if any(term in question_lc for term in ("action", "do", "next", "deadline", "due")):
        answer = _format_actions_answer(actions)
        sources = _sources_from_actions(actions)
    elif any(term in question_lc for term in ("party", "petitioner", "respondent")):
        answer = (
            f"Petitioners: {_field_value(extraction, 'petitioners') or _field_value(extraction, 'parties') or 'not extracted'}. "
            f"Respondents: {_field_value(extraction, 'respondents') or 'not extracted'}."
        )
        sources = _sources_from_fields(extraction, ["petitioners", "respondents", "parties"])
    elif "date" in question_lc or "when" in question_lc:
        answer = f"Judgment date: {_field_value(extraction, 'judgment_date') or 'not extracted'}."
        sources = _sources_from_fields(extraction, ["judgment_date"])
    else:
        answer = (
            f"{_field_value(extraction, 'case_number') or 'This judgment'} from "
            f"{_field_value(extraction, 'court') or 'the extracted court'} has disposition "
            f"{_field_value(extraction, 'disposition') or 'unknown'} and "
            f"{len(actions)} extracted action item(s)."
        )
        sources = _sources_from_fields(extraction, ["case_number", "court", "disposition"])

    return {
        "answer": answer,
        "grounded": True,
        "record_id": record.get("record_id"),
        "sources": sources[:5],
    }


def _format_actions_answer(actions: list[dict]) -> str:
    if not actions:
        return "No action items were extracted from this judgment."
    parts = []
    for action in actions[:5]:
        owner = action.get("responsible_department") or "owner unclear"
        due_date = (action.get("timeline") or {}).get("due_date") or "no explicit due date"
        parts.append(f"{action.get('title', 'Untitled action')} [{action.get('category', 'uncategorized')}], owner: {owner}, due: {due_date}")
    return " Extracted action items: " + " | ".join(parts)


def _field_value(extraction: dict, field_name: str):
    value = (extraction.get(field_name) or {}).get("value")
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return value


def _sources_from_fields(extraction: dict, field_names: list[str]) -> list[dict]:
    sources = []
    for field_name in field_names:
        for evidence in (extraction.get(field_name) or {}).get("evidence", []):
            sources.append(
                {
                    "field": field_name,
                    "page": evidence.get("page"),
                    "snippet": evidence.get("snippet"),
                    "confidence": evidence.get("confidence"),
                }
            )
    return sources


def _sources_from_actions(actions: list[dict]) -> list[dict]:
    sources = []
    for action in actions:
        for evidence in action.get("evidence", []):
            sources.append(
                {
                    "action_id": action.get("action_id"),
                    "page": evidence.get("page"),
                    "snippet": evidence.get("snippet"),
                    "confidence": evidence.get("confidence"),
                }
            )
    return sources
