from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from rag.judgment.dashboard import to_dashboard_record

from .metrics import build_record_metrics
from .serialization import (
    deserialize_review_package,
    serialize_dashboard_record,
    serialize_review_package,
)


_INVALID_DASHBOARD_OWNER_RE = re.compile(
    r"^(?:"
    r"delay(?:\s+in\s+filing\s+the\s+appeal)?|"
    r"compensation|"
    r"costs?|"
    r"appeals?|"
    r"petitions?|"
    r"writ(?:\s+petition)?|"
    r"case|"
    r"matter|"
    r"order|"
    r"judgment|"
    r"relief"
    r")$",
    re.I,
)

_OWNER_FROM_ACTION_RE = re.compile(
    r"\b(?:before|to|through|by)\s+(?:the\s+)?"
    r"(?P<owner>"
    r"Reference\s+Court|"
    r"Special\s+Land\s+Acquisition\s+Officer|"
    r"Spl\.?\s*LAO|"
    r"Registrar(?:\s+General)?|"
    r"High\s+Court(?:\s+Registry)?|"
    r"Registry|"
    r"Deputy\s+Commissioner|"
    r"State\s+Government|"
    r"Government\s+of\s+[A-Z][A-Za-z ]+"
    r")\b",
    re.I,
)


def _clean_dashboard_owner(owner: Any, *context_parts: Any) -> str:
    text = str(owner or "").strip(" .,:;-")
    if text and not _INVALID_DASHBOARD_OWNER_RE.match(text):
        return text
    context = " ".join(str(part or "") for part in context_parts)
    match = _OWNER_FROM_ACTION_RE.search(context)
    if match:
        found = re.sub(r"\s+", " ", match.group("owner").strip(" .,:;-"))
        if re.fullmatch(r"Spl\.?\s*LAO", found, flags=re.I):
            return "Special Land Acquisition Officer"
        return found
    return "Case reviewer"


def _missing_field_count(extraction: dict[str, Any]) -> int:
    count = 0
    for key, field in (extraction or {}).items():
        if key in {"directions", "risk_flags", "legal_phrases"} or not isinstance(field, dict):
            continue
        value = field.get("value")
        if value in (None, "", []) or str(value).strip().lower() == "unknown":
            count += 1
    return count


class JudgmentRepository:
    def __init__(self, storage, canvas_app_id: str):
        self.storage = storage
        self.canvas_app_id = canvas_app_id

    def _records_collection(self, user_id: str):
        return (
            self.storage.collection("artifacts")
            .document(self.canvas_app_id)
            .collection("users")
            .document(user_id)
            .collection("judgment_records")
        )

    def _record_ref(self, user_id: str, record_id: str):
        return self._records_collection(user_id).document(record_id)

    def _audit_collection(self, user_id: str, record_id: str):
        return self._record_ref(user_id, record_id).collection("audit_events")

    async def create_record(
        self,
        user_id: str,
        record_id: str,
        review_package,
        source_metadata: dict[str, Any],
        original_pdf_path: str,
        highlighted_pdf_path: str | None = None,
        processing_state: str = "processed",
        processing_errors: list[str] | None = None,
        document_hash: str | None = None,
        pdf_profile: dict[str, Any] | None = None,
        duplicate_candidates: list[dict[str, Any]] | None = None,
        processing_mode: str = "async",
    ) -> dict[str, Any]:
        serialized_package = serialize_review_package(review_package)
        now = datetime.now(UTC).isoformat()
        dashboard_record = self._dashboard_from_package(serialized_package)
        metrics = build_record_metrics(
            {
                **serialized_package,
                "record_id": record_id,
                "pdf_profile": dict(pdf_profile or {}),
                "duplicate_candidates": list(duplicate_candidates or []),
            }
        )
        payload = {
            "record_id": record_id,
            "user_id": user_id,
            "status": processing_state,
            "source_system": source_metadata.get("source_system", "manual_upload"),
            "ccms_case_id": source_metadata.get("ccms_case_id"),
            "original_pdf_path": original_pdf_path,
            "highlighted_pdf_path": highlighted_pdf_path,
            "processing_errors": list(processing_errors or []),
            "document_hash": document_hash,
            "pdf_profile": dict(pdf_profile or {}),
            "duplicate_candidates": list(duplicate_candidates or []),
            "processing_mode": processing_mode,
            "dashboard_record": dashboard_record,
            "metrics": metrics,
            "created_at": now,
            "updated_at": now,
            **serialized_package,
        }
        await self._record_ref(user_id, record_id).set(payload)
        await self._audit_collection(user_id, record_id).add(
            {
                "event_type": "record_created",
                "reviewer_id": None,
                "notes": "Judgment record created from processing pipeline.",
                "target_type": "record",
                "target_id": record_id,
                "created_at": now,
            }
        )
        return payload

    async def get_record(self, user_id: str, record_id: str) -> dict[str, Any] | None:
        snapshot = await self._record_ref(user_id, record_id).get()
        return snapshot.to_dict() if snapshot.exists else None

    async def delete_record(self, user_id: str, record_id: str) -> None:
        snapshot = await self._record_ref(user_id, record_id).get()
        if not snapshot.exists:
            raise ValueError(f"Judgment record '{record_id}' not found for user '{user_id}'.")
        await self._record_ref(user_id, record_id).delete()

    async def update_record_metadata(self, user_id: str, record_id: str, **updates) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        payload = {**updates, "updated_at": now}
        await self._record_ref(user_id, record_id).update(payload)
        snapshot = await self._record_ref(user_id, record_id).get()
        record = snapshot.to_dict() if snapshot.exists else payload
        if snapshot.exists and any(key in updates for key in {"processing_metrics", "pdf_profile", "duplicate_candidates"}):
            record["metrics"] = build_record_metrics(record, await self.list_audit_events(user_id, record_id))
            await self._record_ref(user_id, record_id).update({"metrics": record["metrics"], "updated_at": now})
        return record

    async def apply_review_decision(
        self,
        user_id: str,
        record_id: str,
        reviewer_id: str,
        decision: str,
        notes: str | None = None,
        extraction_updates: dict[str, Any] | None = None,
        action_updates: list[dict[str, Any]] | None = None,
        reviewer_role: str | None = None,
    ) -> dict[str, Any]:
        snapshot = await self._record_ref(user_id, record_id).get()
        if not snapshot.exists:
            raise ValueError(f"Judgment record '{record_id}' not found for user '{user_id}'.")

        record = snapshot.to_dict() or {}
        before_record = deepcopy(record)
        extraction_updates = extraction_updates or {}
        action_updates = action_updates or []
        now = datetime.now(UTC).isoformat()

        for field_name, update_payload in extraction_updates.items():
            if field_name in record.get("extraction", {}):
                field = record["extraction"][field_name]
                before_value = deepcopy(field)
                if isinstance(update_payload, dict):
                    new_value = update_payload.get("value", field.get("value"))
                    field["status"] = update_payload.get("status", field.get("status", "edited"))
                    field["reason"] = update_payload.get("reason", field.get("reason"))
                    field["reviewer_notes"] = update_payload.get("notes", field.get("reviewer_notes"))
                    field["manual_override"] = bool(update_payload.get("manual_override", field.get("manual_override", False)))
                else:
                    new_value = update_payload
                    field["status"] = "edited"
                field["value"] = new_value
                record["extraction"][field_name]["notes"] = (
                    record["extraction"][field_name].get("notes", []) + [f"Edited by {reviewer_id}"]
                )
                await self._audit_collection(user_id, record_id).add(
                    {
                        "event_type": "field_update",
                        "target_type": "extraction_field",
                        "target_id": field_name,
                        "before": before_value,
                        "after": deepcopy(field),
                        "reviewer_id": reviewer_id,
                        "role": reviewer_role,
                        "reason": field.get("reason"),
                        "notes": notes,
                        "created_at": now,
                    }
                )

        action_index = {
            action.get("action_id"): action for action in record.get("action_items", [])
        }
        for update in action_updates:
            action = action_index.get(update.get("action_id"))
            if not action:
                continue
            before_action = deepcopy(action)
            for key, value in update.items():
                if key == "action_id":
                    continue
                action[key] = value
            await self._audit_collection(user_id, record_id).add(
                {
                    "event_type": "action_update",
                    "target_type": "action_item",
                    "target_id": update.get("action_id"),
                    "before": before_action,
                    "after": action,
                    "reviewer_id": reviewer_id,
                    "role": reviewer_role,
                    "reason": update.get("reason"),
                    "notes": notes,
                    "created_at": now,
                }
            )

        normalized_decision = decision.strip().lower()
        status_map = {
            "approve": "approved",
            "approved": "approved",
            "edit": "edited",
            "edited": "edited",
            "reject": "rejected",
            "rejected": "rejected",
            "escalate": "needs_clarification",
            "needs_clarification": "needs_clarification",
            "complete": "completed",
            "completed": "completed",
        }
        record["review_status"] = status_map.get(normalized_decision, "needs_clarification")
        if record["review_status"] in {"approved", "edited", "completed"}:
            self._validate_approval(record)
        record["reviewer_id"] = reviewer_id
        record["reviewer_role"] = reviewer_role
        record["reviewer_notes"] = notes
        record["reviewed_at"] = now
        record["updated_at"] = now
        audit_for_metrics = await self.list_audit_events(user_id, record_id)
        record["metrics"] = build_record_metrics(record, audit_for_metrics)
        record["dashboard_record"] = self._dashboard_from_package(record)

        await self._record_ref(user_id, record_id).set(record, merge=False)
        await self._audit_collection(user_id, record_id).add(
            {
                "event_type": "review_decision",
                "target_type": "record",
                "target_id": record_id,
                "before": before_record.get("review_status"),
                "after": record["review_status"],
                "reviewer_id": reviewer_id,
                "role": reviewer_role,
                "notes": notes,
                "created_at": now,
            }
        )
        return record

    async def list_audit_events(self, user_id: str, record_id: str) -> list[dict[str, Any]]:
        events = []
        async for snapshot in self._audit_collection(user_id, record_id).stream():
            payload = snapshot.to_dict() or {}
            payload["event_id"] = snapshot.id
            events.append(payload)
        return sorted(events, key=lambda item: str(item.get("created_at", "")))

    async def list_dashboard_records(self, user_id: str) -> list[dict[str, Any]]:
        records = []
        async for snapshot in self._records_collection(user_id).stream():
            payload = snapshot.to_dict() or {}
            dashboard_record = payload.get("dashboard_record")
            if dashboard_record:
                merged = deepcopy(dashboard_record)
                merged["record_id"] = payload.get("record_id", snapshot.id)
                merged["action_register"] = self._clean_action_register(
                    merged.get("action_register") or self._action_register_from_payload(payload),
                    merged,
                )
                merged["departments"] = self._dashboard_departments(merged)
                metrics = dict(payload.get("metrics") or {})
                fresh_metrics = build_record_metrics(payload)
                metrics["ocr_used"] = bool(metrics.get("ocr_used") or fresh_metrics.get("ocr_used"))
                metrics["vision_ocr_used"] = bool(metrics.get("vision_ocr_used") or fresh_metrics.get("vision_ocr_used"))
                metrics["ocr_pages"] = metrics.get("ocr_pages") or fresh_metrics.get("ocr_pages")
                merged["metrics"] = metrics
                merged["overall_confidence"] = payload.get("overall_confidence", 0)
                merged["updated_at"] = payload.get("updated_at")
                records.append(merged)
        return sorted(records, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    async def list_records(
        self,
        user_id: str,
        *,
        status: str | None = None,
        department: str | None = None,
        role: str | None = None,
        case_query: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        async for snapshot in self._records_collection(user_id).stream():
            payload = snapshot.to_dict() or {}
            row = self._queue_row(payload, snapshot.id)
            if status and row.get("review_status") != status:
                continue
            if department and department.lower() not in " ".join(row.get("departments", [])).lower():
                continue
            if role and role != "all" and role not in " ".join(row.get("escalations", [])).lower():
                if not (role == "data_reviewer" and row.get("review_status") == "pending_review"):
                    continue
            if case_query:
                haystack = " ".join(str(row.get(key) or "") for key in ("case_number", "court", "record_id")).lower()
                if case_query.lower() not in haystack:
                    continue
            rows.append(row)
        rows = sorted(rows, key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return rows[: max(1, min(limit, 500))]

    async def find_duplicate_candidates(
        self,
        user_id: str,
        document_hash: str,
        *,
        exclude_record_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if not document_hash:
            return []
        matches = []
        async for snapshot in self._records_collection(user_id).stream():
            payload = snapshot.to_dict() or {}
            record_id = payload.get("record_id", snapshot.id)
            if exclude_record_id and record_id == exclude_record_id:
                continue
            if payload.get("document_hash") == document_hash:
                matches.append(self._queue_row(payload, snapshot.id))
        return matches

    async def get_cached_record_by_hash(
        self,
        user_id: str,
        document_hash: str,
        *,
        exclude_record_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not document_hash:
            return None
        candidates = []
        async for snapshot in self._records_collection(user_id).stream():
            payload = snapshot.to_dict() or {}
            record_id = payload.get("record_id", snapshot.id)
            if exclude_record_id and record_id == exclude_record_id:
                continue
            if payload.get("document_hash") == document_hash and payload.get("extraction") and payload.get("action_items"):
                candidates.append(payload)
        if not candidates:
            return None
        status_rank = {"approved": 0, "edited": 1, "completed": 2, "pending_review": 3}
        return sorted(
            candidates,
            key=lambda item: (
                status_rank.get(item.get("review_status"), 9),
                str(item.get("updated_at", "")),
            ),
        )[0]

    async def create_record_from_cache(
        self,
        *,
        user_id: str,
        record_id: str,
        cached_record: dict[str, Any],
        source_metadata: dict[str, Any],
        original_pdf_path: str,
        highlighted_pdf_path: str | None,
        document_hash: str,
        processing_mode: str = "cache_replay",
    ) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        payload = deepcopy(cached_record)
        cached_source_metadata = payload.get("source_metadata") or {}
        payload.update(
            {
                "record_id": record_id,
                "user_id": user_id,
                "status": "processed",
                "source_system": source_metadata.get("source_system", "manual_upload"),
                "ccms_case_id": source_metadata.get("ccms_case_id"),
                "original_pdf_path": original_pdf_path,
                "highlighted_pdf_path": highlighted_pdf_path,
                "processing_errors": [],
                "document_hash": document_hash,
                "duplicate_candidates": [],
                "processing_mode": processing_mode,
                "source_metadata": {
                    **cached_source_metadata,
                    **source_metadata,
                    "cache_replay": True,
                    "cache_source_record_id": cached_record.get("record_id"),
                },
                "created_at": now,
                "updated_at": now,
            }
        )
        payload["metrics"] = build_record_metrics(payload)
        payload["dashboard_record"] = self._dashboard_from_package(payload)
        await self._record_ref(user_id, record_id).set(payload)
        await self._audit_collection(user_id, record_id).add(
            {
                "event_type": "record_created",
                "reviewer_id": None,
                "notes": "Judgment record loaded from a verified cached processing package.",
                "target_type": "record",
                "target_id": record_id,
                "created_at": now,
            }
        )
        return payload

    async def find_near_duplicate_candidates(
        self,
        user_id: str,
        candidate: dict[str, Any],
        *,
        exclude_record_id: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = []
        async for snapshot in self._records_collection(user_id).stream():
            payload = snapshot.to_dict() or {}
            record_id = payload.get("record_id", snapshot.id)
            if exclude_record_id and record_id == exclude_record_id:
                continue
            score = _duplicate_score(payload, candidate)
            if score >= 0.58:
                row = self._queue_row(payload, snapshot.id)
                row["duplicate_score"] = round(score, 2)
                row["duplicate_reason"] = _duplicate_reason(payload, candidate)
                matches.append(row)
        return sorted(matches, key=lambda item: item["duplicate_score"], reverse=True)

    async def resolve_duplicate(
        self,
        user_id: str,
        record_id: str,
        reviewer_id: str,
        resolution: str,
        duplicate_record_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        snapshot = await self._record_ref(user_id, record_id).get()
        if not snapshot.exists:
            raise ValueError(f"Judgment record '{record_id}' not found for user '{user_id}'.")
        record = snapshot.to_dict() or {}
        now = datetime.now(UTC).isoformat()
        record["duplicate_resolution"] = {
            "resolution": resolution,
            "duplicate_record_id": duplicate_record_id,
            "reviewer_id": reviewer_id,
            "notes": notes,
            "created_at": now,
        }
        record["updated_at"] = now
        await self._record_ref(user_id, record_id).set(record, merge=False)
        await self._audit_collection(user_id, record_id).add(
            {
                "event_type": "duplicate_resolution",
                "target_type": "record",
                "target_id": record_id,
                "after": record["duplicate_resolution"],
                "reviewer_id": reviewer_id,
                "notes": notes,
                "created_at": now,
            }
        )
        return record

    def filter_dashboard_records(
        self,
        records: list[dict[str, Any]],
        *,
        department: str | None = None,
        action_type: str | None = None,
        review_status: str | None = None,
        deadline_from: str | None = None,
        deadline_to: str | None = None,
        case_query: str | None = None,
        priority: str | None = None,
    ) -> list[dict[str, Any]]:
        filtered = []
        for record in records:
            row_filtered_record = deepcopy(record)
            action_rows = [
                row
                for row in row_filtered_record.get("action_register", [])
                if isinstance(row, dict)
            ]
            owner_haystack = " ".join(
                str(value or "")
                for value in [
                    *(record.get("departments") or []),
                    *[
                        item.get("owner") or item.get("department") or ""
                        for item in record.get("action_register", [])
                        if isinstance(item, dict)
                    ],
                ]
            ).lower()
            if department and department.lower() not in owner_haystack:
                continue
            if action_type and action_type not in record.get("action_categories", []):
                continue
            if department or action_type:
                scoped_rows = []
                for row in action_rows:
                    owner_text = " ".join(str(row.get(key) or "") for key in ("owner", "department")).lower()
                    category_matches = not action_type or row.get("category") == action_type
                    department_matches = not department or department.lower() in owner_text
                    if category_matches and department_matches:
                        scoped_rows.append(row)
                if scoped_rows:
                    row_filtered_record["action_register"] = scoped_rows
                    row_filtered_record["departments"] = self._dashboard_departments(row_filtered_record)
            if review_status and record.get("review_status") != review_status:
                continue
            if priority and record.get("highest_priority") != priority:
                continue
            if case_query:
                haystack = " ".join(
                    str(value or "")
                    for value in [
                        record.get("case_number"),
                        record.get("court"),
                        record.get("record_id"),
                        *[
                            item.get("title") or ""
                            for item in record.get("action_register", [])
                            if isinstance(item, dict)
                        ],
                    ]
                ).lower()
                if case_query.lower() not in haystack:
                    continue
            due_dates = [str(item) for item in record.get("due_dates", []) if item]
            if deadline_from and due_dates and min(due_dates) < deadline_from:
                continue
            if deadline_to and due_dates and max(due_dates) > deadline_to:
                continue
            filtered.append(row_filtered_record)
        return filtered

    def _dashboard_from_package(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        dashboard_payload = deepcopy(payload)
        action_items = dashboard_payload.get("action_items", [])
        reviewed_actions = [
            action
            for action in action_items
            if action.get("status") in {"approved", "edited"}
        ]
        if reviewed_actions:
            dashboard_payload["action_items"] = reviewed_actions

        package = deserialize_review_package(dashboard_payload)
        record = to_dashboard_record(package)
        if record is None:
            return None
        return serialize_dashboard_record(record)

    def _action_register_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for action in payload.get("action_items", []):
            if action.get("category") == "internal_review":
                continue
            timeline = action.get("timeline") or {}
            evidence = action.get("evidence") or []
            first_evidence = evidence[0] if evidence else {}
            rows.append(
                {
                    "title": action.get("title"),
                    "owner": action.get("responsible_department"),
                    "department": action.get("responsible_department"),
                    "category": action.get("category"),
                    "priority": action.get("priority"),
                    "deadline": timeline.get("due_date"),
                    "timeline": timeline.get("raw_text"),
                    "timeline_type": timeline.get("timeline_type") or action.get("timeline_type"),
                    "status": payload.get("review_status") or action.get("status"),
                    "evidence_count": len(evidence),
                    "evidence_page": first_evidence.get("page"),
                }
            )
        return rows

    def _clean_action_register(self, rows: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
        cleaned = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            item = deepcopy(row)
            owner = _clean_dashboard_owner(
                item.get("owner") or item.get("department"),
                item.get("title"),
                item.get("timeline"),
            )
            item["owner"] = owner
            item["department"] = owner
            item["status"] = record.get("review_status") or item.get("status")
            cleaned.append(item)
        return cleaned

    def _dashboard_departments(self, record: dict[str, Any]) -> list[str]:
        departments = []
        for row in record.get("action_register", []):
            if not isinstance(row, dict):
                continue
            owner = row.get("owner") or row.get("department")
            if owner and owner not in departments:
                departments.append(owner)
        if departments:
            return departments
        return [item for item in record.get("departments", []) if item]

    def _queue_row(self, payload: dict[str, Any], snapshot_id: str) -> dict[str, Any]:
        extraction = payload.get("extraction", {})
        actions = payload.get("action_items", [])
        departments = _as_list((extraction.get("departments") or {}).get("value"))
        for action in actions:
            dept = action.get("responsible_department")
            if dept and dept not in departments:
                departments.append(dept)
        return {
            "record_id": payload.get("record_id", snapshot_id),
            "case_number": (extraction.get("case_number") or {}).get("value"),
            "court": (extraction.get("court") or {}).get("value"),
            "judgment_date": (extraction.get("judgment_date") or {}).get("value"),
            "departments": departments,
            "review_status": payload.get("review_status", "pending_review"),
            "risk_flags": payload.get("risk_flags", []),
            "missing_field_count": _missing_field_count(extraction),
            "action_count": len(actions),
            "action_categories": sorted({action.get("category") for action in actions if action.get("category")}),
            "escalations": sorted({action.get("escalation_recommendation") for action in actions if action.get("escalation_recommendation")}),
            "document_hash": payload.get("document_hash"),
            "pdf_profile": payload.get("pdf_profile", {}),
            "duplicate_candidates": payload.get("duplicate_candidates", []),
            "duplicate_resolution": payload.get("duplicate_resolution"),
            "original_file_name": (payload.get("source_metadata") or {}).get("original_file_name"),
            "metrics": payload.get("metrics", {}),
            "updated_at": payload.get("updated_at"),
        }

    def _validate_approval(self, record: dict[str, Any]) -> None:
        extraction = record.get("extraction", {})
        required_fields = ["case_number", "court", "judgment_date"]
        if not ((extraction.get("parties") or {}).get("value") or (extraction.get("departments") or {}).get("value")):
            required_fields.append("parties")
        for field_name in required_fields:
            field = extraction.get(field_name, {})
            if field.get("value") in (None, "", []):
                raise ValueError(f"Cannot approve: required field '{field_name}' is missing.")
            if not field.get("evidence") and not field.get("manual_override"):
                raise ValueError(f"Cannot approve: required field '{field_name}' has no source evidence.")
        for action in record.get("action_items", []):
            if action.get("category") == "no_immediate_action":
                continue
            if not action.get("evidence") and not action.get("manual_override"):
                raise ValueError("Cannot approve: every action item needs source evidence or manual override.")


def _duplicate_score(existing: dict[str, Any], candidate: dict[str, Any]) -> float:
    extraction = existing.get("extraction", {})
    existing_hash = existing.get("document_hash")
    if existing_hash and existing_hash == candidate.get("document_hash"):
        return 1.0
    parts = []
    parts.append(_similarity((extraction.get("case_number") or {}).get("value"), candidate.get("case_number")) * 0.45)
    parts.append(_similarity((extraction.get("court") or {}).get("value"), candidate.get("court")) * 0.2)
    parts.append((1.0 if str((extraction.get("judgment_date") or {}).get("value") or "") == str(candidate.get("judgment_date") or "") else 0.0) * 0.2)
    existing_name = (existing.get("source_metadata") or {}).get("original_file_name")
    parts.append(_similarity(existing_name, candidate.get("original_file_name")) * 0.15)
    return sum(parts)


def _duplicate_reason(existing: dict[str, Any], candidate: dict[str, Any]) -> str:
    extraction = existing.get("extraction", {})
    if existing.get("document_hash") == candidate.get("document_hash"):
        return "same_document_hash"
    if _similarity((extraction.get("case_number") or {}).get("value"), candidate.get("case_number")) > 0.92:
        return "same_case_number"
    return "similar_case_metadata"


def _similarity(left: Any, right: Any) -> float:
    left_text = str(left or "").lower().strip()
    right_text = str(right or "").lower().strip()
    if not left_text or not right_text:
        return 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]
