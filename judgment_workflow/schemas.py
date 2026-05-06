from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JudgmentUploadResponse(BaseModel):
    message: str
    task_id: str | None = None
    record_id: str
    state: str = "queued"
    record: dict[str, Any] | None = None


class CCMSFetchRequest(BaseModel):
    user_id: str
    ccms_case_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionUpdateRequest(BaseModel):
    action_id: str
    title: str | None = None
    responsible_department: str | None = None
    priority: str | None = None
    category: str | None = None
    status: str | None = None
    legal_basis: str | None = None
    timeline: dict[str, Any] | None = None
    reason: str | None = None
    notes: str | None = None
    manual_override: bool | None = None


class FieldUpdateRequest(BaseModel):
    field_id: str | None = None
    value: Any | None = None
    status: str | None = None
    reason: str | None = None
    notes: str | None = None
    manual_override: bool | None = None


class JudgmentReviewRequest(BaseModel):
    reviewer_id: str
    decision: str
    reviewer_role: str | None = None
    notes: str | None = None
    extraction_updates: dict[str, Any | FieldUpdateRequest] = Field(default_factory=dict)
    action_updates: list[ActionUpdateRequest] = Field(default_factory=list)


class JudgmentQuestionRequest(BaseModel):
    user_id: str
    question: str


class DuplicateResolutionRequest(BaseModel):
    reviewer_id: str
    resolution: str
    duplicate_record_id: str | None = None
    notes: str | None = None
