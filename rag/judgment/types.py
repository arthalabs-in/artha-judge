from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any


class ReviewStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    NEEDS_CLARIFICATION = "needs_clarification"
    COMPLETED = "completed"


class FieldReviewStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    MANUALLY_ENTERED = "manually_entered"


class ActionStatus(str, Enum):
    PENDING = "pending_review"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    COMPLETED = "completed"


class ReviewerRole(str, Enum):
    DATA_REVIEWER = "data_reviewer"
    LEGAL_REVIEWER = "legal_reviewer"
    DEPARTMENT_OFFICER = "department_officer"
    ADMIN = "admin"


class ActionCategory(str, Enum):
    COMPLIANCE = "compliance"
    APPEAL_CONSIDERATION = "appeal_consideration"
    LEGAL_REVIEW = "legal_review"
    DIRECT_COMPLIANCE = "direct_compliance"
    CONDITIONAL_FOLLOW_UP = "conditional_follow_up"
    RECORD_UPDATE = "record_update"
    INTERNAL_REVIEW = "internal_review"
    NO_OPERATIONAL_ACTION = "no_operational_action"
    INFORMATION_UPDATE = "information_update"
    AFFIDAVIT_REPORT_FILING = "affidavit_report_filing"
    PAYMENT_RELEASE = "payment_release"
    RECONSIDERATION = "reconsideration"
    NO_IMMEDIATE_ACTION = "no_immediate_action"


class Disposition(str, Enum):
    ALLOWED = "allowed"
    DISMISSED = "dismissed"
    DISPOSED = "disposed"
    QUASHED = "quashed"
    REMANDED = "remanded"
    PARTLY_ALLOWED = "partly_allowed"
    LEAVE_GRANTED = "leave_granted"
    UNKNOWN = "unknown"


@dataclass
class SourceEvidence:
    source_id: str
    page: int | None
    chunk_id: str | None
    snippet: str
    confidence: float = 0.9
    extraction_method: str = "deterministic"
    bbox: list[float] | None = None
    locator_confidence: float | None = None
    char_start: int | None = None
    char_end: int | None = None
    source_quality: str = "text"
    quad_points: list[list[float]] | None = None
    match_strategy: str | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    layer: str | None = None
    page_role: str | None = None
    ocr_confidence: float | None = None


@dataclass
class ExtractedField:
    name: str
    value: Any = None
    raw_value: str | None = None
    confidence: float = 0.0
    evidence: list[SourceEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    field_id: str | None = None
    status: FieldReviewStatus = FieldReviewStatus.PENDING
    reason: str | None = None
    reviewer_notes: str | None = None
    manual_override: bool = False
    requires_review: bool = False


@dataclass
class Timeline:
    raw_text: str | None = None
    due_date: date | None = None
    confidence: float = 0.0
    timeline_type: str = "missing"


@dataclass
class JudgmentExtraction:
    case_number: ExtractedField = field(default_factory=lambda: ExtractedField("case_number"))
    case_type: ExtractedField = field(default_factory=lambda: ExtractedField("case_type", value="unknown"))
    court: ExtractedField = field(default_factory=lambda: ExtractedField("court"))
    bench: ExtractedField = field(default_factory=lambda: ExtractedField("bench", value=[]))
    judgment_date: ExtractedField = field(default_factory=lambda: ExtractedField("judgment_date"))
    parties: ExtractedField = field(default_factory=lambda: ExtractedField("parties", value=[]))
    petitioners: ExtractedField = field(default_factory=lambda: ExtractedField("petitioners", value=[]))
    respondents: ExtractedField = field(default_factory=lambda: ExtractedField("respondents", value=[]))
    departments: ExtractedField = field(default_factory=lambda: ExtractedField("departments", value=[]))
    advocates: ExtractedField = field(default_factory=lambda: ExtractedField("advocates", value=[]))
    disposition: ExtractedField = field(default_factory=lambda: ExtractedField("disposition", value=Disposition.UNKNOWN.value))
    legal_phrases: ExtractedField = field(default_factory=lambda: ExtractedField("legal_phrases", value=[]))
    directions: list[ExtractedField] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class ActionItem:
    title: str
    responsible_department: str | None
    timeline: Timeline = field(default_factory=Timeline)
    category: str = ActionCategory.COMPLIANCE.value
    priority: str = "medium"
    legal_basis: str | None = None
    confidence: float = 0.75
    evidence: list[SourceEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    action_id: str | None = None
    direction_summary: str | None = None
    owner_source: str | None = None
    timeline_type: str = "missing"
    ambiguity_flags: list[str] = field(default_factory=list)
    escalation_recommendation: str | None = None
    decision_reason: str | None = None
    review_recommendation: str | None = None
    requires_human_review: bool = False
    status: ActionStatus = ActionStatus.PENDING
    reviewer_notes: str | None = None


@dataclass
class JudgmentReviewPackage:
    extraction: JudgmentExtraction
    action_items: list[ActionItem]
    source_metadata: dict[str, Any] = field(default_factory=dict)
    review_status: ReviewStatus = ReviewStatus.PENDING
    overall_confidence: float = 0.0
    risk_flags: list[str] = field(default_factory=list)
    reviewer_id: str | None = None
    reviewer_notes: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None


@dataclass
class ReviewDecision:
    decision: str
    reviewer_id: str
    notes: str | None = None
    reviewer_role: ReviewerRole | str | None = None
    action_items: list[ActionItem] | None = None
    extraction: JudgmentExtraction | None = None


@dataclass
class DashboardRecord:
    case_number: str | None
    court: str | None
    judgment_date: date | None
    departments: list[str]
    pending_actions: list[str]
    review_status: ReviewStatus
    highest_priority: str
    risk_flags: list[str]
    action_categories: list[str] = field(default_factory=list)
    due_dates: list[date] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    action_register: list[dict[str, Any]] = field(default_factory=list)
