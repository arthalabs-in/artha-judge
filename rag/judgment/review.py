from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

from .types import JudgmentReviewPackage, ReviewDecision, ReviewStatus


_DECISION_TO_STATUS = {
    "approve": ReviewStatus.APPROVED,
    "approved": ReviewStatus.APPROVED,
    "edit": ReviewStatus.EDITED,
    "edited": ReviewStatus.EDITED,
    "reject": ReviewStatus.REJECTED,
    "rejected": ReviewStatus.REJECTED,
    "needs_clarification": ReviewStatus.NEEDS_CLARIFICATION,
    "clarify": ReviewStatus.NEEDS_CLARIFICATION,
    "complete": ReviewStatus.COMPLETED,
    "completed": ReviewStatus.COMPLETED,
}


def apply_review_decision(
    package: JudgmentReviewPackage,
    decision: ReviewDecision,
) -> JudgmentReviewPackage:
    reviewed = deepcopy(package)
    normalized_decision = decision.decision.strip().lower()
    reviewed.review_status = _DECISION_TO_STATUS.get(normalized_decision, ReviewStatus.NEEDS_CLARIFICATION)
    reviewed.reviewer_id = decision.reviewer_id
    reviewed.reviewer_notes = decision.notes
    reviewed.reviewed_at = datetime.now(UTC)

    if decision.extraction is not None:
        reviewed.extraction = decision.extraction
    if decision.action_items is not None:
        reviewed.action_items = decision.action_items

    return reviewed
