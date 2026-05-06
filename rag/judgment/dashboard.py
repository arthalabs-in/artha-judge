from __future__ import annotations

from .types import ActionCategory, DashboardRecord, JudgmentReviewPackage, ReviewStatus


_DASHBOARD_STATUSES = {ReviewStatus.APPROVED, ReviewStatus.EDITED, ReviewStatus.COMPLETED}
_PRIORITY_RANK = {"low": 0, "medium": 1, "high": 2, "urgent": 3}


def to_dashboard_record(package: JudgmentReviewPackage) -> DashboardRecord | None:
    if package.review_status not in _DASHBOARD_STATUSES:
        return None

    departments = []
    pending_actions = []
    action_categories = []
    due_dates = []
    escalations = []
    highest_priority = "low"

    for item in package.action_items:
        if item.category == ActionCategory.INTERNAL_REVIEW.value:
            continue
        if item.responsible_department and item.responsible_department not in departments:
            departments.append(item.responsible_department)
        pending_actions.append(item.title)
        if item.category and item.category not in action_categories:
            action_categories.append(item.category)
        if item.timeline.due_date and item.timeline.due_date not in due_dates:
            due_dates.append(item.timeline.due_date)
        if item.escalation_recommendation and item.escalation_recommendation not in escalations:
            escalations.append(item.escalation_recommendation)
        if _PRIORITY_RANK.get(item.priority, 0) > _PRIORITY_RANK.get(highest_priority, 0):
            highest_priority = item.priority

    return DashboardRecord(
        case_number=package.extraction.case_number.value,
        court=package.extraction.court.value,
        judgment_date=package.extraction.judgment_date.value,
        departments=departments,
        pending_actions=pending_actions,
        review_status=package.review_status,
        highest_priority=highest_priority,
        risk_flags=package.risk_flags,
        action_categories=action_categories,
        due_dates=due_dates,
        escalations=escalations,
    )


def filter_dashboard_records(packages: list[JudgmentReviewPackage]) -> list[DashboardRecord]:
    records = [to_dashboard_record(package) for package in packages]
    return [record for record in records if record is not None]
