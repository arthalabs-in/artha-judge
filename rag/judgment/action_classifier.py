from __future__ import annotations

import re

from .types import ActionCategory


def classify_action(text: str, disposition: str | None = None) -> tuple[str, list[str], str | None]:
    lowered = text.lower()
    ambiguity_flags: list[str] = []
    escalation: str | None = None

    if re.search(r"\bcompliance\s+report\b|\baffidavit\b|\breport\s+(?:before|to)\b", lowered):
        category = ActionCategory.AFFIDAVIT_REPORT_FILING.value
    elif re.search(r"\brelease\b|\bpay(?:ment)?\b|\barrears\b|\bbenefits?\b|\bcompensation\b", lowered):
        category = ActionCategory.PAYMENT_RELEASE.value
    elif re.search(r"\breconsider\b|\bspeaking\s+order\b|\bfresh\s+order\b", lowered):
        category = ActionCategory.RECONSIDERATION.value
    elif re.search(r"\bupdate\b|\bcorrect\b|\bmodify\b", lowered):
        category = ActionCategory.INFORMATION_UPDATE.value
    elif re.search(r"\bliberty\s+to\s+appeal\b|\bappeal\b|\bspecial\s+leave\b", lowered) or disposition in {"dismissed", "allowed", "partly_allowed"}:
        category = ActionCategory.APPEAL_CONSIDERATION.value
        ambiguity_flags.append("appeal_review_required")
        escalation = "legal_reviewer"
    elif disposition in {"unknown", None}:
        category = ActionCategory.LEGAL_REVIEW.value
        ambiguity_flags.append("legal_outcome_unclear")
        escalation = "legal_reviewer"
    else:
        category = ActionCategory.COMPLIANCE.value

    return category, ambiguity_flags, escalation


def priority_for_action(category: str, ambiguity_flags: list[str], has_due_date: bool) -> str:
    if "appeal_review_required" in ambiguity_flags:
        return "high"
    if has_due_date:
        return "high"
    if category in {ActionCategory.LEGAL_REVIEW.value, ActionCategory.APPEAL_CONSIDERATION.value}:
        return "high"
    return "medium"

