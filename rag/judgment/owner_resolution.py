from __future__ import annotations

import re
from typing import Any

from .types import ActionCategory, ActionItem, JudgmentExtraction, Timeline


_INFERABLE_OWNER_CATEGORIES = {
    ActionCategory.APPEAL_CONSIDERATION.value,
    ActionCategory.LEGAL_REVIEW.value,
    ActionCategory.RECORD_UPDATE.value,
    ActionCategory.INFORMATION_UPDATE.value,
    ActionCategory.RECONSIDERATION.value,
    ActionCategory.CONDITIONAL_FOLLOW_UP.value,
}

_PUBLIC_ENTITY_RE = re.compile(
    r"\b(?:state|government|department|authority|board|commissioner|collector|committee|"
    r"ministry|director(?:ate| general)?|registrar|registry|court|tribunal|municipal|"
    r"corporation|council|scrutiny|revenue|customs|police|secretary|officer|union of india)\b",
    re.I,
)

_OWNER_UNCLEAR_FLAGS = {"owner_unclear", "owner unclear", "owner_unresolved", "owner unresolved"}
_TIMELINE_MISSING_FLAGS = {"missing_timeline", "timeline_missing", "timeline unclear", "timeline_unclear"}
_INTERNAL_RECORD_UPDATE_RE = re.compile(
    r"\bupdate\b.{0,80}\b(?:record|case\s+record|customs\s+record|dashboard|review\s+record)\b|"
    r"\breflect\b.{0,80}\b(?:appeals?\s+allowed|dismissal|remand|disposition)\b",
    re.I,
)
_UNSAFE_COURT_STAFF_OWNER_RE = re.compile(
    r"\b(?:court\s+master|p\.?\s*s\.?\s+to\s+registrar|ps\s+to\s+registrar|personal\s+secretary)\b",
    re.I,
)
_REMAND_DESTINATION_RE = re.compile(
    r"\b(?:remit(?:ted)?|remand(?:ed)?|restore(?:d)?)\b.{0,160}?\bto\s+(?:the\s+)?"
    r"(?P<owner>High Court(?:\s+of\s+[A-Z][A-Za-z .&-]+)?|Trial Court|District Court|"
    r"Tribunal|Board of [A-Z][A-Za-z .&-]+|[A-Z][A-Za-z .,&()/-]{3,80}?)"
    r"\s+(?:for|to|with|per|pursuant|under|as directed|in accordance|$)",
    re.I,
)
_PROCEDURAL_REGISTRY_RE = re.compile(
    r"\b(?:tag|tagged|connect|connected|list|listed)\b.{0,120}\b(?:appeal|matter|petition|case)\b|"
    r"\b(?:appeal|matter|petition|case)\b.{0,120}\b(?:tag|tagged|connect|connected|list|listed)\b",
    re.I,
)


def apply_inferred_action_owner(action: ActionItem, extraction: JudgmentExtraction) -> ActionItem:
    """Attach one clear public respondent/department as a review-required owner.

    This is intentionally conservative: explicit action owners win, no-operational
    actions are left alone, and multiple possible public owners stay unresolved.
    """

    if _is_internal_record_update_action(action):
        _apply_internal_review_owner(action)
        return action

    if _is_procedural_registry_action(action):
        _apply_owner(action, "Registry", "procedural_registry")
        _force_timeline_not_specified(action)
        return action

    if action.responsible_department and not _is_unsafe_court_staff_owner(action.responsible_department):
        if _PUBLIC_ENTITY_RE.search(str(action.responsible_department)):
            _clear_owner_unclear(action)
            if "inferred_assignee_review" not in action.ambiguity_flags:
                action.ambiguity_flags.append("inferred_assignee_review")
            action.requires_human_review = True
        return action
    if action.responsible_department:
        action.responsible_department = None
        action.owner_source = None

    owner, source = _owner_from_action_text(action)
    if owner:
        _apply_owner(action, owner, source)
        return action

    return action


def _apply_owner(action: ActionItem, owner: str, source: str) -> None:
    action.responsible_department = owner
    action.owner_source = source
    action.requires_human_review = True
    _clear_owner_unclear(action)
    if source == "remand_destination":
        _mark_timeline_not_specified(action)
    if "inferred_assignee_review" not in action.ambiguity_flags:
        action.ambiguity_flags.append("inferred_assignee_review")


def _apply_internal_review_owner(action: ActionItem) -> None:
    action.category = ActionCategory.INTERNAL_REVIEW.value
    action.responsible_department = "Case reviewer"
    action.owner_source = "system_policy"
    action.requires_human_review = True
    action.timeline = Timeline(raw_text=None, due_date=None, confidence=0.0, timeline_type="not_configured")
    action.timeline_type = "not_configured"
    action.ambiguity_flags = []
    action.escalation_recommendation = action.escalation_recommendation or "case_reviewer"


def _clear_owner_unclear(action: ActionItem) -> None:
    action.ambiguity_flags = [
        flag for flag in action.ambiguity_flags if str(flag).strip().lower() not in _OWNER_UNCLEAR_FLAGS
    ]


def _owner_from_action_text(action: ActionItem) -> tuple[str | None, str | None]:
    text = " ".join(
        str(value or "")
        for value in [
            action.title,
            action.legal_basis,
            action.direction_summary,
            " ".join(evidence.snippet for evidence in action.evidence),
        ]
    )
    match = _REMAND_DESTINATION_RE.search(text)
    if match:
        return _clean_entity(match.group("owner")), "remand_destination"
    return None, None


def remand_owner_from_text(text: str) -> str | None:
    match = _REMAND_DESTINATION_RE.search(str(text or ""))
    return _clean_entity(match.group("owner")) if match else None


def _is_internal_record_update_action(action: ActionItem) -> bool:
    text = " ".join(
        str(value or "")
        for value in [action.title, action.legal_basis, action.direction_summary]
    )
    return bool(_INTERNAL_RECORD_UPDATE_RE.search(text))


def _is_procedural_registry_action(action: ActionItem) -> bool:
    text = " ".join(
        str(value or "")
        for value in [action.title, action.legal_basis, action.direction_summary]
    )
    return bool(_PROCEDURAL_REGISTRY_RE.search(text))


def _mark_timeline_not_specified(action: ActionItem) -> None:
    if action.timeline.timeline_type == "missing":
        action.timeline.timeline_type = "not_specified"
        action.timeline_type = "not_specified"
    action.ambiguity_flags = [
        flag for flag in action.ambiguity_flags if str(flag).strip().lower() not in _TIMELINE_MISSING_FLAGS
    ]
    if "timeline_not_specified" not in action.ambiguity_flags:
        action.ambiguity_flags.append("timeline_not_specified")


def _force_timeline_not_specified(action: ActionItem) -> None:
    action.timeline = Timeline(raw_text=None, due_date=None, confidence=0.0, timeline_type="not_specified")
    action.timeline_type = "not_specified"
    action.ambiguity_flags = [
        flag for flag in action.ambiguity_flags if str(flag).strip().lower() not in _TIMELINE_MISSING_FLAGS
    ]
    if "timeline_not_specified" not in action.ambiguity_flags:
        action.ambiguity_flags.append("timeline_not_specified")


def _is_unsafe_court_staff_owner(owner: str) -> bool:
    return bool(_UNSAFE_COURT_STAFF_OWNER_RE.search(str(owner or "")))


def _single_public_owner(extraction: JudgmentExtraction) -> tuple[str | None, str | None]:
    respondent_candidates = _public_entities(extraction.respondents.value)
    if len(respondent_candidates) == 1:
        return respondent_candidates[0], "inferred_public_respondent"

    department_candidates = _public_entities(extraction.departments.value)
    if len(department_candidates) == 1:
        return department_candidates[0], "inferred_public_department"

    return None, None


def _public_entities(value: Any) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()
    for item in _as_strings(value):
        cleaned = _clean_entity(item)
        if not cleaned or not _PUBLIC_ENTITY_RE.search(cleaned):
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        entities.append(cleaned)
    return entities


def _as_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ("entity", "name", "value", "text", "label"):
            if value.get(key):
                return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        items: list[str] = []
        for item in value:
            items.extend(_as_strings(item))
        return items
    return [str(value)]


def _clean_entity(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" .,:;-")
    cleaned = re.sub(r"^(?:the|said)\s+", "", cleaned, flags=re.I)
    return cleaned
