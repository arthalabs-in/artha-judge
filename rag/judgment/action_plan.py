from __future__ import annotations

import re
from datetime import date, timedelta

from .action_classifier import classify_action, priority_for_action
from .owner_resolution import apply_inferred_action_owner
from .types import ActionCategory, ActionItem, JudgmentExtraction, Timeline


_ACTION_PATTERNS = [
    re.compile(r"\b(?:we\s+direct\s+(?:that\s+)?|the\s+)(?P<dept>.+?)\s+(?:is|are)\s+directed\s+to\s+(?P<action>.+)$", re.I),
    re.compile(r"\bthe\s+(?P<dept>.+?)\s+shall\s+(?P<action>.+)$", re.I),
    re.compile(r"\b(?P<dept>state|government|department|board|authority|respondents?)\s+shall\s+(?P<action>.+)$", re.I),
]


def build_action_plan(extraction: JudgmentExtraction) -> list[ActionItem]:
    action_items: list[ActionItem] = []
    judgment_date = extraction.judgment_date.value if isinstance(extraction.judgment_date.value, date) else None
    disposition = str(extraction.disposition.value or "unknown")

    for index, direction in enumerate(extraction.directions):
        direction_text = str(direction.value or "")
        action_items.append(
            _direction_to_action(direction_text, direction.evidence, judgment_date, disposition, index, extraction)
        )

    if not action_items and disposition not in {"unknown", None}:
        category, flags, escalation = classify_action(str(extraction.disposition.raw_value or disposition), disposition)
        if category == ActionCategory.APPEAL_CONSIDERATION.value:
            evidence = list(extraction.disposition.evidence)
            action = ActionItem(
                title="Review judgment outcome for appeal or compliance decision",
                responsible_department=_first_department(extraction),
                timeline=Timeline(raw_text=None, due_date=None, confidence=0.0, timeline_type="missing"),
                category=category,
                priority="high",
                legal_basis=str(extraction.disposition.raw_value or disposition),
                confidence=0.62,
                evidence=evidence,
                notes=["No explicit operational direction found; legal review suggested from disposition."],
                action_id="action-0",
                direction_summary=str(extraction.disposition.raw_value or disposition),
                owner_source="department_extraction" if _first_department(extraction) else None,
                timeline_type="missing",
                ambiguity_flags=flags + ["no_explicit_operational_direction"],
                escalation_recommendation=escalation or "legal_reviewer",
            )
            action_items.append(apply_inferred_action_owner(action, extraction))

    return action_items


def _direction_to_action(
    direction_text: str,
    evidence,
    judgment_date: date | None,
    disposition: str | None,
    index: int,
    extraction: JudgmentExtraction,
) -> ActionItem:
    owner = None
    action_text = direction_text
    owner_source = None
    for pattern in _ACTION_PATTERNS:
        match = pattern.search(direction_text)
        if match:
            owner = _clean_owner(match.group("dept"))
            action_text = _clean_action(match.group("action"))
            owner_source = "source_text"
            break

    timeline = _parse_timeline(direction_text, judgment_date)
    category, ambiguity_flags, escalation = classify_action(direction_text, disposition)
    if not owner:
        ambiguity_flags.append("owner_unclear")
        escalation = escalation or "department_officer"
    if timeline.timeline_type == "missing":
        ambiguity_flags.append("missing_timeline")
    if any(item.source_quality != "text" or not item.bbox for item in evidence):
        ambiguity_flags.append("page_or_ocr_evidence")

    action = ActionItem(
        title=action_text,
        responsible_department=owner,
        timeline=timeline,
        category=category,
        priority=priority_for_action(category, ambiguity_flags, bool(timeline.due_date)),
        legal_basis=direction_text,
        confidence=0.8 if owner and timeline.raw_text else 0.64,
        evidence=list(evidence),
        notes=[] if timeline.raw_text else ["No explicit timeline found."],
        action_id=f"action-{index}",
        direction_summary=direction_text[:240],
        owner_source=owner_source,
        timeline_type=timeline.timeline_type,
        ambiguity_flags=ambiguity_flags,
        escalation_recommendation=escalation,
    )
    return apply_inferred_action_owner(action, extraction)


def _parse_timeline(text: str | None, judgment_date: date | None) -> Timeline:
    if not text:
        return Timeline()

    match = re.search(
        r"\bwithin\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twelve)\s+(days?|weeks?|months?)\b",
        text,
        re.I,
    )
    if not match:
        return Timeline(timeline_type="missing")

    raw_text = match.group(0).strip(" .")
    due_date = None
    if judgment_date:
        amount = _word_or_int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("day"):
            due_date = judgment_date + timedelta(days=amount)
        elif unit.startswith("week"):
            due_date = judgment_date + timedelta(days=amount * 7)
        elif unit.startswith("month"):
            due_date = judgment_date + timedelta(days=amount * 30)

    return Timeline(raw_text=raw_text, due_date=due_date, confidence=0.82, timeline_type="explicit")


def _clean_action(value: str) -> str:
    value = re.sub(r"\bwithin\s+[^.]+$", "", value.strip(" ."), flags=re.I).strip()
    return value[:1].upper() + value[1:] if value else "Review judgment direction"


def _clean_owner(value: str) -> str | None:
    value = re.sub(r"\s+", " ", value.strip(" .,:;-"))
    value = re.sub(r"^(?:that\s+)?the\s+", "", value, flags=re.I)
    if not value or len(value) > 120:
        return None
    return value


def _first_department(extraction: JudgmentExtraction) -> str | None:
    departments = extraction.departments.value or []
    return str(departments[0]) if departments else None


def _word_or_int(value: str) -> int:
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "twelve": 12,
    }
    normalized = value.lower()
    if normalized in words:
        return words[normalized]
    return int(value)
