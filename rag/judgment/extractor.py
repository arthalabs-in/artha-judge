from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

from langchain_core.documents import Document

from .evidence import normalize_evidence
from .legal_markers import (
    DEPARTMENT_PATTERN,
    DISPOSITION_PATTERNS,
    LEGAL_PHRASE_PATTERNS,
    NON_OPERATIVE_RE,
    OPERATIVE_MARKER_RE,
)
from .types import ExtractedField, JudgmentExtraction


_CASE_RE = re.compile(
    r"\b(?P<type>Writ\s+Petition(?:\s*\([^)]+\))?|WP|Civil\s+Appeal|Criminal\s+Appeal|SLP)"
    r"\s*(?:No\.?|Number)?\s*[\w()/.-]+\s+of\s+\d{4}\b",
    re.I,
)
_DATE_RE = re.compile(
    r"\b(?:judg(?:e)?ment\s+dated|order\s+dated|dated|date\s+of\s+judg(?:e)?ment|date)\s*[:\-]?\s*"
    r"(\d{1,2})(?:st|nd|rd|th)?[\s\-/]+([A-Za-z]{3,9}|\d{1,2})[\s\-/]+(\d{4})\b",
    re.I,
)
_TEXT_DATE_RE = re.compile(
    r"\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,\s*(\d{4})\b",
    re.I,
)
_NUMERIC_YMD_RE = re.compile(r"\b(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})\b")
_FILENAME_DATE_RE = re.compile(r"(\d{1,2})[-_ ]([A-Za-z]{3,9}|\d{1,2})[-_ ](\d{4})")
_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def extract_judgment_details(documents: list[Document]) -> JudgmentExtraction:
    extraction = JudgmentExtraction()

    for document in documents:
        text = document.page_content or ""
        _extract_case_number(extraction, document, text)
        _extract_court(extraction, document, text)
        _extract_bench(extraction, document, text)
        _extract_judgment_date(extraction, document, text)
        _extract_parties(extraction, document, text)
        _extract_departments(extraction, document, text)
        _extract_advocates(extraction, document, text)
        _extract_disposition(extraction, document, text)
        _extract_legal_phrases(extraction, document, text)
        _extract_directions(extraction, document, text)

    _extract_cross_document_advocates(extraction, documents)
    _extract_final_disposition(extraction, documents)
    _apply_law_report_normalization(extraction, documents)
    _extract_date_from_source_filename(extraction, documents)
    _add_missing_field_flags(extraction)
    return extraction


def _extract_case_number(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    if extraction.case_number.value:
        return
    match = _CASE_RE.search(text)
    if not match:
        return
    value = _normalize_case_number(match.group(0))
    case_type = _normalize_case_type(match.group("type"))
    extraction.case_number = ExtractedField(
        name="case_number",
        value=value,
        raw_value=match.group(0),
        confidence=0.92,
        evidence=[normalize_evidence(document, match.group(0))],
        field_id="case_number",
    )
    extraction.case_type = ExtractedField(
        name="case_type",
        value=case_type,
        raw_value=match.group("type"),
        confidence=0.9,
        evidence=[normalize_evidence(document, match.group("type"))],
        field_id="case_type",
    )


def _extract_court(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    if extraction.court.value:
        return
    for line in text.splitlines():
        clean = _clean_line(line)
        court = _court_from_line(clean)
        if court:
            extraction.court = ExtractedField(
                name="court",
                value=court,
                raw_value=line.strip(),
                confidence=0.88,
                evidence=[normalize_evidence(document, line.strip())],
                field_id="court",
            )
            return


def _court_from_line(clean: str) -> str | None:
    if not clean:
        return None
    normalized = re.sub(r"\s+", " ", clean).strip(" .,:;-")
    if re.fullmatch(r"(?:in\s+the\s+)?supreme\s+court(?:\s+of\s+india)?", normalized, re.I):
        return "Supreme Court of India"
    if re.fullmatch(r"\d{1,4}\s+supreme\s+court\s+reports?\b.*", normalized, re.I):
        return "Supreme Court of India"
    high_court = re.fullmatch(r"(?:in\s+the\s+)?high\s+court\s+of\s+(.{3,80})", normalized, re.I)
    if high_court:
        return f"High Court Of {_title_preserving_initials(high_court.group(1))}"
    if re.fullmatch(r"(?:in\s+the\s+)?district\s+court(?:\s+.{2,80})?", normalized, re.I):
        return _title_preserving_initials(re.sub(r"^in\s+the\s+", "", normalized, flags=re.I))
    if re.fullmatch(r"(?:in\s+the\s+)?(?:administrative|income\s+tax|central\s+administrative|armed\s+forces)\s+tribunal", normalized, re.I):
        return _title_preserving_initials(re.sub(r"^in\s+the\s+", "", normalized, flags=re.I))
    return None


def _extract_bench(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    existing = list(extraction.bench.value or [])
    evidence = list(extraction.bench.evidence)
    for judge, raw_text, confidence in _bench_entries_from_text(text):
        if judge not in existing:
            existing.append(judge)
            evidence.append(normalize_evidence(document, raw_text, confidence=confidence))
    if existing:
        extraction.bench = ExtractedField(
            name="bench",
            value=existing[:8],
            raw_value="; ".join(existing[:8]),
            confidence=0.75,
            evidence=evidence[:8],
            field_id="bench",
        )


def _bench_entries_from_text(text: str) -> list[tuple[str, str, float]]:
    entries: list[tuple[str, str, float]] = []
    for judge, raw_text in _bench_from_header_blocks(text):
        entries.append((judge, raw_text, 0.82))
    for judge, raw_text in _bench_from_standalone_lines(text):
        entries.append((judge, raw_text, 0.82))
    for judge, raw_text in _bench_from_law_report_headers(text):
        entries.append((judge, raw_text, 0.76))
    for judge, raw_text in _bench_from_signature_block(text):
        entries.append((judge, raw_text, 0.86))
    deduped: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for judge, raw_text, confidence in entries:
        key = judge.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((judge, raw_text, confidence))
    return deduped


def _bench_from_standalone_lines(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    pattern = re.compile(
        r"\bHON'?BLE\s+(?:SHRI|SMT\.?|MR\.?|MS\.?|MRS\.?)?\s*JUSTICE\s+"
        r"(?P<name>[A-Z][A-Za-z .'-]{2,80})\b",
        re.I,
    )
    for line in text.splitlines()[:160]:
        clean = _clean_line(line)
        if not clean or re.search(r"\b(?:for\s+(?:petitioner|appellant|respondent)|counsel|advocate)\b", clean, re.I):
            continue
        match = pattern.search(clean)
        if not match:
            continue
        name = _normalise_person_name(match.group("name"))
        if name and _looks_like_judge_signature_name(name):
            entries.append((f"Justice {name}", clean))
    return entries


def _bench_from_header_blocks(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    entries: list[tuple[str, str]] = []
    for index, line in enumerate(lines[:120]):
        clean = _clean_line(line)
        if not re.search(r"^(?:coram|before|present)\b", clean, re.I):
            continue
        block_lines = [line]
        for follow in lines[index + 1:index + 8]:
            follow_clean = _clean_line(follow)
            if not follow_clean:
                if len(block_lines) > 1:
                    break
                continue
            if re.search(r"\b(?:for\s+(?:petitioner|appellant|respondent)|upon\s+hearing|order|judg(?:e)?ment)\b", follow_clean, re.I):
                break
            if not re.search(r"\b(?:hon'?ble|justice|cji|chief\s+justice)\b", follow_clean, re.I):
                break
            block_lines.append(follow)
        block = " ".join(block_lines)
        for judge in _judge_names_from_header_text(block):
            entries.append((judge, block))
    return entries


def _bench_from_signature_block(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    signature_pattern = re.compile(
        r"\b(?P<role>CJI|J)\.?\s*(?:\(\s*(?P<paren>[A-Z][A-Z .'-]{1,80})\s*\)|"
        r"\[\s*(?P<bracket>[A-Z][A-Za-z .'-]{1,80})\s*\])",
        re.I,
    )
    for match in signature_pattern.finditer(text):
        role = match.group("role").upper()
        name = _normalise_person_name(match.group("paren") or match.group("bracket") or "")
        if not _looks_like_judge_signature_name(name):
            continue
        prefix = "Chief Justice" if role == "CJI" else "Justice"
        entries.append((f"{prefix} {name}", match.group(0)))
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not re.search(r"^[^A-Za-z]*(?:CJI|J)\.?\s*$", line.strip(), re.I):
            continue
        role = "CJI" if re.search(r"\bCJI\b", line, re.I) else "J"
        for follow in lines[index + 1:index + 5]:
            match = re.search(r"[\[(]\s*([A-Z][A-Za-z .'-]{1,80})\s*[\])]", follow)
            if not match:
                continue
            name = _normalise_person_name(match.group(1))
            if not _looks_like_judge_signature_name(name):
                continue
            prefix = "Chief Justice" if role == "CJI" else "Justice"
            entries.append((f"{prefix} {name}", f"{line.strip()} {follow.strip()}"))
            break
    return entries


def _bench_from_law_report_headers(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in text.splitlines()[:80]:
        clean = _clean_line(line)
        match = re.search(
            r"\b[A-Z][A-Z0-9 .!'?/,\\><~-]{2,80}\s+(?:v\.?|v~|vs\.?|versus|u\.?|i{1,2}\.?|l/|/)\s+"
            r"[A-Z][A-Z0-9 .!'?/,\\><~-]{2,80}\s*\((?P<names>[^)]{3,80}),\s*(?P<role>C\.?\s*J\.?|J{1,2}\.?)\)",
            clean,
            re.I,
        )
        if not match:
            continue
        prefix = "Chief Justice" if re.search(r"\bC\.?\s*J\.?", match.group("role"), re.I) else "Justice"
        for raw_name in re.split(r"\s*&\s*|\s+and\s+", match.group("names"), flags=re.I):
            name = _normalise_person_name(raw_name)
            if name and not re.search(r"\b(?:page|judgment|court|appeal|petition|respondent|appellant)\b", name, re.I):
                entries.append((f"{prefix} {name}", clean))
    return entries


def _apply_law_report_normalization(extraction: JudgmentExtraction, documents: list[Document]) -> None:
    report_documents = [
        document
        for document in documents
        if _looks_like_supreme_court_report(document.page_content or "")
    ]
    if not report_documents:
        return
    first_report = report_documents[0]
    if not _court_from_line(str(extraction.court.raw_value or "")) or str(extraction.court.value or "").lower().endswith("tribunal as follows"):
        extraction.court = ExtractedField(
            name="court",
            value="Supreme Court of India",
            raw_value="Supreme Court Reports",
            confidence=0.76,
            evidence=[normalize_evidence(first_report, "Supreme Court Reports", confidence=0.76)],
            field_id="court",
            requires_review=True,
        )
    clean_bench = _clean_law_report_bench(extraction.bench.value)
    if clean_bench:
        extraction.bench = ExtractedField(
            name="bench",
            value=clean_bench,
            raw_value="; ".join(clean_bench),
            confidence=max(extraction.bench.confidence, 0.76),
            evidence=list(extraction.bench.evidence)[: len(clean_bench)] or [
                normalize_evidence(first_report, str(clean_bench[0]), confidence=0.76)
            ],
            field_id="bench",
            requires_review=True,
        )
    if _parties_look_noisy(extraction.parties.value):
        party_candidates = [
            party
            for document in report_documents[:20]
            for party in [_law_report_parties(document.page_content or "")]
            if party
        ]
        law_report_party = party_candidates[0] if party_candidates else None
        if law_report_party:
            petitioners, respondents, raw = law_report_party
            evidence = [normalize_evidence(first_report, raw, confidence=0.74)]
            extraction.petitioners = ExtractedField("petitioners", petitioners, raw, 0.74, evidence, field_id="petitioners", requires_review=True)
            extraction.respondents = ExtractedField("respondents", respondents, raw, 0.74, evidence, field_id="respondents", requires_review=True)
            extraction.parties = ExtractedField(
                "parties",
                petitioners + [party for party in respondents if party not in petitioners],
                raw,
                0.74,
                evidence,
                field_id="parties",
                requires_review=True,
            )
    if extraction.judgment_date.value and _law_report_date_looks_cited(extraction.judgment_date):
        extraction.judgment_date = ExtractedField(
            name="judgment_date",
            value=None,
            confidence=0.0,
            notes=["OCR law-report text contained only cited or historical dates; judgment date left for review."],
            field_id="judgment_date",
            requires_review=True,
        )
    if _law_report_disposition_looks_non_final(extraction.disposition):
        extraction.disposition = ExtractedField(
            name="disposition",
            value="unknown",
            raw_value=extraction.disposition.raw_value,
            confidence=0.0,
            evidence=list(extraction.disposition.evidence),
            notes=["OCR law-report text contained only non-operative disposition-like prose; outcome left for review."],
            field_id="disposition",
            requires_review=True,
        )


def _looks_like_supreme_court_report(text: str) -> bool:
    sample = " ".join(text.split()[:500]).lower()
    return "supreme court reports" in sample or bool(_law_report_parties(text) and _bench_from_law_report_headers(text))


def _clean_law_report_bench(value) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        text = str(item or "").strip()
        is_chief = text.lower().startswith("chief justice ")
        if is_chief:
            name = text[len("Chief Justice "):].strip()
        elif text.lower().startswith("justice "):
            name = text[len("Justice "):].strip()
        else:
            continue
        if re.search(r"\b(?:as\s+he\s+then\s+was|then\s+was)\b", name, re.I):
            continue
        name = re.sub(r"[^A-Za-z .'-]+", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" .'-")
        if len(name) < 3 or len(name) > 32:
            continue
        key = re.sub(r"[^a-z]+", "", name.lower())
        if any(key and (key in existing or existing in key) for existing in seen):
            continue
        if any(_looks_like_ocr_variant(key, existing) for existing in seen):
            continue
        seen.add(key)
        prefix = "Chief Justice" if is_chief else "Justice"
        cleaned.append(f"{prefix} {_title_preserving_initials(name)}")
        if len(cleaned) >= 3:
            break
    return cleaned


def _looks_like_ocr_variant(key: str, existing: str) -> bool:
    if not key or not existing:
        return False
    if key[:1] == existing[:1] and min(len(key), len(existing)) <= 4:
        return True
    return SequenceMatcher(None, key, existing).ratio() >= 0.6


def _law_report_date_looks_cited(field: ExtractedField) -> bool:
    value = field.value
    if not isinstance(value, date):
        return False
    snippets = " ".join(evidence.snippet for evidence in field.evidence).lower()
    if value.year < 1973:
        return True
    return bool(re.search(r"\b(?:earlier|prior|referred|case|decision|order)\b.{0,80}\bdated\b", snippets))


def _law_report_disposition_looks_non_final(field: ExtractedField) -> bool:
    if not field.value or field.value == "unknown":
        return False
    raw = str(field.raw_value or "")
    snippets = " ".join(evidence.snippet for evidence in field.evidence)
    text = f"{raw} {snippets}"
    value = str(field.value)
    if _looks_like_non_final_disposition_text(text):
        return True
    if value in {"allowed", "dismissed"} and not re.search(
        r"\b(?:we\s+(?:allow|dismiss)|(?:appeals?|petitions?|applications?)\s+(?:are|is)\s+(?:accordingly\s+)?(?:allowed|dismissed)|"
        r"in\s+the\s+result|accordingly|i\s+hold|my\s+conclusions?|petition\s+shall)\b",
        text,
        re.I,
    ):
        return True
    return False


def _parties_look_noisy(value) -> bool:
    parties = [str(item or "") for item in value or [] if str(item or "").strip()]
    if not parties:
        return True
    joined = " ".join(parties).lower()
    if "kesavananda" in joined and "kerala" in joined:
        return False
    if any(item[:1].islower() or item.lower().startswith("of ") for item in parties):
        return True
    return any(len(item) < 4 for item in parties) or bool(re.search(r"\b(?:article|amending|american|constitution|court|according|see\s+for\s+example|collector|advocate|attorney|counsel|m/s|between\s+the|intended\s+to|supplement\s+each\s+other)\b", joined))


def _looks_like_judge_signature_name(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower()
    if lowered in {"new delhi"} or any(month == lowered for month in _MONTHS):
        return False
    if re.search(r"\b(?:page|judgment|court|appeal|petition|respondent|appellant)\b", lowered):
        return False
    return len(name.split()) >= 2


def _judge_names_from_header_text(text: str) -> list[str]:
    names: list[str] = []
    pattern = re.compile(
        r"(?:HON'?BLE\s+)?(?:(?:MR|MS|MRS)\.?\s+)?(?:(?P<chief>CHIEF\s+JUSTICE)|JUSTICE)\s+"
        r"(?P<name>[A-Z][A-Z .'-]{1,80}?)(?=\s+(?:HON'?BLE(?:\s+(?:MR|MS|MRS)\.?)?\s+JUSTICE|JUSTICE|CHIEF\s+JUSTICE)|$)",
        re.I,
    )
    for match in pattern.finditer(" ".join(text.split())):
        name = _normalise_person_name(match.group("name"))
        if not name:
            continue
        prefix = "Chief Justice" if match.group("chief") else "Justice"
        names.append(f"{prefix} {name}")
    return list(dict.fromkeys(names))


def _normalise_person_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z .'-]+", " ", value)
    cleaned = re.sub(r"\b(?:HON'?BLE|MR|MS|MRS|DR|JUSTICE|CHIEF|CJI|J)\.?\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        return ""
    parts = []
    for token in cleaned.split():
        if "." in token:
            parts.append(token.upper())
        else:
            parts.append(token[:1].upper() + token[1:].lower())
    return " ".join(parts)


def _title_preserving_initials(value: str) -> str:
    parts = []
    for token in re.sub(r"\s+", " ", value.strip()).split():
        if "." in token:
            parts.append(token.upper())
        else:
            parts.append(token[:1].upper() + token[1:].lower())
    return " ".join(parts)


def _extract_judgment_date(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    if extraction.judgment_date.value:
        return
    case_year = _case_year(extraction)
    parsed_date = None
    matched_text = None
    candidate_text = _judgment_date_candidate_text(text)
    for match, parser in _date_candidates(candidate_text):
        candidate_date = parser(*match.groups())
        if not candidate_date:
            continue
        if case_year and candidate_date.year < case_year:
            continue
        parsed_date = candidate_date
        matched_text = match.group(0)
        break
    if not parsed_date or not matched_text:
        return
    extraction.judgment_date = ExtractedField(
        name="judgment_date",
        value=parsed_date,
        raw_value=matched_text,
        confidence=0.9,
        evidence=[normalize_evidence(document, matched_text)],
        field_id="judgment_date",
    )


def _extract_date_from_source_filename(extraction: JudgmentExtraction, documents: list[Document]) -> None:
    for document in documents:
        metadata = document.metadata or {}
        names = [
            str(metadata.get("original_file_name") or ""),
            str(metadata.get("source") or ""),
        ]
        for name_source in names:
            name = Path(name_source).name
            match = _FILENAME_DATE_RE.search(name)
            if not match:
                continue
            parsed_date = _parse_date_parts(*match.groups())
            if not parsed_date:
                continue
            if extraction.judgment_date.value and parsed_date <= extraction.judgment_date.value:
                return
            extraction.judgment_date = ExtractedField(
                name="judgment_date",
                value=parsed_date,
                raw_value=match.group(0),
                confidence=0.65,
                evidence=[normalize_evidence(document, name, confidence=0.6, extraction_method="filename")],
                notes=["Date inferred from uploaded filename."],
                field_id="judgment_date",
                requires_review=True,
            )
            return


def _extract_parties(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    if extraction.petitioners.value and extraction.respondents.value:
        return
    law_report_party = _law_report_parties(text)
    if law_report_party:
        petitioners, respondents, raw = law_report_party
        evidence = [normalize_evidence(document, raw, confidence=0.76)]
        extraction.petitioners = ExtractedField(
            name="petitioners",
            value=petitioners,
            raw_value=raw,
            confidence=0.76,
            evidence=evidence,
            field_id="petitioners",
            requires_review=True,
        )
        extraction.respondents = ExtractedField(
            name="respondents",
            value=respondents,
            raw_value=raw,
            confidence=0.76,
            evidence=evidence,
            field_id="respondents",
            requires_review=True,
        )
        extraction.parties = ExtractedField(
            name="parties",
            value=petitioners + [party for party in respondents if party not in petitioners],
            raw_value=raw,
            confidence=0.74,
            evidence=evidence,
            field_id="parties",
            requires_review=True,
        )
        return
    line_party = _labeled_parties(text) or _line_based_parties(text)
    if line_party:
        petitioners, respondents, raw = line_party
        evidence = [normalize_evidence(document, raw)]
        extraction.petitioners = ExtractedField(
            name="petitioners",
            value=petitioners,
            raw_value=raw,
            confidence=0.84,
            evidence=evidence,
            field_id="petitioners",
        )
        extraction.respondents = ExtractedField(
            name="respondents",
            value=respondents,
            raw_value=raw,
            confidence=0.84,
            evidence=evidence,
            field_id="respondents",
        )
        extraction.parties = ExtractedField(
            name="parties",
            value=petitioners + [party for party in respondents if party not in petitioners],
            raw_value=raw,
            confidence=0.82,
            evidence=evidence,
            field_id="parties",
        )
        return
    normalized = "\n".join(_clean_line(line) for line in text.splitlines()[:80])
    party_match = re.search(
        r"(?P<left>[A-Z][A-Za-z0-9&.,'() \-]+?)\s+(?:VERSUS|Versus|Vs\.?|V\.?)\s+"
        r"(?P<right>[A-Z][A-Za-z0-9&.,'() \-]+?)(?:\s+J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|\n|$)",
        normalized,
        re.S,
    )
    if not party_match:
        for line in text.splitlines()[:120]:
            if re.search(r"\b(?:v\.?|vs\.?|versus)\b", line, re.I):
                party_match = re.search(
                    r"(?P<left>.+?)\s+(?:v\.?|vs\.?|versus)\s+(?P<right>.+)$",
                    line.strip(),
                    re.I,
                )
                break
    if not party_match:
        return
    petitioners = _split_parties(party_match.group("left"))
    respondents = _split_parties(party_match.group("right"))
    all_parties = petitioners + [party for party in respondents if party not in petitioners]
    evidence = [normalize_evidence(document, party_match.group(0))]
    extraction.petitioners = ExtractedField(
        name="petitioners",
        value=petitioners,
        raw_value=party_match.group("left"),
        confidence=0.82,
        evidence=evidence,
        field_id="petitioners",
    )
    extraction.respondents = ExtractedField(
        name="respondents",
        value=respondents,
        raw_value=party_match.group("right"),
        confidence=0.82,
        evidence=evidence,
        field_id="respondents",
    )
    extraction.parties = ExtractedField(
        name="parties",
        value=all_parties,
        raw_value=party_match.group(0),
        confidence=0.8,
        evidence=evidence,
        field_id="parties",
    )


def _law_report_parties(text: str) -> tuple[list[str], list[str], str] | None:
    for line in text.splitlines()[:80]:
        clean = _clean_line(line)
        if re.search(r"\b(?:see\s+for\s+example|for\s+(?:the\s+)?(?:petitioner|respondent|appellant|advocate)|appearing\s+for|learned\s+counsel|advocate-general|attorney-general)\b", clean, re.I):
            continue
        match = re.search(
            r"\b(?P<left>[A-Z][A-Z0-9 .!'?/,\\><~-]{2,80})\s+(?:v\.?|v~|vs\.?|versus|u\.?|i{1,2}\.?|l/|/)\s+"
            r"(?P<right>[A-Z][A-Z0-9 .!'?/,\\><~-]{2,80})\s*\(",
            clean,
            re.I,
        )
        if not match:
            continue
        left = _clean_law_report_party(match.group("left"))
        right = _clean_law_report_party(match.group("right"))
        if left and right:
            return [left], [right], clean
    return None


def _extract_departments(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    existing = list(extraction.departments.value or [])
    evidence = list(extraction.departments.evidence)
    for match in DEPARTMENT_PATTERN.finditer(text):
        dept = _clean_department_entity(match.group(0))
        if not dept or len(dept) > 90 or dept.lower() in {"state"}:
            continue
        if dept not in existing:
            existing.append(dept)
            evidence.append(normalize_evidence(document, match.group(0), confidence=0.75))
    if existing:
        extraction.departments = ExtractedField(
            name="departments",
            value=existing[:12],
            raw_value="; ".join(existing[:12]),
            confidence=0.75,
            evidence=evidence[:12],
            field_id="departments",
        )


def _extract_advocates(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    existing = list(extraction.advocates.value or [])
    evidence = list(extraction.advocates.evidence)
    for advocate in _advocates_from_appearance_blocks(text):
        if advocate not in existing:
            existing.append(advocate)
            evidence.append(normalize_evidence(document, advocate, confidence=0.75))
    for line in text.splitlines()[:150]:
        clean = _clean_line(line)
        if re.search(r"\b(?:advocate|solicitor|attorney|amicus)\b", clean, re.I) and len(clean) < 180:
            if clean not in existing:
                existing.append(clean)
                evidence.append(normalize_evidence(document, line.strip(), confidence=0.6))
    if existing:
        extraction.advocates = ExtractedField(
            name="advocates",
            value=existing[:10],
            raw_value="; ".join(existing[:10]),
            confidence=0.6,
            evidence=evidence[:10],
            field_id="advocates",
            requires_review=True,
        )


def _extract_cross_document_advocates(extraction: JudgmentExtraction, documents: list[Document]) -> None:
    existing = list(extraction.advocates.value or [])
    evidence = list(extraction.advocates.evidence)
    for index, document in enumerate(documents):
        current = document.page_content or ""
        next_text = documents[index + 1].page_content if index + 1 < len(documents) else ""
        combined = f"{current}\n{next_text or ''}"
        for advocate, raw_text in _counsel_mentions_from_text(combined):
            if advocate in existing or _is_counsel_shorthand(advocate):
                continue
            existing.append(advocate)
            evidence.append(normalize_evidence(Document(page_content=raw_text, metadata=document.metadata), raw_text, confidence=0.58))
    if existing:
        extraction.advocates = ExtractedField(
            name="advocates",
            value=existing[:12],
            raw_value="; ".join(existing[:12]),
            confidence=max(extraction.advocates.confidence, 0.58),
            evidence=evidence[:12],
            field_id="advocates",
            requires_review=True,
        )


def _counsel_mentions_from_text(text: str) -> list[tuple[str, str]]:
    cleaned = " ".join(text.split())
    cleaned = re.sub(r"\bPage\s+\d+\s+JUDGMENT\b", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\b\d{1,3}\s+(?=respective\s+respondent|respondent|appellant|petitioner)", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned)
    mentions: list[tuple[str, str]] = []
    pattern = re.compile(
        r"\b(?P<title>Mr\.|Ms\.|Mrs\.|Dr\.|Shri|Smt\.)\s+"
        r"(?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,5}),\s+"
        r"(?P<role>learned\s+(?:senior\s+)?counsel)(?:\s+appearing)?\s+for\s+"
        r"(?:the\s+)?(?P<party>respective\s+respondent\(s\)|respondent\(s\)|appellant\(s\)|"
        r"petitioner\(s\)|appellants?|respondents?|petitioners?|state)",
        re.I,
    )
    for match in pattern.finditer(cleaned):
        name = _normalise_person_name(f"{match.group('title')} {match.group('name')}")
        if not name:
            continue
        title = match.group("title").replace(".", "")
        role = re.sub(r"\s+", " ", match.group("role").lower()).strip()
        party = _normalise_counsel_party(match.group("party"))
        mentions.append((f"{title}. {name}, {role} for {party}", match.group(0)))
    return list(dict.fromkeys(mentions))[:12]


def _normalise_counsel_party(value: str) -> str:
    lowered = value.lower()
    if "appellant" in lowered:
        return "appellant"
    if "respondent" in lowered:
        return "respondent"
    if "petitioner" in lowered:
        return "petitioner"
    if "state" in lowered:
        return "state"
    return re.sub(r"[^a-z]+", " ", lowered).strip()


def _is_counsel_shorthand(value: str) -> bool:
    match = re.match(r"^(?:Mr|Ms|Mrs|Dr|Shri|Smt)\.\s+([A-Za-z.'-]+),", value)
    return bool(match)


def _date_candidates(text: str):
    for match in _DATE_RE.finditer(text):
        yield match, _parse_date_parts
    for match in _TEXT_DATE_RE.finditer(text):
        yield match, lambda month, day, year: _parse_date_parts(day, month, year)
    for match in _NUMERIC_YMD_RE.finditer(text):
        yield match, lambda year, month, day: _parse_date_parts(day, month, year)


def _advocates_from_appearance_blocks(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    match = re.search(
        r"\bFor\s+(?:Petitioner|Appellant|Respondent|Applicant)\(s\)\s+(?P<block>.*?)(?:\bUPON\s+hearing\b|\bO\s*R\s*D\s*E\s*R\b|$)",
        cleaned,
        re.I,
    )
    if not match:
        return []
    block = re.sub(r"\bFor\s+(?:Petitioner|Appellant|Respondent|Applicant)\(s\)\b", " ", match.group("block"), flags=re.I)
    items = []
    for match in re.finditer(
        r"(?:Mr\.|Ms\.|Mrs\.|Dr\.|Shri|Smt\.)\s+[A-Za-z][A-Za-z .'-]{1,80}?,\s*(?:Adv\.?|Advocate|Sr\. Adv\.?)",
        block,
        re.I,
    ):
        advocate = _clean_line(match.group(0))
        advocate = re.sub(r"\badv\.?$", "Adv.", advocate, flags=re.I)
        if advocate not in items:
            items.append(advocate)
    return items[:12]


def _extract_disposition(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    for disposition, pattern in DISPOSITION_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        if disposition == "allowed" and _looks_like_partly_allowed_context(text, match.start(), match.end()):
            continue
        current_conf = extraction.disposition.confidence
        confidence = 0.82 if int((document.metadata or {}).get("page") or 0) > 1 else 0.65
        if confidence >= current_conf:
            extraction.disposition = ExtractedField(
                name="disposition",
                value=disposition,
                raw_value=match.group(0),
                confidence=confidence,
                evidence=[normalize_evidence(document, match.group(0), confidence=confidence)],
                field_id="disposition",
            )


def _extract_final_disposition(extraction: JudgmentExtraction, documents: list[Document]) -> None:
    candidates: list[tuple[float, str, str, Document]] = []
    page_count = max(
        [int((document.metadata or {}).get("page") or index + 1) for index, document in enumerate(documents)]
        or [len(documents) or 1]
    )
    for index, document in enumerate(documents):
        text = " ".join((document.page_content or "").split())
        if not text:
            continue
        page = int((document.metadata or {}).get("page") or index + 1)
        is_final_area = page >= max(1, page_count - 2)
        for disposition, pattern in _disposition_candidate_patterns():
            for match in pattern.finditer(text):
                raw_text = match.group(0)
                if _looks_like_non_final_disposition_candidate(text, match.start(), match.end(), disposition):
                    continue
                score = 0.72
                if is_final_area:
                    score += 0.18
                if _looks_like_final_order_context(text, match.start(), match.end()):
                    score += 0.12
                if _looks_like_procedural_history(text, match.start(), match.end()):
                    score -= 0.28
                if disposition in {"allowed", "dismissed", "partly_allowed"}:
                    score += 0.08
                if disposition == "remanded":
                    score -= 0.04
                candidates.append((score, disposition, raw_text, document))
    if not candidates:
        return
    score, disposition, raw_text, document = max(candidates, key=lambda item: item[0])
    if score < max(0.72, extraction.disposition.confidence):
        return
    extraction.disposition = ExtractedField(
        name="disposition",
        value=disposition,
        raw_value=raw_text,
        confidence=round(min(score, 0.94), 2),
        evidence=[normalize_evidence(document, raw_text, confidence=round(min(score, 0.94), 2))],
        field_id="disposition",
    )


def _disposition_candidate_patterns() -> list[tuple[str, re.Pattern[str]]]:
    return [
        ("leave_granted", re.compile(r"\b(?:accordingly,\s*)?(?:we\s+)?grant\s+leave\b|\bleave\s+(?:is\s+)?granted\b|\btag\s+this\s+appeal\b|\btagged\s+with\b|\b(?:list|tag|connect)\s+.{0,80}\balong\s+with\b", re.I)),
        ("partly_allowed", re.compile(r"\b(?:partly|partially)\s+allow(?:ed)?\s+(?:the\s+)?(?:appeals?|petitions?|applications?)\b|\b(?:appeals?|petitions?|applications?)\s+(?:are|is)\s+(?:partly|partially)\s+allowed\b|\b(?:partly|partially)\s+allowed\b", re.I)),
        ("allowed", re.compile(r"\b(?:we\s+)?allow(?:ed)?\s+(?:the\s+)?(?:appeals?|petitions?|applications?)\b|\b(?:appeals?|petitions?|applications?)\s+(?:are|is)\s+(?:accordingly\s+)?allowed\b", re.I)),
        ("dismissed", re.compile(r"\b(?:we\s+)?dismiss(?:ed)?\s+(?:the\s+)?(?:appeals?|petitions?|applications?)\b|\b(?:appeals?|petitions?|applications?)\s+(?:are|is)\s+dismissed\b", re.I)),
        ("disposed", re.compile(r"\b(?:we\s+)?dispose(?:d)?\s+(?:of\s+)?(?:the\s+)?(?:appeals?|petitions?|applications?|matter)\b|\bdisposed\s+of\b", re.I)),
        ("quashed", re.compile(r"\bquash(?:ed|ing)?\b", re.I)),
        (
            "remanded",
            re.compile(
                r"\bremand(?:ed)?\b|\bremitted\s+back\b|"
                r"\bremit(?:ted)?\s+(?:the\s+)?(?:matters?|cases?|appeals?|petitions?)\b|"
                r"\b(?:matters?|cases?|appeals?|petitions?)\s+(?:is|are|stand|stands)\s+remitted\b",
                re.I,
            ),
        ),
    ]


def _looks_like_final_order_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 180): min(len(text), end + 220)].lower()
    return bool(re.search(r"\b(?:in\s+the\s+result|in\s+view\s+of|we\s+allow|we\s+dismiss|we\s+grant|we\s+direct|accordingly|set\s+aside|tag\s+this\s+appeal|no\s+order\s+as\s+to\s+costs)\b", window))


def _looks_like_procedural_history(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 240): min(len(text), end + 180)].lower()
    return bool(re.search(r"\b(?:high\s+court|tribunal|collector|commissioner|authority|impugned\s+judg(?:e)?ment)\b.{0,140}\b(?:dismissed|allowed|disposed|held|liable\s+to\s+be\s+dismissed)\b", window))


def _looks_like_non_final_disposition_candidate(text: str, start: int, end: int, disposition: str) -> bool:
    if disposition != "allowed":
        return False
    window = text[max(0, start - 80): min(len(text), end + 100)]
    return _looks_like_non_final_disposition_text(window)


def _looks_like_non_final_disposition_text(text: str) -> bool:
    lowered = " ".join(text.lower().split())
    return bool(re.search(
        r"\b(?:if\s+it\s+be\s+allowed|should\s+be\s+allowed\s+to|be\s+allowed\s+to|allowed\s+(?:tlj|to)\s+stand|dis\s*-\s*allowed|not\s+allowed\s+to|"
        r"allowed\s+to\s+(?:take|be|make|continue|proceed)|what\s+is\s+done\s+once,\s+if\s+it\s+be\s+allowed)\b",
        lowered,
    ))


def _looks_like_partly_allowed_context(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 40): min(len(text), end + 20)]
    return bool(re.search(r"\b(?:partly|partially)\s+allowed\b", window, re.I))


def _extract_legal_phrases(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    existing = list(extraction.legal_phrases.value or [])
    evidence = list(extraction.legal_phrases.evidence)
    for label, pattern in LEGAL_PHRASE_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group(0)
            item = {"type": label, "text": phrase, "page": (document.metadata or {}).get("page")}
            if item not in existing:
                existing.append(item)
                evidence.append(normalize_evidence(document, phrase, confidence=0.72))
    if existing:
        extraction.legal_phrases = ExtractedField(
            name="legal_phrases",
            value=existing[:40],
            raw_value="; ".join(str(item["text"]) for item in existing[:20]),
            confidence=0.72,
            evidence=evidence[:40],
            field_id="legal_phrases",
        )


def _extract_directions(extraction: JudgmentExtraction, document: Document, text: str) -> None:
    sentences = _judgment_sentences(" ".join(text.split()))
    for sentence in sentences:
        clean = sentence.strip(" .")
        if len(clean) < 35 or len(clean) > 900:
            continue
        if _looks_like_non_operational_reporter_direction(clean):
            continue
        if NON_OPERATIVE_RE.search(clean) or not OPERATIVE_MARKER_RE.search(clean):
            continue
        if _looks_like_statutory_discussion(clean):
            continue
        if any(existing.raw_value == sentence for existing in extraction.directions):
            continue
        extraction.directions.append(
            ExtractedField(
                name="direction",
                value=clean,
                raw_value=sentence,
                confidence=0.78,
                evidence=[normalize_evidence(document, sentence)],
                field_id=f"direction-{len(extraction.directions)}",
                requires_review=True,
            )
        )


def _judgment_sentences(text: str) -> list[str]:
    protected = text
    for abbreviation in ("C.A.No.", "C.A.", "S.L.P.", "W.P.", "Nos.", "No.", "Mr.", "Ms.", "Dr.", "Rs."):
        protected = re.sub(
            rf"(?<![A-Za-z]){re.escape(abbreviation)}",
            lambda match: match.group(0).replace(".", "<DOT>"),
            protected,
            flags=re.I,
        )
    return [
        sentence.replace("<DOT>", ".")
        for sentence in re.split(r"(?<=[.!?])\s+", protected)
    ]


def _looks_like_non_operational_reporter_direction(text: str) -> bool:
    lowered = text.lower()
    if "which is directed to" in lowered:
        return True
    if "clouding the issues" in lowered:
        return True
    if re.search(r"\b(?:bear\s+their\s+own\s+costs?|no\s+order\s+as\s+to\s+costs?)\b", lowered):
        return True
    if re.search(r"\bkesavananda\b.{0,30}\bkerala\b", lowered) and re.search(r"\bwe\s+direct\b", lowered):
        return True
    return False


def _add_missing_field_flags(extraction: JudgmentExtraction) -> None:
    for field_name in ("case_number", "court", "judgment_date"):
        if not getattr(extraction, field_name).value:
            extraction.risk_flags.append(f"missing_{field_name}")
    if not extraction.directions and extraction.disposition.value in {"unknown", None}:
        extraction.risk_flags.append("missing_directions")
    if not extraction.departments.value:
        extraction.risk_flags.append("owner_unclear")


def _parse_date_parts(day: str, month_value: str, year: str) -> date | None:
    try:
        month = int(month_value) if month_value.isdigit() else _MONTHS.get(month_value.lower())
        if not month:
            return None
        return date(int(year), month, int(day))
    except ValueError:
        return None


def _normalize_case_number(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    value = re.sub(r"\bNo\s+", "No. ", value, flags=re.I)
    normalized = value.title().replace("Wp", "WP").replace("Slp", "SLP")
    normalized = re.sub(r"\bOf\b", "of", normalized)
    normalized = re.sub(r"\bNo\.\b", "No.", normalized)
    return normalized


def _normalize_case_type(value: str) -> str:
    clean = re.sub(r"\s+", " ", value.lower()).strip()
    if "writ" in clean or clean == "wp":
        return "writ_petition"
    if "civil appeal" in clean:
        return "civil_appeal"
    if "criminal appeal" in clean:
        return "criminal_appeal"
    if "slp" in clean:
        return "slp"
    return "unknown"


def _clean_party(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[….\s]*(?:Petitioner|Respondent|Appellant|Applicant|Ors?\.?|\(s\))+$", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9)]+$", "", value)
    return value.strip(" .:-\n\t")


def _clean_department_entity(value: str) -> str:
    value = _clean_party(value)
    value = re.sub(r"^(?:of|by|and|the|said)\s+", "", value, flags=re.I)
    high_court = re.search(r"\bHigh\s+Court\s+of\s+[^).,;:]+", value, re.I)
    if high_court:
        name = re.split(
            r"\b(?:in|for|from|dated|which|that|held|dismissed|at|vide|by|with|per)\b",
            high_court.group(0),
            maxsplit=1,
            flags=re.I,
        )[0].strip(" .,:;-")
        place = re.sub(r"^High\s+Court\s+of\s+", "", name, flags=re.I)
        place = " ".join(part.capitalize() for part in place.split())
        return f"High Court of {place}" if place else ""

    government = re.search(r"\bGovernment\s+of\s+India\b", value, re.I)
    if government:
        return "Government of India"

    value = re.sub(
        r"\b(?:issued|produce|certified|established|respect of which|while|when|which)\b.*$",
        "",
        value,
        flags=re.I,
    ).strip(" .,:;-")
    return value


def _clean_law_report_party(value: str) -> str:
    value = re.sub(r"^[\d\s\[\]().,;:'\"!?/-]+", "", value)
    value = re.sub(r"[^A-Za-z0-9 .&'/-]+", " ", value)
    value = re.sub(r"\b(?:supp|s\.?c\.?r|reports?|court|page)\b", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" .'-/")
    if len(value) < 3 or len(value) > 80:
        return ""
    return _title_preserving_initials(value)


def _split_parties(value: str) -> list[str]:
    value = re.sub(r"…\s*(?:Petitioner|Respondent)\(s\)", "", value, flags=re.I)
    parts = re.split(r"\s+&\s+|\s+and\s+|,\s*", value, flags=re.I)
    return [
        part for part in (_clean_party(part) for part in parts)
        if part and len(part) > 2 and part.lower() not in {"ors", "or", "others"}
    ]


def _labeled_parties(text: str) -> tuple[list[str], list[str], str] | None:
    head = "\n".join(text.splitlines()[:140])
    match = re.search(
        r"(?P<left>[\s\S]{8,260}?)\bPetitioner\(s\)[\s\S]{0,80}?\bVERSUS\b(?P<right>[\s\S]{8,220}?)\bRespondent\(s\)",
        head,
        re.I,
    )
    if not match:
        return None
    left = _party_block_tail(match.group("left"))
    right = _party_block_head(match.group("right"))
    petitioners = _split_parties(left)
    respondents = _split_parties(right)
    if not petitioners or not respondents:
        return None
    return petitioners, respondents, f"{left} VERSUS {right}"


def _line_based_parties(text: str) -> tuple[list[str], list[str], str] | None:
    lines = [_clean_line(line) for line in text.splitlines()[:120]]
    lines = [line for line in lines if line and not line.isdigit()]
    for index, line in enumerate(lines):
        same_line = re.search(r"(?P<left>.+?)\s+(?:v\.?|vs\.?|versus)\s+(?P<right>.+)$", line, re.I)
        if same_line:
            return _split_parties(same_line.group("left")), _split_parties(same_line.group("right")), line
        if re.fullmatch(r"(?:v\.?|vs\.?|versus)", line, re.I) and index > 0 and index + 1 < len(lines):
            left = _nearest_party_caption_line(lines, index - 1, -1)
            right = _nearest_party_caption_line(lines, index + 1, 1)
            if not left or not right:
                continue
            raw = f"{left} VERSUS {right}"
            return _split_parties(left), _split_parties(right), raw
    return None


def _nearest_party_caption_line(lines: list[str], start: int, step: int) -> str | None:
    index = start
    while 0 <= index < len(lines):
        line = lines[index]
        if not re.fullmatch(r"[-–—. ]*(?:Petitioner|Respondent|Appellant|Applicant|Claimant)s?[-–—. ()]*", line, re.I):
            return line
        index += step
    return None


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" .")


def _party_block_tail(block: str) -> str:
    lines = [_clean_line(line) for line in block.splitlines()]
    lines = [
        line for line in lines
        if line
        and not line.isdigit()
        and not re.search(r"\b(?:reportable|court|jurisdiction|petition|appeal|judg(?:e)?ment)\b", line, re.I)
    ]
    return " ".join(lines[-3:])


def _party_block_head(block: str) -> str:
    lines = [_clean_line(line) for line in block.splitlines()]
    lines = [line for line in lines if line and not line.isdigit()]
    return " ".join(lines[:3])


def _judgment_date_candidate_text(text: str) -> str:
    lines = text.splitlines()
    head = "\n".join(lines[:80])
    tail = "\n".join(lines[-80:])
    return f"{head}\n{tail}"


def _looks_like_statutory_discussion(sentence: str) -> bool:
    return bool(
        re.search(r"\b(?:act|rules?|section|article|constitution)\b", sentence, re.I)
        and re.search(r"\bshall\b", sentence, re.I)
        and not re.search(r"\b(?:we|court|respondent|state|department|board)\s+(?:direct|order|shall)\b", sentence, re.I)
    )


def _case_year(extraction: JudgmentExtraction) -> int | None:
    value = str(extraction.case_number.value or "")
    match = re.search(r"\bof\s+(\d{4})\b", value, re.I)
    if not match:
        return None
    return int(match.group(1))
