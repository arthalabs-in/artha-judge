from __future__ import annotations

import base64
import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import request

import fitz
from langchain_core.documents import Document

from rag.judgment.action_plan import build_action_plan
from rag.judgment.service import _attach_retrieval_scores
from rag.judgment.types import ExtractedField, JudgmentReviewPackage
from rag.judgment.evidence import normalize_evidence


VISION_FIELD_NAMES = {
    "case_number",
    "case_type",
    "court",
    "bench",
    "judgment_date",
    "parties",
    "petitioners",
    "respondents",
    "departments",
    "advocates",
    "disposition",
}

VISION_DISPOSITIONS = {
    "allowed",
    "dismissed",
    "disposed",
    "quashed",
    "remanded",
    "partly_allowed",
    "leave_granted",
    "unknown",
}


@dataclass
class VisionExtractionResult:
    fields: dict[str, Any] = field(default_factory=dict)
    directions: list[str] = field(default_factory=list)
    evidence_pages: dict[str, int] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)
    provider: str = "minicpm"


class VisionExtractor(Protocol):
    async def extract(
        self,
        *,
        pdf_path: str,
        pages: list[int],
        deterministic_summary: dict[str, Any],
    ) -> VisionExtractionResult:
        ...


class MiniCPMHTTPVisionExtractor:
    def __init__(
        self,
        *,
        endpoint: str,
        timeout_sec: int = 90,
        model: str = "openbmb/MiniCPM-o-4_5",
    ) -> None:
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec
        self.model = model

    async def extract(
        self,
        *,
        pdf_path: str,
        pages: list[int],
        deterministic_summary: dict[str, Any],
    ) -> VisionExtractionResult:
        images = render_pdf_pages_for_vision(pdf_path, pages)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return strict JSON only. Extract court judgment metadata from page images.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _vision_prompt(deterministic_summary, pages)},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                            }
                            for image_b64 in images
                        ],
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 1400,
        }
        raw = await _post_json(self.endpoint, payload, timeout_sec=self.timeout_sec)
        content = _chat_completion_content(raw)
        parsed = _parse_json_object(content)
        return result_from_vision_payload(parsed, provider="minicpm")


class MiniCPMGGUFVisionExtractor:
    def __init__(
        self,
        *,
        cli_path: str,
        model_path: str,
        mmproj_path: str,
        timeout_sec: int = 120,
    ) -> None:
        self.cli_path = cli_path
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.timeout_sec = timeout_sec

    async def extract(
        self,
        *,
        pdf_path: str,
        pages: list[int],
        deterministic_summary: dict[str, Any],
    ) -> VisionExtractionResult:
        return await asyncio.to_thread(
            self._extract_sync,
            pdf_path=pdf_path,
            pages=pages,
            deterministic_summary=deterministic_summary,
        )

    def _extract_sync(
        self,
        *,
        pdf_path: str,
        pages: list[int],
        deterministic_summary: dict[str, Any],
    ) -> VisionExtractionResult:
        merged = VisionExtractionResult(provider="minicpm_gguf")
        with tempfile.TemporaryDirectory(prefix="artha-minicpm-") as tmpdir:
            image_paths = render_pdf_pages_to_files(pdf_path, pages, Path(tmpdir))
            for page, image_path in image_paths:
                prompt = _vision_prompt(deterministic_summary, [page])
                command = [
                    self.cli_path,
                    "-m",
                    self.model_path,
                    "--mmproj",
                    self.mmproj_path,
                    "-c",
                    "4096",
                    "--temp",
                    "0",
                    "--top-p",
                    "1",
                    "--top-k",
                    "1",
                    "--repeat-penalty",
                    "1.05",
                    "--image",
                    str(image_path),
                    "-p",
                    prompt,
                ]
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                    check=True,
                )
                parsed = _parse_json_object(completed.stdout)
                page_result = result_from_vision_payload(parsed, provider="minicpm_gguf")
                _merge_vision_result(merged, page_result, default_page=page)
        return merged


class OllamaVisionExtractor:
    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        model: str = "openbmb/minicpm-o2.6:latest",
        timeout_sec: int = 120,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.timeout_sec = timeout_sec

    async def extract(
        self,
        *,
        pdf_path: str,
        pages: list[int],
        deterministic_summary: dict[str, Any],
    ) -> VisionExtractionResult:
        images = render_pdf_pages_for_vision(pdf_path, pages)
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": _vision_prompt(deterministic_summary, pages),
                    "images": images,
                }
            ],
            "options": {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
                "num_ctx": 4096,
            },
        }
        raw = await _post_json(self.endpoint, payload, timeout_sec=self.timeout_sec)
        content = _ollama_content(raw)
        parsed = _parse_json_object(content)
        result = result_from_vision_payload(parsed, provider="ollama_minicpm")
        if _needs_metadata_repair(result) and images:
            repair_payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": _vision_metadata_repair_prompt(pages[:1]),
                        "images": images[:1],
                    }
                ],
                "options": {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "num_ctx": 4096,
                },
            }
            repair_raw = await _post_json(self.endpoint, repair_payload, timeout_sec=self.timeout_sec)
            repair_content = _ollama_content(repair_raw)
            repair_parsed = _parse_json_object(repair_content)
            repair_result = result_from_vision_payload(repair_parsed, provider="ollama_minicpm")
            _merge_vision_result(result, repair_result, default_page=pages[0] if pages else 1)
        for field_name in _vision_fields_needing_repair(result):
            repair_payload = {
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": _vision_field_repair_prompt(field_name, pages[:2] or pages[:1]),
                        "images": images[:2] or images[:1],
                    }
                ],
                "options": {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": 1,
                    "num_ctx": 4096,
                },
            }
            repair_raw = await _post_json(self.endpoint, repair_payload, timeout_sec=self.timeout_sec)
            repair_content = _ollama_content(repair_raw)
            repair_parsed = _parse_json_object(repair_content)
            repair_result = result_from_vision_payload(
                _payload_from_field_repair(field_name, repair_parsed),
                provider="ollama_minicpm",
            )
            if repair_result.fields:
                result.raw_json.setdefault("field_repairs", {})[field_name] = repair_parsed
                _merge_vision_result(result, repair_result, default_page=pages[0] if pages else 1)
        return result


async def _post_json(endpoint: str, payload: dict[str, Any], *, timeout_sec: int) -> dict[str, Any]:
    return await asyncio.to_thread(_post_json_sync, endpoint, payload, timeout_sec)


def _post_json_sync(endpoint: str, payload: dict[str, Any], timeout_sec: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))


def get_configured_vision_extractor(*, force_ollama: bool = False) -> VisionExtractor | None:
    if os.getenv("JUDGMENT_VISION_FALLBACK", "").strip().lower() not in {"1", "true", "yes", "on"}:
        if force_ollama:
            endpoint = os.getenv("OLLAMA_VISION_ENDPOINT", "http://127.0.0.1:11434/api/chat").strip()
            model = os.getenv("OLLAMA_VISION_MODEL", "openbmb/minicpm-o2.6:latest").strip()
            timeout = int(os.getenv("JUDGMENT_VISION_TIMEOUT_SEC", "120") or "120")
            return OllamaVisionExtractor(endpoint=endpoint, model=model, timeout_sec=timeout)
        return None
    provider = os.getenv("JUDGMENT_VISION_PROVIDER", "minicpm").strip().lower()
    if provider in {"minicpm_gguf", "gguf", "llama_cpp"}:
        cli_path = os.getenv("MINICPM_GGUF_CLI", "").strip()
        model_path = os.getenv("MINICPM_GGUF_MODEL", "").strip()
        mmproj_path = os.getenv("MINICPM_GGUF_MMPROJ", "").strip()
        if not cli_path or not model_path or not mmproj_path:
            return None
        timeout = int(os.getenv("JUDGMENT_VISION_TIMEOUT_SEC", "120") or "120")
        return MiniCPMGGUFVisionExtractor(
            cli_path=cli_path,
            model_path=model_path,
            mmproj_path=mmproj_path,
            timeout_sec=timeout,
        )
    if provider == "ollama":
        endpoint = os.getenv("OLLAMA_VISION_ENDPOINT", "http://127.0.0.1:11434/api/chat").strip()
        model = os.getenv("OLLAMA_VISION_MODEL", "openbmb/minicpm-o2.6:latest").strip()
        timeout = int(os.getenv("JUDGMENT_VISION_TIMEOUT_SEC", "120") or "120")
        return OllamaVisionExtractor(endpoint=endpoint, model=model, timeout_sec=timeout)
    if provider == "lmstudio":
        endpoint = os.getenv("LMSTUDIO_VISION_ENDPOINT", "http://127.0.0.1:1234/v1/chat/completions").strip()
        model = os.getenv("LMSTUDIO_VISION_MODEL", os.getenv("MINICPM_VISION_MODEL", "local-model")).strip()
        timeout = int(os.getenv("JUDGMENT_VISION_TIMEOUT_SEC", "120") or "120")
        return MiniCPMHTTPVisionExtractor(endpoint=endpoint, timeout_sec=timeout, model=model)
    if provider != "minicpm":
        return None
    endpoint = os.getenv("MINICPM_VISION_ENDPOINT", "").strip()
    if not endpoint:
        return None
    timeout = int(os.getenv("JUDGMENT_VISION_TIMEOUT_SEC", "90") or "90")
    model = os.getenv("MINICPM_VISION_MODEL", "openbmb/MiniCPM-o-4_5")
    return MiniCPMHTTPVisionExtractor(endpoint=endpoint, timeout_sec=timeout, model=model)


def should_run_vision_fallback(
    package: JudgmentReviewPackage,
    documents: list[Document],
    pdf_profile: dict[str, Any] | None = None,
) -> bool:
    profile = pdf_profile or {}
    extraction = package.extraction
    if profile.get("profile_type") != "digital":
        return True
    if profile.get("sparse_pages"):
        return True
    if _looks_like_law_report(documents):
        return True
    weak_fields = [
        extraction.parties,
        extraction.petitioners,
        extraction.respondents,
        extraction.bench,
        extraction.judgment_date,
    ]
    return any(_field_is_weak(field) for field in weak_fields)


def select_vision_pages(
    documents: list[Document],
    pdf_profile: dict[str, Any] | None = None,
    *,
    max_pages: int | None = None,
) -> list[int]:
    page_count = int((pdf_profile or {}).get("page_count") or 0)
    known_pages = sorted({
        int((doc.metadata or {}).get("page") or 0)
        for doc in documents
        if int((doc.metadata or {}).get("page") or 0) > 0
    })
    if not page_count and known_pages:
        page_count = max(known_pages)
    selected: list[int] = []

    def add(page: int) -> None:
        if page > 0 and (not page_count or page <= page_count) and page not in selected:
            selected.append(page)

    profile_type = str((pdf_profile or {}).get("profile_type") or "").lower()
    if profile_type and profile_type != "digital":
        for page in (1, 2):
            add(page)
        if page_count:
            add(page_count - 1)
            add(page_count)
    else:
        for page in (1, 2, 3):
            add(page)
    for doc in documents:
        text = doc.page_content or ""
        page = int((doc.metadata or {}).get("page") or 0)
        if re.search(r"\b(?:judgment|order|coram|versus|petitioner|respondent|cji|j\.)\b", text, re.I):
            add(page)
    if page_count:
        add(page_count - 1)
        add(page_count)
    limit = max_pages or int(os.getenv("JUDGMENT_VISION_MAX_PAGES", "6") or "6")
    return selected[:limit]


def merge_vision_extraction(
    package: JudgmentReviewPackage,
    result: VisionExtractionResult,
    *,
    documents: list[Document],
    provider_name: str = "minicpm",
) -> JudgmentReviewPackage:
    changed = False
    should_rebuild_actions = _vision_should_rebuild_actions(package)
    for field_name, value in result.fields.items():
        if field_name not in VISION_FIELD_NAMES:
            continue
        if field_name == "disposition" and not _valid_vision_disposition(value):
            continue
        if value in (None, "", []):
            continue
        target = getattr(package.extraction, field_name, None)
        if target is None or not _field_is_weak(target):
            continue
        page = result.evidence_pages.get(field_name) or _first_page(documents)
        evidence_doc = _document_for_page(documents, page) or (documents[0] if documents else Document(page_content="", metadata={}))
        raw_value = _raw_value(value)
        evidence = [
            normalize_evidence(
                evidence_doc,
                raw_value,
                confidence=0.88,
                extraction_method=f"vision_{provider_name}",
            )
        ]
        for item in evidence:
            item.source_quality = "vision_ocr"
        setattr(
            package.extraction,
            field_name,
            ExtractedField(
                name=field_name,
                value=value,
                raw_value=raw_value,
                confidence=0.88,
                evidence=evidence,
                field_id=field_name,
                requires_review=True,
                notes=[f"Extracted by {provider_name} vision fallback from rendered PDF page."],
            ),
        )
        changed = True

    if result.directions and not package.extraction.directions:
        page = result.evidence_pages.get("directions") or _first_page(documents)
        evidence_doc = _document_for_page(documents, page) or (documents[0] if documents else Document(page_content="", metadata={}))
        for index, direction in enumerate(result.directions):
            if not str(direction).strip():
                continue
            if not _valid_vision_direction(direction):
                continue
            evidence = [
                normalize_evidence(
                    evidence_doc,
                    str(direction),
                    confidence=0.86,
                    extraction_method=f"vision_{provider_name}",
                )
            ]
            for item in evidence:
                item.source_quality = "vision_ocr"
            package.extraction.directions.append(
                ExtractedField(
                    name="direction",
                    value=str(direction),
                    raw_value=str(direction),
                    confidence=0.86,
                    evidence=evidence,
                    field_id=f"direction-{index}",
                    requires_review=True,
                    notes=[f"Extracted by {provider_name} vision fallback from rendered PDF page."],
                )
            )
            changed = True

    if changed:
        if should_rebuild_actions:
            package.action_items = build_action_plan(package.extraction)
            _attach_retrieval_scores(package.extraction, package.action_items, documents)
        _refresh_vision_risk_flags(package)
    package.source_metadata["vision_fallback_used"] = changed
    package.source_metadata["vision_provider"] = provider_name
    package.source_metadata["vision_raw_json"] = result.raw_json
    return package


def _vision_should_rebuild_actions(package: JudgmentReviewPackage) -> bool:
    if not package.action_items:
        return True
    return all(item.category in {"no_operational_action", "no_immediate_action"} for item in package.action_items)


def _valid_vision_disposition(value: Any) -> bool:
    return _normalize_vision_disposition(value) in VISION_DISPOSITIONS


def _normalize_vision_disposition(value: Any) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip().lower())
    text = text.replace("-", "_")
    return text


def _valid_vision_direction(direction: Any) -> bool:
    text = " ".join(str(direction or "").split())
    lowered = text.lower()
    if not text:
        return False
    if re.search(r"\b(?:tribunal|high court|lower court|trial court)\s+held\b", lowered):
        return False
    return bool(
        re.search(
            r"\b(?:accordingly|we\s+(?:direct|order|allow|dismiss|grant|set\s+aside)|"
            r"shall|is\s+directed\s+to|are\s+directed\s+to|remit|remand|restore|tag\s+this\s+appeal|tagged\s+with)\b",
            lowered,
        )
    )


def result_from_vision_payload(payload: dict[str, Any], *, provider: str = "minicpm") -> VisionExtractionResult:
    fields: dict[str, Any] = {}
    case_details = payload.get("case_details") if isinstance(payload.get("case_details"), dict) else payload
    for field in ("case_number", "case_type", "court", "bench", "judgment_date", "disposition", "advocates"):
        if field in case_details and not _vision_field_value_empty(field, case_details[field]):
            if field == "advocates":
                fields[field] = _advocate_values(case_details[field])
            elif field == "bench":
                fields[field] = _as_list(case_details[field])
            else:
                fields[field] = case_details[field]
    date_payload = payload.get("date_of_order")
    if isinstance(date_payload, dict) and date_payload.get("value"):
        fields["judgment_date"] = str(date_payload["value"]).strip()
    elif isinstance(date_payload, dict) and date_payload.get("raw_text"):
        fields["judgment_date"] = str(date_payload["raw_text"]).strip()
    departments = _public_entity_values(case_details.get("departments"))
    departments.extend(item for item in _public_entity_values(case_details.get("responsible_entities")) if item not in departments)
    if departments:
        fields["departments"] = departments
    parties = payload.get("parties") or payload.get("parties_involved") or case_details.get("parties")
    if isinstance(parties, dict):
        if parties.get("petitioners") is not None:
            fields["petitioners"] = _as_list(parties.get("petitioners"))
        if parties.get("respondents") is not None:
            fields["respondents"] = _as_list(parties.get("respondents"))
        combined = _as_list(parties.get("petitioners"))
        combined.extend(p for p in _as_list(parties.get("respondents")) if p not in combined)
        if combined:
            fields["parties"] = combined
    elif parties:
        fields["parties"] = _as_list(parties)
    for party_field in ("petitioners", "respondents"):
        if party_field in case_details:
            role_values = _as_list(case_details.get(party_field))
            if _role_values_consistent_with_parties(role_values, fields.get("parties")):
                fields[party_field] = role_values
    directions = (
        payload.get("key_directions_orders")
        or payload.get("operative_directions")
        or payload.get("directions")
        or case_details.get("key_directions_orders")
        or case_details.get("operative_directions")
        or case_details.get("directions")
    )
    evidence_pages = _normalize_evidence_pages(payload.get("evidence_pages") or case_details.get("evidence_pages"))
    source_pages = _direction_source_pages(directions)
    if source_pages:
        evidence_pages.setdefault("directions", source_pages[-1])
    return VisionExtractionResult(
        fields=fields,
        directions=_as_direction_list(directions),
        evidence_pages=evidence_pages,
        raw_json=payload,
        provider=provider,
    )


def _merge_vision_result(target: VisionExtractionResult, source: VisionExtractionResult, *, default_page: int) -> None:
    for key, value in source.fields.items():
        if value not in (None, "", []):
            if target.fields.get(key) in (None, "", []):
                target.fields[key] = value
                target.evidence_pages[key] = source.evidence_pages.get(key, default_page)
            else:
                target.fields.setdefault(key, value)
                target.evidence_pages.setdefault(key, source.evidence_pages.get(key, default_page))
    for direction in source.directions:
        if direction and direction not in target.directions:
            target.directions.append(direction)
    if source.directions:
        target.evidence_pages.setdefault("directions", source.evidence_pages.get("directions", default_page))
    target.raw_json.setdefault("pages", []).append(source.raw_json)


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        normalized = [_string_value(item) for item in value]
        return [item for item in normalized if item]
    normalized = _string_value(value)
    return [normalized] if normalized else []


def _advocate_values(value: Any) -> list[str]:
    values = []
    if isinstance(value, list):
        for item in value:
            text = _string_value(item)
            combined = f"{text} {_string_value((item or {}).get('role')) if isinstance(item, dict) else ''}"
            if text and _looks_like_advocate(combined):
                values.append(text)
        return values
    text = _string_value(value)
    return [text] if text and _looks_like_advocate(text) else []


def _looks_like_advocate(value: str) -> bool:
    return bool(re.search(r"\b(?:adv\.?|advocate|counsel|solicitor|attorney|for\s+(?:the\s+)?(?:petitioner|respondent|appellant))\b", value or "", re.I))


def _public_entity_values(value: Any) -> list[str]:
    return [item for item in _as_list(value) if _looks_like_public_entity(item)]


def _looks_like_public_entity(value: str) -> bool:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    if not text:
        return False
    if re.search(r"\b(?:hon'?ble|justice|judge|advocate|counsel|petitioner'?s advocate|respondent'?s advocate)\b", lowered):
        return False
    return bool(
        re.search(
            r"\b(?:state|government|department|commissioner|collector|registrar|registry|court|tribunal|"
            r"authority|board|corporation|municipal|panchayat|police|secretary|officer|director|"
            r"committee|council|university|ministry|railway|revenue|tax|customs)\b",
            lowered,
        )
    )


def _as_direction_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        normalized = _direction_text(value)
        return [normalized] if normalized and not _is_direction_heading_only(normalized) else []
    normalized = [_direction_text(item) for item in value]
    return [item for item in normalized if item and not _is_direction_heading_only(item)]


def _is_direction_heading_only(value: str) -> bool:
    text = " ".join(str(value or "").split()).strip(" .:-")
    if re.fullmatch(r"(?i)(?:order|judgment|ordered|the\s+following)", text):
        return True
    if re.fullmatch(r"(?i).*made\s+the\s+following", text):
        return True
    return False


def _direction_text(value: Any) -> str:
    if isinstance(value, dict):
        parts = []
        for key in ("title", "text", "direction", "action", "description", "value", "content"):
            item = value.get(key)
            if item:
                text = str(item).strip()
                if text and text not in parts:
                    parts.append(text)
        return " ".join(parts).strip()
    return _string_value(value)


def _string_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, dict):
        for key in ("name", "organization", "text", "direction", "value", "title", "content"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value).strip()


def _role_values_consistent_with_parties(role_values: list[str], parties: Any) -> bool:
    party_values = _as_list(parties)
    if not role_values or not party_values:
        return True
    normalized_parties = {_normalize_name_for_match(value) for value in party_values}
    for role_value in role_values:
        normalized_role = _normalize_name_for_match(role_value)
        if not any(_names_compatible(normalized_role, party) for party in normalized_parties):
            return False
    return True


def _normalize_name_for_match(value: str) -> str:
    return re.sub(r"\W+", "", value or "").lower()


def _names_compatible(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left in right or right in left:
        shorter = min(len(left), len(right))
        longer = max(len(left), len(right))
        return shorter >= 12 or (shorter / max(longer, 1)) >= 0.55
    return False


def _direction_source_pages(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    pages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_page = item.get("source_page") or item.get("page_number") or item.get("page")
        if str(raw_page).isdigit():
            pages.append(int(raw_page))
    return pages


def _normalize_evidence_pages(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {
            str(key): int(page)
            for key, page in value.items()
            if str(page).isdigit()
        }
    if not isinstance(value, list):
        return {}
    pages = [
        int(item.get("page_number") or item.get("page") or 0)
        for item in value
        if isinstance(item, dict) and str(item.get("page_number") or item.get("page") or "").isdigit()
    ]
    if not pages:
        return {}
    first_page = pages[0]
    last_page = pages[-1]
    return {
        "case_number": first_page,
        "court": first_page,
        "bench": first_page,
        "parties": first_page,
        "petitioners": first_page,
        "respondents": first_page,
        "judgment_date": last_page,
        "disposition": last_page,
        "directions": last_page,
    }


def render_pdf_pages_for_vision(pdf_path: str, pages: list[int], *, zoom: float = 1.8) -> list[str]:
    rendered: list[str] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as pdf:
        for page_number in pages:
            if page_number < 1 or page_number > len(pdf):
                continue
            pix = pdf[page_number - 1].get_pixmap(matrix=matrix, alpha=False)
            rendered.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return rendered


def render_pdf_pages_to_files(pdf_path: str, pages: list[int], output_dir: Path, *, zoom: float = 1.8) -> list[tuple[int, Path]]:
    rendered: list[tuple[int, Path]] = []
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as pdf:
        for page_number in pages:
            if page_number < 1 or page_number > len(pdf):
                continue
            pix = pdf[page_number - 1].get_pixmap(matrix=matrix, alpha=False)
            path = output_dir / f"page-{page_number}.png"
            pix.save(path)
            rendered.append((page_number, path))
    return rendered


def _vision_prompt(deterministic_summary: dict[str, Any], pages: list[int]) -> str:
    return (
        "You are the vision OCR extractor for Artha Judge. Extract the same schema used by the text/LLM extractor "
        "from the attached rendered PDF pages. Use only visible page content. Return strict JSON only with this shape: "
        "{\n"
        '  "case_details": {"case_number": null, "case_type": null, "court": null, "bench": [], '
        '"departments": [], "responsible_entities": [], "advocates": [], "disposition": null, "evidence_snippet": null},\n'
        '  "date_of_order": {"value": null, "raw_text": null, "confidence": 0.0, "evidence_snippet": null},\n'
        '  "parties_involved": {"petitioners": [], "respondents": [], "other_parties": [], "evidence_snippet": null},\n'
        '  "key_directions_orders": [{"text": "...", "confidence": 0.0, "evidence_snippet": "...", "source_page": null}],\n'
        '  "relevant_timelines": [{"text": "...", "confidence": 0.0, "evidence_snippet": "..."}],\n'
        '  "page_notes": [{"page": 1, "readability": "good|moderate|poor", "notes": "..."}],\n'
        '  "uncertainties": [],\n'
        '  "verbatim_final_order_excerpt": null,\n'
        '  "confidence": 0.0\n'
        "}. "
        "For scanned judgments, read first pages for metadata and final pages for the operative order. "
        "Do not infer petitioner/respondent roles unless visible labels such as Petitioner, Respondent, Versus, or party captions support them. "
        "bench must contain judges; do not put judges in departments or responsible_entities. "
        "departments/responsible_entities must be actual public bodies, authorities, courts, tribunals, registries, or officers visible in the judgment; never include advocates, counsel, judges, or private parties there. "
        "date_of_order.value should be the normalized judgment/order date if raw_text visibly contains a date. "
        "For key_directions_orders, capture only final operative orders/directions/outcomes, not prayers, submissions, facts, or lower-court history. "
        "When an ORDER heading appears, continue reading below it and include the final operative sentence such as dismissed, allowed, quashed, remanded, or disposed. "
        "An ORDER heading or 'made the following' sentence is not an operative direction by itself; if that is all you can read, put it in uncertainties instead of key_directions_orders. "
        "verbatim_final_order_excerpt must copy the visible final order sentence when present. "
        "If the final order only dismisses/allows/disposes with no operational task, return that outcome as a direction with the evidence snippet; do not invent an owner. "
        "If a value is unclear, return null or [] and explain in uncertainties instead of guessing. "
        f"Rendered page numbers: {pages}. "
        f"Current deterministic extraction hints: {json.dumps(deterministic_summary, default=str)}"
    )


def _vision_metadata_repair_prompt(pages: list[int]) -> str:
    return (
        "Read only the court judgment header/caption from the attached rendered page image. "
        "Return strict JSON only with this exact schema: "
        "{\"case_details\":{\"case_number\":string|null,\"court\":string|null,\"bench\":string[],"
        "\"judgment_date\":string|null,\"disposition\":null},"
        "\"parties\":{\"petitioners\":string[],\"respondents\":string[],\"other_parties\":string[]},"
        "\"operative_directions\":[],"
        "\"evidence_pages\":{\"case_number\":number|null,\"court\":number|null,\"bench\":number|null,"
        "\"judgment_date\":number|null,\"parties\":number|null,\"directions\":null},"
        "\"uncertainties\":string[]}. "
        "The case number may look like WRIT PETITION NO., WP No., W.P., Civil Appeal No., Criminal Appeal No., SLP, or similar. "
        "Do not return file names such as .odt unless no formal case number is visible. "
        f"Rendered page numbers: {pages}."
    )


def _needs_metadata_repair(result: VisionExtractionResult) -> bool:
    return any(result.fields.get(field) in (None, "", []) for field in ("case_number", "court"))


def _vision_fields_needing_repair(result: VisionExtractionResult) -> list[str]:
    fields = result.fields or {}
    repair_fields = []
    if _vision_field_missing_or_weak("bench", fields.get("bench")):
        repair_fields.append("bench")
    return repair_fields


def _vision_field_missing_or_weak(field_name: str, value: Any) -> bool:
    if value in (None, "", []):
        return True
    values = _as_list(value)
    if field_name == "bench":
        if not values:
            return True
        weak_values = {"before", "division bench", "single judge", "hon'ble", "honble"}
        return all(str(item).strip().lower() in weak_values for item in values)
    return False


def _vision_field_repair_prompt(field_name: str, pages: list[int]) -> str:
    if field_name == "bench":
        return (
            "Repair exactly one missing or weak field: bench.\n"
            "Read only the attached rendered judgment header/caption pages. Return strict JSON only with this shape:\n"
            "{\n"
            '  "field": "bench",\n'
            '  "value": [],\n'
            '  "raw_text": null,\n'
            '  "confidence": 0.0,\n'
            '  "evidence_snippet": null,\n'
            '  "uncertainties": []\n'
            "}\n"
            "Rules:\n"
            "- Extract only judge names from visible bench markers such as BEFORE THE HON'BLE, CORAM, PRESENT, or signature lines.\n"
            "- If the page says BEFORE THE HON'BLE MR.JUSTICE A.B.C., value must be [\"Justice A.B.C.\"].\n"
            "- Do not return generic labels like BEFORE, Single Judge, Division Bench, Court, or Hon'ble.\n"
            "- Do not extract case_number, court, date, parties, advocates, directions, or anything except bench.\n"
            "- If no judge name is visible, return value=[] with low confidence and explain in uncertainties.\n"
            f"Rendered page numbers: {pages}."
        )
    return (
        f"Repair exactly one missing or weak field: {field_name}. "
        "Return strict JSON only with field, value, raw_text, confidence, evidence_snippet, and uncertainties."
    )


def _payload_from_field_repair(field_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("value")
    if field_name == "bench":
        return {
            "case_details": {"bench": value if isinstance(value, list) else _as_list(value)},
            "evidence_pages": {"bench": payload.get("page") or 1},
        }
    return {"case_details": {field_name: value}}


def _chat_completion_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
    if isinstance(payload.get("content"), str):
        return str(payload["content"])
    return json.dumps(payload)


def _ollama_content(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(payload.get("response"), str):
        return payload["response"]
    return json.dumps(payload)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _field_is_weak(field: ExtractedField) -> bool:
    value = field.value
    if value in (None, "", []):
        return True
    if field.name == "case_number" and _vision_field_value_empty("case_number", value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"unknown", "not found", "not available", "n/a"}:
        return True
    if field.requires_review and field.confidence < 0.85:
        return True
    if field.confidence and field.confidence < 0.65:
        return True
    joined = " ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return bool(re.search(r"\b(?:see for example|advocate|attorney|m/s|collector|intended to|supplement each other)\b", joined, re.I))


def _vision_field_value_empty(field_name: str, value: Any) -> bool:
    if value in (None, "", []):
        return True
    if field_name == "case_number":
        text = str(value).strip()
        if not re.search(r"\d", text):
            return True
        if re.fullmatch(r"(?i)\s*(?:writ\s+petition|wp|w\.p\.|civil\s+appeal|criminal\s+appeal|slp|special\s+leave\s+petition)\s*(?:no\.?|number)?\s*[:.-]?\s*", text):
            return True
    return False


def _refresh_vision_risk_flags(package: JudgmentReviewPackage) -> None:
    extraction = package.extraction
    field_risks = {
        "missing_case_number": extraction.case_number,
        "missing_court": extraction.court,
        "missing_judgment_date": extraction.judgment_date,
    }
    refreshed = []
    for flag in package.risk_flags:
        if flag in field_risks and not _field_is_weak(field_risks[flag]):
            continue
        if flag == "missing_directions" and extraction.directions:
            continue
        if flag == "missing_action_items" and package.action_items:
            continue
        refreshed.append(flag)
    if not package.action_items and "missing_action_items" not in refreshed:
        refreshed.append("missing_action_items")
    package.risk_flags = refreshed


def _looks_like_law_report(documents: list[Document]) -> bool:
    sample = "\n".join((doc.page_content or "")[:500] for doc in documents[:3])
    return bool(re.search(r"\bSUPREME\s+COURT\s+REPORTS\b|\[\d{4}\]\s+Supp\.?\s+S\.?C\.?R\.?", sample, re.I))


def _raw_value(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _first_page(documents: list[Document]) -> int:
    if not documents:
        return 1
    return int((documents[0].metadata or {}).get("page") or 1)


def _document_for_page(documents: list[Document], page: int) -> Document | None:
    for document in documents:
        if int((document.metadata or {}).get("page") or 0) == page:
            return document
    return None
