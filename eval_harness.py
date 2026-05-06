from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import httpx


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EvalFieldScore:
    field: str
    expected: Any
    actual: Any
    match: bool
    method: str = "exact"
    note: str | None = None


@dataclass
class EvalActionScore:
    action_index: int
    expected_title: str | None
    actual_title: str | None
    title_match: bool
    owner_match: bool | None = None
    category_match: bool | None = None
    priority_match: bool | None = None
    note: str | None = None


@dataclass
class EvalDirectionScore:
    direction_index: int
    expected_text: str | None
    actual_text: str | None
    match: bool
    method: str = "fuzzy_text"
    note: str | None = None


@dataclass
class EvalMetricsScore:
    metric: str
    expected: Any
    actual: Any
    match: bool
    method: str = "exact"
    note: str | None = None


@dataclass
class EvalResult:
    pdf_name: str
    pdf_path: str
    record_id: str | None
    success: bool
    schema_version: str | None
    response_path: str | None
    agent_brief_path: str | None = None
    field_scores: list[EvalFieldScore] = field(default_factory=list)
    action_scores: list[EvalActionScore] = field(default_factory=list)
    direction_scores: list[EvalDirectionScore] = field(default_factory=list)
    metrics_scores: list[EvalMetricsScore] = field(default_factory=list)
    error: str | None = None
    duration_ms: float | None = None
    processing_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    total: int
    successful: int
    failed: int
    field_accuracy: float
    action_accuracy: float
    direction_accuracy: float
    metrics_accuracy: float
    per_file: list[dict[str, Any]]
    generated_at: str
    wall_clock_ms: float | None = None
    total_file_processing_ms: float = 0.0
    effective_concurrency: float = 0.0
    stage_totals: list[dict[str, Any]] = field(default_factory=list)
    llm_timing_summary: dict[str, Any] = field(default_factory=dict)
    slowest_files: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower().strip().replace("  ", " ")


def _normalise_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d-%m-%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text.lower().strip()


def _text_match(expected: Any, actual: Any, fuzzy: bool = True) -> bool:
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    if not fuzzy:
        return str(expected).strip() == str(actual).strip()
    return _normalise_text(expected) == _normalise_text(actual)


def _date_match(expected: Any, actual: Any) -> bool:
    return _normalise_date(expected) == _normalise_date(actual)


def _list_overlap(expected: list[Any], actual: list[Any]) -> tuple[bool, str]:
    exp_set = {_normalise_text(v) for v in expected if v}
    act_set = {_normalise_text(v) for v in actual if v}
    if not exp_set and not act_set:
        return True, "both empty"
    if not exp_set or not act_set:
        return False, f"expected {len(exp_set)} vs actual {len(act_set)}"
    overlap = exp_set & act_set
    if overlap == exp_set:
        return True, f"full overlap ({len(overlap)}/{len(exp_set)})"
    if overlap:
        return True, f"partial overlap ({len(overlap)}/{len(exp_set)})"
    return False, f"no overlap ({exp_set} vs {act_set})"


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class EvalBackend(Protocol):
    async def evaluate(
        self,
        pdf_path: str,
        llm_enabled: bool = True,
        metadata_json: str | None = None,
    ) -> dict[str, Any]:
        ...


class HttpBackend:
    def __init__(self, base_url: str = "http://127.0.0.1:5000", timeout: float = 900.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def evaluate(
        self,
        pdf_path: str,
        llm_enabled: bool = True,
        metadata_json: str | None = None,
    ) -> dict[str, Any]:
        data = {
            "user_id": "evaluation",
            "include_full_record": "false",
            "llm_enabled": "true" if llm_enabled else "false",
        }
        if metadata_json:
            data["metadata_json"] = metadata_json
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(pdf_path, "rb") as fh:
                response = await client.post(
                    f"{self.base_url}/judgments/evaluate",
                    data=data,
                    files={"file": (Path(pdf_path).name, fh, "application/pdf")},
                )
            response.raise_for_status()
            return response.json()


class LocalBackend:
    """Direct-import backend for testing without a running server."""

    def __init__(self, storage=None, canvas_app_id: str | None = None):
        from judgment_workflow.api import _build_evaluation_response
        from judgment_workflow.config import JUDGMENT_DATA_ROOT
        from judgment_workflow.config import CANVAS_APP_ID
        from judgment_workflow.pipeline import process_judgment_file

        self._build_evaluation_response = _build_evaluation_response
        self._process_judgment_file = process_judgment_file
        self._data_root = Path(JUDGMENT_DATA_ROOT)
        self._storage = storage
        self._db_path = Path("user_data") / "evaluation_outputs" / f"local_eval_{uuid.uuid4().hex}.db"
        self._canvas_app_id = canvas_app_id or CANVAS_APP_ID

    async def _get_storage(self):
        if self._storage is not None:
            return self._storage
        from storage.storage_config import StorageConfig, get_storage_backend

        db_path = self._db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = get_storage_backend(
            StorageConfig(backend_type="sqlite", sqlite_db_path=str(db_path))
        )
        await storage.initialize()
        self._storage = storage
        return storage

    async def evaluate(
        self,
        pdf_path: str,
        llm_enabled: bool = True,
        metadata_json: str | None = None,
    ) -> dict[str, Any]:
        storage = await self._get_storage()
        sanitized_filename = Path(pdf_path).name
        record_id = uuid.uuid4().hex
        record_root = self._data_root / "evaluation" / record_id
        record_root.mkdir(parents=True, exist_ok=True)
        target_pdf = record_root / "original.pdf"
        target_pdf.write_bytes(Path(pdf_path).read_bytes())
        source_metadata = {
            "source_system": "evaluation_upload",
            "original_file_name": sanitized_filename,
        }
        if metadata_json:
            metadata = json.loads(metadata_json)
            if isinstance(metadata, dict):
                source_metadata.update(metadata)
            source_metadata["source_system"] = "evaluation_upload"
            source_metadata["original_file_name"] = sanitized_filename

        record = await self._process_judgment_file(
            user_id="evaluation",
            pdf_path=str(target_pdf),
            record_id=record_id,
            original_file_name=sanitized_filename,
            source_metadata=source_metadata,
            storage=storage,
            canvas_app_id=self._canvas_app_id,
            llm_enabled=llm_enabled,
            processing_mode="evaluation",
        )
        return self._build_evaluation_response(record, include_full_record=False)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    def __init__(
        self,
        backend: EvalBackend,
        output_dir: str | Path = "eval_results",
        llm_enabled: bool = True,
        concurrency: int = 1,
    ):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.llm_enabled = llm_enabled
        self.concurrency = max(1, int(concurrency or 1))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(
        self,
        pdf_path: str,
        expected: dict[str, Any] | None = None,
        metadata_json: str | None = None,
    ) -> EvalResult:
        pdf_name = Path(pdf_path).name
        start = datetime.utcnow()
        try:
            response = await self.backend.evaluate(
                pdf_path,
                llm_enabled=self.llm_enabled,
                metadata_json=metadata_json,
            )
        except Exception as exc:
            return EvalResult(
                pdf_name=pdf_name,
                pdf_path=pdf_path,
                record_id=None,
                success=False,
                schema_version=None,
                response_path=None,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(datetime.utcnow() - start).total_seconds() * 1000,
            )

        duration_ms = (datetime.utcnow() - start).total_seconds() * 1000
        record_id = response.get("record_id")
        schema_version = response.get("schema_version")

        # Save raw response
        response_path = self.output_dir / f"{Path(pdf_name).stem}_response.json"
        response_path.write_text(json.dumps(response, indent=2, default=str), encoding="utf-8")
        agent_brief_path = self.output_dir / f"{Path(pdf_name).stem}_agent_brief.md"
        agent_brief_path.write_text(build_agent_brief(response, pdf_name), encoding="utf-8")

        result = EvalResult(
            pdf_name=pdf_name,
            pdf_path=pdf_path,
            record_id=record_id,
            success=True,
            schema_version=schema_version,
            response_path=str(response_path),
            agent_brief_path=str(agent_brief_path),
            duration_ms=duration_ms,
            processing_metrics=(response.get("quality") or {}).get("processing_metrics") or {},
        )

        if expected:
            result.field_scores = score_fields(response, expected)
            result.action_scores = score_actions(response, expected)
            result.direction_scores = score_directions(response, expected)
            result.metrics_scores = score_quality_metrics(response, expected)

        return result

    async def run_manifest(self, manifest_path: str | Path) -> list[EvalResult]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(manifest, list):
            manifest = [manifest]

        async def run_entry(index: int, entry: dict[str, Any]) -> tuple[int, EvalResult]:
            pdf_path = entry["pdf_path"]
            expected = entry.get("expected")
            metadata_json = entry.get("metadata_json")
            result = await self.run_single(
                pdf_path,
                expected=expected,
                metadata_json=metadata_json,
            )
            meta_path = self.output_dir / f"{Path(pdf_path).stem}_meta.json"
            meta_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
            return index, result

        return await self._run_ordered([lambda i=i, entry=entry: run_entry(i, entry) for i, entry in enumerate(manifest)])

    async def run_directory(self, pdf_dir: str | Path, pattern: str = "*.pdf") -> list[EvalResult]:
        pdf_dir = Path(pdf_dir)
        pdfs = sorted(pdf_dir.glob(pattern))
        async def run_pdf(index: int, pdf_path: Path) -> tuple[int, EvalResult]:
            result = await self.run_single(str(pdf_path))
            meta_path = self.output_dir / f"{pdf_path.stem}_meta.json"
            meta_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
            return index, result

        return await self._run_ordered([lambda i=i, pdf_path=pdf_path: run_pdf(i, pdf_path) for i, pdf_path in enumerate(pdfs)])

    async def _run_ordered(self, jobs) -> list[EvalResult]:
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_job(job):
            async with semaphore:
                return await job()

        completed = await asyncio.gather(*(run_job(job) for job in jobs))
        return [result for _, result in sorted(completed, key=lambda item: item[0])]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _get_extraction_field(response: dict[str, Any], field_name: str) -> Any:
    extraction = response.get("extraction") or {}
    fields = extraction.get("fields") or []
    for f in fields:
        if f.get("field") == field_name or f.get("field_id") == field_name:
            return f.get("value")
    # Fallback to legacy flat extraction
    flat = extraction.get(field_name) or {}
    if isinstance(flat, dict):
        return flat.get("value")
    return flat


def _get_action_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    action_plan = response.get("action_plan") or {}
    return list(action_plan.get("items") or [])


def build_agent_brief(response: dict[str, Any], pdf_name: str) -> str:
    """Create a compact Markdown packet that another AI agent can summarize."""
    input_payload = response.get("input") or {}
    review = response.get("review") or {}
    extraction = response.get("extraction") or {}
    quality = response.get("quality") or {}
    metrics = quality.get("metrics") or {}
    lines = [
        f"# Judgment Evaluation Brief: {pdf_name}",
        "",
        "## Record",
        f"- Record ID: {response.get('record_id') or 'missing'}",
        f"- Schema: {response.get('schema_version') or 'missing'}",
        f"- Original file: {input_payload.get('original_file_name') or pdf_name}",
        f"- Review status: {review.get('status') or 'unknown'}",
        f"- Overall confidence: {review.get('overall_confidence', 'unknown')}",
        "",
        "## Key Extracted Fields",
    ]

    for field_payload in _brief_fields(extraction.get("fields") or []):
        evidence = _first_evidence(field_payload)
        source = _brief_source(evidence)
        lines.append(
            f"- {field_payload.get('label') or _humanize_label(field_payload.get('field', 'field'))}: "
            f"{_brief_value(field_payload.get('value'))} "
            f"(confidence: {field_payload.get('confidence', 'unknown')}, status: {field_payload.get('status') or 'unknown'}{source})"
        )

    lines.extend(["", "## Directions"])
    directions = extraction.get("directions") or []
    if directions:
        for idx, direction in enumerate(directions, start=1):
            evidence = _first_evidence(direction)
            lines.append(f"- {idx}. {_brief_value(direction.get('value') or direction.get('text'))}{_brief_source(evidence)}")
    else:
        lines.append("- None extracted.")

    lines.extend(["", "## Proposed Action Items"])
    actions = _get_action_items(response)
    if actions:
        for idx, action in enumerate(actions, start=1):
            timeline = action.get("timeline") or {}
            due_date = timeline.get("due_date") or timeline.get("raw_text") or "not explicit"
            evidence = _first_evidence(action)
            lines.append(
                f"- {idx}. {action.get('title') or 'Untitled action'} | "
                f"Owner: {action.get('owner') or 'unclear'} | "
                f"Category: {action.get('category') or 'unknown'} | "
                f"Priority: {action.get('priority') or 'unknown'} | "
                f"Timeline: {due_date} | "
                f"Confidence: {action.get('confidence', 'unknown')} | "
                f"Human review: {action.get('requires_human_review', False)}{_brief_source(evidence)}"
            )
            if action.get("decision_reason"):
                lines.append(f"  Decision reason: {_brief_value(action.get('decision_reason'))}")
            if action.get("review_recommendation"):
                lines.append(f"  Review recommendation: {_brief_value(action.get('review_recommendation'))}")
    else:
        lines.append("- None extracted.")

    lines.extend(["", "## Quality Signals"])
    risk_flags = quality.get("risk_flags") or extraction.get("risk_flags") or []
    lines.append(f"- Risk flags: {_brief_value(risk_flags) if risk_flags else 'none'}")
    lines.append(f"- Evidence coverage: {metrics.get('evidence_coverage_percent', 'unknown')}")
    lines.append(f"- Ambiguity count: {metrics.get('ambiguous_count', 'unknown')}")
    lines.append(f"- Duplicate count: {metrics.get('duplicate_count', 'unknown')}")
    lines.append(f"- OCR used: {metrics.get('ocr_used', 'unknown')}")

    timing_lines = _brief_timing_lines(quality.get("processing_metrics") or {})
    if timing_lines:
        lines.extend(["", "## Latency Breakdown"])
        lines.extend(timing_lines)

    lines.extend(["", "## Source Snippets"])
    snippets = _brief_snippets(response)
    if snippets:
        for snippet in snippets[:8]:
            lines.append(f"- p. {snippet.get('page') or '?'}: {snippet.get('snippet')}")
    else:
        lines.append("- No source snippets returned.")

    return "\n".join(lines).strip() + "\n"


def _brief_fields(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    noisy_fields = {"legal_phrases"}
    ordered = []
    for field_payload in fields:
        field_name = field_payload.get("field") or field_payload.get("field_id")
        if field_name in noisy_fields:
            continue
        ordered.append(field_payload)
    return ordered


def _brief_value(value: Any) -> str:
    if value in (None, "", []):
        return "not extracted"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item) or "not extracted"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _first_evidence(payload: dict[str, Any]) -> dict[str, Any] | None:
    evidence = payload.get("evidence") or []
    return evidence[0] if evidence and isinstance(evidence[0], dict) else None


def _brief_source(evidence: dict[str, Any] | None) -> str:
    if not evidence:
        return ""
    page = evidence.get("page")
    confidence = evidence.get("confidence")
    source = f", source p. {page}" if page is not None else ", source page unknown"
    if confidence is not None:
        source += f", source confidence: {confidence}"
    return source


def _brief_snippets(response: dict[str, Any]) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    extraction = response.get("extraction") or {}
    for field_payload in extraction.get("fields") or []:
        for evidence in field_payload.get("evidence") or []:
            if evidence.get("snippet"):
                snippets.append({"page": evidence.get("page"), "snippet": evidence.get("snippet")})
    for direction in extraction.get("directions") or []:
        for evidence in direction.get("evidence") or []:
            if evidence.get("snippet"):
                snippets.append({"page": evidence.get("page"), "snippet": evidence.get("snippet")})
    for action in _get_action_items(response):
        for evidence in action.get("evidence") or []:
            if evidence.get("snippet"):
                snippets.append({"page": evidence.get("page"), "snippet": evidence.get("snippet")})
    seen = set()
    unique = []
    for item in snippets:
        key = (item.get("page"), item.get("snippet"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _brief_timing_lines(processing_metrics: dict[str, Any]) -> list[str]:
    stages = list(processing_metrics.get("stage_timings") or [])
    stages.extend(processing_metrics.get("llm_stage_timings") or [])
    if not stages:
        return []
    lines = [f"- Total processing: {processing_metrics.get('processing_ms', 'unknown')} ms"]
    for stage in sorted(stages, key=lambda item: float(item.get("duration_ms") or 0.0), reverse=True)[:8]:
        lines.append(f"- {stage.get('stage')}: {stage.get('duration_ms')} ms")
    return lines


def score_fields(response: dict[str, Any], expected: dict[str, Any]) -> list[EvalFieldScore]:
    scores: list[EvalFieldScore] = []
    for field_name, exp_val in expected.items():
        if field_name in {"action_items", "directions", "risk_flags"}:
            continue
        act_val = _get_extraction_field(response, field_name)

        if isinstance(exp_val, list):
            act_list = act_val if isinstance(act_val, list) else ([act_val] if act_val else [])
            matched, note = _list_overlap(exp_val, act_list)
            scores.append(
                EvalFieldScore(
                    field=field_name,
                    expected=exp_val,
                    actual=act_list,
                    match=matched,
                    method="set_overlap",
                    note=note,
                )
            )
        elif field_name in {"judgment_date"}:
            scores.append(
                EvalFieldScore(
                    field=field_name,
                    expected=exp_val,
                    actual=act_val,
                    match=_date_match(exp_val, act_val),
                    method="date",
                )
            )
        else:
            scores.append(
                EvalFieldScore(
                    field=field_name,
                    expected=exp_val,
                    actual=act_val,
                    match=_text_match(exp_val, act_val),
                    method="fuzzy_text",
                )
            )
    return scores


def score_actions(response: dict[str, Any], expected: dict[str, Any]) -> list[EvalActionScore]:
    scores: list[EvalActionScore] = []
    expected_actions = expected.get("action_items") or []
    actual_actions = _get_action_items(response)

    for idx, exp in enumerate(expected_actions):
        exp_title = exp.get("title")
        # Try to match by exact title first, then by index fallback
        matched_action = None
        for act in actual_actions:
            if _text_match(exp_title, act.get("title"), fuzzy=True):
                matched_action = act
                break
        if matched_action is None and idx < len(actual_actions):
            matched_action = actual_actions[idx]

        if matched_action is None:
            scores.append(
                EvalActionScore(
                    action_index=idx,
                    expected_title=exp_title,
                    actual_title=None,
                    title_match=False,
                    note="no matching action found",
                )
            )
            continue

        act_title = matched_action.get("title")
        scores.append(
            EvalActionScore(
                action_index=idx,
                expected_title=exp_title,
                actual_title=act_title,
                title_match=_text_match(exp_title, act_title),
                owner_match=_text_match(exp.get("owner"), matched_action.get("owner")),
                category_match=_text_match(exp.get("category"), matched_action.get("category")),
                priority_match=_text_match(exp.get("priority"), matched_action.get("priority")),
                note="matched by title" if _text_match(exp_title, act_title) else "matched by index",
            )
        )
    return scores


def score_directions(response: dict[str, Any], expected: dict[str, Any]) -> list[EvalDirectionScore]:
    scores: list[EvalDirectionScore] = []
    expected_directions = expected.get("directions") or []
    extraction = response.get("extraction") or {}
    actual_directions = extraction.get("directions") or []

    for idx, exp in enumerate(expected_directions):
        exp_text = exp.get("text") if isinstance(exp, dict) else str(exp)
        matched = None
        for act in actual_directions:
            act_text = act.get("value") or act.get("text") or ""
            if _text_match(exp_text, act_text, fuzzy=True):
                matched = act
                break
        if matched is None and idx < len(actual_directions):
            matched = actual_directions[idx]

        if matched is None:
            scores.append(
                EvalDirectionScore(
                    direction_index=idx,
                    expected_text=exp_text,
                    actual_text=None,
                    match=False,
                    note="no matching direction found",
                )
            )
            continue

        act_text = matched.get("value") or matched.get("text") or ""
        scores.append(
            EvalDirectionScore(
                direction_index=idx,
                expected_text=exp_text,
                actual_text=act_text,
                match=_text_match(exp_text, act_text),
                note="matched by text" if _text_match(exp_text, act_text) else "matched by index",
            )
        )
    return scores


def score_quality_metrics(response: dict[str, Any], expected: dict[str, Any]) -> list[EvalMetricsScore]:
    scores: list[EvalMetricsScore] = []
    expected_metrics = expected.get("metrics") or {}
    if not isinstance(expected_metrics, dict):
        return scores

    quality = response.get("quality") or {}
    actual_metrics = quality.get("metrics") or {}
    if not isinstance(actual_metrics, dict):
        return scores

    for metric_name, exp_val in expected_metrics.items():
        act_val = actual_metrics.get(metric_name)
        if metric_name == "evidence_coverage_percent":
            # Allow small tolerance for coverage percentages
            matched = False
            note = None
            if isinstance(exp_val, (int, float)) and isinstance(act_val, (int, float)):
                matched = abs(float(exp_val) - float(act_val)) <= 5.0
                note = f"expected {exp_val} vs actual {act_val} (±5%)"
            scores.append(
                EvalMetricsScore(
                    metric=metric_name,
                    expected=exp_val,
                    actual=act_val,
                    match=matched,
                    method="tolerance_5",
                    note=note,
                )
            )
        elif metric_name in {"ambiguous_count", "duplicate_count", "review_edit_count"}:
            matched = exp_val == act_val
            scores.append(
                EvalMetricsScore(
                    metric=metric_name,
                    expected=exp_val,
                    actual=act_val,
                    match=matched,
                    method="exact",
                )
            )
        else:
            scores.append(
                EvalMetricsScore(
                    metric=metric_name,
                    expected=exp_val,
                    actual=act_val,
                    match=_text_match(exp_val, act_val, fuzzy=False),
                    method="exact",
                )
            )
    return scores


def _aggregate_stage_totals(results: list[EvalResult]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float]] = {}
    for result in results:
        metrics = result.processing_metrics or {}
        stages = list(metrics.get("stage_timings") or [])
        stages.extend(metrics.get("llm_stage_timings") or [])
        for stage in stages:
            if not isinstance(stage, dict) or not stage.get("stage"):
                continue
            name = str(stage["stage"])
            bucket = totals.setdefault(name, {"duration_ms": 0.0, "count": 0.0})
            bucket["duration_ms"] += float(stage.get("duration_ms") or 0.0)
            bucket["count"] += 1.0
    return [
        {
            "stage": name,
            "duration_ms": round(values["duration_ms"], 2),
            "count": int(values["count"]),
            "avg_duration_ms": round(values["duration_ms"] / values["count"], 2) if values["count"] else 0.0,
        }
        for name, values in sorted(totals.items(), key=lambda item: item[1]["duration_ms"], reverse=True)
    ]


def _aggregate_llm_timing_summary(results: list[EvalResult]) -> dict[str, Any]:
    summary = {
        "http_elapsed_ms": 0.0,
        "rate_limiter_wait_ms": 0.0,
        "retry_sleep_ms": 0.0,
        "json_repair_count": 0,
        "transport_retry_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_writes": 0,
        "cache_disabled": 0,
        "singleflight_joins": 0,
        "timeout_or_failure_count": 0,
    }
    for result in results:
        metrics = result.processing_metrics or {}
        for event in metrics.get("llm_stage_timings") or []:
            if not isinstance(event, dict):
                continue
            if event.get("stage_name") == "llm_transport":
                summary["http_elapsed_ms"] += float(event.get("http_elapsed_s") or 0.0) * 1000
                summary["rate_limiter_wait_ms"] += float(event.get("rate_limiter_wait_s") or 0.0) * 1000
                summary["retry_sleep_ms"] += float(event.get("retry_sleep_s") or 0.0) * 1000
                if int(event.get("retry_attempt_index") or 0) > 0:
                    summary["transport_retry_count"] += 1
                cache_event = str(event.get("cache_event") or "")
                if event.get("cache_hit"):
                    summary["cache_hits"] += 1
                elif cache_event == "cache_miss":
                    summary["cache_misses"] += 1
                elif cache_event == "cache_write":
                    summary["cache_writes"] += 1
                    summary["cache_misses"] += 1
                elif cache_event == "cache_disabled":
                    summary["cache_disabled"] += 1
                if event.get("singleflight_joined"):
                    summary["singleflight_joins"] += 1
                if event.get("timeout") or event.get("exception_type"):
                    summary["timeout_or_failure_count"] += 1
            if event.get("stage") == "llm_json_parse":
                if int(event.get("json_attempt_index") or 0) > 0:
                    summary["json_repair_count"] += 1
                if event.get("exception_type"):
                    summary["timeout_or_failure_count"] += 1
    for key in ("http_elapsed_ms", "rate_limiter_wait_ms", "retry_sleep_ms"):
        summary[key] = round(summary[key], 2)
    return summary


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------

class EvalReporter:
    def build_summary(self, results: list[EvalResult], *, wall_clock_ms: float | None = None) -> EvalSummary:
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful

        all_field_scores = [s for r in results for s in r.field_scores]
        field_accuracy = (
            sum(1 for s in all_field_scores if s.match) / len(all_field_scores)
            if all_field_scores
            else 0.0
        )

        all_action_scores = [s for r in results for s in r.action_scores]
        action_accuracy = (
            sum(1 for s in all_action_scores if s.title_match) / len(all_action_scores)
            if all_action_scores
            else 0.0
        )

        all_direction_scores = [s for r in results for s in r.direction_scores]
        direction_accuracy = (
            sum(1 for s in all_direction_scores if s.match) / len(all_direction_scores)
            if all_direction_scores
            else 0.0
        )

        all_metrics_scores = [s for r in results for s in r.metrics_scores]
        metrics_accuracy = (
            sum(1 for s in all_metrics_scores if s.match) / len(all_metrics_scores)
            if all_metrics_scores
            else 0.0
        )

        per_file = []
        for r in results:
            field_total = len(r.field_scores)
            action_total = len(r.action_scores)
            direction_total = len(r.direction_scores)
            metrics_total = len(r.metrics_scores)
            per_file.append({
                "pdf_name": r.pdf_name,
                "success": r.success,
                "record_id": r.record_id,
                "error": r.error,
                "scored": bool(field_total or action_total or direction_total or metrics_total),
                "field_hits": sum(1 for s in r.field_scores if s.match),
                "field_total": field_total,
                "action_hits": sum(1 for s in r.action_scores if s.title_match),
                "action_total": action_total,
                "direction_hits": sum(1 for s in r.direction_scores if s.match),
                "direction_total": direction_total,
                "metrics_hits": sum(1 for s in r.metrics_scores if s.match),
                "metrics_total": metrics_total,
                "duration_ms": r.duration_ms,
            })

        total_file_processing_ms = round(sum(float(r.duration_ms or 0.0) for r in results), 2)
        stage_totals = _aggregate_stage_totals(results)
        llm_timing_summary = _aggregate_llm_timing_summary(results)
        slowest_files = [
            {"pdf_name": r.pdf_name, "duration_ms": r.duration_ms}
            for r in sorted(results, key=lambda item: float(item.duration_ms or 0.0), reverse=True)[:5]
        ]
        effective_concurrency = (
            round(total_file_processing_ms / wall_clock_ms, 2)
            if wall_clock_ms and wall_clock_ms > 0
            else 0.0
        )

        return EvalSummary(
            total=total,
            successful=successful,
            failed=failed,
            field_accuracy=round(field_accuracy, 4),
            action_accuracy=round(action_accuracy, 4),
            direction_accuracy=round(direction_accuracy, 4),
            metrics_accuracy=round(metrics_accuracy, 4),
            per_file=per_file,
            generated_at=datetime.utcnow().isoformat(),
            wall_clock_ms=round(wall_clock_ms, 2) if wall_clock_ms is not None else None,
            total_file_processing_ms=total_file_processing_ms,
            effective_concurrency=effective_concurrency,
            stage_totals=stage_totals,
            llm_timing_summary=llm_timing_summary,
            slowest_files=slowest_files,
        )

    def save_summary(self, summary: EvalSummary, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "summary.json"
        path.write_text(json.dumps(asdict(summary), indent=2, default=str), encoding="utf-8")
        return path

    def print_summary(self, summary: EvalSummary) -> None:
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total files : {summary.total}")
        print(f"Successful  : {summary.successful}")
        print(f"Failed      : {summary.failed}")
        has_scores = any(item.get("scored") for item in summary.per_file)
        if has_scores:
            print(f"Field accuracy    : {summary.field_accuracy:.2%}")
            print(f"Action accuracy   : {summary.action_accuracy:.2%}")
            print(f"Direction accuracy: {summary.direction_accuracy:.2%}")
            print(f"Metrics accuracy  : {summary.metrics_accuracy:.2%}")
        else:
            print("Scoring      : Not scored (no expected manifest provided)")
        if summary.wall_clock_ms is not None:
            print(f"Wall clock   : {summary.wall_clock_ms:.0f} ms")
            print(f"File time sum: {summary.total_file_processing_ms:.0f} ms")
            print(f"Concurrency  : {summary.effective_concurrency:.2f}x effective")
        if summary.stage_totals:
            print("Slow stages  : " + "; ".join(
                f"{item['stage']}={item['duration_ms']:.0f}ms" for item in summary.stage_totals[:5]
            ))
        if summary.llm_timing_summary:
            llm = summary.llm_timing_summary
            print(
                "LLM internals: "
                f"http={llm.get('http_elapsed_ms', 0):.0f}ms; "
                f"rate_wait={llm.get('rate_limiter_wait_ms', 0):.0f}ms; "
                f"retry_sleep={llm.get('retry_sleep_ms', 0):.0f}ms; "
                f"json_repairs={llm.get('json_repair_count', 0)}; "
                f"transport_retries={llm.get('transport_retry_count', 0)}; "
                f"cache_hits={llm.get('cache_hits', 0)}; "
                f"cache_misses={llm.get('cache_misses', 0)}; "
                f"singleflight_joins={llm.get('singleflight_joins', 0)}; "
                f"failures={llm.get('timeout_or_failure_count', 0)}"
            )
        print(f"Generated at      : {summary.generated_at}")
        print(f"{'='*60}")
        for pf in summary.per_file:
            status = "PASS" if pf["success"] else "FAIL"
            if pf.get("scored"):
                score_text = (
                    f"fields {pf['field_hits']}/{pf['field_total']}  "
                    f"actions {pf['action_hits']}/{pf['action_total']}  "
                    f"directions {pf['direction_hits']}/{pf['direction_total']}  "
                    f"metrics {pf['metrics_hits']}/{pf['metrics_total']}"
                )
            else:
                score_text = "outputs generated for hand review"
            print(f"  [{status}] {pf['pdf_name']}  {score_text}  ({pf['duration_ms']:.0f} ms)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Theme 11 judgment extraction evaluation harness",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to JSON manifest with ground-truth expectations",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Directory of PDFs to evaluate (no ground-truth comparison)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:5000",
        help="Base URL of the running judgment API",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run evaluation locally without HTTP server (imports pipeline directly)",
    )
    parser.add_argument(
        "--llm-enabled",
        type=lambda x: x.lower() in ("1", "true", "yes"),
        default=True,
        help="Enable LLM enrichment during evaluation",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="Glob pattern for PDF files when using --pdf-dir",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of PDFs to evaluate concurrently",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="Per-request timeout in seconds for HTTP evaluation",
    )
    return parser


async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.manifest and not args.pdf_dir:
        parser.error("Either --manifest or --pdf-dir is required.")

    backend: EvalBackend
    if args.local:
        backend = LocalBackend()
    else:
        backend = HttpBackend(base_url=args.base_url, timeout=args.request_timeout)

    runner = EvalRunner(
        backend=backend,
        output_dir=args.output_dir,
        llm_enabled=args.llm_enabled,
        concurrency=args.concurrency,
    )
    reporter = EvalReporter()

    wall_started_at = perf_counter()
    if args.manifest:
        results = await runner.run_manifest(args.manifest)
    else:
        results = await runner.run_directory(args.pdf_dir, pattern=args.pattern)
    wall_clock_ms = (perf_counter() - wall_started_at) * 1000

    summary = reporter.build_summary(results, wall_clock_ms=wall_clock_ms)
    reporter.print_summary(summary)
    reporter.save_summary(summary, args.output_dir)
    print(f"\nDetailed outputs saved to: {Path(args.output_dir).resolve()}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
