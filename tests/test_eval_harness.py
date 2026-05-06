import shutil
import asyncio
from datetime import date
from pathlib import Path

import pytest

from eval_harness import (
    EvalActionScore,
    EvalDirectionScore,
    EvalFieldScore,
    EvalMetricsScore,
    EvalReporter,
    EvalResult,
    EvalRunner,
    EvalSummary,
    _date_match,
    _list_overlap,
    _normalise_date,
    _normalise_text,
    _text_match,
    build_agent_brief,
    score_actions,
    score_directions,
    score_fields,
    score_quality_metrics,
)


class _SleepBackend:
    async def evaluate(self, pdf_path: str, llm_enabled: bool = True, metadata_json: str | None = None):
        await asyncio.sleep(0.1)
        return {
            "schema_version": "judgment-evaluation-v1",
            "record_id": Path(pdf_path).stem,
            "input": {"original_file_name": Path(pdf_path).name},
            "review": {"status": "pending_review", "overall_confidence": 0.5},
            "extraction": {"fields": [], "directions": []},
            "action_plan": {"items": []},
            "quality": {
                "metrics": {},
                "processing_metrics": {
                    "processing_ms": 100,
                    "stage_timings": [{"stage": "pdf_text_extraction", "duration_ms": 40}],
                    "llm_stage_timings": [{"stage": "llm_case_details", "duration_ms": 60}],
                },
            },
        }


def test_normalise_text_strips_and_lowercases():
    assert _normalise_text("  HIGH Court  ") == "high court"
    assert _normalise_text(None) == ""
    assert _normalise_text(42) == "42"


def test_normalise_date_parses_multiple_formats():
    assert _normalise_date("15 March 2026") == "2026-03-15"
    assert _normalise_date("2026-03-15") == "2026-03-15"
    assert _normalise_date("March 15, 2026") == "2026-03-15"
    assert _normalise_date("15-03-2026") == "2026-03-15"
    assert _normalise_date(None) is None
    assert _normalise_date("") is None


def test_text_match_exact_and_fuzzy():
    assert _text_match("hello", "hello") is True
    assert _text_match("Hello", "hello", fuzzy=True) is True
    assert _text_match("Hello", "hello", fuzzy=False) is False
    assert _text_match(None, None) is True
    assert _text_match("a", None) is False


def test_date_match_compares_normalised_dates():
    assert _date_match("15 March 2026", "2026-03-15") is True
    assert _date_match("15 March 2026", "14 March 2026") is False
    assert _date_match(None, None) is True


def test_list_overlap_full_partial_none():
    assert _list_overlap(["a", "b"], ["a", "b", "c"]) == (True, "full overlap (2/2)")
    assert _list_overlap(["a", "b"], ["a", "x"]) == (True, "partial overlap (1/2)")
    result = _list_overlap(["a", "b"], ["x", "y"])
    assert result[0] is False
    assert "no overlap" in result[1]
    assert _list_overlap([], []) == (True, "both empty")
    assert _list_overlap(["a"], []) == (False, "expected 1 vs actual 0")


def test_score_fields_with_fuzzy_text_and_date():
    response = {
        "extraction": {
            "fields": [
                {"field": "case_number", "value": "Writ Petition No. 1234 of 2025"},
                {"field": "court", "value": "High Court Of Karnataka At Bengaluru"},
                {"field": "judgment_date", "value": "2026-03-15"},
            ]
        }
    }
    expected = {
        "case_number": "Writ Petition No. 1234 of 2025",
        "court": "High Court of Karnataka at Bengaluru",
        "judgment_date": "15 March 2026",
    }
    scores = score_fields(response, expected)
    assert len(scores) == 3
    assert all(s.match for s in scores)
    assert scores[0].method == "fuzzy_text"
    assert scores[1].method == "fuzzy_text"
    assert scores[2].method == "date"


def test_score_fields_with_list_overlap():
    response = {
        "extraction": {
            "fields": [
                {"field": "departments", "value": ["BBMP", "Urban Development Department"]},
            ]
        }
    }
    expected = {
        "departments": ["bbmp", "Urban Development Dept"],
    }
    scores = score_fields(response, expected)
    assert len(scores) == 1
    assert scores[0].match is True
    assert scores[0].method == "set_overlap"


def test_score_fields_skips_action_items_and_directions_keys():
    response = {"extraction": {"fields": []}}
    expected = {"action_items": [], "directions": [], "risk_flags": []}
    scores = score_fields(response, expected)
    assert scores == []


def test_score_actions_matches_by_title_then_index():
    response = {
        "action_plan": {
            "items": [
                {
                    "action_id": "action-0",
                    "title": "Remove the encroachment",
                    "owner": "BBMP",
                    "category": "compliance",
                    "priority": "high",
                },
                {
                    "action_id": "action-1",
                    "title": "File a compliance report",
                    "owner": "Urban Development Department",
                    "category": "affidavit_report_filing",
                    "priority": "medium",
                },
            ]
        }
    }
    expected = {
        "action_items": [
            {"title": "Remove the encroachment", "owner": "bbmp", "category": "compliance", "priority": "high"},
            {"title": "File a compliance report", "owner": "Urban Development Dept", "category": "affidavit_report_filing", "priority": "medium"},
        ]
    }
    scores = score_actions(response, expected)
    assert len(scores) == 2
    assert scores[0].title_match is True
    assert scores[0].owner_match is True
    assert scores[1].title_match is True


def test_score_actions_notes_no_match():
    response = {"action_plan": {"items": []}}
    expected = {"action_items": [{"title": "Do something"}]}
    scores = score_actions(response, expected)
    assert len(scores) == 1
    assert scores[0].title_match is False
    assert "no matching action found" in scores[0].note


def test_score_directions_matches_by_text():
    response = {
        "extraction": {
            "directions": [
                {"value": "The BBMP is directed to remove the encroachment within four weeks."},
                {"value": "The Urban Development Department shall file a compliance report within 30 days."},
            ]
        }
    }
    expected = {
        "directions": [
            {"text": "The BBMP is directed to remove the encroachment within four weeks."},
            {"text": "The Urban Development Department shall file a compliance report within 30 days."},
        ]
    }
    scores = score_directions(response, expected)
    assert len(scores) == 2
    assert all(s.match for s in scores)
    assert scores[0].note == "matched by text"


def test_score_directions_falls_back_to_index():
    response = {
        "extraction": {
            "directions": [
                {"value": "First direction."},
                {"value": "Second direction."},
            ]
        }
    }
    expected = {
        "directions": [
            {"text": "Completely different first."},
            {"text": "Second direction."},
        ]
    }
    scores = score_directions(response, expected)
    assert scores[0].match is False
    assert scores[0].note == "matched by index"
    assert scores[1].match is True
    assert scores[1].note == "matched by text"


def test_score_quality_metrics_with_tolerance():
    response = {
        "quality": {
            "metrics": {
                "evidence_coverage_percent": 98,
                "ambiguous_count": 0,
                "duplicate_count": 0,
                "ocr_used": False,
            }
        }
    }
    expected = {
        "metrics": {
            "evidence_coverage_percent": 100,
            "ambiguous_count": 0,
            "duplicate_count": 0,
            "ocr_used": False,
        }
    }
    scores = score_quality_metrics(response, expected)
    assert len(scores) == 4
    coverage = next(s for s in scores if s.metric == "evidence_coverage_percent")
    assert coverage.match is True
    assert coverage.method == "tolerance_5"
    ambiguous = next(s for s in scores if s.metric == "ambiguous_count")
    assert ambiguous.match is True
    assert ambiguous.method == "exact"


def test_score_quality_metrics_returns_empty_for_bad_types():
    assert score_quality_metrics({}, {}) == []
    assert score_quality_metrics({"quality": {}}, {"metrics": "bad"}) == []


def test_build_agent_brief_returns_compact_markdown():
    response = {
        "schema_version": "judgment-evaluation-v1",
        "record_id": "record-1",
        "input": {"original_file_name": "sample.pdf"},
        "review": {"status": "pending_review", "overall_confidence": 0.82},
        "extraction": {
            "fields": [
                {
                    "field": "case_number",
                    "label": "Case Number",
                    "value": "WP 1234/2026",
                    "confidence": 0.9,
                    "status": "pending_review",
                    "evidence": [{"page": 1, "snippet": "WP 1234/2026", "confidence": 0.9}],
                },
                {"field": "legal_phrases", "value": ["disposed of"]},
            ],
            "directions": [{"value": "File compliance report", "evidence": [{"page": 2, "snippet": "file report"}]}],
        },
        "action_plan": {
            "items": [
                {
                    "title": "File compliance report",
                    "owner": "BBMP",
                    "category": "affidavit_report_filing",
                    "priority": "high",
                    "timeline": {"raw_text": "within 30 days"},
                    "evidence": [{"page": 2, "snippet": "within 30 days"}],
                }
            ]
        },
        "quality": {
            "risk_flags": ["missing_due_date"],
            "metrics": {"evidence_coverage_percent": 100, "ambiguous_count": 1, "duplicate_count": 0, "ocr_used": False},
            "processing_metrics": {
                "processing_ms": 123,
                "stage_timings": [{"stage": "pdf_text_extraction", "duration_ms": 50}],
                "llm_stage_timings": [{"stage": "llm_case_details", "duration_ms": 70}],
            },
        },
    }

    brief = build_agent_brief(response, "sample.pdf")

    assert "# Judgment Evaluation Brief: sample.pdf" in brief
    assert "Case Number: WP 1234/2026" in brief
    assert "File compliance report | Owner: BBMP" in brief
    assert "missing_due_date" in brief
    assert "Latency Breakdown" in brief
    assert "llm_case_details: 70 ms" in brief
    assert "legal_phrases" not in brief


def test_eval_runner_processes_directory_with_concurrency():
    tmp_path = Path.cwd() / "user_data" / "pytest_tmp" / "eval_harness_concurrency"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)
    for name in ("b.pdf", "a.pdf", "c.pdf"):
        (tmp_path / name).write_bytes(b"%PDF-1.4")
    output_dir = tmp_path / "out"

    async def run():
        runner = EvalRunner(_SleepBackend(), output_dir=output_dir, concurrency=3)
        started = asyncio.get_running_loop().time()
        results = await runner.run_directory(tmp_path)
        elapsed = asyncio.get_running_loop().time() - started
        return results, elapsed

    results, elapsed = asyncio.run(run())

    assert [result.pdf_name for result in results] == ["a.pdf", "b.pdf", "c.pdf"]
    assert elapsed < 0.25
    assert all(result.processing_metrics.get("processing_ms") == 100 for result in results)


def test_build_summary_computes_all_accuracies():
    results = [
        EvalResult(
            pdf_name="a.pdf",
            pdf_path="a.pdf",
            record_id="r1",
            success=True,
            schema_version="judgment-evaluation-v1",
            response_path="a.json",
            field_scores=[
                EvalFieldScore(field="case_number", expected="123", actual="123", match=True),
                EvalFieldScore(field="court", expected="HC", actual="HC", match=True),
            ],
            action_scores=[
                EvalActionScore(action_index=0, expected_title="T1", actual_title="T1", title_match=True),
            ],
            direction_scores=[
                EvalDirectionScore(direction_index=0, expected_text="D1", actual_text="D1", match=True),
            ],
            metrics_scores=[
                EvalMetricsScore(metric="coverage", expected=100, actual=100, match=True),
            ],
            duration_ms=1200.0,
            processing_metrics={
                "stage_timings": [{"stage": "pdf_text_extraction", "duration_ms": 100}],
                "llm_stage_timings": [
                    {"stage": "llm_case_details", "duration_ms": 900},
                    {
                        "stage_name": "llm_transport",
                        "logical_stage_name": "llm_case_details",
                        "http_elapsed_s": 1.2,
                        "rate_limiter_wait_s": 0.3,
                        "retry_sleep_s": 0.0,
                        "retry_attempt_index": 0,
                        "cache_event": "cache_write",
                        "cache_hit": False,
                        "singleflight_joined": False,
                        "timeout": False,
                    },
                    {
                        "stage": "llm_json_parse",
                        "logical_stage_name": "llm_case_details",
                        "json_attempt_index": 1,
                        "json_valid_dict": True,
                        "json_parse_s": 0.04,
                    },
                ],
            },
        ),
        EvalResult(
            pdf_name="b.pdf",
            pdf_path="b.pdf",
            record_id="r2",
            success=True,
            schema_version="judgment-evaluation-v1",
            response_path="b.json",
            field_scores=[
                EvalFieldScore(field="case_number", expected="456", actual="456", match=True),
                EvalFieldScore(field="court", expected="SC", actual="HC", match=False),
            ],
            action_scores=[
                EvalActionScore(action_index=0, expected_title="T2", actual_title="T3", title_match=False),
            ],
            direction_scores=[
                EvalDirectionScore(direction_index=0, expected_text="D2", actual_text="D3", match=False),
            ],
            metrics_scores=[
                EvalMetricsScore(metric="coverage", expected=100, actual=90, match=True),
            ],
            duration_ms=1500.0,
            processing_metrics={
                "stage_timings": [{"stage": "pdf_text_extraction", "duration_ms": 200}],
                "llm_stage_timings": [
                    {"stage": "llm_case_details", "duration_ms": 1000},
                    {
                        "stage_name": "llm_transport",
                        "logical_stage_name": "llm_case_details",
                        "http_elapsed_s": 0.0,
                        "rate_limiter_wait_s": 0.1,
                        "retry_sleep_s": 1.5,
                        "retry_attempt_index": 1,
                        "cache_event": "cache_hit",
                        "cache_hit": True,
                        "singleflight_joined": True,
                        "timeout": True,
                        "exception_type": "ReadTimeout",
                    },
                ],
            },
        ),
    ]
    reporter = EvalReporter()
    summary = reporter.build_summary(results, wall_clock_ms=1600.0)
    assert isinstance(summary, EvalSummary)
    assert summary.total == 2
    assert summary.successful == 2
    assert summary.failed == 0
    assert summary.field_accuracy == 0.75
    assert summary.action_accuracy == 0.5
    assert summary.direction_accuracy == 0.5
    assert summary.metrics_accuracy == 1.0
    assert len(summary.per_file) == 2
    assert summary.wall_clock_ms == 1600.0
    assert summary.total_file_processing_ms == 2700.0
    assert summary.effective_concurrency == 1.69
    assert summary.stage_totals[0]["stage"] == "llm_case_details"
    assert summary.stage_totals[0]["duration_ms"] == 1900.0
    assert summary.llm_timing_summary["http_elapsed_ms"] == 1200.0
    assert summary.llm_timing_summary["rate_limiter_wait_ms"] == 400.0
    assert summary.llm_timing_summary["retry_sleep_ms"] == 1500.0
    assert summary.llm_timing_summary["json_repair_count"] == 1
    assert summary.llm_timing_summary["transport_retry_count"] == 1
    assert summary.llm_timing_summary["cache_hits"] == 1
    assert summary.llm_timing_summary["cache_misses"] == 1
    assert summary.llm_timing_summary["cache_writes"] == 1
    assert summary.llm_timing_summary["singleflight_joins"] == 1
    assert summary.llm_timing_summary["timeout_or_failure_count"] == 1
    assert summary.slowest_files[0]["pdf_name"] == "b.pdf"
    assert summary.per_file[0]["field_hits"] == 2
    assert summary.per_file[0]["action_hits"] == 1
    assert summary.per_file[0]["direction_hits"] == 1
    assert summary.per_file[0]["metrics_hits"] == 1


def test_build_summary_with_failed_result():
    results = [
        EvalResult(
            pdf_name="fail.pdf",
            pdf_path="fail.pdf",
            record_id=None,
            success=False,
            schema_version=None,
            response_path=None,
            error="Connection refused",
            duration_ms=None,
        ),
    ]
    reporter = EvalReporter()
    summary = reporter.build_summary(results)
    assert summary.total == 1
    assert summary.successful == 0
    assert summary.failed == 1
    assert summary.field_accuracy == 0.0


def test_print_summary_marks_unscored_directory_runs(capsys):
    reporter = EvalReporter()
    summary = reporter.build_summary(
        [
            EvalResult(
                pdf_name="raw.pdf",
                pdf_path="raw.pdf",
                record_id="r1",
                success=True,
                schema_version="judgment-evaluation-v1",
                response_path="raw_response.json",
                agent_brief_path="raw_agent_brief.md",
                duration_ms=100.0,
            )
        ]
    )

    reporter.print_summary(summary)

    captured = capsys.readouterr()
    assert "Not scored (no expected manifest provided)" in captured.out
    assert "outputs generated for hand review" in captured.out


def test_save_summary_writes_json():
    reporter = EvalReporter()
    summary = EvalSummary(
        total=1,
        successful=1,
        failed=0,
        field_accuracy=1.0,
        action_accuracy=1.0,
        direction_accuracy=1.0,
        metrics_accuracy=1.0,
        per_file=[],
        generated_at="2026-01-01T00:00:00",
    )
    output_dir = Path.cwd() / "user_data" / "pytest_tmp" / "eval_harness_summary"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    path = reporter.save_summary(summary, output_dir)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["total"] == 1
    assert data["field_accuracy"] == 1.0
