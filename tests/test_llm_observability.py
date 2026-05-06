from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from langchain_core.documents import Document


def _workspace_tmp_path(name: str) -> Path:
    root = Path.cwd() / "user_data" / "pytest_tmp" / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _reset_llm_env(monkeypatch, workspace_path: Path):
    monkeypatch.setenv("ENABLE_LLM_TIMING_SPLIT", "true")
    monkeypatch.setenv("ENABLE_LLM_RAW_CACHE", "false")
    monkeypatch.setenv("ENABLE_LLM_SINGLEFLIGHT", "false")
    monkeypatch.setenv("LLM_RAW_CACHE_DIR", str(workspace_path / "llm_cache"))


def test_llm_complete_reports_lower_level_timing_without_changing_response(monkeypatch):
    import rag.llm as llm

    workspace_path = _workspace_tmp_path("llm_observability_timing")
    _reset_llm_env(monkeypatch, workspace_path)
    events = []

    def fake_call_llm(*, messages, model, temperature, max_tokens):
        return '{"ok": true}'

    monkeypatch.setattr(llm, "call_llm", fake_call_llm)

    async def run():
        result = await llm.llm_complete(
            model_name="deepseek-v4-flash",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            max_tokens=123,
            telemetry_callback=events.append,
        )
        return result

    assert asyncio.run(run()) == '{"ok": true}'
    assert len(events) == 1
    event = events[0]
    assert event["stage_name"] == "llm_transport"
    assert event["model"] == "deepseek-v4-flash"
    assert event["provider"] == "opencode"
    assert event["prompt_chars"] > 0
    assert event["response_chars"] == len('{"ok": true}')
    assert event["max_tokens"] == 123
    assert event["retry_attempt_index"] == 0
    assert event["retry_count"] == 2
    assert event["cache_hit"] is False
    assert event["singleflight_joined"] is False
    assert "rate_limiter_wait_s" in event
    assert "http_elapsed_s" in event


def test_llm_raw_cache_reuses_exact_raw_response_and_still_returns_text(monkeypatch):
    import rag.llm as llm

    workspace_path = _workspace_tmp_path("llm_observability_cache")
    _reset_llm_env(monkeypatch, workspace_path)
    monkeypatch.setenv("ENABLE_LLM_RAW_CACHE", "true")
    calls = {"count": 0}
    events = []

    def fake_call_llm(*, messages, model, temperature, max_tokens):
        calls["count"] += 1
        return '{"cached": true}'

    monkeypatch.setattr(llm, "call_llm", fake_call_llm)

    async def call_once():
        return await llm.llm_complete(
            model_name="deepseek-v4-flash",
            messages=[{"role": "user", "content": "same request"}],
            temperature=0.1,
            max_tokens=456,
            telemetry_callback=events.append,
        )

    assert asyncio.run(call_once()) == '{"cached": true}'
    assert asyncio.run(call_once()) == '{"cached": true}'
    assert calls["count"] == 1
    assert events[0]["cache_hit"] is False
    assert events[1]["cache_hit"] is True
    cache_files = list((workspace_path / "llm_cache").glob("*.json"))
    assert len(cache_files) == 1
    cached = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert cached["raw_response"] == '{"cached": true}'


def test_llm_singleflight_dedupes_identical_inflight_requests(monkeypatch):
    import rag.llm as llm

    workspace_path = _workspace_tmp_path("llm_observability_singleflight")
    _reset_llm_env(monkeypatch, workspace_path)
    monkeypatch.setenv("ENABLE_LLM_SINGLEFLIGHT", "true")
    calls = {"count": 0}
    events = []

    def fake_call_llm(*, messages, model, temperature, max_tokens):
        calls["count"] += 1
        import time

        time.sleep(0.05)
        return '{"singleflight": true}'

    monkeypatch.setattr(llm, "call_llm", fake_call_llm)

    async def call_once():
        return await llm.llm_complete(
            model_name="deepseek-v4-flash",
            messages=[{"role": "user", "content": "same in flight"}],
            temperature=0.1,
            max_tokens=789,
            telemetry_callback=events.append,
        )

    async def run():
        return await asyncio.gather(call_once(), call_once())

    assert asyncio.run(run()) == ['{"singleflight": true}', '{"singleflight": true}']
    assert calls["count"] == 1
    assert any(event["singleflight_owner"] is True for event in events)
    assert any(event["singleflight_joined"] is True for event in events)


def _review_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "IN THE HIGH COURT OF KARNATAKA AT BENGALURU "
                "Writ Petition No. 1234 of 2025 ABC Residents Association v. State of Karnataka and BBMP "
                "Judgment dated 15 March 2026."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content=(
                "The BBMP is directed to remove the encroachment within four weeks. "
                "The Urban Development Department shall file a compliance report within 30 days."
            ),
            metadata={"source": "judgment.pdf", "page": 2, "chunk_id": "p2"},
        ),
    ]


def _normalised_package(package) -> str:
    from judgment_workflow.serialization import serialize_review_package

    payload = serialize_review_package(package)
    payload.pop("source_metadata", None)

    def strip_volatile(value):
        if isinstance(value, dict):
            return {
                key: strip_volatile(item)
                for key, item in value.items()
                if key not in {"extracted_at", "created_at", "updated_at"}
            }
        if isinstance(value, list):
            return [strip_volatile(item) for item in value]
        return value

    payload = strip_volatile(payload)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def test_llm_first_review_package_is_identical_with_cache_and_singleflight(monkeypatch):
    import rag.llm as llm
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    workspace_path = _workspace_tmp_path("llm_observability_review_package")
    _reset_llm_env(monkeypatch, workspace_path)
    responses = [
        (
            '{"case_details":{"case_number":"Writ Petition No. 1234 of 2025",'
            '"case_type":"writ_petition","court":"High Court of Karnataka at Bengaluru",'
            '"bench":[],"departments":["BBMP"],"disposition":"allowed",'
            '"evidence_snippet":"Writ Petition No. 1234 of 2025 ABC Residents Association v. State of Karnataka and BBMP"},'
            '"date_of_order":{"value":"2026-03-15","raw_text":"15 March 2026","confidence":0.91,'
            '"evidence_snippet":"Judgment dated 15 March 2026"},'
            '"parties_involved":{"petitioners":["ABC Residents Association"],"respondents":["State of Karnataka","BBMP"],'
            '"other_parties":[],"evidence_snippet":"ABC Residents Association v. State of Karnataka and BBMP"},'
            '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.9}'
        ),
        (
            '{"context_summary":"Final order contains two directions.","needs_more_context":false,'
            '"action_items":[{"title":"Remove the encroachment","responsible_department":"BBMP",'
            '"category":"direct_compliance","priority":"high",'
            '"timeline":{"raw_text":"within four weeks","timeline_type":"explicit","confidence":0.82},'
            '"legal_basis":"The BBMP is directed to remove the encroachment within four weeks",'
            '"decision_reason":"The final order directly requires BBMP to remove the encroachment.",'
            '"review_recommendation":"Verify owner and deadline before publishing.",'
            '"requires_human_review":true,"confidence":0.89,"ambiguity_flags":[],'
            '"evidence_snippet":"The BBMP is directed to remove the encroachment within four weeks"}]}'
        ),
    ]

    def make_fake_call():
        state = {"index": 0}

        def fake_call_llm(*, messages, model, temperature, max_tokens):
            value = responses[state["index"]]
            state["index"] += 1
            return value

        return fake_call_llm

    monkeypatch.setattr(llm, "call_llm", make_fake_call())
    baseline = asyncio.run(
        build_llm_first_review_package(
            _review_documents(),
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
        )
    )
    baseline_json = _normalised_package(baseline)

    cache_dir = workspace_path / "llm_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    monkeypatch.setenv("ENABLE_LLM_RAW_CACHE", "true")
    monkeypatch.setenv("ENABLE_LLM_SINGLEFLIGHT", "true")

    monkeypatch.setattr(llm, "call_llm", make_fake_call())
    first_cached = asyncio.run(
        build_llm_first_review_package(
            _review_documents(),
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
        )
    )

    def fail_if_called(*, messages, model, temperature, max_tokens):
        raise AssertionError("cache should serve this exact request")

    monkeypatch.setattr(llm, "call_llm", fail_if_called)
    second_cached = asyncio.run(
        build_llm_first_review_package(
            _review_documents(),
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
        )
    )

    assert _normalised_package(first_cached) == baseline_json
    assert _normalised_package(second_cached) == baseline_json
