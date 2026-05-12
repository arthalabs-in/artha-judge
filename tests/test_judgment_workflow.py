from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langchain_core.documents import Document

from rag.judgment import build_judgment_review_package
from storage.storage_config import StorageConfig, get_storage_backend


@pytest.fixture
def workspace_tmp_path() -> Path:
    root = Path.cwd() / "user_data" / "pytest_tmp"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "judgment_workflow"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sample_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "IN THE HIGH COURT OF KARNATAKA AT BENGALURU\n"
                "Writ Petition No. 1234 of 2025\n"
                "ABC Residents Association v. State of Karnataka and BBMP\n"
                "Judgment dated 15 March 2026."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content=(
                "The BBMP is directed to remove the encroachment within four weeks. "
                "The Urban Development Department shall file a compliance report "
                "within 30 days from the date of this order."
            ),
            metadata={"source": "judgment.pdf", "page": 8, "chunk_id": "p8", "approx_bbox": [10, 20, 110, 60]},
        ),
    ]


def test_judgment_repository_round_trip_and_dashboard_listing(workspace_tmp_path: Path):
    from judgment_workflow.repository import JudgmentRepository

    async def run():
        storage = get_storage_backend(
            StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "judgments.db"))
        )
        await storage.initialize()
        repository = JudgmentRepository(storage=storage, canvas_app_id="test-app")

        package = build_judgment_review_package(_sample_documents(), {"source_system": "mock_ccms"})
        stored = await repository.create_record(
            user_id="user-1",
            record_id="record-1",
            review_package=package,
            source_metadata={"source_system": "mock_ccms"},
            original_pdf_path=str(workspace_tmp_path / "original.pdf"),
        )
        assert stored["record_id"] == "record-1"
        assert stored["extraction"]["case_number"]["value"] == "Writ Petition No. 1234 of 2025"

        reviewed = await repository.apply_review_decision(
            user_id="user-1",
            record_id="record-1",
            reviewer_id="reviewer-1",
            decision="edit",
            notes="Adjusted owner name",
            extraction_updates={"court": "High Court of Karnataka at Bengaluru"},
            action_updates=[
                {
                    "action_id": "action-0",
                    "responsible_department": "Bruhat Bengaluru Mahanagara Palike",
                    "status": "approved",
                }
            ],
        )
        assert reviewed["review_status"] == "edited"

        audit_events = await repository.list_audit_events("user-1", "record-1")
        assert len(audit_events) >= 2

        dashboard_records = await repository.list_dashboard_records("user-1")
        assert len(dashboard_records) == 1
        assert dashboard_records[0]["departments"] == ["Bruhat Bengaluru Mahanagara Palike"]
        await storage.close()

    asyncio.run(run())


def test_repository_duplicate_detection_and_approval_guard(workspace_tmp_path: Path):
    from judgment_workflow.repository import JudgmentRepository

    async def run():
        storage = get_storage_backend(
            StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "judgments.db"))
        )
        await storage.initialize()
        repository = JudgmentRepository(storage=storage, canvas_app_id="test-app")

        package = build_judgment_review_package(_sample_documents(), {"source_system": "manual_upload"})
        await repository.create_record(
            user_id="user-1",
            record_id="record-1",
            review_package=package,
            source_metadata={"source_system": "manual_upload"},
            original_pdf_path=str(workspace_tmp_path / "original.pdf"),
            document_hash="same-hash",
        )
        duplicates = await repository.find_duplicate_candidates("user-1", "same-hash", exclude_record_id="record-2")
        assert [item["record_id"] for item in duplicates] == ["record-1"]

        no_evidence_package = build_judgment_review_package(_sample_documents())
        no_evidence_package.action_items[0].evidence = []
        await repository.create_record(
            user_id="user-1",
            record_id="record-2",
            review_package=no_evidence_package,
            source_metadata={"source_system": "manual_upload"},
            original_pdf_path=str(workspace_tmp_path / "original.pdf"),
        )
        with pytest.raises(ValueError, match="every action item needs source evidence"):
            await repository.apply_review_decision(
                user_id="user-1",
                record_id="record-2",
                reviewer_id="reviewer-1",
                decision="approve",
            )
        approved = await repository.apply_review_decision(
            user_id="user-1",
            record_id="record-2",
            reviewer_id="reviewer-1",
            decision="approve",
            action_updates=[{"action_id": "action-0", "manual_override": True, "status": "approved"}],
            extraction_updates={
                "case_number": {"status": "approved"},
                "court": {"status": "approved"},
                "judgment_date": {"status": "approved"},
                "parties": {"status": "approved"},
            },
        )
        assert approved["review_status"] == "approved"
        await storage.close()

    asyncio.run(run())


def test_question_answer_is_grounded_in_record():
    from judgment_workflow.api import _answer_from_record
    from judgment_workflow.serialization import serialize_review_package

    package = build_judgment_review_package(_sample_documents())
    record = {"record_id": "record-1", **serialize_review_package(package)}

    response = _answer_from_record(record, "What action is required and by when?")

    assert response["grounded"] is True
    assert "Remove the encroachment" in response["answer"]
    assert response["sources"]


def test_llm_extractor_merges_valid_json_and_ignores_invalid():
    from judgment_workflow.llm_extractor import enrich_review_package_with_llm

    package = build_judgment_review_package(_sample_documents())
    enriched = asyncio.run(
        enrich_review_package_with_llm(
            package,
            llm_callable=lambda prompt: (
                '{"extraction":{"court":{"value":"High Court of Karnataka at Bengaluru","confidence":0.96}},'
                '"action_items":[{"action_id":"action-0","category":"compliance","priority":"high"}]}'
            ),
        )
    )
    assert enriched.extraction.court.value == "High Court of Karnataka at Bengaluru"
    assert enriched.action_items[0].category == "compliance"
    assert enriched.action_items[0].priority == "high"

    unchanged = asyncio.run(
        enrich_review_package_with_llm(package, llm_callable=lambda prompt: "not-json")
    )
    assert unchanged.extraction.case_number.value == "Writ Petition No. 1234 of 2025"


def test_llm_extractor_researches_low_confidence_fields_before_filling():
    from judgment_workflow.llm_extractor import enrich_review_package_with_llm

    package = build_judgment_review_package(_sample_documents())
    package.extraction.court.value = None
    package.extraction.court.confidence = 0.1
    package.extraction.court.evidence = []

    def fake_llm(prompt: str) -> str:
        assert "Retrieval context" in prompt
        assert "IN THE HIGH COURT OF KARNATAKA" in prompt
        return (
            '{"extraction":{"court":{'
            '"value":"High Court of Karnataka at Bengaluru",'
            '"confidence":0.93,'
            '"evidence_snippet":"IN THE HIGH COURT OF KARNATAKA AT BENGALURU"}}}'
        )

    enriched = asyncio.run(
        enrich_review_package_with_llm(package, documents=_sample_documents(), llm_callable=fake_llm)
    )

    assert enriched.extraction.court.value == "High Court of Karnataka at Bengaluru"
    assert enriched.extraction.court.confidence == 0.93
    assert enriched.extraction.court.evidence
    assert enriched.extraction.court.evidence[-1].extraction_method == "llm_research"


def test_llm_extractor_can_add_missing_action_items_from_research_evidence():
    from judgment_workflow.llm_extractor import enrich_review_package_with_llm

    package = build_judgment_review_package(_sample_documents())
    package.action_items = []
    package.extraction.directions = []

    enriched = asyncio.run(
        enrich_review_package_with_llm(
            package,
            documents=_sample_documents(),
            llm_callable=lambda prompt: (
                '{"action_items":[{'
                '"title":"Remove the encroachment",'
                '"responsible_department":"BBMP",'
                '"category":"compliance",'
                '"priority":"high",'
                '"confidence":0.88,'
                '"timeline":{"raw_text":"within four weeks","timeline_type":"explicit","confidence":0.82},'
                '"evidence_snippet":"The BBMP is directed to remove the encroachment within four weeks"'
                '}]}'
            ),
        )
    )

    assert len(enriched.action_items) == 1
    assert enriched.action_items[0].title == "Remove the encroachment"
    assert enriched.action_items[0].responsible_department == "BBMP"
    assert enriched.action_items[0].evidence[0].extraction_method == "llm_research"


def test_llm_extractor_accepts_fuzzy_supported_research_evidence():
    from judgment_workflow.llm_extractor import enrich_review_package_with_llm

    package = build_judgment_review_package(_sample_documents())
    package.action_items = []
    package.extraction.directions = []

    enriched = asyncio.run(
        enrich_review_package_with_llm(
            package,
            documents=_sample_documents(),
            llm_callable=lambda prompt: (
                '{"action_items":[{'
                '"title":"File a compliance report",'
                '"responsible_department":"Urban Development Department",'
                '"category":"affidavit_report_filing",'
                '"priority":"high",'
                '"confidence":0.87,'
                '"evidence_snippet":"Urban Development Department shall file compliance report within 30 days"'
                '}]}'
            ),
        )
    )

    assert len(enriched.action_items) == 1
    assert enriched.action_items[0].title == "File a compliance report"
    assert enriched.action_items[0].evidence[0].extraction_method == "llm_research"


def test_vision_fallback_merges_low_confidence_ocr_fields():
    from judgment_workflow.vision_extraction import VisionExtractionResult, merge_vision_extraction

    documents = [
        Document(
            page_content="632 SUPREME COURT REPORTS [1973] Supp. S.C.R.\nQam v. Kbiala (KlilHlna, J.)",
            metadata={"source": "ocr.pdf", "page": 1, "chunk_id": "p1", "source_quality": "text"},
        )
    ]
    package = build_judgment_review_package(documents)
    package.extraction.parties.value = ["Qam", "Kbiala"]
    package.extraction.parties.confidence = 0.4
    package.extraction.parties.requires_review = True

    merged = merge_vision_extraction(
        package,
        VisionExtractionResult(
            fields={
                "petitioners": ["Kesavananda Bharati"],
                "respondents": ["State of Kerala"],
                "parties": ["Kesavananda Bharati", "State of Kerala"],
                "bench": ["Justice H.R. Khanna"],
            },
            evidence_pages={"parties": 1, "bench": 1},
            raw_json={"confidence": 0.92},
        ),
        documents=documents,
        provider_name="minicpm",
    )

    assert merged.extraction.parties.value == ["Kesavananda Bharati", "State of Kerala"]
    assert merged.extraction.petitioners.value == ["Kesavananda Bharati"]
    assert merged.extraction.respondents.value == ["State of Kerala"]
    assert merged.extraction.bench.value == ["Justice H.R. Khanna"]
    assert merged.extraction.parties.evidence[0].extraction_method == "vision_minicpm"
    assert merged.source_metadata["vision_fallback_used"] is True


def test_vision_fallback_does_not_overwrite_strong_deterministic_fields():
    from judgment_workflow.vision_extraction import VisionExtractionResult, merge_vision_extraction

    package = build_judgment_review_package(_sample_documents())
    original_parties = list(package.extraction.parties.value)

    merged = merge_vision_extraction(
        package,
        VisionExtractionResult(
            fields={"parties": ["Wrong OCR Party", "Wrong OCR Respondent"]},
            evidence_pages={"parties": 1},
            raw_json={},
        ),
        documents=_sample_documents(),
        provider_name="minicpm",
    )

    assert merged.extraction.parties.value == original_parties
    assert merged.source_metadata["vision_fallback_used"] is False


def test_vision_fallback_adds_operational_directions_for_action_plan():
    from judgment_workflow.vision_extraction import VisionExtractionResult, merge_vision_extraction

    documents = [
        Document(
            page_content="ORDER\nThe scanned page contains the operative order.",
            metadata={"source": "ocr.pdf", "page": 3, "chunk_id": "p3"},
        )
    ]
    package = build_judgment_review_package(documents)
    package.extraction.parties.requires_review = True
    package.action_items = []

    merged = merge_vision_extraction(
        package,
        VisionExtractionResult(
            fields={"court": "Supreme Court of India"},
            directions=["The State Government shall file a compliance report within 30 days."],
            evidence_pages={"directions": 3, "court": 3},
            raw_json={"ok": True},
        ),
        documents=documents,
        provider_name="minicpm",
    )

    assert merged.extraction.directions
    assert merged.extraction.directions[0].evidence[0].extraction_method == "vision_minicpm"
    assert any(item.title == "File a compliance report" for item in merged.action_items)
    assert "missing_directions" not in merged.risk_flags
    assert "missing_action_items" not in merged.risk_flags


def test_vision_fallback_trigger_selects_noisy_ocr_pages():
    from judgment_workflow.vision_extraction import should_run_vision_fallback, select_vision_pages

    documents = [
        Document(
            page_content="632 SUPREME COURT REPORTS [1973] Supp. S.C.R.\nQam v. Kbiala (KlilHlna, J.)",
            metadata={"source": "ocr.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content="Historical discussion continues.",
            metadata={"source": "ocr.pdf", "page": 2, "chunk_id": "p2"},
        ),
        Document(
            page_content="ORDER\nThe final operative order is set out here.",
            metadata={"source": "ocr.pdf", "page": 7, "chunk_id": "p7"},
        ),
    ]
    package = build_judgment_review_package(documents)
    package.extraction.parties.value = ["Qam", "Kbiala"]
    package.extraction.parties.confidence = 0.4
    package.extraction.parties.requires_review = True

    assert should_run_vision_fallback(package, documents, {"profile_type": "digital"}) is True
    assert select_vision_pages(documents, {"page_count": 7}, max_pages=4) == [1, 2, 3, 7]


def test_vision_merge_does_not_overwrite_existing_action_plan():
    from rag.judgment.types import ActionItem, Timeline
    from judgment_workflow.vision_extraction import VisionExtractionResult, merge_vision_extraction

    documents = [
        Document(
            page_content="The appeals are allowed and the matters are remitted to the High Court.",
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]
    package = build_judgment_review_package(documents)
    package.action_items = [
        ActionItem(
            title="Fresh consideration by High Court",
            responsible_department="High Court",
            timeline=Timeline(timeline_type="not_specified"),
            category="direct_compliance",
        )
    ]
    package.extraction.judgment_date.value = None
    package.extraction.judgment_date.confidence = 0.0

    merged = merge_vision_extraction(
        package,
        VisionExtractionResult(
            fields={"judgment_date": "2009-04-08"},
            directions=["Leave granted."],
            evidence_pages={"judgment_date": 1, "directions": 1},
            raw_json={"ok": True},
        ),
        documents=documents,
        provider_name="minicpm",
    )

    assert [item.title for item in merged.action_items] == ["Fresh consideration by High Court"]


def test_vision_merge_rejects_non_disposition_and_lower_court_history_directions():
    from judgment_workflow.vision_extraction import VisionExtractionResult, merge_vision_extraction

    documents = [
        Document(
            page_content="Accordingly, we grant leave and tag this appeal with connected civil appeals.",
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]
    package = build_judgment_review_package(documents)
    package.extraction.disposition.value = None
    package.extraction.disposition.confidence = 0.0
    package.extraction.directions = []

    merged = merge_vision_extraction(
        package,
        VisionExtractionResult(
            fields={"disposition": "Petition for Special Leave to Appeal"},
            directions=[
                "The Tribunal held that the short listing process was not satisfactory.",
                "Accordingly, we grant leave and tag this appeal with connected civil appeals.",
            ],
            evidence_pages={"disposition": 1, "directions": 1},
            raw_json={"ok": True},
        ),
        documents=documents,
        provider_name="minicpm",
    )

    assert merged.extraction.disposition.value is None
    assert [field.value for field in merged.extraction.directions] == [
        "Accordingly, we grant leave and tag this appeal with connected civil appeals."
    ]


def test_vision_page_selection_prioritizes_final_pages_for_scanned_pdfs():
    from judgment_workflow.vision_extraction import select_vision_pages

    documents = [
        Document(page_content="", metadata={"source": "scan.pdf", "page": 1}),
    ]

    assert select_vision_pages(documents, {"profile_type": "scanned", "page_count": 135}, max_pages=4) == [
        1,
        2,
        134,
        135,
    ]


def test_pipeline_helper_applies_configured_vision_fallback():
    from judgment_workflow.pipeline import _apply_vision_fallback_if_needed
    from judgment_workflow.vision_extraction import VisionExtractionResult

    class FakeVisionExtractor:
        def __init__(self):
            self.calls = []

        async def extract(self, *, pdf_path, pages, deterministic_summary):
            self.calls.append((pdf_path, pages, deterministic_summary))
            return VisionExtractionResult(
                fields={
                    "petitioners": ["Kesavananda Bharati"],
                    "respondents": ["State of Kerala"],
                    "parties": ["Kesavananda Bharati", "State of Kerala"],
                },
                evidence_pages={"parties": 1, "petitioners": 1, "respondents": 1},
                raw_json={"ok": True},
            )

    documents = [
        Document(
            page_content="632 SUPREME COURT REPORTS [1973] Supp. S.C.R.\nQam v. Kbiala (KlilHlna, J.)",
            metadata={"source": "ocr.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]
    package = build_judgment_review_package(documents)
    package.extraction.parties.value = ["Qam", "Kbiala"]
    package.extraction.parties.confidence = 0.4
    package.extraction.parties.requires_review = True
    fake = FakeVisionExtractor()

    merged = asyncio.run(
        _apply_vision_fallback_if_needed(
            package,
            documents=documents,
            pdf_path="ocr.pdf",
            pdf_profile={"profile_type": "digital", "page_count": 1},
            vision_extractor=fake,
        )
    )

    assert fake.calls
    assert fake.calls[0][1] == [1]
    assert merged.extraction.parties.value == ["Kesavananda Bharati", "State of Kerala"]
    assert merged.source_metadata["vision_fallback_enabled"] is True
    assert merged.source_metadata["vision_fallback_used"] is True


def test_pipeline_allows_vision_only_scanned_pdf_when_vision_enabled():
    from judgment_workflow.pipeline import _prepare_review_documents

    documents = _prepare_review_documents(
        documents=[],
        table_documents=[],
        ocr_documents=[],
        pdf_path="scan.pdf",
        original_file_name="scan.pdf",
        source_metadata={"source_system": "evaluation_upload"},
        pdf_profile={"profile_type": "image_only", "page_count": 3},
        vision_enabled=True,
    )

    assert len(documents) == 1
    assert documents[0].page_content == ""
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["source_quality"] == "vision_only"


def test_vision_payload_parser_keeps_string_values_whole():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {"bench": "Justice A. Example"},
            "parties": {"petitioners": "Alpha Ltd.", "respondents": "State of Example"},
            "operative_directions": "Tag the appeal with connected matters.",
        }
    )

    assert result.fields["petitioners"] == ["Alpha Ltd."]
    assert result.fields["respondents"] == ["State of Example"]
    assert result.directions == ["Tag the appeal with connected matters."]


def test_vision_payload_parser_normalizes_nested_party_objects_and_page_lists():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "case_number": "WP No. 2558 of 2018",
                "parties": [{"name": "Alpha"}, {"name": "State"}],
                "petitioners": [{"name": "Alpha"}],
                "respondents": [{"name": "State"}],
                "operative_directions": [{"text": "Issue validity certificate within one month.", "page_number": 3}],
                "evidence_pages": [{"page_number": 1}, {"page_number": 3}],
            }
        }
    )

    assert result.fields["parties"] == ["Alpha", "State"]
    assert result.fields["petitioners"] == ["Alpha"]
    assert result.fields["respondents"] == ["State"]
    assert result.directions == ["Issue validity certificate within one month."]
    assert result.evidence_pages["case_number"] == 1
    assert result.evidence_pages["directions"] == 3


def test_vision_payload_parser_accepts_direction_key_and_rejects_contradictory_roles():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "parties": [
                    {"name": "Kartik Mangare"},
                    {"name": "Grishma Mangare"},
                ],
                "petitioners": [
                    {"name": "The Vice-Chairman, Scheduled Tribe Caste Scrutiny Committee"}
                ],
                "respondents": [],
            },
            "operative_directions": [
                {"direction": "The impugned order is quashed and set aside."},
                {"direction": "Issue validity certificate to the petitioners within 15 days."},
            ],
        }
    )

    assert result.fields["parties"] == ["Kartik Mangare", "Grishma Mangare"]
    assert result.fields.get("petitioners") in (None, [])
    assert result.directions == [
        "The impugned order is quashed and set aside.",
        "Issue validity certificate to the petitioners within 15 days.",
    ]


def test_vision_payload_parser_combines_action_description_and_accepts_fuzzy_roles():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "parties": [
                    {"name": "The Scheduled Tribe-Caste Certificate Scrutiny Committee, through its Member-Secretary"},
                    {"name": "Ratiram Sonba Chaudhari"},
                ],
                "petitioners": [{"name": "Ratiram Sonba Chaudhari"}],
                "respondents": [{"name": "The Scheduled Tribe-Caste Certificate Scrutiny Committee"}],
                "operative_directions": [
                    {
                        "action": "Writ Petition is allowed.",
                        "description": "The impugned order is quashed and set aside.",
                        "source_page": 3,
                    },
                    {
                        "action": "Respondent Committee shall issue validity certificate within four weeks.",
                        "source_page": 3,
                    },
                ],
            }
        }
    )

    assert result.fields["petitioners"] == ["Ratiram Sonba Chaudhari"]
    assert result.fields["respondents"] == ["The Scheduled Tribe-Caste Certificate Scrutiny Committee"]
    assert result.directions == [
        "Writ Petition is allowed. The impugned order is quashed and set aside.",
        "Respondent Committee shall issue validity certificate within four weeks.",
    ]
    assert result.evidence_pages["directions"] == 3


def test_vision_prompt_uses_llm_extractor_schema():
    from judgment_workflow.vision_extraction import _vision_prompt

    prompt = _vision_prompt({"case_number": None}, [1, 2, 4])

    assert '"case_type": null' in prompt
    assert '"date_of_order": {"value": null' in prompt
    assert '"parties_involved": {"petitioners": []' in prompt
    assert '"key_directions_orders": [{"text": "...", "confidence": 0.0' in prompt
    assert '"relevant_timelines": [{"text": "...", "confidence": 0.0' in prompt
    assert '"verbatim_final_order_excerpt": null' in prompt
    assert "An ORDER heading or 'made the following' sentence is not an operative direction by itself" in prompt
    assert "same schema used by the text/LLM extractor" in prompt


def test_vision_result_merge_overwrites_empty_primary_values():
    from judgment_workflow.vision_extraction import VisionExtractionResult, _merge_vision_result

    target = VisionExtractionResult(
        fields={"case_number": None, "court": "High Court"},
        evidence_pages={"court": 1},
    )
    source = VisionExtractionResult(
        fields={"case_number": "Writ Petition No. 6840 of 2018", "court": "Different Court"},
        evidence_pages={"case_number": 1, "court": 1},
    )

    _merge_vision_result(target, source, default_page=1)

    assert target.fields["case_number"] == "Writ Petition No. 6840 of 2018"
    assert target.fields["court"] == "High Court"


def test_vision_payload_parser_rejects_incomplete_case_number_labels():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload({"case_details": {"case_number": "WRIT PETITION NO."}})

    assert "case_number" not in result.fields


def test_vision_payload_parser_does_not_treat_observed_judge_entity_as_department():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "case_number": "CIVIL REVISION PETITION NO.2982/1997",
                "court": "High Court of Karnataka at Bangalore",
                "responsible_entities": ["The Hon'ble Mr. Justice Y.Bhaskar Rao"],
                "advocates": [{"name": "Sri T.Narayanaswamy, Adv.", "role": "Petitioner's Advocate"}],
            },
            "parties_involved": {
                "petitioners": [{"name": "K.Nissar Ahmed"}],
                "respondents": [{"name": "M/s.Karnataka Agro Industries Corp.Ltd."}],
            },
        }
    )

    assert "departments" not in result.fields
    assert result.fields["advocates"] == ["Sri T.Narayanaswamy, Adv."]


def test_vision_payload_parser_rejects_standalone_order_heading_as_direction():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload({"key_directions_orders": [{"text": "ORDER"}]})

    assert result.directions == []


def test_vision_payload_parser_keeps_respondent_organization_and_filters_advocate_roles():
    from judgment_workflow.vision_extraction import result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "advocates": [
                    {"name": "Sri.T.Narayanaswamy, Adv.", "role": "Petitioner"},
                    {
                        "name": "General Manager (P) and Secretary",
                        "organization": "M/s.Karnataka Agro Industries Corp.Ltd.",
                        "role": "Respondent",
                    },
                ]
            },
            "parties_involved": {
                "petitioners": [{"name": "K.Nissar Ahmed"}],
                "respondents": [{"organization": "M/s.Karnataka Agro Industries Corp.Ltd."}],
            },
        }
    )

    assert result.fields["advocates"] == ["Sri.T.Narayanaswamy, Adv."]
    assert result.fields["respondents"] == ["M/s.Karnataka Agro Industries Corp.Ltd."]


def test_vision_field_repair_targets_missing_bench_from_observed_wp16426_output():
    from judgment_workflow.vision_extraction import _vision_fields_needing_repair, result_from_vision_payload

    result = result_from_vision_payload(
        {
            "case_details": {
                "case_number": "W.P.16426/96",
                "case_type": "Civil Writ Petition",
                "court": "High Court of Karnataka at Bangalore",
                "bench": [],
                "departments": [],
                "responsible_entities": [],
                "advocates": [
                    {"name": "Smt Suman Hegde, Adv.", "role": "Petitioner"},
                    {"name": "Sri N.S. Venugopal, Adv", "role": "Respondent"},
                ],
                "disposition": None,
            },
            "date_of_order": {"value": "18th June 1998", "raw_text": "DATED THIS THE 18TH DAY OF JUNE 1998"},
            "parties_involved": {
                "petitioners": [{"name": "Sufala Devidas Rane"}],
                "respondents": [{"name": "Deputy Director, Department of Public Education, Karwar (U.K), Zilla Parishad"}],
            },
        }
    )

    assert _vision_fields_needing_repair(result) == ["bench"]


def test_vision_field_repair_prompt_is_single_field_and_schema_bound():
    from judgment_workflow.vision_extraction import _vision_field_repair_prompt

    prompt = _vision_field_repair_prompt("bench", [1, 2])

    assert "Repair exactly one missing or weak field: bench" in prompt
    assert '"field": "bench"' in prompt
    assert '"value": []' in prompt
    assert "Do not extract case_number" in prompt
    assert "BEFORE THE HON'BLE" in prompt
    assert "G.C.BHARUKA" not in prompt


def test_configured_vision_extractor_supports_minicpm_gguf(monkeypatch):
    from judgment_workflow.vision_extraction import MiniCPMGGUFVisionExtractor, get_configured_vision_extractor

    monkeypatch.setenv("JUDGMENT_VISION_FALLBACK", "1")
    monkeypatch.setenv("JUDGMENT_VISION_PROVIDER", "minicpm_gguf")
    monkeypatch.setenv("MINICPM_GGUF_CLI", "llama-minicpmv-cli")
    monkeypatch.setenv("MINICPM_GGUF_MODEL", "Model-7.6B-Q4_K_M.gguf")
    monkeypatch.setenv("MINICPM_GGUF_MMPROJ", "mmproj-model-f16.gguf")

    extractor = get_configured_vision_extractor()

    assert isinstance(extractor, MiniCPMGGUFVisionExtractor)


def test_configured_vision_extractor_supports_ollama(monkeypatch):
    from judgment_workflow.vision_extraction import OllamaVisionExtractor, get_configured_vision_extractor

    monkeypatch.setenv("JUDGMENT_VISION_FALLBACK", "1")
    monkeypatch.setenv("JUDGMENT_VISION_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_VISION_MODEL", "openbmb/minicpm-o2.6:latest")

    extractor = get_configured_vision_extractor()

    assert isinstance(extractor, OllamaVisionExtractor)
    assert extractor.model == "openbmb/minicpm-o2.6:latest"


def test_configured_vision_extractor_supports_lmstudio(monkeypatch):
    from judgment_workflow.vision_extraction import MiniCPMHTTPVisionExtractor, get_configured_vision_extractor

    monkeypatch.setenv("JUDGMENT_VISION_FALLBACK", "1")
    monkeypatch.setenv("JUDGMENT_VISION_PROVIDER", "lmstudio")
    monkeypatch.setenv("LMSTUDIO_VISION_MODEL", "minicpm-o-2_6-q4")

    extractor = get_configured_vision_extractor()

    assert isinstance(extractor, MiniCPMHTTPVisionExtractor)


def test_llm_first_workflow_uses_case_context_and_second_action_pass():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE HIGH COURT OF KARNATAKA AT BENGALURU "
                "Writ Petition No. 1234 of 2025 ABC Residents Association v. State of Karnataka and BBMP "
                "Judgment dated 15 March 2026."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(page_content="Background facts.", metadata={"source": "judgment.pdf", "page": 2, "chunk_id": "p2"}),
        Document(page_content="More submissions.", metadata={"source": "judgment.pdf", "page": 3, "chunk_id": "p3"}),
        Document(
            page_content="Earlier context says BBMP is the municipal authority for the encroachment.",
            metadata={"source": "judgment.pdf", "page": 4, "chunk_id": "p4"},
        ),
        Document(page_content="The final order begins but refers to prior context.", metadata={"source": "judgment.pdf", "page": 5, "chunk_id": "p5"}),
        Document(
            page_content=(
                "The BBMP is directed to remove the encroachment within four weeks. "
                "The Urban Development Department shall file a compliance report within 30 days."
            ),
            metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
        ),
    ]
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "extracting only high-confidence case metadata" in prompt:
            assert "first_page" in prompt
            assert "top 10 reranked" in prompt
            assert "agentic_context" in prompt
            assert "directed ordered shall within weeks days compliance report" in prompt
            return (
                '{"case_details":{"case_number":"Writ Petition No. 1234 of 2025",'
                '"case_type":"writ_petition","court":"High Court of Karnataka at Bengaluru",'
                '"bench":[]},"date_of_order":{"value":"2026-03-15","raw_text":"15 March 2026",'
                '"confidence":0.91,"evidence_snippet":"Judgment dated 15 March 2026"},'
                '"parties_involved":{"petitioners":["ABC Residents Association"],'
                '"respondents":["State of Karnataka","BBMP"],"other_parties":[],'
                '"evidence_snippet":"ABC Residents Association v. State of Karnataka and BBMP"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.9}'
            )
        if "first pass over the last 3 pages" in prompt:
            return (
                '{"context_summary":"Final pages mention BBMP removal and compliance report directions.",'
                '"needs_more_context":true,"missing_context_reason":"Need prior owner context",'
                '"no_immediate_action":false,"action_items":[]}'
            )
        assert "second and final pass" in prompt
        assert "Final pages mention BBMP removal" in prompt
        assert "previous_3_pages" in prompt
        return (
            '{"context_summary":"Action plan resolved.","needs_more_context":false,'
            '"second_pass_used":true,"no_immediate_action":false,"action_items":[{'
            '"title":"Remove the encroachment","responsible_department":"BBMP",'
            '"category":"direct_compliance","priority":"high",'
            '"timeline":{"raw_text":"within four weeks","timeline_type":"explicit","confidence":0.82},'
            '"legal_basis":"The BBMP is directed to remove the encroachment within four weeks",'
            '"decision_reason":"The final order directly requires BBMP to remove the encroachment.",'
            '"review_recommendation":"Verify owner and deadline before publishing.",'
            '"requires_human_review":true,'
            '"confidence":0.89,"ambiguity_flags":[],'             
            '"evidence_snippet":"The BBMP is directed to remove the encroachment within four weeks"}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 6},
            llm_callable=fake_llm,
        )
    )

    assert len(prompts) == 3
    assert package.source_metadata["llm_review_mode"] == "llm_first"
    assert package.source_metadata["llm_context_plan_source"] == "deterministic_default"
    assert package.source_metadata["llm_action_second_pass"] is True
    assert package.extraction.case_number.value == "Writ Petition No. 1234 of 2025"
    assert package.extraction.court.value == "High Court of Karnataka at Bengaluru"
    assert package.extraction.departments.value == []
    assert "missing_case_number" not in package.risk_flags
    assert package.action_items[0].title == "Remove the encroachment"
    assert package.action_items[0].category == "direct_compliance"
    assert package.action_items[0].decision_reason
    assert package.action_items[0].evidence[0].extraction_method == "llm_first"


def test_llm_first_workflow_does_not_copy_deterministic_department_noise():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "Civil Appeal No. 7298 of 2022 China Development Bank v. Doha Bank Q.P.S.C. "
                "Supreme Court of India. Order dated 20 December 2024."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content=(
                "The Adjudicating Authority and State of Karnataka are discussed only as cited legal context. "
                "The appeal is quashed."
            ),
            metadata={"source": "judgment.pdf", "page": 2, "chunk_id": "p2"},
        ),
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 7298 of 2022",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":[],'
                '"departments":[],"advocates":[],"disposition":"quashed",'
                '"evidence_snippet":"Civil Appeal No. 7298 of 2022 China Development Bank v. Doha Bank Q.P.S.C."},'
                '"date_of_order":{"value":"2024-12-20","raw_text":"20 December 2024",'
                '"confidence":0.9,"evidence_snippet":"Order dated 20 December 2024"},'
                '"parties_involved":{"petitioners":["China Development Bank"],'
                '"respondents":["Doha Bank Q.P.S.C."],"other_parties":[],'
                '"evidence_snippet":"China Development Bank v. Doha Bank Q.P.S.C."},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.9}'
            )
        return (
            '{"context_summary":"Appeal quashed with no operational government task.",'
            '"needs_more_context":false,"action_items":[{'
            '"title":"Review effect of quashed appeal order",'
            '"responsible_department":null,"category":"legal_review","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeal is quashed.","decision_reason":"The quashing outcome may require legal review but no direct owner is named.",'
            '"review_recommendation":"Legal reviewer should decide whether any record update or response is required.",'
            '"requires_human_review":true,"confidence":0.62,'
            '"ambiguity_flags":["owner unclear"],"evidence_snippet":"The appeal is quashed."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.departments.value == []
    assert package.extraction.departments.confidence <= 0.3
    assert package.extraction.departments.requires_review is True
    assert package.extraction.disposition.value == "quashed"
    assert package.action_items[0].category == "legal_review"
    assert package.action_items[0].title == "Review effect of quashed appeal order"
    assert "missing_action_items" not in package.risk_flags
    assert "owner_unclear" in package.risk_flags


def test_llm_first_debug_trace_diagnoses_deterministic_bench_lost():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Criminal Appeal No. 5579 of 2024\n"
                "Shambhu Debnath v. The State of Bihar & Ors.\n"
                "CORAM:\n"
                "HON'BLE MR. JUSTICE VIKRAM NATH\n"
                "HON'BLE MR. JUSTICE PRASANNA B. VARALE\n"
                "Judgment dated 20 December 2024."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content="The appeal is allowed and the matter requires legal review by the State.",
            metadata={"source": "judgment.pdf", "page": 2, "chunk_id": "p2"},
        ),
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Criminal Appeal No. 5579 of 2024",'
                '"case_type":"criminal_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["State of Bihar"],"disposition":"allowed",'
                '"evidence_snippet":"Criminal Appeal No. 5579 of 2024 Shambhu Debnath v. The State of Bihar & Ors."},'
                '"date_of_order":{"value":"2024-12-20","raw_text":"20 December 2024",'
                '"confidence":0.9,"evidence_snippet":"Judgment dated 20 December 2024"},'
                '"parties_involved":{"petitioners":["Shambhu Debnath"],'
                '"respondents":["The State of Bihar & Ors."],"other_parties":[],'
                '"evidence_snippet":"Shambhu Debnath v. The State of Bihar & Ors."},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.88}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Review effect of allowed appeal",'
            '"responsible_department":"State of Bihar","category":"legal_review","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeal is allowed","decision_reason":"The final outcome affects legal posture.",'
            '"review_recommendation":"Legal reviewer should verify next steps.",'
            '"requires_human_review":true,"confidence":0.7,"ambiguity_flags":[],'
            '"evidence_snippet":"The appeal is allowed"}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
            llm_callable=fake_llm,
        )
    )

    trace = package.source_metadata["extraction_debug"]
    assert trace["summary"]["deterministic_found_but_final_missing"] == ["bench"]
    assert trace["summary"]["llm_returned_empty_fields"] == ["bench"]
    assert trace["summary"]["context_contains_keywords"]["coram"] is True
    bench_trace = trace["field_trace"]["bench"]
    assert bench_trace["deterministic"]["value"]
    assert bench_trace["llm_output"]["value"] == []
    assert bench_trace["final"]["value"] == []
    assert bench_trace["diagnosis"] == "deterministic_found_but_llm_final_empty"


def test_llm_first_repairs_empty_bench_when_deterministic_coram_exists():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Civil Appeal No. 2245 of 2009\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "CORAM :\n"
                "HON'BLE MR. JUSTICE S.H. KAPADIA\n"
                "HON'BLE MR. JUSTICE AFTAB ALAM\n"
                "The appeal is allowed."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai"],"disposition":"allowed",'
                '"evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        if "Repair one missing case-detail field" in prompt:
            assert '"field": "bench"' in prompt
            assert "CORAM" in prompt
            return (
                '{"field":"bench","value":["Justice S.H. Kapadia","Justice Aftab Alam"],'
                '"confidence":0.94,'
                '"evidence_snippet":"CORAM : HON\'BLE MR. JUSTICE S.H. KAPADIA HON\'BLE MR. JUSTICE AFTAB ALAM",'
                '"reason":"CORAM names the judges who heard the appeal."}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Review effect of allowed appeal",'
            '"responsible_department":"Commissioner of Customs, Mumbai","category":"legal_review","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeal is allowed.","decision_reason":"The allowed appeal affects legal posture.",'
            '"review_recommendation":"Legal reviewer should verify next steps.",'
            '"requires_human_review":true,"confidence":0.72,"ambiguity_flags":["timeline unclear"],'
            '"evidence_snippet":"The appeal is allowed."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    assert any("Repair one missing case-detail field" in prompt for prompt in prompts)
    assert package.extraction.bench.value == ["Justice S.H. Kapadia", "Justice Aftab Alam"]
    assert package.extraction.bench.evidence
    assert package.extraction.bench.evidence[0].extraction_method == "llm_first"
    trace = package.source_metadata["extraction_debug"]
    assert trace["summary"]["field_repairs_attempted"] == ["bench"]
    assert trace["summary"]["field_repairs_applied"] == ["bench"]
    assert trace["field_trace"]["bench"]["diagnosis"] == "llm_value_written"


def test_llm_action_plan_infers_single_public_respondent_owner_for_review_action():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Civil Appeal No. 2245 of 2009\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "The appeals are allowed with no order as to costs."
            ),
            metadata={"source": "judgment.pdf", "page": 7, "chunk_id": "p7"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai","Director General of Health Service"],'
                '"responsible_entities":["Commissioner of Customs, Mumbai","Director General of Health Service"],'
                '"disposition":"allowed","evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],'
                '"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Update case record to reflect Supreme Court allowance of appeal",'
            '"responsible_department":null,"category":"legal_review","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeals are allowed with no order as to costs.",'
            '"decision_reason":"The allowed appeal affects the department case record.",'
            '"review_recommendation":"Legal reviewer should verify next steps.",'
            '"requires_human_review":true,"confidence":0.72,'
            '"ambiguity_flags":["owner unclear","timeline unclear"],'
            '"evidence_snippet":"The appeals are allowed with no order as to costs."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    action = package.action_items[0]
    assert action.responsible_department == "Case reviewer"
    assert action.owner_source == "system_policy"
    assert action.category == "internal_review"
    assert action.requires_human_review is True
    assert action.ambiguity_flags == []
    assert "owner_unclear" not in package.risk_flags


def test_llm_action_plan_rejects_court_staff_owner_and_uses_remand_destination():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "we set aside the impugned judgment and remit the cases to the High Court for fresh consideration.\n"
                "(Madhu Saxena) Court Master"
            ),
            metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai"],"disposition":"allowed",'
                '"evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],'
                '"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        return (
            '{"context_summary":"Remand ordered.","needs_more_context":false,'
            '"action_items":[{"title":"Remit case to High Court for fresh consideration",'
            '"responsible_department":"Registrar (Madhu Saxena) Court Master",'
            '"category":"direct_compliance","priority":"high",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"we set aside the impugned judgment and remit the cases to the High Court for fresh consideration.",'
            '"decision_reason":"The file must be remitted.",'
            '"review_recommendation":"Verify transmission to the receiving court.",'
            '"requires_human_review":true,"confidence":0.73,'
            '"ambiguity_flags":["timeline unclear"],'
            '"evidence_snippet":"we set aside the impugned judgment and remit the cases to the High Court for fresh consideration."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    action = package.action_items[0]
    assert action.responsible_department == "High Court"
    assert action.owner_source == "remand_destination"
    assert "owner_unclear" not in action.ambiguity_flags


def test_llm_action_plan_clears_owner_unclear_when_public_owner_is_present():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "The Director General of Health Service categorized the hospital. The appeals are allowed."
            ),
            metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Director General of Health Service"],"disposition":"allowed",'
                '"evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],'
                '"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Legal review of DGHS categorisation decision",'
            '"responsible_department":"Director General of Health Service",'
            '"category":"legal_review","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeals are allowed.",'
            '"decision_reason":"Departmental records may need review.",'
            '"review_recommendation":"Verify department action.",'
            '"requires_human_review":true,"confidence":0.73,'
            '"ambiguity_flags":["owner unclear","timeline unclear"],'
            '"evidence_snippet":"The appeals are allowed."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    action = package.action_items[0]
    assert action.responsible_department == "Director General of Health Service"
    assert "owner_unclear" not in action.ambiguity_flags
    assert "owner_unclear" not in package.risk_flags


def test_llm_action_plan_recovers_remand_and_separates_internal_record_update():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "On this ground alone, we set aside the impugned judgment dated October 1, 2008 "
                "in Customs Appeal Nos. 52, 53 and 55 of 2008 and remit the cases to the High Court "
                "for fresh consideration in accordance with law."
            ),
            metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai","Director General of Health Service"],'
                '"disposition":"allowed","evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],'
                '"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Update customs record to reflect appeal allowed",'
            '"responsible_department":"Commissioner of Customs, Mumbai",'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeals are allowed.",'
            '"decision_reason":"The case record should reflect the disposition.",'
            '"review_recommendation":"Verify record update.",'
            '"requires_human_review":true,"confidence":0.73,'
            '"ambiguity_flags":["timeline unclear"],'
            '"evidence_snippet":"we set aside the impugned judgment dated October 1, 2008 in Customs Appeal Nos. 52, 53 and 55 of 2008 and remit the cases to the High Court for fresh consideration in accordance with law."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    legal_actions = [item for item in package.action_items if item.category != "internal_review"]
    internal_actions = [item for item in package.action_items if item.category == "internal_review"]

    assert internal_actions[0].responsible_department == "Case reviewer"
    assert internal_actions[0].timeline.timeline_type == "not_configured"
    remand_action = next(item for item in legal_actions if "fresh consideration" in item.title.lower())
    assert remand_action.responsible_department == "High Court"
    assert remand_action.owner_source == "remand_destination"
    assert "missing_timeline" not in remand_action.ambiguity_flags
    assert "timeline_not_specified" in remand_action.ambiguity_flags
    assert remand_action.responsible_department != "Commissioner of Customs, Mumbai"


def test_llm_action_repair_uses_broader_document_context_for_owner_and_internal_actions():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "On this ground alone, we set aside the impugned judgment dated October 1, 2008 "
                "in Customs Appeal Nos. 52, 53 and 55 of 2008 and remit the cases to the High Court "
                "for fresh consideration in accordance with law."
            ),
            metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
        )
    ]
    repair_prompts = []

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai","Director General of Health Service"],'
                '"disposition":"allowed","evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],'
                '"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        if "Repair action ownership and timelines" in prompt:
            repair_prompts.append(prompt)
            assert "remit the cases to the High Court" in prompt
            return (
                '{"context_summary":"Final order remits the customs appeals to the High Court.",'
                '"action_items":['
                '{"title":"Fresh consideration of Customs Appeal Nos. 52, 53 and 55 of 2008 by High Court",'
                '"responsible_department":"High Court","category":"direct_compliance","priority":"medium",'
                '"timeline":{"raw_text":null,"timeline_type":"not_specified","confidence":0.8},'
                '"legal_basis":"The cases are remitted to the High Court for fresh consideration.",'
                '"decision_reason":"The receiving court is the owner of the legal action.",'
                '"review_recommendation":"Verify receiving court before publishing.",'
                '"requires_human_review":true,"confidence":0.86,'
                '"ambiguity_flags":["timeline_not_specified"],'
                '"evidence_snippet":"remit the cases to the High Court for fresh consideration in accordance with law."},'
                '{"title":"Update case record to reflect appeal allowed and remand",'
                '"responsible_department":"Case reviewer","category":"internal_review","priority":"medium",'
                '"timeline":{"raw_text":null,"timeline_type":"not_configured","confidence":1.0},'
                '"legal_basis":"System review task derived from final disposition.",'
                '"decision_reason":"Record update is internal workflow, not a court-directed public owner action.",'
                '"review_recommendation":"Reviewer should confirm publication state.",'
                '"requires_human_review":true,"confidence":0.9,'
                '"ambiguity_flags":[],"evidence_snippet":"The appeals are allowed with no order as to costs."}]}'
            )
        return (
            '{"context_summary":"Appeal allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Update customs record to reflect appeal allowed",'
            '"responsible_department":"Commissioner of Customs, Mumbai",'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeals are allowed.",'
            '"decision_reason":"The case record should reflect the disposition.",'
            '"review_recommendation":"Verify record update.",'
            '"requires_human_review":true,"confidence":0.73,'
            '"ambiguity_flags":["timeline unclear"],'
            '"evidence_snippet":"The appeals are allowed."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    assert repair_prompts
    legal_actions = [item for item in package.action_items if item.category != "internal_review"]
    internal_actions = [item for item in package.action_items if item.category == "internal_review"]
    assert legal_actions[0].responsible_department == "High Court"
    assert legal_actions[0].timeline.timeline_type == "not_specified"
    assert internal_actions[0].responsible_department == "Case reviewer"
    assert internal_actions[0].timeline.timeline_type == "not_configured"


def test_llm_action_plan_prunes_background_actions_when_final_order_remands():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "The notification required hospitals to produce CDEC evidence. "
                "On this ground alone, we set aside the impugned judgment and remit the cases "
                "to the High Court for fresh consideration in accordance with law."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Health Services Department"],"disposition":"allowed",'
                '"evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[]},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        return (
            '{"context_summary":"Mistakenly uses background notification.",'
            '"needs_more_context":false,'
            '"action_items":[{"title":"Consider re-categorization request",'
            '"responsible_department":"Health Services Department","category":"conditional_follow_up",'
            '"priority":"medium","timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The notification required hospitals to produce CDEC evidence.",'
            '"decision_reason":"The appeal was allowed and the cases were remitted, so background notification duties may require follow-up.",'
            '"review_recommendation":"Review background notification.",'
            '"requires_human_review":true,"confidence":0.5,'
            '"ambiguity_flags":["missing_timeline"],'
            '"evidence_snippet":"The notification required hospitals to produce CDEC evidence."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    legal_titles = [item.title for item in package.action_items if item.category != "internal_review"]
    assert any("High Court" in title for title in legal_titles)
    assert not any("re-categorization" in title for title in legal_titles)


def test_llm_package_does_not_flag_owner_unclear_when_action_owner_is_present():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "The appeals are allowed. The matters are remitted to the High Court "
                "for fresh consideration in accordance with law."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":[],"disposition":"allowed","evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[]},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        return (
            '{"context_summary":"Final order remits the matter.",'
            '"needs_more_context":false,'
            '"action_items":[{"title":"Fresh consideration by High Court",'
            '"responsible_department":"High Court","category":"direct_compliance",'
            '"priority":"medium","timeline":{"raw_text":null,"timeline_type":"not_specified","confidence":0.8},'
            '"legal_basis":"The matters are remitted to the High Court for fresh consideration.",'
            '"decision_reason":"The receiving court is the action owner.",'
            '"review_recommendation":"Verify before publishing.",'
            '"requires_human_review":true,"confidence":0.86,'
            '"ambiguity_flags":["timeline_not_specified"],'
            '"evidence_snippet":"remitted to the High Court for fresh consideration in accordance with law."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    assert package.action_items[0].responsible_department == "High Court"
    assert "owner_unclear" not in package.risk_flags


def test_llm_first_uses_deterministic_action_fallback_when_action_llm_returns_empty():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Civil Appeal No. 2245 of 2009\n"
                "The appeals are allowed. On this ground alone, we set aside the impugned judgment "
                "and remit the cases to the High Court for fresh consideration in accordance with law."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":[],"disposition":"allowed","evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[]},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        return "{}"

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    debug = package.source_metadata["extraction_debug"]
    assert debug["llm_outputs"]["deterministic_action_fallback"]["action_items"]
    assert package.action_items


def test_llm_first_repairs_empty_date_and_advocates_when_last_page_has_them():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    documents = [
        Document(
            page_content=(
                "IN THE SUPREME COURT OF INDIA\n"
                "Civil Appeal No. 2245 of 2009\n"
                "Dr. B.N. Hospital v. Commissioner of Customs, Mumbai\n"
                "The appeals are allowed with no order as to costs."
            ),
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content=(
                "RECORD OF PROCEEDINGS\n"
                "Petition(s) for Special Leave to Appeal (Civil) No(s).3554/2009\n"
                "(From the judgment and order dated 01/10/2008 in CA No. 52/2008)\n"
                "Date: 08/04/2009 This Petition was called on for hearing today.\n"
                "CORAM : HON'BLE MR. JUSTICE S.H. KAPADIA\n"
                "HON'BLE MR. JUSTICE AFTAB ALAM\n"
                "For Petitioner(s) Mr. Bharat Sangal, Adv.\n"
                "Mr. Prasenjit Das, Adv.\n"
                "Ms. Mrinalini Oinam, Adv.\n"
                "For Respondent(s) None appears."
            ),
            metadata={"source": "judgment.pdf", "page": 7, "chunk_id": "p7"},
        ),
    ]
    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 2245 of 2009",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":["Commissioner of Customs, Mumbai"],"advocates":[],"disposition":null,'
                '"evidence_snippet":"Civil Appeal No. 2245 of 2009"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":["Dr. B.N. Hospital"],'
                '"respondents":["Commissioner of Customs, Mumbai"],"other_parties":[],"evidence_snippet":"Dr. B.N. Hospital v. Commissioner of Customs, Mumbai"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "judgment_date"' in prompt:
            assert "Date: 08/04/2009" in prompt
            return (
                '{"field":"judgment_date","value":"2009-04-08","raw_text":"Date: 08/04/2009",'
                '"confidence":0.95,"evidence_snippet":"Date: 08/04/2009 This Petition was called on for hearing today.",'
                '"reason":"The proceedings date is explicitly printed before CORAM."}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "advocates"' in prompt:
            assert "For Petitioner" in prompt
            return (
                '{"field":"advocates","value":["Mr. Bharat Sangal, Adv.","Mr. Prasenjit Das, Adv.","Ms. Mrinalini Oinam, Adv."],'
                '"confidence":0.9,'
                '"evidence_snippet":"For Petitioner(s) Mr. Bharat Sangal, Adv. Mr. Prasenjit Das, Adv. Ms. Mrinalini Oinam, Adv.",'
                '"reason":"The listed names appear under For Petitioner(s)."}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "disposition"' in prompt:
            assert "appeals are allowed" in prompt.lower()
            return (
                '{"field":"disposition","value":"allowed","confidence":0.92,'
                '"evidence_snippet":"The appeals are allowed with no order as to costs.",'
                '"reason":"The operative order expressly allows the appeals."}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "bench"' in prompt:
            return (
                '{"field":"bench","value":["Justice S.H. Kapadia","Justice Aftab Alam"],'
                '"confidence":0.94,"evidence_snippet":"CORAM : HON\'BLE MR. JUSTICE S.H. KAPADIA HON\'BLE MR. JUSTICE AFTAB ALAM",'
                '"reason":"CORAM lists the bench."}'
            )
        return (
            '{"context_summary":"Appeals allowed.","needs_more_context":false,'
            '"action_items":[{"title":"Update case record to reflect appeals allowed",'
            '"responsible_department":"Commissioner of Customs, Mumbai","category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The appeals are allowed with no order as to costs.",'
            '"decision_reason":"The final order changes case status.",'
            '"review_recommendation":"Verify record update.",'
            '"requires_human_review":true,"confidence":0.76,"ambiguity_flags":["timeline unclear"],'
            '"evidence_snippet":"The appeals are allowed with no order as to costs."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            documents,
            {"source_system": "test"},
            pdf_profile={"page_count": 7},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.judgment_date.value.isoformat() == "2009-04-08"
    assert package.extraction.advocates.value == [
        "Mr. Bharat Sangal, Adv.",
        "Mr. Prasenjit Das, Adv.",
        "Ms. Mrinalini Oinam, Adv.",
    ]
    assert package.extraction.disposition.value == "allowed"
    trace = package.source_metadata["extraction_debug"]
    assert "judgment_date" in trace["summary"]["field_repairs_applied"]
    assert "advocates" in trace["summary"]["field_repairs_applied"]
    assert "disposition" in trace["summary"]["field_repairs_applied"]


def test_debug_extraction_endpoint_returns_stored_trace(monkeypatch):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    trace = {
        "summary": {"missing_final_fields": ["bench"]},
        "field_trace": {"bench": {"diagnosis": "deterministic_found_but_llm_final_empty"}},
    }

    class FakeRepository:
        def __init__(self, *args, **kwargs):
            pass

        async def get_record(self, user_id, record_id):
            assert user_id == "debug-user"
            assert record_id == "record-1"
            return {"record_id": record_id, "user_id": user_id, "extraction_debug": trace}

    monkeypatch.setattr(judgment_api, "JudgmentRepository", FakeRepository)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.get("/judgments/record-1/debug-extraction?user_id=debug-user")

    assert response.status_code == 200
    assert response.json()["extraction_debug"] == trace


def test_llm_json_parser_accepts_fenced_json():
    from judgment_workflow.llm_review_workflow import _parse_llm_json_object

    assert _parse_llm_json_object('```json\n{"ok": true}\n```') == {"ok": True}
    assert _parse_llm_json_object('Here is the JSON:\n{"ok": true}\nDone') == {"ok": True}


def test_llm_source_confidence_is_match_based_not_hardcoded():
    from judgment_workflow.llm_review_workflow import _evidence_for_snippet

    documents = [
        Document(
            page_content="The BBMP is directed to remove the encroachment within four weeks.",
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        )
    ]

    exact = _evidence_for_snippet("BBMP is directed to remove the encroachment", documents)
    fuzzy = _evidence_for_snippet("BBMP shall remove public road encroachment", documents)

    assert exact is not None
    assert fuzzy is not None
    assert exact.confidence == 0.95
    assert 0.45 <= fuzzy.confidence < exact.confidence


def test_llm_no_operational_action_is_explicit_reviewable_decision():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"SLP 1/2026","case_type":"SLP",'
                '"court":"Supreme Court of India","bench":[],"departments":[],"disposition":"dismissed",'
                '"evidence_snippet":"SLP 1/2026 Supreme Court of India"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[{"text":"The special leave petition is dismissed.",'
                '"confidence":0.91,"evidence_snippet":"The special leave petition is dismissed."}],'
                '"relevant_timelines":[],"confidence":0.9}'
            )
        return (
            '{"context_summary":"The special leave petition is dismissed.",'
            '"needs_more_context":false,"action_items":[{'
            '"title":"Record no operational government action",'
            '"responsible_department":null,"category":"no_immediate_action","priority":"low",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The special leave petition is dismissed.",'
            '"decision_reason":"Dismissal creates no direct compliance task in the supplied order.",'
            '"review_recommendation":"Verify dismissal before closing as no operational action.",'
            '"requires_human_review":true,"confidence":0.86,'
            '"ambiguity_flags":[],"evidence_snippet":"The special leave petition is dismissed."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [Document(page_content="SLP 1/2026 Supreme Court of India. The special leave petition is dismissed.", metadata={"page": 1, "chunk_id": "p1"})],
            {"source_system": "test"},
            pdf_profile={"page_count": 1},
            llm_callable=fake_llm,
        )
    )

    assert package.action_items[0].category == "no_operational_action"
    assert package.action_items[0].decision_reason
    assert package.action_items[0].requires_human_review is True
    assert "missing_action_items" not in package.risk_flags


def test_llm_first_workflow_forces_decision_when_action_pass_dodges():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 1 of 2026",'
                '"case_type":"civil_appeal","court":"Supreme Court of India","bench":[],'
                '"departments":[],"disposition":"allowed","evidence_snippet":"Civil Appeal No. 1 of 2026"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":["State"],"other_parties":[],"evidence_snippet":"State"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.8}'
            )
        if "previous action-plan attempt" in prompt:
            return (
                '{"context_summary":"Appeal allowed and impugned order set aside.",'
                '"needs_more_context":false,"forced_decision_pass":true,"action_items":[{'
                '"title":"Review effect of allowed appeal","responsible_department":"State",'
                '"category":"legal_review","priority":"medium",'
                '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
                '"legal_basis":"The appeal is allowed and the impugned order is set aside.",'
                '"decision_reason":"The allowed appeal changes the legal posture and may require record or litigation action.",'
                '"review_recommendation":"Legal reviewer should confirm whether any record update or follow-up is needed.",'
                '"requires_human_review":true,"confidence":0.72,"ambiguity_flags":["timeline unclear"],'
                '"evidence_snippet":"The appeal is allowed and the impugned order is set aside."}]}'
            )
        return '{"context_summary":"Need more final-order context.","needs_more_context":true,"action_items":[]}'

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(page_content="Civil Appeal No. 1 of 2026 State matter.", metadata={"page": 1, "chunk_id": "p1"}),
                Document(
                    page_content="The appeal is allowed and the impugned order is set aside.",
                    metadata={"page": 2, "chunk_id": "p2"},
                ),
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
            llm_callable=fake_llm,
        )
    )

    assert any("previous action-plan attempt" in prompt for prompt in prompts)
    assert package.source_metadata["llm_action_forced_pass"] is True
    assert package.action_items[0].title == "Review effect of allowed appeal"
    assert package.action_items[0].decision_reason


def test_llm_first_repairs_authoring_judge_to_signature_bench():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    prompts = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal Nos. 3778-3780 of 2016",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":["Justice Banumathi"],'
                '"departments":["Board of Revenue, Bihar"],"advocates":[],"disposition":"allowed",'
                '"evidence_snippet":"J U D G M E N T R. BANUMATHI, J."},'
                '"date_of_order":{"value":"2016-04-12","raw_text":"April 12, 2016","confidence":0.9,'
                '"evidence_snippet":"New Delhi; April 12, 2016"},'
                '"parties_involved":{"petitioners":["Kedar Mishra"],"respondents":["State of Bihar"],'
                '"other_parties":[],"evidence_snippet":"KEDAR MISHRA VERSUS THE STATE OF BIHAR"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "bench"' in prompt:
            assert "T.S. THAKUR" in prompt
            assert "UDAY UMESH LALIT" in prompt
            return (
                '{"field":"bench","value":["Chief Justice T.S. Thakur","Justice R. Banumathi",'
                '"Justice Uday Umesh Lalit"],"confidence":0.93,'
                '"evidence_snippet":"CJI. (T.S. THAKUR) J. (R. BANUMATHI) J. (UDAY UMESH LALIT)",'
                '"reason":"The signature block lists the judges on the bench."}'
            )
        return (
            '{"context_summary":"Appeals allowed and matter remitted.",'
            '"needs_more_context":false,"action_items":[{"title":"Reconsider matter by Board of Revenue",'
            '"responsible_department":"Board of Revenue, Bihar","category":"conditional_follow_up",'
            '"priority":"medium","timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The matter is remitted back to the Board of Revenue, Bihar to reconsider.",'
            '"decision_reason":"The final order remits the matter.","review_recommendation":"Verify follow-up.",'
            '"requires_human_review":true,"confidence":0.78,"ambiguity_flags":["missing_timeline"],'
            '"evidence_snippet":"The matter is remitted back to the Board of Revenue, Bihar to reconsider."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "IN THE SUPREME COURT OF INDIA\n"
                        "CIVIL APPEAL NOS. 3778-3780 OF 2016\n"
                        "J U D G M E N T\n"
                        "R. BANUMATHI, J.\n"
                    ),
                    metadata={"page": 1, "chunk_id": "p1"},
                ),
                Document(
                    page_content=(
                        "The matter is remitted back to the Board of Revenue, Bihar to reconsider.\n"
                        "The appeals are accordingly allowed.\n"
                        "The parties to bear their respective costs.\n"
                        "...................CJI. (T.S. THAKUR)\n"
                        "......................J. (R. BANUMATHI)\n"
                        ".....................J. (UDAY UMESH LALIT)\n"
                        "New Delhi; April 12, 2016"
                    ),
                    metadata={"page": 11, "chunk_id": "p11"},
                ),
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 11},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.bench.value == [
        "Chief Justice T.S. Thakur",
        "Justice R. Banumathi",
        "Justice Uday Umesh Lalit",
    ]
    trace = package.source_metadata["extraction_debug"]
    assert "bench" in trace["summary"]["field_repairs_attempted"]
    assert "bench" in trace["summary"]["field_repairs_applied"]


def test_llm_action_quality_removes_cost_notice_when_substantive_actions_exist():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal Nos. 3778-3780 of 2016",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":[],"departments":["Board of Revenue, Bihar"],'
                '"advocates":[],"disposition":"allowed","evidence_snippet":"Civil Appeal Nos. 3778-3780 of 2016"},'
                '"date_of_order":{"value":"2016-04-12","raw_text":"April 12, 2016","confidence":0.9,'
                '"evidence_snippet":"New Delhi; April 12, 2016"},'
                '"parties_involved":{"petitioners":["Kedar Mishra"],"respondents":["State of Bihar"],'
                '"other_parties":[],"evidence_snippet":"KEDAR MISHRA VERSUS THE STATE OF BIHAR"},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.86}'
            )
        return (
            '{"context_summary":"Appeals allowed, remand ordered, costs left to parties.",'
            '"needs_more_context":false,"action_items":['
            '{"title":"Reconsider matter by Board of Revenue","responsible_department":"Board of Revenue, Bihar",'
            '"category":"conditional_follow_up","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The matter is remitted back to the Board of Revenue, Bihar to reconsider.",'
            '"decision_reason":"The final order remits the matter.","review_recommendation":"Verify follow-up.",'
            '"requires_human_review":true,"confidence":0.78,"ambiguity_flags":["missing_timeline"],'
            '"evidence_snippet":"The matter is remitted back to the Board of Revenue, Bihar to reconsider."},'
            '{"title":"Notify parties of cost order","responsible_department":null,"category":"compliance",'
            '"priority":"medium","timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"The parties to bear their respective costs.",'
            '"decision_reason":"The Court ordered each party to bear its own costs.",'
            '"review_recommendation":"Draft and send cost notices.","requires_human_review":true,'
            '"confidence":0.8,"ambiguity_flags":["owner_unclear","timeline_missing"],'
            '"evidence_snippet":"The parties to bear their respective costs."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "The matter is remitted back to the Board of Revenue, Bihar to reconsider.\n"
                        "The appeals are accordingly allowed.\n"
                        "The parties to bear their respective costs.\n"
                        "New Delhi; April 12, 2016"
                    ),
                    metadata={"page": 11, "chunk_id": "p11"},
                )
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 11},
            llm_callable=fake_llm,
        )
    )

    assert [item.title for item in package.action_items] == ["Reconsider matter by Board of Revenue"]
    trace = package.source_metadata["extraction_debug"]
    assert trace["action_trace"]["removed_cost_only_actions"] == ["Notify parties of cost order"]


def test_llm_action_quality_rewrites_sole_cost_clause_to_no_operational_action():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 1 of 2026",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":[],"departments":[],'
                '"advocates":[],"disposition":"disposed","evidence_snippet":"Civil Appeal No. 1 of 2026"},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.8}'
            )
        return (
            '{"context_summary":"Only costs are addressed.","needs_more_context":false,"action_items":[{'
            '"title":"Notify parties of cost order","responsible_department":null,"category":"direct_compliance",'
            '"priority":"medium","timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"No order as to costs.","decision_reason":"The order addresses costs.",'
            '"review_recommendation":"Notify parties.","requires_human_review":true,"confidence":0.72,'
            '"ambiguity_flags":["owner_unclear"],"evidence_snippet":"No order as to costs."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [Document(page_content="No order as to costs.", metadata={"page": 2, "chunk_id": "p2"})],
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
            llm_callable=fake_llm,
        )
    )

    assert package.action_items[0].title == "Record no cost recovery required"
    assert package.action_items[0].category == "no_operational_action"
    assert package.action_items[0].priority == "low"
    assert package.action_items[0].requires_human_review is True
    trace = package.source_metadata["extraction_debug"]
    assert trace["action_trace"]["rewritten_cost_only_actions"] == ["Notify parties of cost order"]


def test_llm_first_repairs_authoring_judge_to_bracket_signature_bench():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 1410 of 2007",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":["Dipak Misra, J."],'
                '"departments":[],"advocates":[],"disposition":"allowed",'
                '"evidence_snippet":"J U D G M E N T Dipak Misra, J."},'
                '"date_of_order":{"value":"2016-03-29","raw_text":"March 29, 2016","confidence":0.9,'
                '"evidence_snippet":"New Delhi. March 29, 2016."},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.84}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "bench"' in prompt:
            assert "[Dipak Misra]" in prompt
            assert "[Shiva Kirti Singh]" in prompt
            return (
                '{"field":"bench","value":["Justice Dipak Misra","Justice Shiva Kirti Singh"],'
                '"confidence":0.92,"evidence_snippet":"J. [Dipak Misra] J. [Shiva Kirti Singh]",'
                '"reason":"The signature block names both judges."}'
            )
        return (
            '{"context_summary":"Appeals allowed.","needs_more_context":false,"action_items":[{'
            '"title":"Update records for allowed appeal","responsible_department":null,'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"We allow the appeals.","decision_reason":"Appeal outcome changed.",'
            '"review_recommendation":"Verify record update.","requires_human_review":true,'
            '"confidence":0.72,"ambiguity_flags":["owner_unclear"],'
            '"evidence_snippet":"We allow the appeals."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "J U D G M E N T\n"
                        "Dipak Misra, J.\n"
                        "We allow the appeals.\n"
                        "...............................J.\n"
                        "[Dipak Misra]\n"
                        "...............................J.\n"
                        "[Shiva Kirti Singh]\n"
                    ),
                    metadata={"page": 28, "chunk_id": "p28"},
                )
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 28},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.bench.value == ["Justice Dipak Misra", "Justice Shiva Kirti Singh"]
    assert "bench" in package.source_metadata["extraction_debug"]["summary"]["field_repairs_applied"]


def test_llm_first_repairs_body_text_counsel_and_conflicting_disposition():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Civil Appeal No. 1410 of 2007",'
                '"case_type":"Civil Appeal","court":"Supreme Court of India","bench":[],'
                '"departments":[],"advocates":[],"disposition":"dismissed",'
                '"evidence_snippet":"The tribunal dismissed the appeal."},'
                '"date_of_order":{"value":null,"raw_text":null,"confidence":0.0,"evidence_snippet":null},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.8}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "advocates"' in prompt:
            assert "Mr. Balbir Singh" in prompt
            assert "Mr. Sanjay Kumar Visen" in prompt
            return (
                '{"field":"advocates","value":["Mr. Balbir Singh, learned senior counsel for appellant",'
                '"Mr. Sanjay Kumar Visen, learned counsel for respondent"],"confidence":0.72,'
                '"evidence_snippet":"Mr. Balbir Singh, learned senior counsel appearing for the appellant. '
                'Mr. Sanjay Kumar Visen, learned counsel for the respective respondent(s)",'
                '"reason":"Counsel names and roles are present in body text."}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "disposition"' in prompt:
            assert "we allow the appeals" in prompt.lower()
            return (
                '{"field":"disposition","value":"allowed","confidence":0.91,'
                '"evidence_snippet":"we allow the appeals and set aside all the impugned orders",'
                '"reason":"The final operative order allows the appeals."}'
            )
        return (
            '{"context_summary":"Appeals allowed.","needs_more_context":false,"action_items":[{'
            '"title":"Update records for allowed appeal","responsible_department":null,'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"we allow the appeals","decision_reason":"Final outcome is allowed.",'
            '"review_recommendation":"Verify record update.","requires_human_review":true,'
            '"confidence":0.72,"ambiguity_flags":["owner_unclear"],'
            '"evidence_snippet":"we allow the appeals"}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "The tribunal dismissed the appeal. "
                        "Mr. Balbir Singh, learned senior counsel appearing for the appellant, submitted the case."
                    ),
                    metadata={"page": 10, "chunk_id": "p10"},
                ),
                Document(
                    page_content="11. Mr. Sanjay Kumar Visen, learned counsel for the",
                    metadata={"page": 12, "chunk_id": "p12"},
                ),
                Document(
                    page_content=(
                        "respective respondent(s), per contra, supported the order. "
                        "In view of aforesaid analysis, we allow the appeals and set aside all the impugned orders."
                    ),
                    metadata={"page": 13, "chunk_id": "p13"},
                ),
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 13},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.advocates.value == [
        "Mr. Balbir Singh, learned senior counsel for appellant",
        "Mr. Sanjay Kumar Visen, learned counsel for respondent",
    ]
    assert package.extraction.advocates.requires_review is True
    assert package.extraction.disposition.value == "allowed"
    trace = package.source_metadata["extraction_debug"]["summary"]
    assert "advocates" in trace["field_repairs_applied"]
    assert "disposition" in trace["field_repairs_applied"]


def test_llm_repair_does_not_apply_prior_history_dismissal_over_grant_leave():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Petition for Special Leave to Appeal (C) No. 19898 of 2014",'
                '"case_type":"SLP","court":"Supreme Court of India","bench":["Madan B. Lokur, J."],'
                '"departments":[],"advocates":[],"disposition":"dismissed",'
                '"evidence_snippet":"it is liable to be dismissed"},'
                '"date_of_order":{"value":"2016-03-28","raw_text":"March 28, 2016","confidence":0.9,'
                '"evidence_snippet":"March 28, 2016"},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "disposition"' in prompt:
            assert "grant leave" in prompt.lower()
            return (
                '{"field":"disposition","value":"leave_granted","confidence":0.91,'
                '"evidence_snippet":"Accordingly, we grant leave and tag this appeal",'
                '"reason":"The final operative order grants leave and tags the appeal."}'
            )
        return (
            '{"context_summary":"Leave granted and appeal tagged; prior High Court dismissal discussed.",'
            '"needs_more_context":false,"action_items":['
            '{"title":"Update case status to dismissed in court registry","responsible_department":null,'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"it is liable to be dismissed",'
            '"decision_reason":"The writ petition was liable to be dismissed.",'
            '"review_recommendation":"Update registry status.","requires_human_review":true,'
            '"confidence":0.8,"ambiguity_flags":["owner_unclear"],'
            '"evidence_snippet":"it is liable to be dismissed"},'
            '{"title":"Tag appeal with CA No. 7295 of 2012 and CA No. 11895 of 2014",'
            '"responsible_department":null,"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"Accordingly, we grant leave and tag this appeal with C.A.No.7295 of 2012 and C.A.No.11895 of 2014.",'
            '"decision_reason":"The final order tags this appeal with connected civil appeals.",'
            '"review_recommendation":"Verify connected appeal tagging.","requires_human_review":true,'
            '"confidence":0.84,"ambiguity_flags":["owner_unclear"],'
            '"evidence_snippet":"Accordingly, we grant leave and tag this appeal with C.A.No.7295 of 2012 and C.A.No.11895 of 2014."}'
            ']}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "The High Court held that we do not find any merit in this writ petition "
                        "and it is liable to be dismissed."
                    ),
                    metadata={"page": 6, "chunk_id": "p6"},
                ),
                Document(
                    page_content=(
                        "Accordingly, we grant leave and tag this appeal with C.A.No.7295 of 2012 "
                        "and C.A.No.11895 of 2014.\n"
                        ".……………………..J\n"
                        "(Madan B. Lokur)\n"
                        "..……………………J\n"
                        "March 28, 2016\n"
                        "(S. A. Bobde)"
                    ),
                    metadata={"page": 8, "chunk_id": "p8"},
                ),
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 8},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.disposition.value == "leave_granted"
    assert [item.title for item in package.action_items] == [
        "Tag appeal with CA No. 7295 of 2012 and CA No. 11895 of 2014"
    ]
    trace = package.source_metadata["extraction_debug"]
    assert "disposition" in trace["summary"]["field_repairs_applied"]
    assert trace["action_trace"]["removed_stale_dismissal_actions"] == [
        "Update case status to dismissed in court registry"
    ]


def test_llm_bench_repair_keeps_deterministic_multi_signature_when_repair_is_partial():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    def fake_llm(prompt: str) -> str:
        if "extracting only high-confidence case metadata" in prompt:
            return (
                '{"case_details":{"case_number":"Petition for Special Leave to Appeal (C) No. 19898 of 2014",'
                '"case_type":"SLP","court":"Supreme Court of India","bench":["Justice Madan B. Lokur"],'
                '"departments":[],"advocates":[],"disposition":"leave_granted",'
                '"evidence_snippet":"Madan B. Lokur, J."},'
                '"date_of_order":{"value":"2016-03-28","raw_text":"March 28, 2016","confidence":0.9,'
                '"evidence_snippet":"March 28, 2016"},'
                '"parties_involved":{"petitioners":[],"respondents":[],"other_parties":[],"evidence_snippet":null},'
                '"key_directions_orders":[],"relevant_timelines":[],"confidence":0.82}'
            )
        if "Repair one missing case-detail field" in prompt and '"field": "bench"' in prompt:
            assert "S. A. Bobde" in prompt
            return (
                '{"field":"bench","value":["Justice Madan B. Lokur"],"confidence":0.91,'
                '"evidence_snippet":"Madan B. Lokur, J.",'
                '"reason":"The authoring judge line names Justice Lokur."}'
            )
        return (
            '{"context_summary":"Leave granted and appeal tagged.","needs_more_context":false,"action_items":[{'
            '"title":"Tag appeal with connected civil appeals","responsible_department":null,'
            '"category":"record_update","priority":"medium",'
            '"timeline":{"raw_text":null,"timeline_type":"missing","confidence":0.0},'
            '"legal_basis":"Accordingly, we grant leave and tag this appeal.",'
            '"decision_reason":"The final order tags the appeal.","review_recommendation":"Verify tagging.",'
            '"requires_human_review":true,"confidence":0.8,"ambiguity_flags":["owner_unclear"],'
            '"evidence_snippet":"Accordingly, we grant leave and tag this appeal."}]}'
        )

    package = asyncio.run(
        build_llm_first_review_package(
            [
                Document(
                    page_content=(
                        "Accordingly, we grant leave and tag this appeal.\n"
                        ".……………………..J\n"
                        "(Madan B. Lokur)\n"
                        "..……………………J\n"
                        "March 28, 2016\n"
                        "(S. A. Bobde)"
                    ),
                    metadata={"page": 8, "chunk_id": "p8"},
                )
            ],
            {"source_system": "test"},
            pdf_profile={"page_count": 8},
            llm_callable=fake_llm,
        )
    )

    assert package.extraction.bench.value == [
        "Justice Madan B. Lokur",
        "Justice S. A. Bobde",
    ]


def test_llm_first_workflow_uses_reviewable_fallback_when_llm_fails():
    from judgment_workflow.llm_review_workflow import build_llm_first_review_package

    async def run():
        package = await build_llm_first_review_package(
            _sample_documents(),
            {"source_system": "test"},
            pdf_profile={"page_count": 2},
            llm_callable=lambda prompt: (_ for _ in ()).throw(ConnectionError("blocked")),
        )
        assert package.action_items
        assert package.source_metadata["extraction_debug"]["llm_outputs"]["deterministic_action_fallback"]["action_items"]

    asyncio.run(run())


def test_default_llm_model_routes_to_groq_provider():
    from llm_router import get_active_provider
    from rag.config import DEFAULT_LLM_MODEL

    assert DEFAULT_LLM_MODEL == "openai/gpt-oss-120b"
    assert get_active_provider(model=DEFAULT_LLM_MODEL) == "groq"


def test_ollama_provider_identity_is_local():
    from llm_router import get_active_provider, get_request_identity

    assert get_active_provider(provider="ollama") == "ollama"
    identity = get_request_identity(model="llama3.1:8b", provider="ollama")
    assert identity["provider"] == "ollama"
    assert identity["base_url"].startswith("http://127.0.0.1:")


def test_ollama_chat_call_uses_local_api(monkeypatch):
    from llm_router import call_ollama

    calls = []

    class FakeResponse:
        status_code = 200
        text = '{"message":{"content":"ok"}}'

        def json(self):
            return {"message": {"content": "ok"}}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr("llm_router._post", fake_post)

    result = call_ollama(
        messages=[{"role": "user", "content": "Return JSON"}],
        model="llama3.1:8b",
        temperature=0.1,
        max_tokens=256,
    )

    assert result == "ok"
    assert calls[0][0] == "http://127.0.0.1:11434/api/chat"
    assert calls[0][1]["json"]["stream"] is False


def test_highlight_positions_use_bbox_when_present(workspace_tmp_path: Path):
    from judgment_workflow.pdf_highlights import build_highlight_positions

    package = build_judgment_review_package(_sample_documents())
    positions = build_highlight_positions(package)

    assert positions
    assert positions[0]["page"] == 1 or positions[0]["page"] == 8
    assert any(position.get("bbox") for position in positions)
    assert all(position.get("label") for position in positions)
    assert all(position.get("source_label") for position in positions)


def test_highlight_enrichment_adds_search_quads_and_strategy(workspace_tmp_path: Path):
    from judgment_workflow.pdf_highlights import enrich_highlight_positions

    pdf_path = workspace_tmp_path / "source.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The BBMP is directed to remove the encroachment within four weeks.")
    doc.save(pdf_path)
    doc.close()

    positions = enrich_highlight_positions(
        str(pdf_path),
        [{"page": 1, "text": "BBMP is directed to remove the encroachment", "type": "direction"}],
    )

    assert positions[0]["bbox"]
    assert positions[0]["quad_points"]
    assert positions[0]["match_strategy"] == "pymupdf_quad_search"
    assert positions[0]["locator_confidence"] >= 0.8


def test_highlight_enrichment_falls_back_to_fuzzy_word_windows(workspace_tmp_path: Path):
    from judgment_workflow.pdf_highlights import enrich_highlight_positions

    pdf_path = workspace_tmp_path / "source-fuzzy.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Respondent nos. 2 to 4 shall surrender before the Trial Court within four weeks.")
    doc.save(pdf_path)
    doc.close()

    positions = enrich_highlight_positions(
        str(pdf_path),
        [{"page": 1, "text": "respondents two to four surrender before trial court within 4 weeks", "type": "action_item"}],
    )

    assert positions[0]["bbox"]
    assert positions[0]["quad_points"]
    assert positions[0]["match_strategy"] == "word_window_fuzzy"
    assert positions[0]["locator_confidence"] >= 0.58


def test_render_highlighted_page_png_returns_visible_preview(workspace_tmp_path: Path):
    from judgment_workflow.pdf_highlights import generate_highlighted_pdf, render_highlighted_page_png

    source_pdf = workspace_tmp_path / "source-preview.pdf"
    highlighted_pdf = workspace_tmp_path / "highlighted-preview.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The compliance report shall be filed within 30 days.")
    doc.save(source_pdf)
    doc.close()

    generate_highlighted_pdf(
        str(source_pdf),
        str(highlighted_pdf),
        {"extraction": {"directions": [{"evidence": [{"page": 1, "snippet": "compliance report shall be filed"}]}]}, "action_items": []},
    )

    png = render_highlighted_page_png(str(highlighted_pdf), 1)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000


def test_generate_highlighted_pdf_is_standalone(workspace_tmp_path: Path):
    from judgment_workflow.pdf_highlights import generate_highlighted_pdf

    source_pdf = workspace_tmp_path / "source.pdf"
    highlighted_pdf = workspace_tmp_path / "highlighted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The BBMP is directed to remove the encroachment within four weeks.")
    doc.save(source_pdf)
    doc.close()

    result = generate_highlighted_pdf(
        str(source_pdf),
        str(highlighted_pdf),
        {
            "extraction": {
                "directions": [
                    {
                        "evidence": [
                            {
                                "page": 1,
                                "snippet": "BBMP is directed to remove the encroachment",
                                "extraction_method": "deterministic",
                            }
                        ]
                    }
                ]
            },
            "action_items": [],
        },
    )

    assert result == str(highlighted_pdf)
    assert highlighted_pdf.exists()
    highlighted_doc = fitz.open(highlighted_pdf)
    try:
        annotations = list(highlighted_doc.load_page(0).annots() or [])
        assert annotations
        contents = "\n".join(annotation.info.get("content", "") for annotation in annotations)
        assert "Direction" in contents
        assert "p. 1" in contents
        assert "deterministic" in contents
    finally:
        highlighted_doc.close()


def test_highlighted_page_endpoint_generates_deferred_pdf(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    source_pdf = workspace_tmp_path / "source-lazy.pdf"
    highlighted_pdf = workspace_tmp_path / "records" / "record-lazy" / "highlighted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The BBMP is directed to remove the encroachment within four weeks.")
    doc.save(source_pdf)
    doc.close()

    record = {
        "record_id": "record-lazy",
        "user_id": "reviewer",
        "original_pdf_path": str(source_pdf),
        "highlighted_pdf_path": str(highlighted_pdf),
        "pdf_profile": {"page_count": 1},
        "extraction": {
            "directions": [
                {
                    "evidence": [
                        {
                            "page": 1,
                            "snippet": "BBMP is directed to remove the encroachment",
                            "extraction_method": "deterministic",
                        }
                    ]
                }
            ]
        },
        "action_items": [],
    }
    metadata_updates = []

    class FakeRepository:
        def __init__(self, *args, **kwargs):
            pass

        async def get_record(self, user_id, record_id):
            assert user_id == "reviewer"
            assert record_id == "record-lazy"
            return record

        async def update_record_metadata(self, user_id, record_id, **kwargs):
            metadata_updates.append(kwargs)
            record.update(kwargs)

    monkeypatch.setattr(judgment_api, "JudgmentRepository", FakeRepository)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.get("/judgments/record-lazy/highlighted-page/1?user_id=reviewer")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")
    assert highlighted_pdf.exists()
    assert metadata_updates == [{"highlighted_pdf_path": str(highlighted_pdf)}]


def test_process_judgment_file_defers_highlight_generation(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import pipeline as judgment_pipeline
    from judgment_workflow.pipeline import process_judgment_file

    source_pdf = workspace_tmp_path / "source-deferred.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "IN THE HIGH COURT OF KARNATAKA AT BENGALURU. "
        "Writ Petition No. 1234 of 2025. "
        "The BBMP is directed to remove the encroachment within four weeks.",
    )
    doc.save(source_pdf)
    doc.close()

    storage = get_storage_backend(
        StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "deferred.db"))
    )
    asyncio.run(storage.initialize())

    monkeypatch.setattr(judgment_pipeline, "JUDGMENT_DATA_ROOT", workspace_tmp_path / "judgments")

    try:
        record = asyncio.run(
            process_judgment_file(
                user_id="user-1",
                pdf_path=str(source_pdf),
                record_id="record-deferred",
                original_file_name=source_pdf.name,
                source_metadata={"source_system": "test"},
                storage=storage,
                canvas_app_id="theme11-local",
                llm_enabled=False,
                processing_mode="test",
            )
        )
    finally:
        asyncio.run(storage.close())

    highlighted_path = Path(record["highlighted_pdf_path"])
    assert highlighted_path.name == "highlighted.pdf"
    assert not highlighted_path.exists()
    assert record["source_metadata"]["highlight_generation_mode"] == "deferred"


def test_process_judgment_file_reuses_ocr_detection_for_profile(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import document_profile, pipeline as judgment_pipeline
    from judgment_workflow.pipeline import process_judgment_file

    calls = {"detect": 0}

    def fake_detect_ocr_need(pdf_path):
        calls["detect"] += 1
        return {
            "needs_ocr": False,
            "page_count": 2,
            "sparse_pages": [],
            "total_text_chars": 180,
            "ocr_available": False,
        }

    storage = get_storage_backend(
        StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "single-profile-pass.db"))
    )
    asyncio.run(storage.initialize())

    monkeypatch.setattr(judgment_pipeline, "JUDGMENT_DATA_ROOT", workspace_tmp_path / "judgments")
    monkeypatch.setattr(judgment_pipeline, "compute_document_hash", lambda pdf_path: "hash-single-pass")
    monkeypatch.setattr(judgment_pipeline, "extract_layered_pdf_documents", lambda pdf_path: _sample_documents())
    monkeypatch.setattr(judgment_pipeline, "detect_ocr_need", fake_detect_ocr_need)
    monkeypatch.setattr(document_profile, "detect_ocr_need", fake_detect_ocr_need)

    try:
        record = asyncio.run(
            process_judgment_file(
                user_id="user-1",
                pdf_path=str(workspace_tmp_path / "source.pdf"),
                record_id="record-single-profile-pass",
                original_file_name="source.pdf",
                source_metadata={"source_system": "test"},
                storage=storage,
                canvas_app_id="theme11-local",
                llm_enabled=False,
                processing_mode="test",
            )
        )
    finally:
        asyncio.run(storage.close())

    assert calls["detect"] == 1
    assert record["pdf_profile"]["profile_type"] == "digital"
    assert record["pdf_profile"]["total_text_chars"] == 180


def test_detect_ocr_need_flags_corrupted_embedded_text(workspace_tmp_path: Path):
    from judgment_workflow.ocr import detect_ocr_need

    pdf_path = workspace_tmp_path / "corrupted-text-layer.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "0 IN tat n0R00PS CF KLNa2dAAtbA:i3L.CLL "
        "wP.no lqj qj1997 c/w :: ; ;: ee ee "
        "By thls Ccurt ar the respondents 1 and 2",
    )
    doc.save(pdf_path)
    doc.close()

    result = detect_ocr_need(str(pdf_path), min_text_chars_per_page=20)

    assert result["needs_ocr"] is True
    assert result["text_layer_reliable"] is False
    assert result["unreliable_text_pages"] == [1]


def test_process_judgment_file_uses_minicpm_vision_ocr_only_for_corrupted_text_layer(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import pipeline as judgment_pipeline
    from judgment_workflow.pipeline import process_judgment_file
    from judgment_workflow.vision_extraction import VisionExtractionResult

    def fake_detect_ocr_need(pdf_path):
        return {
            "needs_ocr": True,
            "page_count": 1,
            "sparse_pages": [],
            "unreliable_text_pages": [1],
            "total_text_chars": 140,
            "text_layer_reliable": False,
            "ocr_available": True,
        }

    class FakeMiniCPMExtractor:
        model = "openbmb/minicpm-o2.6:latest"

        async def extract(self, *, pdf_path, pages, deterministic_summary):
            assert pages == [1]
            return VisionExtractionResult(
                fields={
                    "case_number": "Writ Petition No. 1234 of 2025",
                    "court": "High Court of Karnataka at Bengaluru",
                },
                directions=["The BBMP is directed to remove the encroachment within four weeks."],
                evidence_pages={"case_number": 1, "court": 1, "directions": 1},
                raw_json={"provider": "fake-minicpm"},
                provider="ollama_minicpm",
            )

    bad_text_documents = [
        Document(
            page_content="0 IN tat n0R00PS CF KLNa2dAAtbA:i3L.CLL wP.no lqj qj1997",
            metadata={"source": "source.pdf", "page": 1, "chunk_id": "bad-p1"},
        )
    ]

    storage = get_storage_backend(
        StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "ocr-only-routing.db"))
    )
    asyncio.run(storage.initialize())

    monkeypatch.setattr(judgment_pipeline, "JUDGMENT_DATA_ROOT", workspace_tmp_path / "judgments")
    monkeypatch.setattr(judgment_pipeline, "compute_document_hash", lambda pdf_path: "hash-ocr-only-routing")
    monkeypatch.setattr(judgment_pipeline, "detect_ocr_need", fake_detect_ocr_need)
    monkeypatch.setattr(judgment_pipeline, "extract_layered_pdf_documents", lambda pdf_path: bad_text_documents)
    monkeypatch.setattr(judgment_pipeline, "ocr_pdf_with_tesseract", lambda pdf_path, target_pages=None: pytest.fail("Tesseract OCR should not run for corrupted text-layer PDFs"))
    monkeypatch.setattr(judgment_pipeline, "get_configured_vision_extractor", lambda **kwargs: FakeMiniCPMExtractor())

    try:
        record = asyncio.run(
            process_judgment_file(
                user_id="user-1",
                pdf_path=str(workspace_tmp_path / "source.pdf"),
                record_id="record-ocr-only-routing",
                original_file_name="source.pdf",
                source_metadata={"source_system": "test"},
                storage=storage,
                canvas_app_id="theme11-local",
                llm_enabled=False,
                processing_mode="test",
            )
        )
    finally:
        asyncio.run(storage.close())

    assert record["source_metadata"]["text_layer_rejected"] is True
    assert record["source_metadata"]["ocr_routing"] == "vision_ocr_only"
    assert record["source_metadata"]["vision_ocr_model"] == "openbmb/minicpm-o2.6:latest"
    assert record["pdf_profile"]["profile_type"] == "ocr_required"
    assert record["extraction"]["case_number"]["value"] == "Writ Petition No. 1234 of 2025"
    assert record["processing_metrics"]["extraction_methods"] == {"vision_ocr": 1}
    assert "ocr_review_required" not in record["risk_flags"]
    assert "ocr_unavailable" not in record["risk_flags"]


def test_process_judgment_file_passes_vision_ocr_context_to_llm_for_corrupted_text(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import pipeline as judgment_pipeline
    from judgment_workflow.pipeline import process_judgment_file
    from judgment_workflow.vision_extraction import VisionExtractionResult

    def fake_detect_ocr_need(pdf_path):
        return {
            "needs_ocr": True,
            "page_count": 2,
            "sparse_pages": [],
            "unreliable_text_pages": [1, 2],
            "total_text_chars": 240,
            "text_layer_reliable": False,
            "ocr_available": False,
        }

    class FakeMiniCPMExtractor:
        model = "openbmb/minicpm-o2.6:latest"

        async def extract(self, *, pdf_path, pages, deterministic_summary):
            assert pages == [1, 2]
            return VisionExtractionResult(
                fields={
                    "case_number": "CIVIL REVISION PETITION NO.2982/1997",
                    "case_type": "Civil Revision Petition",
                    "court": "High Court of Karnataka at Bangalore",
                    "bench": ["Justice Y. Bhaskar Rao"],
                    "judgment_date": "16TH DAY OF JANUARY, 1998",
                    "advocates": ["Sri.T.Narayanaswamy, Adv."],
                },
                directions=["The petition is dismissed."],
                evidence_pages={"case_number": 1, "court": 1, "directions": 2},
                raw_json={
                    "case_details": {
                        "case_number": "CIVIL REVISION PETITION NO.2982/1997",
                        "case_type": "Civil Revision Petition",
                        "court": "High Court of Karnataka at Bangalore",
                    },
                    "date_of_order": {"raw_text": "DATED THE 16TH DAY OF JANUARY, 1998"},
                    "key_directions_orders": [{"text": "The petition is dismissed.", "source_page": 2}],
                },
                provider="ollama_minicpm",
            )

    bad_text_documents = [
        Document(
            page_content="0 IN tat n0R00PS CF KLNa2dAAtbA:i3L.CLL wP.no lqj qj1997",
            metadata={"source": "source.pdf", "page": 1, "chunk_id": "bad-p1"},
        )
    ]
    llm_seen = {}

    async def fake_build_llm_first_review_package(documents, case_metadata, *, pdf_profile=None, **kwargs):
        llm_seen["texts"] = [doc.page_content for doc in documents]
        llm_seen["metadata"] = [dict(doc.metadata or {}) for doc in documents]
        assert any('"case_details"' in text and "CIVIL REVISION PETITION NO.2982/1997" in text for text in llm_seen["texts"])
        assert not any("KLNa2dAAtbA" in text for text in llm_seen["texts"])
        return build_judgment_review_package(documents, case_metadata)

    storage = get_storage_backend(
        StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "vision-to-llm.db"))
    )
    asyncio.run(storage.initialize())

    monkeypatch.setattr(judgment_pipeline, "JUDGMENT_DATA_ROOT", workspace_tmp_path / "judgments")
    monkeypatch.setattr(judgment_pipeline, "compute_document_hash", lambda pdf_path: "hash-vision-to-llm")
    monkeypatch.setattr(judgment_pipeline, "detect_ocr_need", fake_detect_ocr_need)
    monkeypatch.setattr(judgment_pipeline, "extract_layered_pdf_documents", lambda pdf_path: bad_text_documents)
    monkeypatch.setattr(judgment_pipeline, "get_configured_vision_extractor", lambda **kwargs: FakeMiniCPMExtractor())
    monkeypatch.setattr(judgment_pipeline, "build_llm_first_review_package", fake_build_llm_first_review_package)

    try:
        record = asyncio.run(
            process_judgment_file(
                user_id="user-1",
                pdf_path=str(workspace_tmp_path / "source.pdf"),
                record_id="record-vision-to-llm",
                original_file_name="source.pdf",
                source_metadata={"source_system": "test"},
                storage=storage,
                canvas_app_id="theme11-local",
                llm_enabled=True,
                processing_mode="test",
            )
        )
    finally:
        asyncio.run(storage.close())

    assert llm_seen["metadata"][0]["extraction_method"] == "vision_ocr"
    assert record["source_metadata"]["ocr_routing"] == "vision_ocr_only"
    assert record["source_metadata"]["vision_fallback_used"] is True


def test_judgment_evidence_index_retrieves_and_reranks_legal_evidence():
    from rag.judgment.retrieval import JudgmentEvidenceIndex

    documents = [
        Document(
            page_content="Background facts and submissions by learned counsel.",
            metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
        ),
        Document(
            page_content="The Urban Development Department shall file a compliance report within 30 days.",
            metadata={"source": "judgment.pdf", "page": 8, "chunk_id": "p8"},
        ),
        Document(
            page_content="No order as to costs.",
            metadata={"source": "judgment.pdf", "page": 9, "chunk_id": "p9"},
        ),
    ]

    index = JudgmentEvidenceIndex.from_documents(documents)
    results = index.search("department compliance report deadline", top_k=2)

    assert results[0].page == 8
    assert "compliance report" in results[0].snippet
    assert results[0].retrieval_score > 0
    assert results[0].match_strategy in {"hybrid_lexical", "hybrid_dense", "hybrid_fused"}


def test_duplicate_detection_finds_case_metadata_near_match(workspace_tmp_path: Path):
    from judgment_workflow.repository import JudgmentRepository

    async def run():
        storage = get_storage_backend(
            StorageConfig(backend_type="sqlite", sqlite_db_path=str(workspace_tmp_path / "judgments.db"))
        )
        await storage.initialize()
        repository = JudgmentRepository(storage=storage, canvas_app_id="test-app")

        package = build_judgment_review_package(_sample_documents(), {"source_system": "manual_upload"})
        await repository.create_record(
            user_id="user-1",
            record_id="record-1",
            review_package=package,
            source_metadata={"source_system": "manual_upload", "original_file_name": "wp-1234-2025.pdf"},
            original_pdf_path=str(workspace_tmp_path / "original.pdf"),
            document_hash="hash-1",
        )
        near_matches = await repository.find_near_duplicate_candidates(
            "user-1",
            {
                "document_hash": "different-hash",
                "case_number": "Writ Petition No. 1234 of 2025",
                "court": "High Court Of Karnataka At Bengaluru",
                "judgment_date": "2026-03-15",
                "original_file_name": "writ-petition-1234-2025-copy.pdf",
            },
            exclude_record_id="record-2",
        )

        assert near_matches
        assert near_matches[0]["record_id"] == "record-1"
        assert near_matches[0]["duplicate_score"] >= 0.75
        await storage.close()

    asyncio.run(run())


def test_metrics_summarize_evidence_review_and_duplicate_state(workspace_tmp_path: Path):
    from judgment_workflow.metrics import build_record_metrics
    from judgment_workflow.serialization import serialize_review_package

    package = build_judgment_review_package(_sample_documents())
    record = {
        "record_id": "record-1",
        **serialize_review_package(package),
        "pdf_profile": {"page_count": 2, "ocr_used": True},
        "duplicate_candidates": [{"record_id": "record-0"}],
        "processing_metrics": {"processing_ms": 1234},
    }

    metrics = build_record_metrics(
        record,
        audit_events=[
            {"event_type": "field_update"},
            {"event_type": "action_update"},
            {"event_type": "duplicate_resolution"},
        ],
    )

    assert metrics["evidence_coverage_percent"] == 100
    assert metrics["review_edit_count"] == 2
    assert metrics["duplicate_count"] == 1
    assert metrics["ocr_used"] is True


def test_detect_scanned_pdf_flags_image_only_pages(workspace_tmp_path: Path):
    from judgment_workflow.ocr import detect_ocr_need

    pdf_path = workspace_tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect((20, 20, 200, 200), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    doc.save(pdf_path)
    doc.close()

    result = detect_ocr_need(str(pdf_path))
    assert result["needs_ocr"] is True
    assert result["page_count"] == 1


def test_ccms_client_returns_mock_case_data(workspace_tmp_path: Path):
    from judgment_workflow.ccms_client import CCMSClient

    sample_pdf = workspace_tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Mock CCMS judgment")
    doc.save(sample_pdf)
    doc.close()

    client = CCMSClient(
        base_url=None,
        api_key=None,
        mock_cases={
            "CCMS-101": {
                "pdf_path": str(sample_pdf),
                "metadata": {"department": "BBMP"},
            }
        },
    )
    payload = asyncio.run(client.fetch_case("CCMS-101"))
    assert payload["metadata"]["department"] == "BBMP"
    assert payload["pdf_bytes"]


def test_evaluate_endpoint_returns_clean_review_schema(monkeypatch, workspace_tmp_path: Path):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    async def fake_process_judgment_file(**kwargs):
        assert kwargs["processing_mode"] == "evaluation"
        return {
            "record_id": kwargs["record_id"],
            "user_id": kwargs["user_id"],
            "review_status": "pending_review",
            "overall_confidence": 0.84,
            "risk_flags": ["missing_timeline"],
            "source_metadata": {
                "source_system": "evaluation_upload",
                "original_file_name": "sample.pdf",
            },
            "document_hash": "hash-123",
            "pdf_profile": {"page_count": 2, "ocr_used": False},
            "processing_metrics": {"processing_ms": 321},
            "metrics": {
                "evidence_coverage_percent": 100,
                "ambiguous_count": 1,
                "duplicate_count": 0,
            },
            "duplicate_candidates": [],
            "extraction": {
                "case_number": {
                    "field_id": "case_number",
                    "name": "case_number",
                    "value": "WP 1234/2026",
                    "confidence": 0.95,
                    "status": "pending_review",
                    "evidence": [
                        {
                            "page": 1,
                            "snippet": "WP 1234/2026",
                            "confidence": 0.95,
                            "bbox": [10, 20, 100, 40],
                            "source_id": "p1",
                        }
                    ],
                },
                "directions": [
                    {
                        "field_id": "direction-0",
                        "name": "direction",
                        "value": "File compliance report",
                        "confidence": 0.8,
                        "status": "pending_review",
                        "evidence": [],
                    }
                ],
                "risk_flags": ["missing_timeline"],
            },
            "action_items": [
                {
                    "action_id": "action-0",
                    "title": "File compliance report",
                    "responsible_department": "BBMP",
                    "category": "affidavit_report_filing",
                    "priority": "high",
                    "status": "pending",
                    "timeline": {"raw_text": "within 30 days", "due_date": None, "timeline_type": "explicit"},
                    "evidence": [
                        {
                            "page": 2,
                            "snippet": "file a compliance report within 30 days",
                            "confidence": 0.88,
                            "source_id": "p2",
                        }
                    ],
                    "ambiguity_flags": ["missing_due_date"],
                }
            ],
        }

    monkeypatch.setattr(judgment_api, "process_judgment_file", fake_process_judgment_file)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.post(
        "/judgments/evaluate",
        data={"user_id": "eval-user", "include_full_record": "false"},
        files={"file": ("sample.pdf", b"%PDF-1.4\n%", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "judgment-evaluation-v1"
    assert payload["record_id"]
    assert payload["input"]["original_file_name"] == "sample.pdf"
    assert payload["review"]["status"] == "pending_review"
    assert payload["extraction"]["fields"][0]["field"] == "case_number"
    assert payload["extraction"]["fields"][0]["evidence"][0]["page"] == 1
    assert payload["action_plan"]["items"][0]["owner"] == "BBMP"
    assert payload["quality"]["metrics"]["evidence_coverage_percent"] == 100
    assert "record" not in payload


def test_upload_endpoint_sanitizes_user_id_before_writing_pdf(monkeypatch, workspace_tmp_path):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    data_root = workspace_tmp_path / "judgment_data"
    captured = {}

    async def fake_process_judgment_file(**kwargs):
        captured.update(kwargs)
        return {"record_id": kwargs["record_id"], "user_id": kwargs["user_id"]}

    monkeypatch.setattr(judgment_api, "JUDGMENT_DATA_ROOT", data_root)
    monkeypatch.setattr(judgment_api, "process_judgment_file", fake_process_judgment_file)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.post(
        "/judgments/upload?sync=true",
        data={"user_id": "../demo reviewer"},
        files={"file": ("sample.pdf", b"%PDF-1.4\n%", "application/pdf")},
    )

    assert response.status_code == 200
    assert captured["user_id"] == "demo_reviewer"
    assert Path(captured["pdf_path"]).resolve().is_relative_to(data_root.resolve())


def test_upload_progress_endpoint_reports_real_pipeline_stages(monkeypatch):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    async def fake_process_judgment_file(**kwargs):
        progress_callback = kwargs["progress_callback"]
        progress_callback(stage="judgment_processing", message="Extracting judgment content...", pct=15.0)
        progress_callback(stage="judgment_extraction", message="Building review package...", pct=55.0)
        progress_callback(stage="complete", message="Judgment workflow complete.", pct=100.0)
        return {"record_id": kwargs["record_id"], "user_id": kwargs["user_id"]}

    monkeypatch.setattr(judgment_api, "process_judgment_file", fake_process_judgment_file)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.post(
        "/judgments/upload-progress",
        data={"user_id": "progress-user"},
        files={"file": ("sample.pdf", b"%PDF-1.4\n%", "application/pdf")},
    )

    assert response.status_code == 200
    created = response.json()
    assert created["job_id"]
    assert created["record_id"]
    assert created["state"] in {"queued", "running", "success"}

    import time

    status = {}
    for _ in range(20):
        status = client.get(f"/judgments/jobs/{created['job_id']}").json()
        if status["state"] == "success":
            break
        time.sleep(0.05)

    assert status["state"] == "success"
    assert status["stage"] == "complete"
    assert status["pct"] == 100.0
    assert status["result"]["record_id"] == created["record_id"]
    assert [stage["key"] for stage in status["stages"]] == [
        "upload",
        "judgment_processing",
        "ocr_detection",
        "judgment_extraction",
        "highlight_generation",
        "record_storage",
        "complete",
    ]
    assert status["stages"][-1]["state"] == "complete"


def test_progress_job_reports_failure(monkeypatch):
    from judgment_workflow import api as judgment_api
    from judgment_workflow.api import judgment_router

    async def fake_process_judgment_file(**kwargs):
        kwargs["progress_callback"](stage="judgment_processing", message="Extracting judgment content...", pct=15.0)
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(judgment_api, "process_judgment_file", fake_process_judgment_file)
    monkeypatch.setattr(judgment_api, "get_storage", lambda: None)

    app = FastAPI()
    app.include_router(judgment_router)
    client = TestClient(app)

    response = client.post(
        "/judgments/upload-progress",
        data={"user_id": "progress-user"},
        files={"file": ("sample.pdf", b"%PDF-1.4\n%", "application/pdf")},
    )

    assert response.status_code == 200
    created = response.json()

    import time

    status = {}
    for _ in range(20):
        status = client.get(f"/judgments/jobs/{created['job_id']}").json()
        if status["state"] == "failure":
            break
        time.sleep(0.05)

    assert status["state"] == "failure"
    assert status["stage"] == "failed"
    assert status["error"] == "pipeline exploded"
