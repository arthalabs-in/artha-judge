from datetime import date

from langchain_core.documents import Document

from rag.judgment import (
    ReviewDecision,
    ReviewStatus,
    apply_review_decision,
    build_judgment_review_package,
    filter_dashboard_records,
    normalize_evidence,
)
from rag.judgment.action_plan import build_action_plan
from rag.judgment.types import ActionItem, ExtractedField, JudgmentExtraction, SourceEvidence, Timeline


def _sample_documents():
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
            metadata={"source": "judgment.pdf", "page": 8, "chunk_id": "p8"},
        ),
    ]


def test_normalize_evidence_keeps_source_page_chunk_and_snippet():
    evidence = normalize_evidence(_sample_documents()[1], "BBMP is directed")

    assert evidence.source_id == "judgment.pdf"
    assert evidence.page == 8
    assert evidence.chunk_id == "p8"
    assert "BBMP is directed" in evidence.snippet
    assert evidence.confidence == 0.9


def test_build_review_package_extracts_case_details_and_action_items():
    package = build_judgment_review_package(
        _sample_documents(),
        case_metadata={"source_system": "mock_ccms", "ccms_case_id": "CCMS-42"},
    )

    assert package.extraction.case_number.value == "Writ Petition No. 1234 of 2025"
    assert package.extraction.court.value == "High Court Of Karnataka At Bengaluru"
    assert package.extraction.judgment_date.value == date(2026, 3, 15)
    assert package.extraction.parties.value == [
        "ABC Residents Association",
        "State of Karnataka",
        "BBMP",
    ]
    assert package.source_metadata["ccms_case_id"] == "CCMS-42"

    titles = [item.title for item in package.action_items]
    assert "Remove the encroachment" in titles
    assert "File a compliance report" in titles

    bbmp_action = next(item for item in package.action_items if item.title == "Remove the encroachment")
    assert bbmp_action.responsible_department == "BBMP"
    assert bbmp_action.timeline.raw_text == "within four weeks"
    assert bbmp_action.timeline.due_date == date(2026, 4, 12)
    assert bbmp_action.evidence[0].page == 8


def test_action_plan_infers_single_public_respondent_for_review_action():
    extraction = JudgmentExtraction(
        respondents=ExtractedField("respondents", value=["Commissioner of Customs, Mumbai"]),
        departments=ExtractedField(
            "departments",
            value=["Commissioner of Customs, Mumbai", "Director General of Health Service"],
        ),
        disposition=ExtractedField("disposition", value="allowed", raw_value="The appeal is allowed."),
        directions=[
            ExtractedField(
                "direction",
                value="Update case record to reflect the allowed appeal.",
                evidence=[SourceEvidence("judgment.pdf", 7, "p7", "The appeal is allowed.")],
            )
        ],
    )

    action = build_action_plan(extraction)[0]

    assert action.responsible_department == "Case reviewer"
    assert action.owner_source == "system_policy"
    assert action.category == "internal_review"
    assert action.requires_human_review is True
    assert action.ambiguity_flags == []


def test_action_plan_keeps_record_update_as_internal_review():
    extraction = JudgmentExtraction(
        respondents=ExtractedField("respondents", value=["Commissioner of Customs, Mumbai"]),
        disposition=ExtractedField("disposition", value="allowed", raw_value="The appeal is allowed."),
        directions=[
            ExtractedField(
                "direction",
                value="Update customs record to reflect appeal allowed.",
                evidence=[SourceEvidence("judgment.pdf", 7, "p7", "The appeal is allowed.")],
            )
        ],
    )

    action = build_action_plan(extraction)[0]

    assert action.category == "internal_review"
    assert action.responsible_department == "Case reviewer"
    assert action.owner_source == "system_policy"
    assert action.timeline.timeline_type == "not_configured"
    assert action.ambiguity_flags == []


def test_action_plan_does_not_guess_owner_when_public_entities_conflict():
    extraction = JudgmentExtraction(
        respondents=ExtractedField("respondents", value=["State of Bihar", "Board of Revenue, Bihar"]),
        departments=ExtractedField("departments", value=["State of Bihar", "Board of Revenue, Bihar"]),
        disposition=ExtractedField("disposition", value="allowed", raw_value="The appeal is allowed."),
        directions=[
            ExtractedField(
                "direction",
                value="Update case record to reflect the allowed appeal.",
                evidence=[SourceEvidence("judgment.pdf", 7, "p7", "The appeal is allowed.")],
            )
        ],
    )

    action = build_action_plan(extraction)[0]

    assert action.responsible_department == "Case reviewer"
    assert action.owner_source == "system_policy"
    assert action.category == "internal_review"
    assert action.ambiguity_flags == []


def test_action_plan_uses_remand_destination_before_public_respondent():
    extraction = JudgmentExtraction(
        respondents=ExtractedField("respondents", value=["Commissioner of Customs, Mumbai"]),
        disposition=ExtractedField("disposition", value="allowed", raw_value="The appeal is allowed."),
        directions=[
            ExtractedField(
                "direction",
                value="We set aside the impugned judgment and remit the cases to the High Court for fresh consideration.",
                evidence=[
                    SourceEvidence(
                        "judgment.pdf",
                        6,
                        "p6",
                        "we set aside the impugned judgment and remit the cases to the High Court for fresh consideration.",
                    )
                ],
            )
        ],
    )

    action = build_action_plan(extraction)[0]

    assert action.responsible_department == "High Court"
    assert action.owner_source == "remand_destination"
    assert "owner_unclear" not in action.ambiguity_flags


def test_action_plan_resolves_remit_to_high_court_per_order():
    extraction = JudgmentExtraction(
        respondents=ExtractedField("respondents", value=["Commissioner of Customs, Mumbai"]),
        directions=[
            ExtractedField(
                "direction",
                value="Update case status and remit to High Court per Supreme Court order.",
                evidence=[SourceEvidence("judgment.pdf", 6, "p6", "remit to High Court per Supreme Court order")],
            )
        ],
    )

    action = build_action_plan(extraction)[0]

    assert action.responsible_department == "High Court"
    assert action.owner_source == "remand_destination"
    assert action.timeline.timeline_type == "not_specified"
    assert "timeline_not_specified" in action.ambiguity_flags


def test_supreme_court_labeled_parties_and_filename_date_are_preferred():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "REPORTABLE\n"
                    "IN THE SUPREME COURT OF INDIA\n"
                    "WRIT PETITION (CIVIL) NO. 373 OF 2006\n"
                    "Indian Young Lawyers Association\n"
                    "Petitioner(s)\n"
                    "& Ors.\n"
                    "VERSUS\n"
                    "The State of Kerala & Ors.\n"
                    "Respondent(s)\n"
                    "Digitally signed by CHETAN KUMAR Date: 2018.09.28 15:49:38 IST\n"
                    "The Court considered an older order dated 13 October 2017."
                ),
                metadata={
                    "source": "original.pdf",
                    "original_file_name": "18956_2006_Judgement_28-Sep-2018.pdf",
                    "page": 1,
                    "chunk_id": "p1",
                },
            )
        ]
    )

    assert package.extraction.case_number.value == "Writ Petition (Civil) No. 373 of 2006"
    assert package.extraction.judgment_date.value == date(2018, 9, 28)
    assert package.extraction.petitioners.value == ["Indian Young Lawyers Association"]
    assert package.extraction.respondents.value == ["The State of Kerala"]


def test_court_extraction_ignores_supreme_court_mentions_in_prose():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "Writ Petition No. 1 of 2026\n"
                    "The High Court considered a prior Supreme Court judgment dated 20 April 1972. "
                    "The respondent shall file a compliance report within four weeks."
                ),
                metadata={"source": "judgment.pdf", "page": 3, "chunk_id": "p3"},
            )
        ]
    )

    assert package.extraction.court.value is None


def test_court_extraction_ignores_generic_tribunal_mentions_in_prose():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "Writ Petition No. 2 of 2026\n"
                    "The members composing the Tribunal were not validly appointed to the Tribunal. "
                    "The respondent shall file a compliance report within four weeks."
                ),
                metadata={"source": "judgment.pdf", "page": 4, "chunk_id": "p4"},
            )
        ]
    )

    assert package.extraction.court.value is None


def test_scr_ocr_law_report_header_extracts_court_bench_and_parties():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "292 SUPREME COURT REPORTS [1973] Supp. S.C.R.\n"
                    "KESAVANANDA v. KERALA (Hegde & Mukherjea, JJ.)\n"
                    "The circumstances of the case are considered in this opinion."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.court.value == "Supreme Court of India"
    assert package.extraction.bench.value == ["Justice Hegde", "Justice Mukherjea"]
    assert package.extraction.petitioners.value == ["Kesavananda"]
    assert package.extraction.respondents.value == ["Kerala"]


def test_scr_ocr_law_report_cleans_repeated_header_bench_variants_and_citation_dates():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "KESAVANANDA v. KERALA (Ray, J.)\n"
                    "The Court referred to an earlier order dated 3 April 1947."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            ),
            Document(
                page_content="KESAVANANDA v. KERALA (Rily, J.)\nFurther discussion follows.",
                metadata={"source": "scr.pdf", "page": 2, "chunk_id": "p2"},
            ),
        ]
    )

    assert package.extraction.court.value == "Supreme Court of India"
    assert package.extraction.bench.value == ["Justice Ray"]
    assert package.extraction.judgment_date.value is None


def test_scr_ocr_law_report_deduplicates_close_judge_ocr_variants():
    package = build_judgment_review_package(
        [
            Document(
                page_content="KESAVANANDA v. KERALA (Chandrachud, J.)\nThe opinion begins.",
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            ),
            Document(
                page_content="KESAVANANDA v. KERALA (Charniruchud, J.)\nThe reporter header repeats.",
                metadata={"source": "scr.pdf", "page": 2, "chunk_id": "p2"},
            ),
        ]
    )

    assert package.extraction.bench.value == ["Justice Chandrachud"]


def test_scr_ocr_law_report_recovers_noisy_title_line_parties_without_canonical_dictionary():
    package = build_judgment_review_package(
        [
            Document(
                page_content="KESAVANA!'DA v~ KEIV.LA (Palekar, J.)\nThe opinion begins.",
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.parties.value == ["Kesavana 'da", "KEIV.LA"]


def test_scr_ocr_law_report_prefers_report_title_over_citation_v_match():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "632 SUfllEME oom:r lil'O&TS. [1973] Supp, S.C.R.\n"
                    "See for example N. B. Jeejeebhoy v. Assistl(nt . Collector, Thana(').\n"
                    "ltESAVA]!qAM>,\\ u. KBIALA (KlilHlna, J.)\n"
                    "The necessary facts may now be set out."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.parties.value == ["Qam", "Kbiala"]
    assert package.extraction.bench.value == ["Justice Klilhlna"]


def test_scr_ocr_law_report_prefers_title_over_counsel_list():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "86a ORIGINAL JURISDICTION: Writ Petition No. 135'of 1970.\n"
                    "For Respondent No. 1: Advocate-General for the State of Kerala.\n"
                    "USAVANANDA II. KEIW.A (Sikri, C.J.)\n"
                    "These cases raise questions of constitutional amendment."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.parties.value == ["Usavananda", "KEIW.A"]
    assert package.extraction.bench.value == ["Chief Justice Sikri"]


def test_scr_ocr_law_report_prose_disposition_does_not_create_review_action():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "KESAVANANDA v. KERALA (Mathew, J.)\n"
                    "The struggle for a just economic order should be allowed to take priority "
                    "over the struggle for individual rights."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            ),
            Document(
                page_content=(
                    "SUPREME COURT REPORTS [1973] Supp. S.C.R.\n"
                    "The argument, if it be allowed, may be repeated in later cases. "
                    "The provision was described elsewhere as dis- allowed."
                ),
                metadata={"source": "scr.pdf", "page": 2, "chunk_id": "p2"},
            ),
        ]
    )

    assert package.extraction.disposition.value == "unknown"
    assert package.action_items == []


def test_scr_ocr_law_report_historical_dismissal_does_not_create_review_action():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "KESAVANANDA v. KERALA (Chandrachud, J.)\n"
                    "Charles had to summon a new Parliament and one grievance was dismissed. "
                    "The passage is historical background, not the operative order."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.disposition.value == "unknown"
    assert package.action_items == []


def test_scr_ocr_law_report_collapses_close_repeated_header_variants():
    package = build_judgment_review_package(
        [
            Document(
                page_content="KESAVANANDA v. KERALA (Beg, J.)\nThe opinion begins.",
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            ),
            Document(
                page_content="XESAVANANDA II. JtEJW.A (Dwivcdi, J.)\nThe header repeats.",
                metadata={"source": "scr.pdf", "page": 2, "chunk_id": "p2"},
            ),
        ]
    )

    assert package.extraction.bench.value == ["Justice Beg", "Justice Dwivcdi"]


def test_scr_ocr_law_report_cost_and_reporter_prose_do_not_create_actions():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "KESAVANANDA v. KERALA (Ray, J.)\n"
                    "In the circumstances of the case we direct the parties to bear their own costs "
                    "in these cases up till this stage."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            ),
            Document(
                page_content=(
                    "The amendment, namely that which is directed to removing from art. 31, "
                    "was considered by the Court."
                ),
                metadata={"source": "scr.pdf", "page": 2, "chunk_id": "p2"},
            ),
        ]
    )

    assert package.action_items == []


def test_scr_ocr_law_report_clouding_issues_prose_does_not_create_action():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "KESAVANANDA v. KERALA (Hegde & Mukherjea, JJ.)\n"
                    "Hence in our opinion, we will be clouding the issues, if we allow ourselves "
                    "to be drawn into the larger debate."
                ),
                metadata={"source": "scr.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.action_items == []


def test_supreme_court_record_proceedings_date_and_advocates_are_extracted():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "IN THE SUPREME COURT OF INDIA\n"
                    "CIVIL APPEAL NO. 2245 OF 2009\n"
                    "(From the judgment and order dated 01/10/2008 in CA No. 52/2008)\n"
                    "DR.B.N.HOSPITAL & N.HOSPITAL RES.CENTRE Petitioner(s)\n"
                    "VERSUS\n"
                    "COMMISSIONER OF CUSTOMS, MUMBAI Respondent(s)\n"
                    "Date: 08/04/2009 This Petition was called on for hearing today.\n"
                    "CORAM : HON'BLE MR. JUSTICE S.H. KAPADIA\n"
                    "HON'BLE MR. JUSTICE AFTAB ALAM\n"
                    "For Petitioner(s) Mr. Bharat Sangal, Adv.\n"
                    "Mr. Prasenjit Das, Adv.\n"
                    "Ms. Mrinalini Oinam, Adv.\n"
                    "For Respondent(s) UPON hearing counsel the Court made the following ORDER"
                ),
                metadata={"source": "judgment.pdf", "page": 7, "chunk_id": "p7"},
            )
        ]
    )

    assert package.extraction.judgment_date.value == date(2009, 4, 8)
    assert "Mr. Bharat Sangal, Adv." in package.extraction.advocates.value
    assert "Mr. Prasenjit Das, Adv." in package.extraction.advocates.value
    assert "UPON hearing counsel" not in package.extraction.advocates.value


def test_supreme_court_signature_block_extracts_full_bench():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "IN THE SUPREME COURT OF INDIA\n"
                    "CIVIL APPEAL NOS. 3778-3780 OF 2016\n"
                    "J U D G M E N T\n"
                    "R. BANUMATHI, J.\n"
                    "The appeals are accordingly allowed.\n"
                    "The parties to bear their respective costs.\n"
                    "...................CJI. (T.S. THAKUR)\n"
                    "......................J. (R. BANUMATHI)\n"
                    ".....................J. (UDAY UMESH LALIT)\n"
                    "New Delhi; April 12, 2016"
                ),
                metadata={"source": "judgment.pdf", "page": 11, "chunk_id": "p11"},
            )
        ]
    )

    assert package.extraction.bench.value == [
        "Chief Justice T.S. Thakur",
        "Justice R. Banumathi",
        "Justice Uday Umesh Lalit",
    ]
    assert package.extraction.bench.requires_review is False


def test_bench_extraction_ignores_before_in_ordinary_prose():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "IN THE SUPREME COURT OF INDIA\n"
                    "CIVIL APPEAL NO. 1 OF 2026\n"
                    "The revision was filed before the Board of Revenue.\n"
                    "The matter came before the Divisional Commissioner after hearing parties.\n"
                    "J U D G M E N T\n"
                    "R. BANUMATHI, J."
                ),
                metadata={"source": "judgment.pdf", "page": 1, "chunk_id": "p1"},
            )
        ]
    )

    assert package.extraction.bench.value == []


def test_bracketed_signature_block_extracts_bench_without_coram():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "IN THE SUPREME COURT OF INDIA\n"
                    "CIVIL APPEAL NO. 1410 OF 2007\n"
                    "J U D G M E N T\n"
                    "Dipak Misra, J.\n"
                    "We allow the appeals and set aside all the impugned orders.\n"
                    "...............................J.\n"
                    "[Dipak Misra]\n"
                    "...............................J.\n"
                    "[Shiva Kirti Singh]\n"
                    "New Delhi. March 29, 2016."
                ),
                metadata={"source": "judgment.pdf", "page": 28, "chunk_id": "p28"},
            )
        ]
    )

    assert package.extraction.bench.value == [
        "Justice Dipak Misra",
        "Justice Shiva Kirti Singh",
    ]


def test_body_text_counsel_names_are_extracted_as_reviewable_advocates():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "Mr. Balbir Singh, learned senior counsel appearing for\n"
                    "the appellant, submitted that the notification applies.\n"
                    "Mr. Singh later drew attention to the proviso.\n"
                    "11. Mr. Sanjay Kumar Visen, learned counsel for the"
                ),
                metadata={"source": "judgment.pdf", "page": 12, "chunk_id": "p12"},
            ),
            Document(
                page_content=(
                    "Page 13\n"
                    "JUDGMENT\n"
                    "respective respondent(s), per contra, supported the order.\n"
                    "Learned counsel would submit that strict construction applies."
                ),
                metadata={"source": "judgment.pdf", "page": 13, "chunk_id": "p13"},
            ),
        ]
    )

    assert "Mr. Balbir Singh, learned senior counsel for appellant" in package.extraction.advocates.value
    assert "Mr. Sanjay Kumar Visen, learned counsel for respondent" in package.extraction.advocates.value
    assert all("Mr. Singh" not in advocate for advocate in package.extraction.advocates.value)
    assert package.extraction.advocates.requires_review is True


def test_final_operative_disposition_beats_procedural_history():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "The tribunal dismissed the appeal by its order dated September 9, 2002. "
                    "The High Court dismissed the writ petition thereafter."
                ),
                metadata={"source": "judgment.pdf", "page": 3, "chunk_id": "p3"},
            ),
            Document(
                page_content=(
                    "In view of aforesaid analysis, we allow the appeals and set aside all "
                    "the impugned orders. There shall be no order as to costs."
                ),
                metadata={"source": "judgment.pdf", "page": 28, "chunk_id": "p28"},
            ),
        ]
    )

    assert package.extraction.disposition.value == "allowed"
    assert "allow the appeals" in package.extraction.disposition.raw_value.lower()


def test_final_disposition_prefers_allowed_outcome_over_remand_direction():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "In the result, the impugned order is set aside and the matter is "
                    "remitted back to the Board of Revenue. The appeals are accordingly allowed."
                ),
                metadata={"source": "judgment.pdf", "page": 11, "chunk_id": "p11"},
            )
        ]
    )

    assert package.extraction.disposition.value == "allowed"


def test_final_order_grant_leave_beats_prior_dismissal_history():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "The High Court held that we do not find any merit in this writ "
                    "petition and it is liable to be dismissed."
                ),
                metadata={"source": "judgment.pdf", "page": 6, "chunk_id": "p6"},
            ),
            Document(
                page_content=(
                    "Accordingly, we grant leave and tag this appeal with "
                    "C.A.No.7295 of 2012 and C.A.No.11895 of 2014."
                ),
                metadata={"source": "judgment.pdf", "page": 8, "chunk_id": "p8"},
            ),
        ]
    )

    assert package.extraction.disposition.value == "leave_granted"
    assert "grant leave" in package.extraction.disposition.raw_value.lower()
    assert package.extraction.disposition.evidence[0].page == 8


def test_signature_block_extracts_multiple_judges_with_date_between():
    package = build_judgment_review_package(
        [
            Document(
                page_content=(
                    "IN THE SUPREME COURT OF INDIA\n"
                    "Petition for Special Leave to Appeal (C) No. 19898 of 2014\n"
                    "J U D G M E N T\n"
                    "Madan B. Lokur, J.\n"
                    "Accordingly, we grant leave and tag this appeal with C.A.No.7295 of 2012.\n"
                    ".……………………..J\n"
                    "(Madan B. Lokur)\n"
                    "..……………………J\n"
                    "March 28, 2016\n"
                    "(S. A. Bobde)"
                ),
                metadata={"source": "judgment.pdf", "page": 8, "chunk_id": "p8"},
            )
        ]
    )

    assert package.extraction.bench.value == [
        "Justice Madan B. Lokur",
        "Justice S. A. Bobde",
    ]


def test_tag_connected_appeal_uses_registry_owner_not_party():
    from rag.judgment.owner_resolution import apply_inferred_action_owner

    action = ActionItem(
        title="Tag civil appeal C.A. No.7295 of 2012 and Civil Appeal no. 11895 of 2014",
        responsible_department="Union of India",
        timeline=Timeline(raw_text="C.A. No.7295 of 2012", timeline_type="date", confidence=0.4),
        category="direct_compliance",
        legal_basis=(
            "Accordingly, we grant leave and tag this appeal with "
            "C.A.No.7295 of 2012 and C.A.No.11895 of 2014."
        ),
    )

    resolved = apply_inferred_action_owner(action, JudgmentExtraction())

    assert resolved.responsible_department == "Registry"
    assert resolved.owner_source == "procedural_registry"
    assert resolved.timeline.timeline_type == "not_specified"


def test_review_decision_edit_approves_package_and_dashboard_projection():
    package = build_judgment_review_package(_sample_documents())
    edited_first_action = package.action_items[0]
    edited_first_action.responsible_department = "Bruhat Bengaluru Mahanagara Palike"

    reviewed = apply_review_decision(
        package,
        ReviewDecision(
            decision="edit",
            reviewer_id="reviewer-1",
            notes="Expanded department name.",
            action_items=[edited_first_action],
        ),
    )

    records = filter_dashboard_records([package, reviewed])

    assert reviewed.review_status == ReviewStatus.EDITED
    assert reviewed.reviewer_id == "reviewer-1"
    assert len(records) == 1
    assert records[0].case_number == "Writ Petition No. 1234 of 2025"
    assert records[0].departments == ["Bruhat Bengaluru Mahanagara Palike"]
    assert records[0].pending_actions == ["Remove the encroachment"]


def test_rejected_package_is_excluded_from_dashboard():
    package = build_judgment_review_package(_sample_documents())

    reviewed = apply_review_decision(
        package,
        ReviewDecision(
            decision="reject",
            reviewer_id="reviewer-1",
            notes="Wrong source document.",
        ),
    )

    assert reviewed.review_status == ReviewStatus.REJECTED
    assert filter_dashboard_records([reviewed]) == []
